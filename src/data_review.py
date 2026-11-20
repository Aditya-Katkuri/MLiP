#!/usr/bin/env python3
"""Manually assign dataset classes, saving a CSV in each image directory.

Run: python src/data_review.py --port 8768
Forward port 8768 in VS Code Remote SSH, then open http://localhost:8768.
No third-party packages are required. Images and dataset manifests are not edited.
"""

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import stat
import tempfile
from urllib.parse import parse_qs, quote, urlsplit

CSV_NAME = 'manual_class_labels.csv'
CLASSES = ('Crosswalk', 'CurbRamp', 'Obstacle', 'SurfaceProblem', 'no_obstacles', 'remove_data')
LABELS = set(CLASSES)
IMAGE_EXTENSIONS = {'.png', '.webp', '.jpg', '.jpeg', '.bmp', '.gif'}
SPLITS = ('other-cities-train', 'pitt-train', 'pitt-val', 'pitt-test')


class ReviewError(Exception):
    def __init__(self, message, status=400, **details):
        super().__init__(message)
        self.status, self.details = status, details


def natural_key(value):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r'(\d+)', value)]


class ReviewStore:
    def __init__(self, root):
        self.root = Path(root).resolve(strict=True)

    def directory(self, name):
        parts = name.split('/') if isinstance(name, str) else []
        if len(parts) != 2 or parts[0] not in SPLITS or parts[1] in {'', '.', '..'}:
            raise ReviewError('Choose a valid split and class directory.')
        path = self.root.joinpath(*parts)
        if path.is_symlink() or path.parent.is_symlink() or not path.is_dir():
            raise ReviewError('Class directory not found.', 404)
        if not path.resolve().is_relative_to(self.root):
            raise ReviewError('Directory is outside the dataset.', 400)
        return path

    def image(self, directory, filename):
        if not isinstance(filename, str) or filename in {'', '.', '..'} or Path(filename).name != filename:
            raise ReviewError('Invalid image filename.')
        path = directory / filename
        if path.is_symlink() or not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ReviewError('Image not found.', 404)
        return path

    def filenames(self, directory):
        return sorted((p.name for p in directory.iterdir()
                       if p.is_file() and not p.is_symlink() and p.suffix.lower() in IMAGE_EXTENSIONS), key=natural_key)

    def directories(self):
        result = []
        for split in SPLITS:
            parent = self.root / split
            if parent.is_symlink() or not parent.is_dir():
                continue
            for path in sorted(parent.iterdir()):
                if path.is_dir() and not path.is_symlink():
                    result.append({'id': f'{split}/{path.name}', 'split': split, 'class': path.name,
                                   'image_count': len(self.filenames(path))})
        return {'root': str(self.root), 'csv_name': CSV_NAME, 'classes': list(CLASSES), 'directories': result}

    @contextmanager
    def locked(self, directory):
        lock = directory / '.manual_class_labels.lock'
        fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o644)
        with os.fdopen(fd, 'a') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def read_labels(self, directory):
        path = directory / CSV_NAME
        if path.is_symlink():
            raise ReviewError('The labels CSV must be a regular file.')
        if not path.exists():
            return ['filename', 'label', 'updated_at'], {}
        with path.open(newline='', encoding='utf-8-sig') as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            if not fields or len(set(fields)) != len(fields) or not {'filename', 'label'} <= set(fields):
                raise ReviewError(f'{CSV_NAME} needs unique filename and label columns; the existing file was left unchanged.')
            rows = {}
            for row in reader:
                filename = row.get('filename')
                if (None in row or any(v is None for v in row.values()) or not filename
                        or Path(filename).name != filename or filename in rows
                        or row['label'] not in LABELS | {''}):
                    raise ReviewError(f'Invalid or duplicate row in {CSV_NAME}; the existing file was left unchanged.')
                rows[filename] = row
        return fields, rows

    def listing(self, name):
        directory = self.directory(name)
        with self.locked(directory):
            _, rows = self.read_labels(directory)
        images = []
        for filename in self.filenames(directory):
            row = rows.get(filename, {})
            images.append({'filename': filename, 'label': row.get('label') or None,
                           'updated_at': row.get('updated_at', ''),
                           'url': f'/image?directory={quote(name, safe="")}&filename={quote(filename, safe="")}'})
        return {'directory': name, 'csv_path': str(directory / CSV_NAME), 'images': images}

    def save(self, name, filename, label, expected_label):
        directory = self.directory(name)
        self.image(directory, filename)
        if not isinstance(label, str) or label not in LABELS:
            raise ReviewError('Label must be one of: ' + ', '.join(CLASSES) + '.')
        if expected_label is not None and (not isinstance(expected_label, str) or expected_label not in LABELS):
            raise ReviewError('Invalid previous label.')
        with self.locked(directory):
            fields, rows = self.read_labels(directory)
            existing = rows.get(filename, {})
            current = existing.get('label') or None
            if current != expected_label:
                raise ReviewError('This image was labeled in another session. Its latest label is now shown; choose again to change it.',
                                  409, label=current, updated_at=existing.get('updated_at', ''))
            if current == label:
                return {'filename': filename, 'label': label, 'updated_at': existing.get('updated_at', '')}
            if 'updated_at' not in fields:
                fields = fields + ['updated_at']
            updated_at = datetime.now(timezone.utc).isoformat(timespec='microseconds')
            rows[filename] = {**dict.fromkeys(fields, ''), **existing,
                              'filename': filename, 'label': label, 'updated_at': updated_at}
            destination = directory / CSV_NAME
            temporary = None
            try:
                with tempfile.NamedTemporaryFile('w', newline='', encoding='utf-8', dir=directory,
                                                 prefix='.manual_class_labels.', suffix='.tmp', delete=False) as handle:
                    temporary = Path(handle.name)
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows[key] for key in sorted(rows, key=natural_key))
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(temporary, stat.S_IMODE(destination.stat().st_mode) if destination.exists() else 0o644)
                os.replace(temporary, destination)
                directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                if temporary is not None and temporary.exists():
                    temporary.unlink()
            return {'filename': filename, 'label': label, 'updated_at': updated_at}


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, store, html, **kwargs):
        self.store, self.html = store, html
        super().__init__(*args, **kwargs)

    def send(self, body, content_type='application/json; charset=utf-8', status=200):
        if isinstance(body, dict):
            body = json.dumps(body, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store' if content_type.startswith('application/json') else 'no-cache')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()
        if self.command != 'HEAD':
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def run_request(self, action):
        try:
            action()
        except ReviewError as error:
            self.send({'error': str(error), **error.details}, status=error.status)
        except (OSError, csv.Error, UnicodeError) as error:
            self.log_error('%s', error)
            self.send({'error': 'Could not read or save the file. Check the CSV format and directory permissions.'}, status=500)

    def do_GET(self):
        self.run_request(self.get)

    def do_HEAD(self):
        self.run_request(self.get)

    def get(self):
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        name = query.get('directory', [''])[0]
        if url.path == '/':
            return self.send(self.html, 'text/html; charset=utf-8')
        if url.path == '/health':
            return self.send({'status': 'ok'})
        if url.path == '/api/directories':
            return self.send(self.store.directories())
        if url.path == '/api/images':
            return self.send(self.store.listing(name))
        if url.path == '/image':
            path = self.store.image(self.store.directory(name), query.get('filename', [''])[0])
            return self.send(path.read_bytes(), mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
        raise ReviewError('Not found.', 404)

    def do_POST(self):
        self.run_request(self.post)

    def post(self):
        if urlsplit(self.path).path != '/api/label':
            raise ReviewError('Not found.', 404)
        origin = self.headers.get('Origin')
        if origin and urlsplit(origin).netloc != self.headers.get('Host'):
            raise ReviewError('Cross-origin saves are not allowed.', 403)
        if self.headers.get_content_type() != 'application/json':
            raise ReviewError('Send JSON.', 415)
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 8192:
                raise ValueError()
            body = json.loads(self.rfile.read(size))
        except (ValueError, UnicodeError):
            raise ReviewError('Invalid JSON request.')
        if not isinstance(body, dict) or not {'directory', 'filename', 'label', 'expected_label'} <= body.keys():
            raise ReviewError('Missing image or label information.')
        result = self.store.save(body['directory'], body['filename'], body['label'], body['expected_label'])
        self.send(result)

    def log_message(self, fmt, *args):
        if len(args) > 1 and str(args[1]) == '200':
            return
        super().log_message(fmt, *args)


def make_server(root, host='127.0.0.1', port=8768):
    store = ReviewStore(root)
    html = Path(__file__).with_suffix('.html').read_bytes()
    return ThreadingHTTPServer((host, port), partial(Handler, store=store, html=html))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data/datasets/sidewalk-data'))
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8768)
    args = parser.parse_args()
    with make_server(args.root, args.host, args.port) as server:
        print(f'Class review: http://{args.host}:{server.server_port}', flush=True)
        print(f'Dataset: {args.root}; labels: <split>/<class>/{CSV_NAME}', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
