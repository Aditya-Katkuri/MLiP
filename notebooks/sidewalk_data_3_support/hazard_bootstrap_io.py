"""Dataset download and result ZIPs without mounting or authenticating Google Drive."""
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit
import hashlib
import json
import os
import stat
import tempfile
import zipfile


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download_drive_archive(url, destination, expected_sha256):
    """Download a publicly readable file; never load browser/account cookies."""
    destination = Path(destination)
    if destination.exists():
        if file_sha(destination) != expected_sha256:
            raise ValueError(f'{destination} exists but is not the expected archive; move it aside before downloading.')
        return destination
    parsed = urlsplit(str(url).strip())
    if parsed.scheme != 'https' or parsed.hostname not in {'drive.google.com', 'docs.google.com'}:
        raise ValueError('Paste the HTTPS Google Drive share link for sidewalk-data-3.tar.')
    if '/folders/' in parsed.path:
        raise ValueError('Supply the archive file link, not a folder link.')
    import gdown
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name+'.download')
    try:
        result = gdown.download(url=str(url).strip(), output=str(partial), use_cookies=False, quiet=False, resume=True)
    except Exception:
        raise RuntimeError('Could not download the archive anonymously. The file must allow Anyone with the link / Viewer. '
                           'Alternatively upload it to the Colab runtime and select DATA_SOURCE="local_archive". '
                           'No Google sign-in or browser cookies are used.') from None
    if result is None or not partial.is_file():
        raise RuntimeError('No archive was downloaded. Check the file link, or use DATA_SOURCE="local_archive".')
    if file_sha(partial) != expected_sha256:
        partial.unlink(missing_ok=True)
        raise ValueError('Downloaded file failed the pinned dataset SHA-256 check; it is not the expected dataset-3 archive.')
    os.replace(partial, destination)
    return destination


def make_results_zip(run_dir, destination_dir):
    """Archive only the run directory, including weights, features and provenance."""
    root = Path(run_dir).resolve()
    destination_dir = Path(destination_dir).resolve()
    assert (root/'config.json').is_file(), 'Expected a run directory containing config.json.'
    assert not destination_dir.is_relative_to(root), 'Put downloads outside the run directory.'
    paths = sorted(p for p in root.rglob('*') if p.is_file() and not p.name.endswith('.tmp'))
    assert all(not p.is_symlink() and p.resolve().is_relative_to(root) for p in paths)
    inventory = {str(p.relative_to(root)):dict(bytes=p.stat().st_size, sha256=file_sha(p))
                 for p in paths if p.name != 'archive_manifest.json'}
    (root/'archive_manifest.json').write_text(json.dumps(inventory, indent=2)+'\n')
    paths = sorted(p for p in root.rglob('*') if p.is_file() and not p.name.endswith('.tmp'))
    destination_dir.mkdir(parents=True, exist_ok=True)
    output = destination_dir/(root.name+'.zip')
    pending = output.with_name(output.name+'.tmp')
    try:
        with zipfile.ZipFile(pending, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
            for path in paths:
                assert not path.is_symlink() and path.resolve().is_relative_to(root)
                archive.write(path, arcname=str(Path(root.name)/path.relative_to(root)))
        with zipfile.ZipFile(pending) as archive:
            assert archive.testzip() is None, 'ZIP integrity check failed.'
        os.replace(pending, output)
    finally:
        pending.unlink(missing_ok=True)
    checksum = file_sha(output)
    output.with_suffix('.zip.sha256').write_text(f'{checksum}  {output.name}\n')
    return output


def restore_results_zip(archive_path, runs_root):
    """Restore a previously downloaded run ZIP into a fresh runtime."""
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        names = []
        for info in archive.infolist():
            p = PurePosixPath(info.filename)
            if p.is_absolute() or '..' in p.parts or '\\' in info.filename or not p.parts:
                raise ValueError('Unsafe path in results archive.')
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError('Results archive must not contain symbolic links.')
            names.append(p)
        roots = {p.parts[0] for p in names}
        if len(roots) != 1:
            raise ValueError('Results archive must contain exactly one run directory.')
        run_name = next(iter(roots))
        if f'{run_name}/config.json' not in archive.namelist() or f'{run_name}/archive_manifest.json' not in archive.namelist():
            raise ValueError('Results ZIP lacks its configuration or integrity manifest.')
        target = runs_root/run_name
        if target.exists():
            raise FileExistsError(f'{target} already exists; set RESUME_RUN_DIR to it instead.')
        with tempfile.TemporaryDirectory(dir=runs_root, prefix='.restore-') as temporary:
            archive.extractall(temporary)
            candidate = Path(temporary)/run_name
            inventory = json.loads((candidate/'archive_manifest.json').read_text())
            for relative, expected in inventory.items():
                path = candidate/relative
                if not path.resolve().is_relative_to(candidate.resolve()) or not path.is_file():
                    raise ValueError('Invalid inventory path in results ZIP.')
                if path.stat().st_size != expected['bytes'] or file_sha(path) != expected['sha256']:
                    raise ValueError(f'Result failed integrity verification: {relative}')
            candidate.rename(target)
    return target
