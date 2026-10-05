#!/usr/bin/env python3
"""Read-only image review dashboard for saved Pittsburgh VLM predictions.

Run from the repository (no extra Python packages or API key required):
    python src/baseline/vlm_dashboard.py --port 8765

Forward port 8765 using VS Code Remote SSH's Ports panel, then open its URL.
The server binds to loopback by default. It serves only its HTML, prediction
metadata, and the specific dataset images referenced by the prediction CSV.
It neither modifies results nor calls the OpenAI API.
"""

import argparse
import csv
import json
import math
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


CLASSES = ["missing_curb_ramp", "obstacle", "surface_problem", "no_barrier"]
DEFAULT_RUN = Path("/data/eda/openai_baseline/gpt61_sol_low_001")
DEFAULT_DATA = Path("/data/datasets")
DIRS = {"nocurbramp", "obstacle", "surfaceproblem"}
IDENTITY = ["dir", "split", "verdict", "city", "label_id"]


def load_run(run_dir, data_root):
    """Validate saved rows and whitelist the images the HTTP server may serve."""
    data_root = data_root.resolve()
    run_dir = run_dir.resolve()
    config_path = run_dir / "run_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    required = {*IDENTITY, "y", "predicted_class", *("p_" + c for c in CLASSES)}
    items, image_paths, seen = [], [], set()
    matrix = [[0 for _ in CLASSES] for _ in CLASSES]
    with (run_dir / "pittsburgh_predictions.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        if required - set(reader.fieldnames or []):
            raise ValueError(f"Missing CSV columns: {sorted(required - set(reader.fieldnames or []))}")
        for row in reader:
            if row["city"] != "pittsburgh":
                raise ValueError("This dashboard expects the Pittsburgh evaluation CSV.")
            if row["dir"] not in DIRS or row["split"] not in {"train", "val", "test"}:
                raise ValueError("Unexpected dataset or split in prediction CSV.")
            if row["verdict"] not in {"correct", "incorrect"} or not row["label_id"].isdigit():
                raise ValueError("Unexpected verdict or label ID in prediction CSV.")
            if row["y"] not in CLASSES or row["predicted_class"] not in CLASSES:
                raise ValueError("Unknown class in prediction CSV.")
            probabilities = {c: float(row["p_" + c]) for c in CLASSES}
            if not all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities.values()):
                raise ValueError(f"Invalid probabilities for label {row['label_id']}.")
            if not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-6, rel_tol=0):
                raise ValueError(f"Probabilities do not sum to one for label {row['label_id']}.")
            if max(CLASSES, key=probabilities.get) != row["predicted_class"]:
                raise ValueError(f"Prediction disagrees with probabilities for label {row['label_id']}.")
            relative = Path(f"sidewalk-validator-ai-dataset-{row['dir']}") / row["split"] / row["verdict"] / f"{row['city']}_{row['label_id']}.webp"
            path = (data_root / relative).resolve()
            if not path.is_relative_to(data_root) or str(relative) in seen:
                raise ValueError("Unsafe or duplicate image path in prediction CSV.")
            seen.add(str(relative))
            index = len(items)
            items.append({
                **{key: row[key] for key in IDENTITY},
                "id": index, "truth": row["y"], "prediction": row["predicted_class"],
                "probabilities": probabilities,
                "confidence": probabilities[row["predicted_class"]],
                "correct": row["y"] == row["predicted_class"],
                "image_url": f"/image/{index}", "image_path": str(path),
                "image_available": path.is_file(),
            })
            image_paths.append(path)
            matrix[CLASSES.index(row["y"])][CLASSES.index(row["predicted_class"]) ] += 1
    if not items:
        raise ValueError("The prediction CSV contains no images.")
    correct = sum(item["correct"] for item in items)
    payload = {
        "run": run_dir.name, "run_dir": str(run_dir),
        "model": config.get("model_requested", "Unknown model"),
        "reasoning": config.get("reasoning_effort", "unspecified"),
        "detail": config.get("image_detail", "unspecified"),
        "prompt": config.get("prompt", "No prompt was recorded."),
        "created_at": config.get("created_at"), "classes": CLASSES,
        "total": len(items), "correct": correct, "incorrect": len(items) - correct,
        "accuracy": correct / len(items), "matrix": matrix,
        "missing_images": sum(not item["image_available"] for item in items),
        "items": items,
    }
    return json.dumps(payload, allow_nan=False).encode(), image_paths


class DashboardHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, html, payload, image_paths, **kwargs):
        self.html, self.payload, self.image_paths = html, payload, image_paths
        super().__init__(*args, **kwargs)

    def do_GET(self):
        self.serve()

    def do_HEAD(self):
        self.serve(head=True)

    def serve(self, head=False):
        path = urlsplit(self.path).path
        if path == "/":
            body, content_type, cache = self.html, "text/html; charset=utf-8", "no-cache"
        elif path == "/api/predictions":
            body, content_type, cache = self.payload, "application/json; charset=utf-8", "no-store"
        elif path == "/health":
            body, content_type, cache = b'{"status":"ok"}', "application/json", "no-store"
        elif path.startswith("/image/") and path.removeprefix("/image/").isdigit():
            index = int(path.removeprefix("/image/"))
            if index >= len(self.image_paths):
                self.send_error(404, "Unknown image")
                return
            try:
                body = self.image_paths[index].read_bytes()
            except OSError:
                self.send_error(404, "Image is unavailable on this server")
                return
            content_type, cache = "image/webp", "private, max-age=3600"
        elif path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        else:
            self.send_error(404, "Not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()
        if not head:
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    def log_message(self, fmt, *args):
        # Keep the server log useful without logging every thumbnail request.
        if len(args) >= 2 and str(args[1]) == "200":
            return
        super().log_message(fmt, *args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    payload, images = load_run(args.run_dir, args.data_root)
    html = Path(__file__).with_suffix(".html").read_bytes()
    handler = partial(DashboardHandler, html=html, payload=payload, image_paths=images)
    with ThreadingHTTPServer((args.host, args.port), handler) as server:
        print(f"VLM dashboard: http://{args.host}:{args.port} ({len(images)} images)", flush=True)
        print(f"Reading {args.run_dir}; forward port {args.port} in VS Code.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
