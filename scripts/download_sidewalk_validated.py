#!/usr/bin/env python3
"""Download the contents of the validated tagger dataset, preserving the archives.

Run in server-side tmux. Partial downloads resume using Hugging Face's local
download cache. Completed files are verified against the published SHA-256.
"""

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.request


REPO = "projectsidewalk/sidewalk-tagger-ai-validated"
OUT = Path("/data/datasets/sidewalk-data")
MANIFEST = Path("/data/logs/sidewalk-tagger-validated-manifest.json")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_XET_CACHE", str(OUT / ".download" / "xet"))

from huggingface_hub import hf_hub_download


def metadata(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response), response.headers.get("Link", "")


def download_file(entry, revision):
    relative = Path(entry["path"]).relative_to("Validated")
    destination = OUT / relative
    expected_hash = entry.get("lfs", {}).get("oid")
    print(f"START {relative}: {entry['size']:,} bytes", flush=True)
    for attempt in range(1, 6):
        try:
            source = destination if destination.is_file() else Path(hf_hub_download(
                repo_id=REPO, repo_type="dataset", revision=revision,
                filename=entry["path"], local_dir=OUT / ".download", token=False,
            ))
            if source.stat().st_size != entry["size"]:
                raise ValueError(f"Size mismatch for {relative}")
            sha = hashlib.sha256()
            with source.open("rb") as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    sha.update(chunk)
            if expected_hash and sha.hexdigest() != expected_hash:
                raise ValueError(f"Checksum mismatch for {relative}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source != destination:
                source.replace(destination)
            print(f"DONE {relative}: size and SHA-256 verified", flush=True)
            return str(relative)
        except Exception as exc:
            print(f"ATTEMPT {attempt}/5 {relative}: {type(exc).__name__}: {exc}", flush=True)
            if attempt == 5:
                raise
            time.sleep(min(15 * attempt, 60))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Reuse a pinned revision on restart rather than mixing repository versions.
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
        revision, files = manifest["revision"], manifest["files"]
    else:
        info, _ = metadata(f"https://huggingface.co/api/datasets/{REPO}")
        revision = info["sha"]
        url = f"https://huggingface.co/api/datasets/{REPO}/tree/{revision}/Validated?recursive=true&limit=1000"
        files = []
        while url:
            entries, links = metadata(url)
            files.extend(entry for entry in entries if entry["type"] == "file")
            following = re.search(r'<([^>]+)>;\s*rel="next"', links)
            url = following.group(1) if following else None
        if not files:
            raise ValueError("The upstream Validated folder has no files")
        for entry in files:
            relative = Path(entry["path"]).relative_to("Validated")
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Invalid upstream file path")
        manifest = {"repository": REPO, "revision": revision, "files": files, "completed": []}
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"SOURCE {REPO} revision={revision}", flush=True)
    print(f"DOWNLOAD {len(files)} files, {sum(f['size'] for f in files):,} bytes -> {OUT}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download_file, entry, revision) for entry in files]
        for future in concurrent.futures.as_completed(futures):
            name = future.result()
            manifest["completed"] = sorted(set(manifest["completed"]) | {name})
            MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print("ALL DOWNLOADS COMPLETE AND VERIFIED", flush=True)


if __name__ == "__main__":
    main()
