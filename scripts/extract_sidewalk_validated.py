#!/usr/bin/env python3
"""Extract the verified validated tagger ZIPs, retaining the original archives."""

import concurrent.futures
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import threading
import time
import zipfile


ROOT = Path("/data/datasets/sidewalk-data")
LOG_ROOT = Path("/data/logs")
MANIFEST = LOG_ROOT / "sidewalk-tagger-validated-manifest.json"
PROGRESS = LOG_ROOT / "sidewalk-tagger-validated-extraction.json"
LOCK = threading.Lock()
STATE = {}


def save_progress():
    # Caller holds LOCK. Atomic replacement permits safe progress polling.
    temporary = PROGRESS.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(STATE, indent=2) + "\n")
    temporary.replace(PROGRESS)


def extract_one(path, staging):
    name = path.stem
    finished, byte_count = 0, 0
    with zipfile.ZipFile(path) as archive:
        for entry in archive.infolist():
            member = PurePosixPath(entry.filename)
            if member.is_absolute() or ".." in member.parts or "\\" in entry.filename:
                raise ValueError(f"Unsafe archive member: {entry.filename}")
            if member.parts[0] != name or stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError(f"Unexpected archive member: {entry.filename}")
            # ZipFile verifies each member's CRC as it is read to EOF.
            output = Path(archive.extract(entry, staging))
            if not entry.is_dir():
                if output.stat().st_size != entry.file_size:
                    raise ValueError(f"Extracted size mismatch: {entry.filename}")
                finished += 1
                byte_count += entry.file_size
                if finished % 100 == 0:
                    with LOCK:
                        STATE["archives"][name].update(files_done=finished, bytes_done=byte_count)
                        save_progress()
                if finished % 1000 == 0:
                    print(f"PROGRESS {name}: {finished} files, {byte_count:,} bytes", flush=True)
    destination = ROOT / name
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    (staging / name).rename(destination)
    with LOCK:
        STATE["archives"][name].update(files_done=finished, bytes_done=byte_count, status="complete")
        save_progress()
    print(f"DONE {name}: {finished} files, {byte_count:,} bytes; ZIP CRC checks passed", flush=True)


def main():
    manifest = json.loads(MANIFEST.read_text())
    paths = [ROOT / Path(entry["path"]).name for entry in manifest["files"]]
    if set(manifest["completed"]) != {p.name for p in paths}:
        raise RuntimeError("Wait for all ZIP downloads and SHA-256 checks to finish")
    if PROGRESS.exists():
        raise FileExistsError(f"Extraction already has a progress record: {PROGRESS}")
    required = 0
    for path in paths:
        if (ROOT / path.stem).exists():
            raise FileExistsError(f"Refusing to overwrite {ROOT / path.stem}")
        with zipfile.ZipFile(path) as archive:
            members = [entry for entry in archive.infolist() if not entry.is_dir()]
            sizes = sum(entry.file_size for entry in members)
            # Round file allocations up to 4 KiB and allow room for directory metadata.
            required += sum(((entry.file_size + 4095) // 4096) * 4096 for entry in members) + 1024 * 1024
            STATE.setdefault("archives", {})[path.stem] = {
                "files_total": len(members), "bytes_total": sizes,
                "files_done": 0, "bytes_done": 0, "status": "extracting",
            }
    free = shutil.disk_usage(ROOT).free
    if required * 1.1 > free:
        raise RuntimeError(f"Not enough space: need {required:,} bytes plus 10% margin, have {free:,}")
    staging = Path(tempfile.mkdtemp(prefix=".extracting-", dir=ROOT))
    STATE.update(status="extracting", staging=str(staging), started_at=time.time(), free_bytes_before=free)
    with LOCK:
        save_progress()
    print(f"SPACE CHECK: {required:,} bytes to extract; {free:,} bytes free", flush=True)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(extract_one, path, staging) for path in paths]
            for future in concurrent.futures.as_completed(futures):
                future.result()
        staging.rmdir()
        with LOCK:
            STATE.update(status="complete", finished_at=time.time(), free_bytes_after=shutil.disk_usage(ROOT).free)
            save_progress()
        print("ALL ARCHIVES EXTRACTED AND CRC-VERIFIED", flush=True)
    except Exception as exc:
        with LOCK:
            STATE.update(status="failed", error=str(exc))
            save_progress()
        raise


if __name__ == "__main__":
    main()
