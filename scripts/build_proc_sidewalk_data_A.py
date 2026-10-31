#!/usr/bin/env python3
"""Build one processed copy of sidewalk-data-2 from a preprocessing config.

The config (configs/preprocess/*.yaml) sets the output folder and how every image is transformed: an optional
centred crop to `aspect`, then a direct resize to `size`, ignoring aspect ratio. Classes listed under `drop` are
left out of that split's folders and CSV. Classes listed under the optional `merge` are renamed in every split, so
their images share a folder and class. No labels are derived here; that is the training notebook's job.

Output: <split>/<class>/<name>.png for the 4 splits, <split>.csv for each (the source CSV, with image_path
rewritten, source_image_path and source_class kept and the source geometry added), and config.yaml recording the
config used, the class counts and when it was built. The output is built in a temporary sibling directory, verified,
then renamed into place. Existing outputs are never overwritten.

usage: python scripts/build_proc_sidewalk_data_A.py configs/preprocess/naive.yaml [--src DIR] [--workers N]
"""
import argparse, shutil, subprocess, tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import pandas as pd, yaml
from PIL import Image

SPLITS = ["pitt-train", "pitt-val", "pitt-test", "other-cities-train"]
FRAME = (1440, 960)
RESAMPLE = {"bicubic": Image.Resampling.BICUBIC, "bilinear": Image.Resampling.BILINEAR,
            "lanczos": Image.Resampling.LANCZOS}


def crop_box(w, h, aspect):
    """Largest centred box of the given aspect inside a w x h image, as PIL (left, top, right, bottom)."""
    if aspect is None:
        return 0, 0, w, h
    if w / h > aspect:
        cw, ch = round(h * aspect), h
    else:
        cw, ch = w, round(w / aspect)
    left, top = (w - cw) // 2, (h - ch) // 2
    return left, top, left + cw, top + ch


def process(cfg, job):
    src, dst = job
    with Image.open(src) as im:
        w, h = im.size
        src_format, src_mode = im.format, im.mode
        if src_mode == "RGBA":
            lo, hi = im.getchannel("A").getextrema()
            if lo != 255:
                raise ValueError(f"{src}: transparent pixels, alpha {lo}-{hi}")
        rgb = im.convert("RGB")
    box = crop_box(w, h, cfg["aspect"])
    size = (cfg["size"], cfg["size"])
    rgb.crop(box).resize(size, RESAMPLE[cfg["resample"]]).save(dst, format="PNG")
    return dict(source_w=w, source_h=h, source_format=src_format, source_mode=src_mode, is_frame=(w, h) == FRAME,
                crop_left=box[0], crop_top=box[1], crop_right=box[2], crop_bottom=box[3])


def select(df, split, cfg):
    """Apply the config's drop and merge to one split's rows."""
    df = df[~df["class"].isin(cfg["drop"].get(split, []))].reset_index(drop=True)
    return df.assign(source_class=df["class"], **{"class": df["class"].replace(cfg.get("merge", {}))})


def build_split(split, cfg, src_root, out_root, pool):
    df = select(pd.read_csv(src_root / f"{split}.csv", low_memory=False), split, cfg)
    df = df.rename(columns={"image_path": "source_image_path"})
    df.insert(df.columns.get_loc("source_image_path"), "image_path",
              [str(Path(split, c, Path(p).with_suffix(".png").name)) for c, p in zip(df["class"], df.source_image_path)])
    if df.image_path.duplicated().any():
        raise ValueError(f"{split}: output paths collide")
    for path in df.image_path:
        (out_root / path).parent.mkdir(parents=True, exist_ok=True)
    jobs = list(zip([src_root / p for p in df.source_image_path], [out_root / p for p in df.image_path]))
    geometry = pd.DataFrame(pool.map(partial(process, cfg), jobs, chunksize=32), index=df.index)
    df = pd.concat([df, geometry], axis=1)
    df.to_csv(out_root / f"{split}.csv", index=False)
    return df


def verify(df, cfg, src_root, out_root):
    for path in df.image_path:
        with Image.open(out_root / path) as im:
            if im.size != (cfg["size"], cfg["size"]) or im.mode != "RGB" or im.format != "PNG":
                raise ValueError(f"{path}: {im.size} {im.mode} {im.format}")
    counts = df.groupby(["split", "class"]).size().unstack(fill_value=0)
    src = pd.concat([select(pd.read_csv(src_root / f"{s}.csv", usecols=["class"]), s, cfg).assign(split=s)
                     for s in SPLITS])
    if not counts.equals(src.groupby(["split", "class"]).size().unstack(fill_value=0)):
        raise ValueError("output counts differ from the source CSVs")
    return counts


def git_sha():
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True)
    return r.stdout.strip() or None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--src", type=Path, default=Path("/data/datasets/sidewalk-data-2"))
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    out = Path(cfg["output"])
    if out.exists():
        raise SystemExit(f"{out} exists; refusing to overwrite")

    tmp = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    try:
        with ProcessPoolExecutor(a.workers) as pool:
            frames = [build_split(s, cfg, a.src, tmp, pool) for s in SPLITS]
        df = pd.concat(frames, ignore_index=True)
        counts = verify(df, cfg, a.src, tmp)
        built = dict(built_at_utc=datetime.now(timezone.utc).isoformat(), source_dataset=str(a.src),
                     config_file=str(a.config), git_sha=git_sha(), n_images=len(df),
                     class_counts={s: {c: int(n) for c, n in row.items() if n} for s, row in counts.iterrows()})
        (tmp / "config.yaml").write_text(yaml.safe_dump({**cfg, "build": built}, sort_keys=False))
        tmp.chmod(0o2775)
        tmp.rename(out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    print(counts.to_string())
    print(f"{len(df)} images -> {out}")


if __name__ == "__main__":
    main()
