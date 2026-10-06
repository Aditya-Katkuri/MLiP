#!/usr/bin/env python3
"""Build the dataset-explorer assets for the review dashboard from /data/datasets/sidewalk-data.

Writes manifest.json first (atomically), then thumbs/<id>.webp (max width 480) and view/<id>.webp (full
resolution) for every image of every class. Existing WebP files are skipped, so reruns only fill gaps. Each item's
split is set by city (SPLIT_CITIES). Ids are positions in a fixed order that must not change: first the original
classes sorted by (Pittsburgh first, class, city, label_id), then each class in APPENDED, in the order it was added,
sorted by (split train/val/test, city, label_id). A rerun asserts every existing id keeps the same filename as any
existing manifest.json.

usage: python src/baseline/build_dashboard_assets.py --out /data/eda/dashboard/explorer
"""
import argparse, glob, json, os, re, time
from datetime import datetime, timezone
from multiprocessing import Pool
import pandas as pd
from PIL import Image

CLASSES = ["Crosswalk", "CurbRamp", "Obstacle", "SurfaceProblem", "no_obstacles"]
APPENDED = ["no_obstacles", "CurbRamp"]  # added after ids 0-13150 were published, in this order; append-only
EXCLUDED = {}  # class -> reason; reported with per-split counts
FN = re.compile(r"^gsv-(.+)-(\d+)-([A-Za-z_]+)\.(?:png|webp)$")
SOURCES = "test_no_obstacles_sources.csv"
GREY = (128, 128, 128)
SPLIT_CITIES = {"train": ["cdmx", "chicago", "newberg", "oradell", "seattle", "spgg", "walla_walla"],
                "val": ["amsterdam", "columbus"], "test": ["pittsburgh"]}
ABOUT = (
    "Project Sidewalk's cleaned tagger dataset (Dataset 1 in Liu et al., ASSETS '24): labels volunteers placed on "
    "Google Street View, each re-checked by the paper's research assistants. Every Crosswalk, "
    "CurbRamp, Obstacle and SurfaceProblem image is a 1440x960 Street View frame with one labelled point (the ring), "
    "which can sit anywhere in the frame; a CurbRamp label marks a curb ramp that is present. Tags are optional "
    "details a labeller can add, such as missing-tactile-warning or steep for a curb ramp, cracks or grass for a "
    "surface problem, or a pole or parked car for an obstacle. "
    "Splits are by city: test is Pittsburgh, val is Columbus and Amsterdam, and train is the other seven cities. "
    "Test also has a fifth class, No obstacles (proxy): 204 Pittsburgh crops with no label point or tags.")
CLASS_NOTES = {"no_obstacles": (
    "Made from Pittsburgh labels the crowd rejected as incorrect in the archived validator-AI crops; test split only. "
    "A rejected label does not prove the scene is free of obstacles or other issues, and some images visibly show "
    "poles or crosswalks.")}


def convert(job):
    i, src, out, tw, tq, vq = job
    im = Image.open(src)
    size, amin = im.size, 255
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        amin = im.getchannel("A").getextrema()[0]
        if amin < 255:  # composite real transparency on neutral grey, not black
            bg = Image.new("RGB", im.size, GREY)
            bg.paste(im, mask=im.getchannel("A"))
            im = bg
    im = im.convert("RGB")
    for sub, img, q in [("view", im, vq),
                        ("thumbs", im.resize((tw, round(im.height * tw / im.width)), Image.LANCZOS) if im.width > tw else im, tq)]:
        dst = f"{out}/{sub}/{i}.webp"
        if not os.path.exists(dst):
            img.save(dst + ".tmp", "WEBP", quality=q, method=4)
            os.replace(dst + ".tmp", dst)
    return i, size, amin


def img_size(p):
    with Image.open(p) as im:
        return im.size


def du(d):
    return sum(os.path.getsize(p) for p in glob.glob(f"{d}/*.webp"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/datasets/sidewalk-data")
    ap.add_argument("--out", default="/data/eda/dashboard/explorer")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--thumb-width", type=int, default=480)
    ap.add_argument("--thumb-quality", type=int, default=78)
    ap.add_argument("--view-quality", type=int, default=85)
    a = ap.parse_args()
    t0 = time.time()
    splits = [s for s in SPLIT_CITIES if os.path.exists(f"{a.root}/{s}.csv")]
    D = pd.concat([pd.read_csv(f"{a.root}/{s}.csv", dtype={"filename": str}) for s in splits], ignore_index=True)
    tags = list(D.columns[6:])
    city_split = {c: s for s, cs in SPLIT_CITIES.items() for c in cs}
    csplit = D.filename.str.extract(FN)[0].map(city_split)
    excluded = {c: {s: int(((D["class"] == c) & (csplit == s)).sum()) for s in SPLIT_CITIES} | {"reason": r}
                for c, r in EXCLUDED.items()}
    D = D[D["class"].isin(CLASSES)].reset_index(drop=True)

    # CSV vs disk
    m = D.filename.str.extract(FN)
    bad = D[m[0].isna() | (m[2] != D["class"]) | (D.image_path != D.split + "/" + D["class"] + "/" + D.filename)]
    on_disk = {os.path.relpath(p, a.root) for c in CLASSES for s in splits for p in glob.glob(f"{a.root}/{s}/{c}/*") if FN.match(os.path.basename(p))}
    missing, extra = sorted(set(D.image_path) - on_disk), sorted(on_disk - set(D.image_path))
    print(f"{len(D)} CSV rows, {len(on_disk)} files on disk; bad filename/path rows {len(bad)}, "
          f"missing files {len(missing)}, files without CSV row {len(extra)}, duplicate paths {D.image_path.duplicated().sum()}",
          flush=True)
    for p in (missing + extra)[:20]:
        print("  mismatch:", p)
    D = D.assign(city=m[0], label_id=m[1].astype("Int64"))[~D.image_path.isin(missing) & ~D.index.isin(bad.index)]
    assert set(D.city) <= set(city_split), set(D.city) - set(city_split)
    block = D["class"].map({c: k + 1 for k, c in enumerate(APPENDED)}).fillna(0).astype(int)
    rank = (D.city != "pittsburgh").astype(int).where(block == 0, D.city.map(city_split).map({s: k for k, s in enumerate(SPLIT_CITIES)}))
    D = (D.assign(block=block, rank=rank).sort_values(["block", "rank", "class", "city", "label_id"])
         .reset_index(drop=True))  # id order; do not change
    print("rows whose CSV split differs from the city split:", int((D.split != D.city.map(city_split)).sum()))
    D["split"] = D.city.map(city_split)

    with Pool(a.workers) as pool:
        sizes = pool.map(img_size, [f"{a.root}/{p}" for p in D.image_path], chunksize=64)
    S = pd.read_csv(f"{a.root}/{SOURCES}") if os.path.exists(f"{a.root}/{SOURCES}") else pd.DataFrame(columns=["filename"])
    n_src, S = S.filename.value_counts(), S.drop_duplicates("filename").set_index("filename")  # first row per image
    rnd = lambda v: None if pd.isna(v) else round(float(v), 4)
    items = []
    for i, ((_, r), (w, h)) in enumerate(zip(D.iterrows(), sizes)):
        it = dict(id=i, file=r.image_path, **{"class": r["class"]}, split=r.split, city=r.city, label_id=int(r.label_id),
                  x=rnd(r.normalized_x), y=rnd(r.normalized_y), tags=[t for t in tags if r[t] == 1], w=w, h=h)
        if r.filename in S.index:
            src = S.loc[r.filename]
            it["source"] = dict(dataset=src.source_dataset.removeprefix("sidewalk-validator-ai-dataset-"),
                                split=src.source_split, verdict=src.source_verdict, label_id=int(src.source_label_id))
            if n_src[r.filename] > 1:
                it["sources_count"] = int(n_src[r.filename])
        items.append(it)
    man = dict(generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
               source_root=a.root, about=ABOUT, classes=CLASSES, excluded=excluded, split_cities=SPLIT_CITIES,
               tags_by_class={c: [t for t in tags if D.loc[D["class"] == c, t].notna().any()] for c in CLASSES},
               class_notes=CLASS_NOTES, items=items)
    if os.path.exists(f"{a.out}/manifest.json"):
        old = json.load(open(f"{a.out}/manifest.json"))["items"]
        key = lambda it: (it["id"], it["file"].rsplit("/", 1)[-1], it["class"])
        assert len(items) >= len(old) and [key(o) for o in old] == [key(it) for it in items[:len(old)]], "ids changed"
        print(f"all {len(old)} existing ids match the existing manifest; {len(items) - len(old)} new; file paths changed "
              f"for {sum(o['file'] != it['file'] for o, it in zip(old, items))}", flush=True)
    os.makedirs(f"{a.out}/thumbs", exist_ok=True)
    os.makedirs(f"{a.out}/view", exist_ok=True)
    with open(f"{a.out}/manifest.json.tmp", "w") as f:
        json.dump(man, f, separators=(",", ":"))
    os.replace(f"{a.out}/manifest.json.tmp", f"{a.out}/manifest.json")
    print(f"manifest: {len(items)} items  {pd.crosstab(D['class'], D.split).to_dict()}  ({time.time() - t0:.1f}s)", flush=True)

    jobs = [(it["id"], f"{a.root}/{it['file']}", a.out, a.thumb_width, a.thumb_quality, a.view_quality) for it in items
            if not (os.path.exists(f"{a.out}/thumbs/{it['id']}.webp") and os.path.exists(f"{a.out}/view/{it['id']}.webp"))]
    print(f"converting {len(jobs)} images ({len(items) - len(jobs)} already done) with {a.workers} workers", flush=True)
    t1, odd, alpha = time.time(), [], []
    with Pool(a.workers) as pool:
        for k, (i, size, amin) in enumerate(pool.imap_unordered(convert, jobs, chunksize=4), 1):
            if list(size) != [items[i]["w"], items[i]["h"]]:
                odd.append((i, size))
            if amin < 255:
                alpha.append((i, amin))
            if k % 1000 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)}  {k / (time.time() - t1):.0f} img/s", flush=True)
    print(f"done in {time.time() - t1:.0f}s (total {time.time() - t0:.0f}s); size changed since manifest: {len(odd)} {odd[:5]}; "
          f"with transparency (composited on grey): {len(alpha)} {alpha[:5]}")
    print(f"thumbs {du(a.out + '/thumbs') / 2**30:.2f} GiB, view {du(a.out + '/view') / 2**30:.2f} GiB, "
          f"manifest {os.path.getsize(a.out + '/manifest.json') / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
