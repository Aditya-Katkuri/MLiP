#!/usr/bin/env python3
"""EDA of the RampNet corpus: training shards, Stage-1 inputs, the paper's gold set, the
post-publication benchmark, the crop-model datasets and the checkpoints.

Stages (all by default, or pick with --stages):
  scan       per-shard metadata + points + image-size sample -> <out>/cache/   (cached)
  dataset    rampnet-dataset aggregates (rows, ramps/pano, leakage, city, geometry)
  stage1     gov records -> manifests -> published funnel; tests the "43%" claim
  gold       manual_labels/ (the 1,000-pano gold set behind P 0.949 / R 0.873)
  benchmark  rampnet-benchmark records: per-city P/R recomputed from verdicts
  cropds     rampnet-crop-model-dataset round1 / round2
  models     checkpoint configs + parameter counts from safetensors headers
  summary    merge <out>/parts/*.json -> <out>/summary.json

Each analysis stage writes <out>/parts/<stage>.json and CSVs under <out>/tables/, so a
single stage can be re-run without redoing the rest.

usage:
  python src/eda/rampnet_eda.py                       # everything, default paths
  python src/eda/rampnet_eda.py --stages benchmark,summary
"""
import argparse, collections, glob, io, json, math, os, re, struct, subprocess, sys, time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SPLITS = ["train", "val", "test"]
# generous bounding boxes around the three training cities (lat_min, lat_max, lon_min, lon_max)
CITY_BOXES = {"nyc": (40.45, 41.0, -74.3, -73.65), "portland": (45.35, 45.7, -122.9, -122.45),
              "bend": (43.95, 44.2, -121.45, -121.2)}


def city_of(lat, lon):
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    conds = [(lat >= a) & (lat <= b) & (lon >= c) & (lon <= d) for a, b, c, d in CITY_BOXES.values()]
    return np.select(conds, list(CITY_BOXES), "other")


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (round(c - h, 4), round(c + h, 4))


def save_part(a, name, obj):
    os.makedirs(f"{a.out}/parts", exist_ok=True)
    with open(f"{a.out}/parts/{name}.json", "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def table(a, df, name):
    df.to_csv(f"{a.out}/tables/{name}.csv", index=isinstance(df.index, pd.MultiIndex) or df.index.name is not None)


def records(df):
    return json.loads(df.to_json(orient="records"))


# ----------------------------------------------------------------------------- scan
def _scan_shard(job):
    split, path, cache = job
    shard = os.path.basename(path).split(".")[0]
    meta_p = f"{cache}/meta/{split}_{shard}.parquet"
    pts_p = f"{cache}/points/{split}_{shard}.parquet"
    img_p = f"{cache}/images/{split}_{shard}.parquet"
    pf = pq.ParquetFile(path)
    info = dict(split=split, shard=shard, rows=pf.metadata.num_rows, row_groups=pf.metadata.num_row_groups,
                file_bytes=os.path.getsize(path))
    if all(os.path.exists(p) for p in (meta_p, pts_p, img_p)):
        return info

    t = pf.read(columns=["pano_id", "record_creation_time", "curb_ramp_points_normalized",
                         "pano_coord", "curb_ramp_coords", "pano_azimuth"])
    pts = t["curb_ramp_points_normalized"].combine_chunks()
    n_pts = pc.fill_null(pc.list_value_length(pts), 0)
    n_coords = pc.fill_null(pc.list_value_length(t["curb_ramp_coords"].combine_chunks()), 0)
    coord = pc.list_flatten(t["pano_coord"].combine_chunks()).to_numpy(zero_copy_only=False).reshape(-1, 2)
    meta = pa.table(dict(
        split=pa.array([split] * t.num_rows), shard=pa.array([shard] * t.num_rows),
        pano_id=t["pano_id"], record_creation_time=t["record_creation_time"],
        n_points=n_pts, n_coords=n_coords, coord0=coord[:, 0], coord1=coord[:, 1],
        pano_azimuth=t["pano_azimuth"]))
    # one row per point; each point is a [x, y] list
    idx = pc.list_parent_indices(pts)
    flat = pc.list_flatten(pts)
    plen = pc.list_value_length(flat).to_numpy(zero_copy_only=False)
    xy = pc.list_flatten(flat).to_numpy(zero_copy_only=False)
    ok = plen == 2
    if not ok.all():
        print(f"WARNING {split}/{shard}: {int((~ok).sum())} points without 2 coords", flush=True)
    starts = np.concatenate([[0], np.cumsum(plen)[:-1]]).astype(int)
    x = np.where(ok, xy[np.minimum(starts, len(xy) - 1)] if len(xy) else np.nan, np.nan)
    y = np.where(ok, xy[np.minimum(starts + 1, len(xy) - 1)] if len(xy) else np.nan, np.nan)
    pano_ids = t["pano_id"].combine_chunks().take(idx)
    points = pa.table(dict(split=pa.array([split] * len(x)), pano_id=pano_ids, x=x, y=y))

    # image-size sample: first 5 rows of the middle row group (header parse only)
    from PIL import Image
    rg = pf.metadata.num_row_groups // 2
    imgs = pf.read_row_group(rg, columns=["pano_id", "image"]).slice(0, 5)
    rows = []
    for pid, im in zip(imgs["pano_id"].to_pylist(), imgs["image"].to_pylist()):
        b = im["bytes"]
        with Image.open(io.BytesIO(b)) as I:
            rows.append(dict(split=split, shard=shard, pano_id=pid, width=I.size[0], height=I.size[1],
                             format=I.format, mode=I.mode, bytes=len(b)))
    for d in ("meta", "points", "images"):
        os.makedirs(f"{cache}/{d}", exist_ok=True)
    pq.write_table(meta, meta_p)
    pq.write_table(points, pts_p)
    pq.write_table(pa.Table.from_pylist(rows), img_p)
    return info


def stage_scan(a):
    cache = f"{a.out}/cache"
    jobs = [(s, p, cache) for s in SPLITS for p in sorted(glob.glob(f"{a.dataset}/{s}/*.parquet"))]
    t0, infos = time.time(), []
    with Pool(a.workers) as pool:
        for i, info in enumerate(pool.imap_unordered(_scan_shard, jobs), 1):
            infos.append(info)
            if i % 32 == 0:
                print(f"  scan {i}/{len(jobs)} shards  {time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(infos).sort_values(["split", "shard"]).to_csv(f"{cache}/shards.csv", index=False)
    print(f"scan done: {len(jobs)} shards in {time.time()-t0:.0f}s", flush=True)


_CACHE = {}


def load_cache(a):
    if not _CACHE:
        c = f"{a.out}/cache"
        cat = lambda d: pa.concat_tables([pq.read_table(f) for f in sorted(glob.glob(f"{c}/{d}/*.parquet"))]).to_pandas()
        meta = cat("meta")
        # pano_coord order: lat has the smaller magnitude in all three US cities
        lat_first = meta.coord0.abs().median() < meta.coord1.abs().median()
        meta["lat"], meta["lon"] = (meta.coord0, meta.coord1) if lat_first else (meta.coord1, meta.coord0)
        meta["city"] = city_of(meta.lat, meta.lon)
        _CACHE.update(meta=meta, points=cat("points"), images=cat("images"),
                      shards=pd.read_csv(f"{c}/shards.csv"))
    return _CACHE["meta"], _CACHE["points"], _CACHE["images"], _CACHE["shards"]


# ----------------------------------------------------------------------------- dataset
def stage_dataset(a):
    meta, pts, imgs, shards = load_cache(a)
    R = {}

    sp = shards.groupby("split").agg(shards=("shard", "count"), rows_footer=("rows", "sum"),
                                     rows_per_shard_min=("rows", "min"), rows_per_shard_max=("rows", "max"),
                                     bytes=("file_bytes", "sum"), row_groups=("row_groups", "sum")).reindex(SPLITS)
    pm = meta.groupby("split").agg(panos=("pano_id", "size"), unique_pano_ids=("pano_id", "nunique"),
                                   ramps=("n_points", "sum"), gov_coords=("n_coords", "sum"),
                                   zero_ramp_panos=("n_points", lambda s: int((s == 0).sum())),
                                   mean_ramps=("n_points", "mean"), median_ramps=("n_points", "median"),
                                   max_ramps=("n_points", "max")).reindex(SPLITS)
    splits = sp.join(pm)
    splits["gb"] = (splits.bytes / 1e9).round(2)
    splits["share_of_panos"] = (splits.panos / splits.panos.sum()).round(4)
    splits["zero_ramp_frac"] = (splits.zero_ramp_panos / splits.panos).round(4)
    splits["kb_per_pano"] = (splits.bytes / splits.panos / 1e3).round(0)
    splits.loc["total"] = splits.sum(numeric_only=True)
    for c in ["mean_ramps", "median_ramps", "share_of_panos", "zero_ramp_frac", "kb_per_pano", "gb"]:
        splits.loc["total", c] = None
    splits.loc["total", "mean_ramps"] = meta.n_points.mean()
    splits.loc["total", "zero_ramp_frac"] = (meta.n_points == 0).mean()
    splits.loc["total", "gb"] = shards.file_bytes.sum() / 1e9
    splits.index.name = "split"
    table(a, splits.reset_index(), "dataset_splits")
    R["splits"] = records(splits.reset_index())
    R["points_rows_in_cache"] = int(len(pts))
    R["card"] = dict(panos=214376, labels=849895, splits_text="150k / 43k / 21k")
    R["footer_rows_total"] = int(shards.rows.sum())
    R["ramps_total"] = int(meta.n_points.sum())

    # ramps-per-pano histogram (0..14, 15+)
    cap = 15
    h = (meta.assign(k=meta.n_points.clip(upper=cap)).groupby(["split", "k"]).size().unstack(0)
         .reindex(columns=SPLITS).fillna(0).astype(int))
    h.index = [str(i) if i < cap else f"{cap}+" for i in h.index]
    h.index.name = "ramps_per_pano"
    table(a, h, "ramps_per_pano_hist")
    R["ramps_per_pano_hist"] = {s: dict(zip(h.index, h[s].tolist())) for s in SPLITS}

    # auto-labels vs the gov coordinates attached to each pano
    d = meta.n_points - meta.n_coords
    R["points_vs_gov_coords"] = dict(
        equal=round(float((d == 0).mean()), 4), fewer_points=round(float((d < 0).mean()), 4),
        more_points=round(float((d > 0).mean()), 4), total_points=int(meta.n_points.sum()),
        total_gov_coords=int(meta.n_coords.sum()),
        diff_hist={int(k): int(v) for k, v in d.clip(-10, 10).value_counts().sort_index().items()})

    # placeholder rows: pano_id "file_N", pano_coord (0, 0), record_creation_time 0
    ph = (meta.lat == 0) & (meta.lon == 0)
    ph_pts = pts.pano_id.isin(set(meta.pano_id[ph]))
    R["placeholder_rows"] = dict(
        rows=int(ph.sum()), points=int(meta.n_points[ph].sum()),
        rows_by_split=meta[ph].split.value_counts().to_dict(), shards=sorted(set(meta.shard[ph])),
        pano_ids=sorted(set(meta.pano_id[ph])), record_creation_time_zero=int((meta.record_creation_time[ph] == 0).sum()),
        out_of_unit_range_points_on_placeholders=int(((pts.x[ph_pts] < 0) | (pts.x[ph_pts] > 1) |
                                                      (pts.y[ph_pts] < 0) | (pts.y[ph_pts] > 1)).sum()))
    clean = meta[~ph]
    R["clean"] = dict(panos=int(len(clean)), unique_pano_ids=int(clean.pano_id.nunique()), ramps=int(clean.n_points.sum()),
                      by_split={s: dict(panos=int((clean.split == s).sum()), ramps=int(clean.n_points[clean.split == s].sum()),
                                        share=round(float((clean.split == s).mean()), 4)) for s in SPLITS},
                      zero_ramp_frac=round(float((clean.n_points == 0).mean()), 4),
                      matches_card=bool(len(clean) == 214376 and clean.n_points.sum() == 849895))

    # leakage (placeholders excluded; they are reported above)
    dup_within = clean.groupby("split").pano_id.apply(lambda s: int(s.duplicated().sum())).reindex(SPLITS)
    sets = {s: set(clean.pano_id[clean.split == s]) for s in SPLITS}
    R["leakage"] = dict(
        duplicate_pano_ids_within_split=dup_within.to_dict(),
        duplicate_rows_example=meta[meta.duplicated("pano_id", keep=False)].sort_values("pano_id")
        [["split", "shard", "pano_id", "n_points"]].head(10).to_dict("records"),
        overlap_train_val=len(sets["train"] & sets["val"]), overlap_train_test=len(sets["train"] & sets["test"]),
        overlap_val_test=len(sets["val"] & sets["test"]),
        unique_pano_ids_total=int(meta.pano_id.nunique()))

    # is one shard a fair sample of its split? (the handoff says "pull ONE shard")
    sh = clean.assign(nyc=clean.city == "nyc", portland=clean.city == "portland", bend=clean.city == "bend",
                      zero=clean.n_points == 0).groupby(["split", "shard"])[["nyc", "portland", "bend", "zero"]].mean()
    R["shard_uniformity"] = {s: {c: dict(min=round(float(sh.loc[s][c].min()), 4), max=round(float(sh.loc[s][c].max()), 4),
                                         std=round(float(sh.loc[s][c].std()), 4)) for c in sh.columns} for s in SPLITS}
    R["pano_ids_sorted_across_shards"] = {
        s: bool((g := clean[clean.split == s].sort_values(["shard"], kind="stable")).pano_id.is_monotonic_increasing)
        for s in SPLITS}

    # city (from pano_coord bbox)
    cs = meta.groupby(["city", "split"]).agg(panos=("pano_id", "size"), ramps=("n_points", "sum")).unstack("split")
    cs.columns = [f"{m}_{s}" for m, s in cs.columns]
    cs = cs.fillna(0).astype(int)
    cs["panos_total"] = cs[[f"panos_{s}" for s in SPLITS if f"panos_{s}" in cs]].sum(axis=1)
    cs["ramps_total"] = cs[[f"ramps_{s}" for s in SPLITS if f"ramps_{s}" in cs]].sum(axis=1)
    cs["zero_ramp_frac"] = meta.groupby("city").n_points.apply(lambda s: round(float((s == 0).mean()), 4))
    cs["ramps_per_positive_pano"] = meta[meta.n_points > 0].groupby("city").n_points.mean().round(3)
    cs.index.name = "city"
    table(a, cs.reset_index(), "dataset_city_split")
    R["city"] = records(cs.reset_index())
    other = meta[meta.city == "other"]
    R["city_other_examples"] = other[["split", "pano_id", "lat", "lon", "n_points", "n_coords"]].head(10).to_dict("records")

    # when records were created
    ts = pd.to_datetime(meta.record_creation_time, unit="s", utc=True)
    R["record_creation_time"] = dict(min=str(ts.min()), max=str(ts.max()),
                                     by_month={str(k): int(v) for k, v in ts.dt.strftime("%Y-%m").value_counts().sort_index().items()})
    R["pano_azimuth"] = dict(min=float(meta.pano_azimuth.min()), max=float(meta.pano_azimuth.max()))

    # image sample
    ims = imgs.groupby(["split", "width", "height", "format", "mode"]).agg(n=("pano_id", "size"),
                                                                           median_kb=("bytes", lambda s: round(s.median() / 1e3)))
    table(a, ims.reset_index(), "dataset_image_sample")
    R["image_sample"] = dict(n=int(len(imgs)), groups=records(ims.reset_index()),
                             bytes_quantiles_kb={str(q): round(float(imgs.bytes.quantile(q)) / 1e3) for q in (0.05, 0.5, 0.95)})

    # where the labels sit in the equirectangular frame
    pts_ok = pts.dropna(subset=["x", "y"])
    ybins = np.round(np.arange(0, 1.0001, 0.02), 2)
    xbins = np.round(np.arange(0, 1.0001, 0.05), 2)
    yh = np.histogram(pts_ok.y.clip(0, 1), ybins)[0]
    xh = np.histogram(pts_ok.x.clip(0, 1), xbins)[0]
    table(a, pd.DataFrame(dict(y_lo=ybins[:-1], y_hi=ybins[1:], points=yh)), "point_y_hist")
    table(a, pd.DataFrame(dict(x_lo=xbins[:-1], x_hi=xbins[1:], points=xh)), "point_x_hist")
    R["points_xy"] = dict(
        n=int(len(pts_ok)), nan=int(len(pts) - len(pts_ok)),
        out_of_unit_range=int(((pts_ok.x < 0) | (pts_ok.x > 1) | (pts_ok.y < 0) | (pts_ok.y > 1)).sum()),
        y_quantiles={str(q): round(float(pts_ok.y.quantile(q)), 4) for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)},
        above_horizon_frac=round(float((pts_ok.y < 0.5).mean()), 4),
        y_hist=dict(bins=ybins.tolist(), counts=yh.tolist()), x_hist=dict(bins=xbins.tolist(), counts=xh.tolist()))

    save_part(a, "dataset", R)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        os.makedirs(f"{a.out}/figures", exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
        (h / h.sum()).plot.bar(ax=ax[0], width=0.8); ax[0].set_title("ramps per panorama (share of split)")
        ax[1].bar(ybins[:-1], yh, width=0.02, align="edge"); ax[1].axvline(0.5, color="k", lw=0.8)
        ax[1].set_title("label y (0=top, 0.5=horizon)")
        fig.tight_layout(); fig.savefig(f"{a.out}/figures/dataset_ramps_and_y.png", dpi=130); plt.close(fig)
    except Exception as e:
        print("figure skipped:", e)
    print(json.dumps({k: R[k] for k in ("footer_rows_total", "ramps_total", "leakage", "points_vs_gov_coords")}, indent=1, default=str))


# ----------------------------------------------------------------------------- stage 1
def _jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def stage_stage1(a):
    meta, _, _, _ = load_cache(a)
    S, R = a.stage1, {}
    gov = {c: len(json.load(open(f"{S}/location_data/{c}.geojson"))["features"]) for c in ("bend", "portland")}
    gov["nyc"] = len(pd.read_csv(f"{S}/location_data/nyc.csv", usecols=["the_geom"]))
    R["gov_records"] = dict(gov, total=sum(gov.values()))
    R["paper_table1"] = dict(nyc=217680, portland=45324, bend=13611, total=276615)

    al = pd.read_csv(f"{S}/manifests/all_locations.csv", dtype={"date": str}, keep_default_na=False,
                     float_precision="round_trip")
    al["city"] = city_of(al.latitude, al.longitude)
    R["all_locations"] = dict(rows=len(al), by_city=al.city.value_counts().to_dict(),
                              unique_coords=int(al[["latitude", "longitude"]].drop_duplicates().shape[0]),
                              date_2000_01_01=int((al.date == "2000-01-01").sum()), date_empty=int((al.date == "").sum()),
                              by_year={k: int(v) for k, v in al.date.str[:4].value_counts().sort_index().items()})

    pos = _jsonl(f"{S}/manifests/dataset.jsonl")
    final = _jsonl(f"{S}/manifests/finaldataset.jsonl")
    negpool = _jsonl(f"{S}/manifests/negativepanos.jsonl")
    negused = _jsonl(f"{S}/manifests/negativepanosSHORTENED.jsonl")
    ids = lambda L: [r["pano_id"] for r in L]
    pos_ids, final_ids, neg_ids = set(ids(pos)), ids(final), set(ids(negused))
    published = set(meta.pano_id)
    R["manifests"] = dict(
        positive_panos=len(pos), positive_unique=len(pos_ids), negative_pool=len(negpool), negatives_used=len(negused),
        final=len(final), final_unique=len(set(final_ids)),
        final_equals_positive_plus_negatives=(set(final_ids) == pos_ids | neg_ids) and len(final) == len(pos) + len(negused),
        positive_negative_overlap=len(pos_ids & neg_ids),
        coords_per_positive_pano_mean=round(float(np.mean([len(r["curb_ramps_coords"]) for r in pos])), 3),
        negatives_with_coords=sum(1 for r in negused if r["curb_ramps_coords"]))

    fset = set(final_ids)
    R["published_vs_final"] = dict(
        published_rows=int(len(meta)), published_unique=len(published),
        final_not_published=len(fset - published), published_not_in_final=len(published - fset),
        positives_published=len(pos_ids & published), negatives_published=len(neg_ids & published),
        zero_ramp_published=int((meta.n_points == 0).sum()),
        zero_ramp_published_that_are_positive_manifest=int(meta[(meta.n_points == 0) & meta.pano_id.isin(pos_ids)].pano_id.nunique()),
        docs_stage1_generation_cost=dict(intended=219170, written=214599, never_written=4571, published=214376, unexplained=223))

    # which government records became training labels: exact coordinate join, as gov_provenance.py does
    by_pid = {r["pano_id"]: r["curb_ramps_coords"] for r in pos}
    c_int = {tuple(c) for r in pos for c in r["curb_ramps_coords"]}
    c_pub = {tuple(c) for pid, cs in by_pid.items() if pid in published for c in cs}
    keys = list(zip(al.latitude.tolist(), al.longitude.tolist()))
    al["in_intended"] = [k in c_int for k in keys]
    al["in_published"] = [k in c_pub for k in keys]
    all_keys = set(keys)

    # The exact float join above is what RampNet's gov_provenance.py does ("float(repr(x)) == x, so
    # this is exact"). But dataset.jsonl wrote some coordinates one ulp away from all_locations.csv
    # (e.g. 40.7608151405334 vs 40.760815140533396), and an exact join silently drops those records.
    # A 1 mm KD-tree join catches them; the count is flat from 1 mm to 1 m, so nothing is ambiguous.
    from scipy.spatial import cKDTree

    def _metres(arr):
        arr = np.asarray(arr, float)
        return np.c_[arr[:, 0] * 111320.0, arr[:, 1] * 111320.0 * np.cos(np.radians(arr[:, 0]))]

    A = _metres(al[["latitude", "longitude"]].to_numpy())
    R["tolerance_sweep_m"] = {}
    for name, cs in (("intended", c_int), ("published", c_pub)):
        d, _ = cKDTree(_metres(np.array(sorted(cs)))).query(A, distance_upper_bound=2.0)
        al[f"tol_{name}"] = d <= 1e-3
        R["tolerance_sweep_m"][name] = {"exact": int(al[f"in_{name}"].sum()),
                                        **{str(t): int((d <= t).sum()) for t in (1e-6, 1e-3, 1e-2, 1e-1, 1.0)}}
    un = np.array([c for c in c_int if c not in all_keys])
    if len(un):
        dd, ii = cKDTree(A).query(_metres(un))
        near = al[["latitude", "longitude"]].to_numpy()[ii]
        R["manifest_coords_without_exact_match"] = dict(
            n=int(len(un)), of_unique_manifest_coords=len(c_int), max_distance_m=float(dd.max()),
            max_abs_degree_diff=float(np.abs(un - near).max()),
            by_city=pd.Series(city_of(un[:, 0], un[:, 1])).value_counts().to_dict(),
            example=dict(dataset_jsonl=[float(v) for v in un[0]], all_locations_csv=[float(v) for v in near[0]]))

    fun = al.groupby("city").agg(records=("date", "size"),
                                 consumed_intended_exact=("in_intended", "sum"), consumed_published_exact=("in_published", "sum"),
                                 consumed_intended=("tol_intended", "sum"), consumed_published=("tol_published", "sum"))
    fun.loc["total"] = fun.sum()
    for k in ("intended_exact", "published_exact", "intended", "published"):
        fun[f"never_label_{k}_frac"] = (1 - fun[f"consumed_{k}"] / fun.records).round(4)
    fun.index.name = "city"
    table(a, fun.reset_index(), "stage1_record_consumption")
    R["record_consumption"] = records(fun.reset_index())
    R["stage1_inputs_card_says"] = dict(records=276071, consumed=156712, consumption_rate=0.5677, never=0.4323,
                                        bend=(13357, 5110), portland=(45035, 21075), nyc=(217679, 130527))

    # undated records (paper-era convert_date mapped unknown -> 2000-01-01, so they always passed)
    und = al[al.date == "2000-01-01"]
    R["undated_records"] = dict(n=len(und), consumed_published_exact=int(und.in_published.sum()),
                                consumed_published=int(und.tol_published.sum()),
                                by_city=und.city.value_counts().to_dict())

    # gov coords attached to each published row vs the manifest
    m = meta.set_index("pano_id")
    mc = pd.Series({pid: len(cs) for pid, cs in by_pid.items() if pid in published})
    j = m.loc[~m.index.duplicated()].join(mc.rename("manifest_coords"), how="inner")
    R["published_coords_match_manifest"] = round(float((j.n_coords == j.manifest_coords).mean()), 4)

    funnel = [
        ("government curb-ramp records (location_data/)", R["gov_records"]["total"]),
        ("records consumed by >=1 intended positive pano (exact float join, as upstream)", int(fun.loc["total", "consumed_intended_exact"])),
        ("records consumed by >=1 intended positive pano (1 mm join)", int(fun.loc["total", "consumed_intended"])),
        ("records consumed by >=1 published pano (1 mm join)", int(fun.loc["total", "consumed_published"])),
        ("intended positive panos (dataset.jsonl)", len(pos)),
        ("negative candidates sampled from streets", len(negpool)),
        ("negatives used (SHORTENED)", len(negused)),
        ("intended panos (finaldataset.jsonl)", len(final)),
        ("written by download_dataset (docs)", 214599),
        ("published rows (parquet footers)", int(len(meta))),
        ("published unique pano_ids", len(published)),
        ("auto-placed curb-ramp points (labels)", int(meta.n_points.sum())),
    ]
    R["funnel"] = [dict(step=s, n=n) for s, n in funnel]
    table(a, pd.DataFrame(R["funnel"]), "stage1_funnel")
    save_part(a, "stage1", R)
    print(json.dumps({k: R[k] for k in ("gov_records", "manifests", "published_vs_final", "record_consumption",
                                        "published_coords_match_manifest")}, indent=1, default=str))


# ----------------------------------------------------------------------------- gold
def stage_gold(a):
    meta, _, _, _ = load_cache(a)
    rows, boxes = [], []
    for f in sorted(glob.glob(f"{a.rampnet_repo}/manual_labels/*.txt")):
        pid = os.path.basename(f)[:-4]
        L = [l.split() for l in open(f) if l.strip()]
        rows.append(dict(pano_id=pid, ramps=len(L)))
        for l in L:
            boxes.append(dict(pano_id=pid, cls=int(float(l[0])), cx=float(l[1]), cy=float(l[2]), w=float(l[3]), h=float(l[4])))
    g, b = pd.DataFrame(rows), pd.DataFrame(boxes)
    split_of = meta.groupby("pano_id").split.agg(lambda s: ",".join(sorted(set(s))))
    g["split"] = g.pano_id.map(split_of).fillna("absent")
    g = g.merge(meta.drop_duplicates("pano_id")[["pano_id", "city", "n_points"]], on="pano_id", how="left")
    hist = g.ramps.clip(upper=10).value_counts().sort_index()
    R = dict(
        panos=len(g), ramps=int(g.ramps.sum()), negative_panos=int((g.ramps == 0).sum()),
        mean_ramps=round(float(g.ramps.mean()), 3), classes=b.cls.value_counts().to_dict(),
        ramps_per_pano_hist={("10+" if k == 10 else str(k)): int(v) for k, v in hist.items()},
        split_membership=g.split.value_counts().to_dict(), city=g.city.fillna("absent").value_counts().to_dict(),
        box_px_at_4096x2048=dict(w_median=round(float((b.w * 4096).median()), 1), h_median=round(float((b.h * 2048).median()), 1),
                                 w_p5_p95=[round(float((b.w * 4096).quantile(q)), 1) for q in (0.05, 0.95)],
                                 h_p5_p95=[round(float((b.h * 2048).quantile(q)), 1) for q in (0.05, 0.95)]),
        box_cy_quantiles={str(q): round(float(b.cy.quantile(q)), 4) for q in (0.05, 0.5, 0.95)},
        gold_vs_autolabel=dict(gold_ramps=int(g.ramps.sum()), autolabel_ramps_same_panos=int(g.n_points.fillna(0).sum()),
                               panos_equal_count=int((g.ramps == g.n_points).sum())),
        paper_numbers=dict(stage2_precision=0.949, stage2_recall=0.873, stage1_precision=0.9403, stage1_recall=0.9245))
    table(a, g, "gold_panos")
    save_part(a, "gold", R)
    print(json.dumps(R, indent=1, default=str))


# ----------------------------------------------------------------------------- benchmark
def _score(recs, exclude_top=False, lenient=False):
    """Re-implementation of rampnet.validation.collect on the HF `records` rows."""
    judged, recall, missed, missed_unsure = [], [], 0, 0
    n_seen = n_judged = n_unconf = n_unsure = n_dup = 0
    for r in recs:
        if exclude_top and r["review_group"] == "top":
            continue
        n_seen += 1
        v = [d["verdict"] for d in r["detections"]]
        if any(x is None for x in v):
            continue
        n_judged += 1
        dec = []
        for x in v:
            if x == "unsure":
                n_unsure += 1; continue
            if x == "duplicate":
                n_dup += 1
                if lenient:
                    continue
                dec.append(False); continue
            dec.append(x == "correct")
        judged += dec
        if r["no_missed"] or len(r["missed"]) > 0:
            recall += dec
            missed += sum(1 for m in r["missed"] if not m["unsure"])
            missed_unsure += sum(1 for m in r["missed"] if m["unsure"])
        else:
            n_unconf += 1
    tp, n = sum(judged), len(judged)
    tpr = sum(recall)
    P = tp / n if n else None
    Rc = tpr / (tpr + missed) if (tpr + missed) else None
    return dict(panos_seen=n_seen, panos_judged=n_judged, recall_pool_panos=n_judged - n_unconf,
                detections_decided=n, tp=tp, fp=n - tp, duplicates=n_dup, unsure=n_unsure,
                missed_confident=missed, missed_unsure=missed_unsure,
                precision=round(P, 4) if P is not None else None, precision_ci=wilson(tp, n),
                recall=round(Rc, 4) if Rc is not None else None, recall_ci=wilson(tpr, tpr + missed))


def stage_benchmark(a):
    meta, _, _, _ = load_cache(a)
    B = f"{a.benchmark}/data"
    split_of = meta.groupby("pano_id").split.agg(lambda s: ",".join(sorted(set(s))))
    cities = sorted(os.path.basename(p)[:-8] for p in glob.glob(f"{B}/records/*.parquet"))
    rows, overlap = [], {}
    for c in cities:
        recs = pq.read_table(f"{B}/records/{c}.parquet").to_pylist()
        nat = pq.read_table(f"{B}/native/{c}.parquet", columns=["width", "height"]).to_pandas()
        low = pq.read_table(f"{B}/4096x2048/{c}.parquet", columns=["width", "height"]).to_pandas()
        gal = pq.ParquetFile(f"{B}/galleries/{c}.parquet").metadata.num_rows
        dates = sorted(r["capture_date"] for r in recs if r["capture_date"])
        verdicts = collections.Counter(d["verdict"] for r in recs for d in r["detections"])
        allp, unb = _score(recs), _score(recs, exclude_top=True)
        ov = [(r["pano_id"], split_of[r["pano_id"]]) for r in recs if r["pano_id"] in split_of.index]
        overlap[c] = ov
        rows.append(dict(
            city=c, panos=len(recs), source=dict(collections.Counter(r["source"] for r in recs)),
            capture_min=dates[0] if dates else None, capture_max=dates[-1] if dates else None,
            native_sizes=dict(collections.Counter(f"{w}x{h}" for w, h in zip(nat.width, nat.height)).most_common(3)),
            model_input_sizes=dict(collections.Counter(f"{w}x{h}" for w, h in zip(low.width, low.height))),
            gallery_crops=gal, review_groups=dict(collections.Counter(r["review_group"] for r in recs)),
            detections=sum(len(r["detections"]) for r in recs), verdicts=dict(verdicts),
            missed_confident=sum(1 for r in recs for m in r["missed"] if not m["unsure"]),
            missed_unsure=sum(1 for r in recs for m in r["missed"] if m["unsure"]),
            label_types=dict(collections.Counter(r["label_type"] for r in recs)),
            model_ids=dict(collections.Counter(f'{r["model_id"]}@{r["model_training_date"]}' for r in recs)),
            train_overlap_panos=len(ov),
            p_all=allp["precision"], r_all=allp["recall"], p_all_ci=allp["precision_ci"], r_all_ci=allp["recall_ci"],
            p_unbiased=unb["precision"], r_unbiased=unb["recall"], n_unbiased=unb["panos_seen"],
            tp=allp["tp"], fp=allp["fp"], recall_pool=allp["recall_pool_panos"]))

    # the git scorer, run as the benchmark's own README says (image-free)
    git = {}
    scorer = f"{a.rampnet_repo}/scripts/score_validation.py"
    for d in sorted(glob.glob(f"{a.rampnet_repo}/benchmark/*/verdicts.json")):
        c = os.path.basename(os.path.dirname(d))
        try:
            out = subprocess.run([sys.executable, scorer, os.path.dirname(d)], capture_output=True, text=True,
                                 timeout=300, cwd=a.rampnet_repo).stdout
            P = re.findall(r"^Precision:\s+([\d.]+)", out, re.M)
            Rr = re.findall(r"^Recall:\s+([\d.]+)", out, re.M)
            J = re.findall(r"^Panos fully judged:\s+(\d+)", out, re.M)
            git[c] = dict(p_all=float(P[0]), r_all=float(Rr[0]), p_unbiased=float(P[1]), r_unbiased=float(Rr[1]),
                          panos_judged=int(J[0]), on_hf=c in cities)
        except Exception as e:
            git[c] = dict(error=f"{type(e).__name__}: {e}")
    try:
        sha = subprocess.run(["git", "-C", a.rampnet_repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        sha = None

    df = pd.DataFrame(rows)
    for c in ("p_all", "r_all", "p_unbiased", "r_unbiased"):
        df[f"git_{c}"] = df.city.map(lambda x: git.get(x, {}).get(c))
    flat = df.drop(columns=["source", "native_sizes", "model_input_sizes", "review_groups", "verdicts",
                            "label_types", "model_ids", "p_all_ci", "r_all_ci"]).copy()
    flat["source"] = df.source.map(lambda d: ",".join(f"{k}:{v}" for k, v in d.items()))
    flat["native_size_top"] = df.native_sizes.map(lambda d: next(iter(d)))
    table(a, flat, "benchmark_cities")
    pooled = dict(tp=int(df.tp.sum()), fp=int(df.fp.sum()))
    pooled["precision_micro_9"] = round(pooled["tp"] / (pooled["tp"] + pooled["fp"]), 4)
    R = dict(cities=records(df), git_scorer=git, rampnet_repo_sha=sha, pooled=pooled,
             reimplementation_matches_git=bool(all(
                 abs(r["p_all"] - git[r["city"]]["p_all"]) < 6e-4 and abs(r["r_all"] - git[r["city"]]["r_all"]) < 6e-4
                 for r in rows if r["city"] in git and "p_all" in git[r["city"]])),
             training_overlap=overlap,
             paper_gold_reference=dict(precision=0.949, recall=0.873, note="manual_labels gold set; in-domain, not this benchmark"))
    save_part(a, "benchmark", R)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        os.makedirs(f"{a.out}/figures", exist_ok=True)
        d = df.sort_values("r_all")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.errorbar(d.r_all, d.p_all, xerr=np.abs(np.array(d.r_all_ci.tolist()).T - d.r_all.values),
                    yerr=np.abs(np.array(d.p_all_ci.tolist()).T - d.p_all.values), fmt="o", ms=4, lw=0.8)
        for _, r in d.iterrows():
            ax.annotate(r.city, (r.r_all, r.p_all), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.scatter([0.873], [0.949], marker="*", s=120, color="k"); ax.annotate("paper gold (in-domain)", (0.873, 0.949), fontsize=7, xytext=(-40, -12), textcoords="offset points")
        ax.set_xlabel("recall"); ax.set_ylabel("precision"); ax.set_title("RampNet on the post-publication benchmark (95% Wilson CI)")
        fig.tight_layout(); fig.savefig(f"{a.out}/figures/benchmark_pr.png", dpi=130); plt.close(fig)
    except Exception as e:
        print("figure skipped:", e)
    print(flat.to_string())
    print(json.dumps({k: R[k] for k in ("git_scorer", "pooled", "reimplementation_matches_git", "training_overlap")}, indent=1, default=str))


# ----------------------------------------------------------------------------- crop-model datasets
def stage_cropds(a):
    from PIL import Image
    R = {}
    d2 = a.crop_round2
    if os.path.isdir(d2):
        r2 = {}
        for s in SPLITS:
            fs = sorted(f for f in os.listdir(f"{d2}/{s}") if f.endswith(".jpg"))
            n = [len(f[:-4].split("_-_")) - 1 for f in fs]
            sizes = collections.Counter()
            for f in fs:
                with Image.open(f"{d2}/{s}/{f}") as I:
                    sizes[f"{I.size[0]}x{I.size[1]}"] += 1
            r2[s] = dict(crops=len(fs), points=sum(n), points_per_crop_hist=dict(sorted(collections.Counter(n).items())),
                         empty_crops=n.count(0), sizes=dict(sizes.most_common(4)),
                         bytes=sum(os.path.getsize(f"{d2}/{s}/{f}") for f in fs))
        r2["total"] = dict(crops=sum(r2[s]["crops"] for s in SPLITS), points=sum(r2[s]["points"] for s in SPLITS))
        r2["card_says"] = dict(panoramas=312, labels=1212)
        R["round2"] = r2
    d1 = a.crop_round1
    files = sorted(glob.glob(f"{d1}/data/*/*.parquet"))
    done = os.path.isdir(d1) and files and not glob.glob(f"{d1}/.cache/huggingface/download/**/*.incomplete", recursive=True)
    if done:
        r1, schema = {}, None
        for f in files:
            s = f.split("/")[-2]
            pf = pq.ParquetFile(f)
            schema = str(pf.schema_arrow)
            cols = [c for c in pf.schema_arrow.names if c not in ("image",)]
            t = pf.read(columns=cols)
            e = r1.setdefault(s, dict(files=0, rows=0, bytes=0, points=0, empty=0))
            e["files"] += 1; e["rows"] += t.num_rows; e["bytes"] += os.path.getsize(f)
            for c in cols:
                if pa.types.is_list(t.schema.field(c).type):
                    L = pc.fill_null(pc.list_value_length(t[c].combine_chunks()), 0).to_numpy(zero_copy_only=False)
                    e["points"] += int(L.sum()); e["empty"] += int((L == 0).sum()); e["list_column"] = c
        R["round1"] = dict(splits=r1, schema=schema, card_says=dict(panoramas=20698, labels=27704),
                           total_rows=sum(v["rows"] for v in r1.values()))
        want = ["crop_uid", "n_keypoints", "width", "height", "sha256"]
        m1 = pd.concat([pq.read_table(f, columns=[c for c in want if c in pq.ParquetFile(f).schema_arrow.names])
                        .to_pandas().assign(split=f.split("/")[-2]) for f in files], ignore_index=True)
        if {"n_keypoints", "width", "height", "sha256"} <= set(m1.columns):
            kh = m1.n_keypoints.clip(upper=6).value_counts().sort_index()
            by_sha = m1.groupby("sha256").split.agg(lambda s: len(set(s)))
            R["round1"].update(
                keypoints_total=int(m1.n_keypoints.sum()),
                keypoints_per_crop_hist={("6+" if k == 6 else str(int(k))): int(v) for k, v in kh.items()},
                keypoints_by_split=m1.groupby("split").n_keypoints.sum().astype(int).to_dict(),
                empty_crops=int((m1.n_keypoints == 0).sum()),
                sizes=dict(collections.Counter(f"{w}x{h}" for w, h in zip(m1.width, m1.height)).most_common(4)),
                duplicate_images_within_split=int(m1.duplicated(["split", "sha256"]).sum()),
                images_in_more_than_one_split=int((by_sha > 1).sum()),
                **{f"{s}_rows_with_identical_train_image": int(m1[m1.split == s].sha256.isin(set(m1.sha256[m1.split == "train"])).sum())
                   for s in ("val", "test")},
                duplicate_crop_uids=int(m1.crop_uid.duplicated().sum()) if "crop_uid" in m1 else None,
                split_share={s: round(float(v), 4) for s, v in m1.split.value_counts(normalize=True).items()})
        # sample sizes from first rows
        try:
            pf = pq.ParquetFile(files[0]); t = pf.read_row_group(0).slice(0, 20)
            imgc = [c for c in t.schema.names if c == "image"]
            if imgc:
                sz = collections.Counter()
                for im in t["image"].to_pylist():
                    with Image.open(io.BytesIO(im["bytes"])) as I:
                        sz[f"{I.size[0]}x{I.size[1]}"] += 1
                R["round1"]["sample_sizes"] = dict(sz)
                R["round1"]["sample_row"] = {k: (v if k != "image" else "<bytes>") for k, v in t.slice(0, 1).to_pylist()[0].items()}
        except Exception as ex:
            R["round1"]["sample_error"] = str(ex)
    else:
        R["round1"] = dict(status="not downloaded when this ran", path=d1)
    save_part(a, "cropds", R)
    print(json.dumps(R, indent=1, default=str))


# ----------------------------------------------------------------------------- models
def _safetensors_summary(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    params = sum(math.prod(v["shape"]) for v in h.values())
    prefixes = collections.Counter(k.split(".")[0] for k in h)
    return dict(file=os.path.basename(path), bytes=os.path.getsize(path), tensors=len(h), params=params,
                dtypes=dict(collections.Counter(v["dtype"] for v in h.values())), top_level=dict(prefixes))


def stage_models(a):
    R = {}
    for d in a.models:
        if not os.path.isdir(d):
            R[d] = "missing"; continue
        e = dict(files={f: os.path.getsize(f"{d}/{f}") for f in sorted(os.listdir(d)) if os.path.isfile(f"{d}/{f}")})
        if os.path.exists(f"{d}/config.json"):
            e["config"] = json.load(open(f"{d}/config.json"))
        e["safetensors"] = [_safetensors_summary(p) for p in sorted(glob.glob(f"{d}/*.safetensors"))]
        for p in glob.glob(f"{d}/*.py"):
            m = re.search(r"BACKBONE_NAME\s*=\s*['\"]([^'\"]+)", open(p).read())
            if m:
                e["backbone"] = m.group(1)
        if os.path.exists(f"{d}/README.md") and "backbone" not in e:
            m = re.search(r"timm/([\w.\-]+)", open(f"{d}/README.md").read())
            if m:
                e["backbone"] = m.group(1)
        R[os.path.basename(d.rstrip("/"))] = e
    save_part(a, "models", R)
    print(json.dumps(R, indent=1, default=str))


# ----------------------------------------------------------------------------- summary
def stage_summary(a):
    parts = {os.path.basename(p)[:-5]: json.load(open(p)) for p in sorted(glob.glob(f"{a.out}/parts/*.json"))}
    S = dict(generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), inputs=dict(
        dataset=a.dataset, benchmark=a.benchmark, stage1=a.stage1, rampnet_repo=a.rampnet_repo,
        crop_round1=a.crop_round1, crop_round2=a.crop_round2, models=a.models))
    h = {}
    if "dataset" in parts:
        d = parts["dataset"]
        c = d["clean"]
        h.update(parquet_rows=d["footer_rows_total"], placeholder_rows=d["placeholder_rows"]["rows"],
                 panos=c["panos"], autolabel_points=c["ramps"], matches_card=c["matches_card"],
                 panos_by_split={s: c["by_split"][s]["panos"] for s in SPLITS},
                 zero_ramp_frac=c["zero_ramp_frac"],
                 cross_split_overlap=sum(d["leakage"][k] for k in ("overlap_train_val", "overlap_train_test", "overlap_val_test")))
    if "stage1" in parts:
        t = {r["city"]: r for r in parts["stage1"]["record_consumption"]}["total"]
        h.update(gov_records=parts["stage1"]["gov_records"]["total"],
                 gov_never_label_frac_exact_join_intended=t["never_label_intended_exact_frac"],
                 gov_consumed_intended=int(t["consumed_intended"]),
                 gov_never_label_frac_intended=t["never_label_intended_frac"],
                 gov_consumed_published=int(t["consumed_published"]),
                 gov_never_label_frac_published=t["never_label_published_frac"])
    if "gold" in parts:
        h.update(gold_panos=parts["gold"]["panos"], gold_ramps=parts["gold"]["ramps"], gold_negatives=parts["gold"]["negative_panos"])
    if "benchmark" in parts:
        h.update(benchmark_cities_hf=len(parts["benchmark"]["cities"]),
                 benchmark_recall_range=[min(c["r_all"] for c in parts["benchmark"]["cities"]),
                                         max(c["r_all"] for c in parts["benchmark"]["cities"])])
    S["headline"] = h
    S.update(parts)
    with open(f"{a.out}/summary.json", "w") as f:
        json.dump(S, f, indent=1, default=str)
    print(json.dumps(h, indent=1))


STAGES = {"scan": stage_scan, "dataset": stage_dataset, "stage1": stage_stage1, "gold": stage_gold,
          "benchmark": stage_benchmark, "cropds": stage_cropds, "models": stage_models, "summary": stage_summary}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="/mnt/scratch/datasets/rampnet-dataset")
    p.add_argument("--benchmark", default="/mnt/scratch/datasets/rampnet-benchmark")
    p.add_argument("--stage1", default="/data/datasets/rampnet-stage1-inputs")
    p.add_argument("--rampnet-repo", default="/data/code/RampNet")
    p.add_argument("--crop-round1", default="/mnt/scratch/datasets/rampnet-crop-model-dataset-round1")
    p.add_argument("--crop-round2", default="/data/datasets/rampnet-crop-model-dataset-round2")
    p.add_argument("--models", nargs="+", default=["/mnt/scratch/datasets/models/rampnet",
                                                   "/data/datasets/models/rampnet-crop-model"])
    p.add_argument("--out", default="/data/eda/rampnet")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--stages", default=",".join(STAGES))
    a = p.parse_args()
    os.makedirs(f"{a.out}/tables", exist_ok=True)
    for s in a.stages.split(","):
        t0 = time.time()
        print(f"== stage {s}", flush=True)
        STAGES[s](a)
        print(f"== stage {s} done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
