#!/usr/bin/env python3
"""Maps and tier assignments for the 3-minute pitch deck.

Writes to reports/presentation/build/:
  map_coverage.png   city streets, audited vs never audited (slide 1)
  map_tiers.png      city streets coloured by location tier (slide 2)
  crop_tiers.csv     tier of every Pittsburgh crop (3 barrier classes), for per-tier evaluation
  assets.json        the numbers quoted on the slides

Tier rule (one rule, fixed before looking at any model output):
  Tier 1  within 150 m of a hospital, health centre, school, child-care centre, senior centre or nursing home,
          or within 50 m of a busy bus stop (top quarter of weekday trips)
  Tier 2  otherwise within 400 m of a daily destination (grocery, pharmacy, library, park, rec centre, ...)
          or within 200 m of any bus stop
  Tier 3  everything else

usage: /opt/miniconda/envs/sidewalk/bin/python reports/presentation/build_assets.py
"""
import glob, json, os, sys, warnings
from pathlib import Path

os.environ.setdefault("PROJ_DATA", os.path.join(sys.prefix, "share", "proj"))
warnings.filterwarnings("ignore")

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

for f in glob.glob(os.path.expanduser("~/.local/share/fonts/Carlito-*.ttf")):
    font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = "Carlito"

D = Path("/data/datasets/pittsburgh")
OUT = Path(__file__).parent / "build"
UTM = 32617
AUDITED, UNAUDITED = "#0F766E", "#A9B2BC"
TIER_COLORS = {1: "#DC2626", 2: "#F59E0B", 3: "#7C8A9A"}
ESSENTIAL = ["hospitals", "health_centers", "schools", "child_care_centers", "senior_centers", "nursing_homes"]
DAILY = ["supermarkets", "pharmacies", "libraries", "parks_and_facilities", "rec_centers", "affordable_housing",
         "food_banks", "doctors_offices", "universities"]
R_ESSENTIAL, R_BUSY_STOP, R_DAILY, R_STOP = 150, 50, 400, 200


def xy(g):
    return np.column_stack([g.geometry.x, g.geometry.y])


def nearest(src, pts):
    return cKDTree(xy(src)).query(pts)[0]


def tiers(pts, P, stops, busy):
    t1 = (nearest(P[P.asset_type.isin(ESSENTIAL)], pts) <= R_ESSENTIAL) | (nearest(busy, pts) <= R_BUSY_STOP)
    t2 = (nearest(P[P.asset_type.isin(DAILY)], pts) <= R_DAILY) | (nearest(stops, pts) <= R_STOP)
    return np.where(t1, 1, np.where(t2, 2, 3))


def map_axes(boundary):
    fig, ax = plt.subplots(figsize=(6.2, 5.4), dpi=300)
    gpd.GeoSeries([boundary], crs=UTM).plot(ax=ax, color="#F4F5F7", edgecolor="#9AA3AD", linewidth=0.6, zorder=0)
    ax.set_axis_off()
    ax.set_aspect("equal")
    return fig, ax


def zoom_map(C, P, busy, lon, lat, half=600):
    """Tier-coloured streets in a square around one label, with the essential destinations and busy stops that set Tier 1."""
    pt = gpd.GeoSeries(gpd.points_from_xy([lon], [lat]), crs=4326).to_crs(UTM).iloc[0]
    x0, x1, y0, y1 = pt.x - half, pt.x + half, pt.y - half, pt.y + half
    fig, ax = plt.subplots(figsize=(3, 3), dpi=400)
    fig.patch.set_facecolor("#F4F5F7")
    sub = C.cx[x0:x1, y0:y1]
    for t in [3, 2, 1]:
        sub[sub.tier == t].plot(ax=ax, color=TIER_COLORS[t], linewidth=2.4, zorder=t)
    e = P[P.asset_type.isin(ESSENTIAL)].cx[x0:x1, y0:y1]
    ax.scatter(e.geometry.x, e.geometry.y, marker="s", s=26, color="#111827", linewidths=0, zorder=5)
    b = busy.cx[x0:x1, y0:y1]
    ax.scatter(b.geometry.x, b.geometry.y, s=18, facecolor="white", edgecolor="#111827", linewidths=0.9, zorder=5)
    ax.scatter([pt.x], [pt.y], s=420, facecolor="white", edgecolor="#111827", linewidths=2.6, zorder=6)
    ax.scatter([pt.x], [pt.y], s=70, color="#111827", zorder=7)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(OUT / "map_tiers_zoom.png", facecolor=fig.get_facecolor())
    plt.close(fig)


EXAMPLE_LABEL = 23869  # confirmed missing curb ramp, Squirrel Hill North; the photo in the slide 2 flowchart


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    city = gpd.read_file(D / "wprdc/city_boundary.geojson").to_crs(UTM).union_all()
    S = gpd.read_file(D / "projectsidewalk/streets.geojson").to_crs(UTM)
    S["km"] = S.length / 1000
    C = S[S.geometry.interpolate(0.5, normalized=True).within(city)].copy()
    C["audited"] = C.audit_count > 0
    stats = dict(city_km=C.km.sum(), audited_km=C.km[C.audited].sum())
    stats["audited_share"] = stats["audited_km"] / stats["city_km"]

    fig, ax = map_axes(city)
    C[~C.audited].plot(ax=ax, color=UNAUDITED, linewidth=0.45, zorder=1)
    C[C.audited].plot(ax=ax, color=AUDITED, linewidth=0.9, zorder=2)
    fig.savefig(OUT / "map_coverage.png", bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)

    A = pd.read_csv(D / "wprdc/county_assets.csv", low_memory=False, usecols=["asset_type", "latitude", "longitude"]).dropna()
    H = pd.read_csv(D / "wprdc/hospitals.csv").rename(columns={"Y": "latitude", "X": "longitude"}).assign(asset_type="hospitals")
    P = pd.concat([A, H[["asset_type", "latitude", "longitude"]]])
    P = gpd.GeoDataFrame(P, geometry=gpd.points_from_xy(P.longitude, P.latitude), crs=4326).to_crs(UTM)
    stops = gpd.read_file(D / "wprdc/prt_stops.geojson").to_crs(UTM)
    stops["trips_wd"] = pd.to_numeric(stops.trips_wd, errors="coerce").fillna(0)
    busy_thr = float(stops.trips_wd.quantile(0.75))
    busy = stops[stops.trips_wd >= busy_thr]

    mid = C.geometry.interpolate(0.5, normalized=True)
    C["tier"] = tiers(np.column_stack([mid.x, mid.y]), P, stops, busy)
    stats["busy_stop_min_weekday_trips"] = busy_thr
    stats["km_by_tier"] = C.groupby("tier").km.sum().round(1).to_dict()
    stats["km_share_by_tier"] = (C.groupby("tier").km.sum() / C.km.sum()).round(3).to_dict()

    fig, ax = map_axes(city)
    for t in [3, 2, 1]:
        C[C.tier == t].plot(ax=ax, color=TIER_COLORS[t], linewidth=0.45 if t == 3 else 0.6, zorder=t)
    fig.savefig(OUT / "map_tiers.png", bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)

    rows = []
    for d in ["nocurbramp", "obstacle", "surfaceproblem"]:
        for p in glob.glob(f"/data/datasets/sidewalk-validator-ai-dataset-{d}/*/*/pittsburgh_*.webp"):
            split, verdict, fn = p.split("/")[-3:]
            rows.append(dict(dir=d, split=split, verdict=verdict, label_id=int(fn[:-5].rsplit("_", 1)[1])))
    K = pd.DataFrame(rows)
    L = pd.read_csv(D / "projectsidewalk/rawLabels.csv", usecols=["label_id", "latitude", "longitude"])
    ex = L[L.label_id == EXAMPLE_LABEL].iloc[0]
    zoom_map(C, P, busy, ex.longitude, ex.latitude)
    K = K.merge(L, on="label_id", how="left")
    stats["crops_without_coordinates"] = int(K.latitude.isna().sum())
    K = K.dropna(subset=["latitude"])
    g = gpd.GeoDataFrame(K, geometry=gpd.points_from_xy(K.longitude, K.latitude), crs=4326).to_crs(UTM)
    K["tier"] = tiers(xy(g), P, stops, busy)
    K.to_csv(OUT / "crop_tiers.csv", index=False)
    stats["crops_by_tier"] = K.tier.value_counts().sort_index().to_dict()
    stats["confirmed_crops_by_tier"] = K[K.verdict == "correct"].tier.value_counts().sort_index().to_dict()
    json.dump(json.loads(pd.Series(stats).to_json()), open(OUT / "assets.json", "w"), indent=1)
    print(json.dumps(json.load(open(OUT / "assets.json")), indent=1))


if __name__ == "__main__":
    main()
