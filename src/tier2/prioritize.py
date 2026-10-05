#!/usr/bin/env python3
"""Tier 2: rank Pittsburgh's known sidewalk barriers for repair, from existing human labels only.

No imagery and no model. Each physical barrier (a Project Sidewalk label cluster inside the city) gets

    priority = magnitude x validity x (floor + (1 - floor) x demand)

  magnitude  how bad the barrier is, using Project Sidewalk's own published access-score constants
  validity   how far human validators back the label (AI votes ignored by default)
  demand     how many people, and which people, need that spot: transit, destinations, a
             vulnerability index of the tract, pedestrian safety

All weights live in the YAML config. src/tier2/evaluate.py measures how much the ranking depends on
them, compares it with Project Sidewalk's access score, and audits it for equity.

Writes to <out>:
  barriers_ranked.{csv,geojson}   one row per barrier, ranked, with every component
  barriers_rejected.csv           barriers dropped because human validators disagreed
  segments_ranked.{csv,geojson}   street edges, ranked by summed barrier priority
  neighborhoods.csv               per-neighbourhood totals next to audit coverage
  components.parquet              the unweighted components, for evaluate.py
  run_info.json                   config, git state, input hashes, counts

usage:
  python src/tier2/prioritize.py --config configs/tier2_default.yaml
"""
import argparse, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PROJ_DATA", os.path.join(sys.prefix, "share", "proj"))
os.environ.setdefault("GDAL_DATA", os.path.join(sys.prefix, "share", "gdal"))

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree

UTM = 32617  # UTM 17N, metres
BARRIER_311 = {"Broken Sidewalk", "Sidewalk/Curb/ADA Ramp Maintenance", "Sidewalk Repair Program",
               "Blocked or Closed Sidewalks or Trails", "Sidewalk has Ice or Litter", "Curb Cuts",
               "ADA Ramp Installation", "Americans with Disabilities Act (ADA) Concerns",
               "Accessibility Related Construction Concerns"}


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def pct_pos(x):
    """Percentile among positive values, in (0, 1]; zero stays zero. Most spots have no crash or
    stop nearby, and a plain percentile would hand all of those the same middling score."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    pos = x > 0
    if pos.any():
        out[pos] = pd.Series(x[pos]).rank(pct=True, method="average").to_numpy()
    return out


def decay_sum(src_xy, src_w, dst_xy, radius):
    """For each destination point, sum of source weights with linear decay to zero at `radius` m."""
    out = np.zeros(len(dst_xy))
    if len(src_xy) == 0:
        return out
    tree = cKDTree(src_xy)
    for i, js in enumerate(tree.query_ball_point(dst_xy, r=radius)):
        if js:
            d = np.hypot(*(src_xy[js] - dst_xy[i]).T)
            out[i] = float(np.sum(src_w[js] * np.clip(1 - d / radius, 0, None)))
    return out


def xy(g):
    return np.column_stack([g.geometry.x.to_numpy(), g.geometry.y.to_numpy()])


# --------------------------------------------------------------------------------------
def load_clusters(D, city):
    raw = json.load(open(D / "projectsidewalk/labelClusters.geojson"))
    keep = ["label_cluster_id", "label_type", "street_edge_id", "intersection_id", "region_name", "avg_image_capture_date",
            "avg_label_date", "median_severity", "cluster_size", "label_ids", "tag_counts"]
    rows = [dict({k: f["properties"].get(k) for k in keep}, lon=f["geometry"]["coordinates"][0], lat=f["geometry"]["coordinates"][1])
            for f in raw["features"]]
    C = pd.DataFrame(rows)
    C = gpd.GeoDataFrame(C, geometry=gpd.points_from_xy(C.lon, C.lat), crs=4326).to_crs(UTM)
    C["in_city"] = C.within(city)
    return C


def barrier_magnitude(C, ps, cfg):
    """-(access-score term) per cluster, from Project Sidewalk's engine constants."""
    tw, sm, qm = ps["type_weights"], ps["severity_multiplier"], ps["quality_multiplier"]
    preset = ps["presets"][cfg["barrier"]["weights_preset"]]
    deltas = {(t["label_type"], t["tag"]): t["delta"] for t in ps["tag_adjustments"]}
    thr, sat = ps["tag_active_threshold"], ps["street_condition_saturation_count"]
    mags, buckets, tags_active = [], [], []
    for t, sev, tc, n in zip(C.label_type, C.median_severity, C.tag_counts, C.cluster_size):
        sign = np.sign(tw[t]["base_weight"])
        w = sign * preset[t]
        bucket = "null" if sev is None or pd.isna(sev) else str(int(np.floor(sev + 0.5)))
        active = sorted(tag for tag, k in (tc or {}).items() if k / max(n, 1) >= thr)
        d = sum(deltas.get((t, tag), 0.0) for tag in active)
        mode = tw[t]["scoring"]
        if mode == "negative_severity":
            term = w * sm[bucket] + d
        elif mode == "positive_quality":
            term = w * qm[bucket] + d
        elif mode == "street_condition":
            term = w / sat + d
        else:
            term = w + d
        mags.append(max(0.0, -term)); buckets.append(bucket); tags_active.append("; ".join(active))
    C["severity_bucket"], C["active_tags"], C["magnitude"] = buckets, tags_active, mags
    return C


def attach_validity(C, D, cfg):
    v = cfg["validity"]
    V = pd.read_csv(D / "projectsidewalk/validations.csv", usecols=["label_id", "validation_result", "validator_type"])
    if not v["use_ai_votes"]:
        V = V[V.validator_type == "Human"]
    votes = V.pivot_table(index="label_id", columns="validation_result", aggfunc="size", fill_value=0)
    for c in ["Agree", "Disagree", "Unsure"]:
        if c not in votes:
            votes[c] = 0
    L = pd.read_csv(D / "projectsidewalk/rawLabels.csv", usecols=["label_id", "high_quality_user", "pano_url"])
    L["hq"] = L.high_quality_user.astype(str).str.lower().eq("true")
    ex = C[["label_cluster_id", "label_ids"]].explode("label_ids").rename(columns={"label_ids": "label_id"})
    ex["label_id"] = ex.label_id.astype("int64")
    ex = ex.merge(votes, left_on="label_id", right_index=True, how="left").merge(L, on="label_id", how="left").fillna(
        {"Agree": 0, "Disagree": 0, "Unsure": 0})
    g = ex.groupby("label_cluster_id").agg(votes_agree=("Agree", "sum"), votes_disagree=("Disagree", "sum"),
                                          votes_unsure=("Unsure", "sum"), any_hq_labeller=("hq", "any"),
                                          gsv_url=("pano_url", "first"))
    C = C.merge(g, left_on="label_cluster_id", right_index=True, how="left")
    agree, disagree = C.votes_agree.fillna(0), C.votes_disagree.fillna(0)
    C["validation"] = np.select([disagree > agree, agree > disagree, (agree + disagree) > 0],
                                ["rejected", "confirmed", "tied"], default="unvalidated")
    C["validity"] = np.select([C.validation == "confirmed", C.any_hq_labeller.fillna(False).astype(bool)],
                              [v["weight_confirmed"], v["weight_unvalidated"]], default=v["weight_unvalidated_low_quality"])
    C.loc[C.validation == "rejected", "validity"] = 0.0
    return C


def attach_street_context(C, D, hoods):
    S = gpd.read_file(D / "projectsidewalk/streets.geojson", columns=["street_edge_id", "way_type", "status", "audit_count", "outdated",
                                                                      "last_label_date"])
    S = S.to_crs(UTM)
    props = S.drop(columns="geometry").copy()
    props["outdated"] = props.outdated.astype(str).str.lower().eq("true")
    C = C.merge(props[["street_edge_id", "way_type", "outdated", "last_label_date"]], on="street_edge_id", how="left")
    C = gpd.GeoDataFrame(C, geometry="geometry", crs=UTM)
    C["needs_reverification"] = C.outdated.fillna(False).astype(bool)
    j = gpd.sjoin(C[["geometry"]], hoods[["hood", "geometry"]], how="left", predicate="within")
    C["neighborhood"] = j[~j.index.duplicated()].hood
    return C, S


def acs_tracts(D, city):
    tracts = gpd.read_file(D / "wprdc/census_tracts_2020.geojson").to_crs(UTM)
    tracts = tracts[tracts.representative_point().within(city)][["GEOID", "ALAND", "geometry"]].copy()

    def est(table):
        d = json.load(open(D / f"census/acs5_tract_{table}.json"))
        return pd.DataFrame.from_dict({g[7:]: v[table]["estimate"] for g, v in d["data"].items()}, orient="index")
    b01003, b01001, b18101, b08201 = est("B01003"), est("B01001"), est("B18101"), est("B08201")
    b03002, b19013 = est("B03002"), est("B19013")
    a = pd.DataFrame(index=b01003.index)
    a["population"] = b01003.B01003001
    older = [f"B01001{i:03d}" for i in list(range(20, 26)) + list(range(44, 50))]
    a["pct_65_plus"] = b01001[older].sum(axis=1) / b01001.B01001001
    disab = [f"B18101{i:03d}" for i in (4, 7, 10, 13, 16, 19, 23, 26, 29, 32, 35, 38)]
    a["pct_disability"] = b18101[disab].sum(axis=1) / b18101.B18101001
    a["pct_zero_vehicle_hh"] = b08201.B08201002 / b08201.B08201001
    a["pct_black_nh"] = b03002.B03002004 / b03002.B03002001
    a["median_hh_income"] = b19013.B19013001
    tracts = tracts.merge(a, left_on="GEOID", right_index=True, how="left")
    tracts["pop_density"] = tracts.population / (pd.to_numeric(tracts.ALAND) / 1e6)
    return tracts


def demand_components(C, D, city, tracts, cfg):
    dm = cfg["demand"]
    R = dm["radius_m"]
    pts = xy(C)

    stops = gpd.read_file(D / "wprdc/prt_stops.geojson").to_crs(UTM)
    stops["trips_wd"] = pd.to_numeric(stops.trips_wd, errors="coerce").fillna(0)
    C["transit_raw"] = decay_sum(xy(stops), stops.trips_wd.to_numpy(), pts, R)
    _, idx = cKDTree(xy(stops)).query(pts)
    C["nearest_stop"] = stops.stop_name.to_numpy()[idx]
    C["nearest_stop_m"] = np.hypot(*(xy(stops)[idx] - pts).T).round(0)

    assets = pd.read_csv(D / "wprdc/county_assets.csv", low_memory=False, usecols=["asset_type", "name", "latitude", "longitude"])
    hosp = pd.read_csv(D / "wprdc/hospitals.csv").rename(columns={"Facility": "name", "Y": "latitude", "X": "longitude"})
    hosp["asset_type"] = "hospitals"
    dest = pd.concat([assets, hosp[["asset_type", "name", "latitude", "longitude"]]], ignore_index=True)
    dest = dest[dest.asset_type.isin(dm["destination_weights"]) & dest.latitude.notna() & dest.longitude.notna()]
    dest = gpd.GeoDataFrame(dest, geometry=gpd.points_from_xy(dest.longitude, dest.latitude), crs=4326).to_crs(UTM)
    dw = dest.asset_type.map(dm["destination_weights"]).to_numpy(dtype=float)
    C["destinations_raw"] = decay_sum(xy(dest), dw, pts, R)
    tree = cKDTree(xy(dest))
    near = []
    for i, js in enumerate(tree.query_ball_point(pts, r=R)):
        if not js:
            near.append(""); continue
        d = np.hypot(*(xy(dest)[js] - pts[i]).T)
        best = js[int(np.argmax(dw[js] * np.clip(1 - d / R, 0, None)))]
        near.append(f"{dest.name.iloc[best]} ({dest.asset_type.iloc[best].replace('_', ' ')})")
    C["key_destination"] = near

    j = gpd.sjoin(C[["geometry"]], tracts.drop(columns=["ALAND"]), how="left", predicate="within")
    j = j[~j.index.duplicated()]
    for c in ["GEOID"] + dm["vulnerability_vars"] + ["pct_black_nh", "median_hh_income", "population"]:
        C[c if c != "GEOID" else "tract"] = j[c].to_numpy()
    tr = tracts.set_index("GEOID")
    vuln_tract = pd.concat([tr[v].rank(pct=True) for v in dm["vulnerability_vars"]], axis=1).mean(axis=1)
    C["vulnerability"] = C.tract.map(vuln_tract).fillna(vuln_tract.median()).to_numpy()

    frames = []
    for y in dm["crash_years"]:
        p = D / f"wprdc/crashes_{y}.csv"
        if p.exists():
            frames.append(pd.read_csv(p, low_memory=False, usecols=lambda c: c in {
                "PED_COUNT", "PED_DEATH_COUNT", "PED_MAJ_INJ_COUNT", "DEC_LAT", "DEC_LONG"}))
    cr = pd.concat(frames, ignore_index=True)
    cr = cr[(cr.PED_COUNT > 0) & cr.DEC_LAT.notna() & cr.DEC_LONG.notna()]
    cr = gpd.GeoDataFrame(cr, geometry=gpd.points_from_xy(cr.DEC_LONG, cr.DEC_LAT), crs=4326).to_crs(UTM)
    cr = cr[np.isfinite(xy(cr)).all(axis=1)]   # a few PennDOT records carry junk coordinates
    cw = 1 + 2 * (cr.PED_DEATH_COUNT.fillna(0) + cr.PED_MAJ_INJ_COUNT.fillna(0)).clip(upper=2).to_numpy()
    C["crash_raw"] = decay_sum(xy(cr), cw, pts, dm["crash_radius_m"])
    hin = gpd.read_file(D / "wprdc/high_injury_network.geojson").to_crs(UTM)
    C["on_high_injury_network"] = C.geometry.distance(hin.union_all()) <= dm["hin_radius_m"]

    if dm.get("use_311"):
        c311 = pd.read_csv(D / "wprdc/pgh_311.csv", low_memory=False, usecols=["subject", "latitude", "longitude"])
        c311 = c311[c311.subject.isin(BARRIER_311) & c311.latitude.notna()]
        c311 = gpd.GeoDataFrame(c311, geometry=gpd.points_from_xy(c311.longitude, c311.latitude), crs=4326).to_crs(UTM)
        C["complaints_raw"] = decay_sum(xy(c311), np.ones(len(c311)), pts, 100)
    return C


def score(C, cfg, weights=None, floor=None):
    """Combine components into demand and priority. Cheap: evaluate.py calls it for every perturbation."""
    dm = cfg["demand"]
    w = dict(weights or dm["weights"])
    comp = {"transit": pct_pos(C.transit_raw), "destinations": pct_pos(C.destinations_raw),
            "vulnerability": C.vulnerability.to_numpy(dtype=float),
            "safety": 0.5 * pct_pos(C.crash_raw) + 0.5 * C.on_high_injury_network.to_numpy(dtype=float)}
    if dm.get("use_311") and "complaints_raw" in C:
        comp["complaints"] = pct_pos(C.complaints_raw)
    total = sum(w.get(k, 0) for k in comp)
    demand = sum(w.get(k, 0) * v for k, v in comp.items()) / total
    f = cfg["priority"]["demand_floor"] if floor is None else floor
    priority = C.magnitude.to_numpy() * C.validity.to_numpy() * (f + (1 - f) * demand)
    return comp, demand, priority


def aggregate(B, S, hoods, D, city):
    seg = B.groupby(["list", "street_edge_id"]).agg(barriers=("label_cluster_id", "size"), priority_sum=("priority", "sum"),
                                                   priority_max=("priority", "max"), top_barrier=("label_cluster_id", "first"),
                                                   types=("label_type", lambda s: "; ".join(f"{k} {v}" for k, v in s.value_counts().items())),
                                                   neighborhood=("neighborhood", "first"),
                                                   needs_reverification=("needs_reverification", "any")).reset_index()
    seg = S[["street_edge_id", "way_type", "geometry"]].merge(seg, on="street_edge_id", how="inner")
    seg["length_m"] = seg.length.round(1)
    seg["priority_per_100m"] = seg.priority_sum / np.maximum(seg.length_m, 25) * 100
    seg = seg.sort_values(["list", "priority_sum"], ascending=[False, False])
    seg.insert(0, "rank", seg.groupby("list").cumcount() + 1)

    # Curb ramps get built per corner, so rank intersections too (only clusters Project Sidewalk placed on one).
    ix = B[B.intersection_id.notna()].groupby(["list", "intersection_id"]).agg(
        barriers=("label_cluster_id", "size"), priority_sum=("priority", "sum"), priority_max=("priority", "max"),
        top_barrier=("label_cluster_id", "first"),
        types=("label_type", lambda s: "; ".join(f"{k} {v}" for k, v in s.value_counts().items())),
        neighborhood=("neighborhood", "first"), nearest_stop=("nearest_stop", "first"),
        needs_reverification=("needs_reverification", "any"), lat=("lat", "mean"), lon=("lon", "mean")).reset_index()
    ix["intersection_id"] = ix.intersection_id.astype("int64")
    ix = ix.sort_values(["list", "priority_sum"], ascending=[False, False])
    ix.insert(0, "rank", ix.groupby("list").cumcount() + 1)

    Sc = S[S.geometry.interpolate(0.5, normalized=True).within(city)].copy()
    Sc["km"] = Sc.length / 1000
    mid = gpd.GeoDataFrame(Sc[["km", "status", "audit_count"]], geometry=Sc.geometry.interpolate(0.5, normalized=True), crs=UTM)
    mid = gpd.sjoin(mid, hoods[["hood", "geometry"]], how="inner", predicate="within")
    cov = mid.groupby("hood").agg(street_km=("km", "sum"),
                                  audited_km=("km", lambda s: s[mid.loc[s.index, "audit_count"] > 0].sum()),
                                  never_opened_km=("km", lambda s: s[mid.loc[s.index, "status"] == "closed"].sum()))
    parts = []
    for name, g in B.groupby("list"):
        t = g.groupby("neighborhood").agg(**{f"{name}_barriers": ("label_cluster_id", "size"), f"{name}_priority": ("priority", "sum")})
        t[f"{name}_in_top_100"] = g[g["rank"] <= 100].groupby("neighborhood").size()
        parts.append(t)
    nb = cov.join(pd.concat(parts, axis=1), how="left").fillna(0)
    nb["audited_share"] = nb.audited_km / nb.street_km
    nb["unaudited_km"] = nb.street_km - nb.audited_km
    for name in B.list.unique():
        nb[f"{name}_priority_per_audited_km"] = nb[f"{name}_priority"] / nb.audited_km.where(nb.audited_km >= 1)
    return seg, ix, nb.sort_values("repair_priority" if "repair_priority" in nb else nb.columns[0], ascending=False)


def git_state(repo):
    try:
        sha = subprocess.check_output(["git", "-C", repo, "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", repo, "-c", "core.fileMode=false", "status", "--porcelain",
                                              "src/tier2", "configs"], text=True).strip())
        return dict(git_sha=sha, tier2_code_or_config_uncommitted=dirty)
    except Exception as e:  # noqa: BLE001
        return dict(git_error=str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tier2_default.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    D, OUT = Path(cfg["data"]), Path(cfg["out"])
    OUT.mkdir(parents=True, exist_ok=True)

    city = gpd.read_file(D / "wprdc/city_boundary.geojson").to_crs(UTM).union_all()
    hoods = gpd.read_file(D / "wprdc/neighborhoods.geojson").to_crs(UTM)
    ps = json.load(open(D / "projectsidewalk/accessScoreConfig.json"))

    print("clusters ...", flush=True)
    C = load_clusters(D, city)
    list_of = {t: name for name, ts in cfg["candidates"]["lists"].items() for t in ts}
    types = list(list_of)
    counts = dict(clusters_total=len(C), clusters_in_city=int(C.in_city.sum()))
    C = C[C.in_city & C.label_type.isin(types)].copy()
    C = barrier_magnitude(C, ps, cfg)
    counts["candidate_type_clusters_in_city"] = len(C)
    counts["zero_magnitude_dropped"] = int((C.magnitude <= 0).sum())
    counts["zero_magnitude_by_type"] = C[C.magnitude <= 0].label_type.value_counts().to_dict()
    C = C[C.magnitude > 0].copy()

    print("validity ...", flush=True)
    C = attach_validity(C, D, cfg)
    counts["validation"] = C.validation.value_counts().to_dict()
    rejected = C[C.validation == "rejected"]
    if cfg["validity"]["drop_if_humans_disagree"]:
        C = C[C.validation != "rejected"].copy()

    print("street context ...", flush=True)
    C, S = attach_street_context(C, D, hoods)
    tracts = acs_tracts(D, city)
    print("demand ...", flush=True)
    C = demand_components(C, D, city, tracts, cfg)
    comp, demand, priority = score(C, cfg)
    for k, v in comp.items():
        C[f"{k}_score"] = v
    C["demand"], C["priority"] = demand, priority
    C["list"] = C.label_type.map(list_of)
    C = C.sort_values(["list", "priority"], ascending=[False, False])
    C.insert(0, "rank", C.groupby("list").cumcount() + 1)
    # Under Project Sidewalk's constants a missing ramp outweighs any obstacle, so the repair list's top is
    # all ramps; a per-type rank lets a crew or budget line for obstacles read its own ordering.
    C["type_rank"] = C.groupby("label_type").cumcount() + 1
    counts["ranked_barriers"] = C.list.value_counts().to_dict()
    counts["ranked_by_type"] = C.label_type.value_counts().to_dict()

    cols = ["list", "rank", "type_rank", "label_cluster_id", "label_type", "severity_bucket", "active_tags", "magnitude", "validation", "votes_agree",
            "votes_disagree", "validity", "transit_score", "destinations_score", "vulnerability_score", "safety_score", "demand",
            "priority", "neighborhood", "tract", "street_edge_id", "intersection_id", "way_type", "needs_reverification",
            "avg_image_capture_date", "avg_label_date", "cluster_size", "nearest_stop", "nearest_stop_m", "key_destination",
            "on_high_injury_network", "gsv_url", "lat", "lon"]
    C["vulnerability_score"] = C["vulnerability_score"] if "vulnerability_score" in C else C.vulnerability
    for c in ["magnitude", "validity", "transit_score", "destinations_score", "vulnerability_score", "safety_score", "demand", "priority"]:
        C[c] = C[c].astype(float).round(4)
    C[cols].to_csv(OUT / "barriers_ranked.csv", index=False)
    for name, g in C.groupby("list"):
        g[cols].to_csv(OUT / f"{name}_ranked.csv", index=False)
    C[cols + ["geometry"]].to_crs(4326).to_file(OUT / "barriers_ranked.geojson", driver="GeoJSON", COORDINATE_PRECISION=6)
    rejected[["label_cluster_id", "label_type", "severity_bucket", "active_tags", "votes_agree", "votes_disagree", "region_name",
              "gsv_url", "lat", "lon"]].to_csv(OUT / "barriers_rejected.csv", index=False)

    comp_cols = ["list", "rank", "label_cluster_id", "label_type", "magnitude", "validity", "validation", "transit_raw", "destinations_raw",
                 "vulnerability", "crash_raw", "on_high_injury_network", "neighborhood", "tract", "street_edge_id",
                 "needs_reverification", "pct_black_nh", "median_hh_income", "pct_disability", "pct_65_plus", "population"]
    comp_cols += ["complaints_raw"] if "complaints_raw" in C else []
    C[comp_cols].to_parquet(OUT / "components.parquet", index=False)

    print("aggregate ...", flush=True)
    seg, ix, nb = aggregate(C, S, hoods, D, city)
    ix.to_csv(OUT / "intersections_ranked.csv", index=False)
    counts["intersections_ranked"] = ix["list"].value_counts().to_dict()
    seg.drop(columns="geometry").to_csv(OUT / "segments_ranked.csv", index=False)
    seg.to_crs(4326).to_file(OUT / "segments_ranked.geojson", driver="GeoJSON", COORDINATE_PRECISION=6)
    nb.to_csv(OUT / "neighborhoods.csv", index_label="neighborhood")
    counts["segments_ranked"] = len(seg)

    manifest = json.load(open(D / "MANIFEST.json"))
    used = ["projectsidewalk/labelClusters.geojson", "projectsidewalk/validations.csv", "projectsidewalk/rawLabels.csv",
            "projectsidewalk/streets.geojson", "projectsidewalk/accessScoreConfig.json", "wprdc/prt_stops.geojson",
            "wprdc/county_assets.csv", "wprdc/hospitals.csv", "wprdc/census_tracts_2020.geojson", "wprdc/city_boundary.geojson",
            "wprdc/neighborhoods.geojson", "wprdc/high_injury_network.geojson"] + [f"wprdc/crashes_{y}.csv" for y in cfg["demand"]["crash_years"]]
    used += [f"census/acs5_tract_{t}.json" for t in ["B01003", "B01001", "B18101", "B08201", "B03002", "B19013"]]
    info = dict(run_id=cfg["run_id"], finished_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"), config=cfg,
                **git_state(str(Path(__file__).resolve().parents[2])), counts=counts,
                inputs={k: dict(sha256=manifest[k]["sha256"], retrieved_at=manifest[k]["retrieved_at"]) for k in used if k in manifest})
    json.dump(info, open(OUT / "run_info.json", "w"), indent=1, default=str)
    print(json.dumps(counts, indent=1, default=str))
    for name, g in C.groupby("list", sort=False):
        print(f"top 10 — {name}:")
        print(g.head(10)[["rank", "label_type", "severity_bucket", "active_tags", "validation", "demand", "priority", "neighborhood",
                          "nearest_stop", "key_destination"]].to_string(index=False, max_colwidth=34))


if __name__ == "__main__":
    main()
