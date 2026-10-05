#!/usr/bin/env python3
"""EDA of the Pittsburgh snapshot: Project Sidewalk's labels and audit coverage, plus the
pedestrian-demand and demographic layers Tier 2 will weight them by.

Coverage is measured against the city's whole street network, not Project Sidewalk's
"explorable" km. Those are different things: "explorable" means *opened for auditing*, it
includes four neighbouring boroughs, and roughly half the city has never been opened.
Every street edge inside the city boundary is therefore put in one of four states:
audited / open, not audited / not opened / no imagery.

Reads the snapshot written by scripts/fetch_pittsburgh_data.py and writes
  <out>/summary.json                    headline numbers + chart-ready aggregates
  <out>/tables/*.csv                    the tables behind them
  <out>/neighborhood_coverage.geojson   simplified polygons with coverage, for maps
Every number in reports/eda/pittsburgh.md comes from one of those.

usage:
  python src/eda/pittsburgh_eda.py --data /data/datasets/pittsburgh --out /data/eda/pittsburgh
"""
import argparse, json, os, sys, warnings
from pathlib import Path

# Running the env's python without `conda activate` skips the hooks that point GDAL at PROJ.
os.environ.setdefault("PROJ_DATA", os.path.join(sys.prefix, "share", "proj"))
os.environ.setdefault("GDAL_DATA", os.path.join(sys.prefix, "share", "gdal"))

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

UTM = 32617  # UTM 17N, metres
BARRIERS = ["NoCurbRamp", "Obstacle", "SurfaceProblem", "NoSidewalk"]
SEVERITY_BARRIERS = ["NoCurbRamp", "Obstacle", "SurfaceProblem"]
STATES = ["audited", "open, not audited", "not opened", "no imagery"]
NO_STREET = "no street nearby"


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())) if n and x.sum() else float("nan")


def truthy(s):
    return s.astype(str).str.lower().isin(["true", "1", "t"])


def clean(o):
    """Recursively turn pandas/numpy values into JSON-native ones (NaN -> null)."""
    if o is None or o is pd.NA or o is pd.NaT:
        return None
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, pd.DataFrame):
        return clean(o.reset_index().to_dict(orient="records"))
    if isinstance(o, pd.Series):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(o).isoformat()
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        if not np.isfinite(o):
            return None
        return round(float(o), 3) if abs(o) >= 1 else float(f"{o:.4g}")
    return o


class Tables:
    def __init__(self, out):
        self.dir = Path(out) / "tables"
        self.dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, name, df, index=True):
        df.to_csv(self.dir / f"{name}.csv", index=index)
        return df


def points(df, lat, lon):
    df = df[df[lat].notna() & df[lon].notna()]
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon], df[lat]), crs=4326).to_crs(UTM)


def nearest_edge(pts, edges, max_m):
    """Attach the nearest city street edge within max_m metres; `state` is NO_STREET if none."""
    j = gpd.sjoin_nearest(pts, edges[["street_edge_id", "state", "geometry"]], how="left",
                          max_distance=max_m, distance_col="dist_m")
    j = j[~j.index.duplicated(keep="first")]
    j["state"] = j.state.fillna(NO_STREET)
    return j


def state_shares(states, weights=None):
    s = pd.Series(1.0 if weights is None else weights.values, index=states.index).groupby(states).sum()
    return (s / s.sum()).reindex(STATES + [NO_STREET]).fillna(0.0)


# --------------------------------------------------------------------------------------
def section_labels(L, T, snapshot):
    out = {}
    L = L.copy()
    L["verdict"] = L.correct.astype(str).str.lower().map({"true": "correct", "false": "incorrect"}).fillna("unvalidated")
    out["n_labels"] = len(L)
    out["n_with_severity"] = int(L.severity.notna().sum())
    out["severity_values"] = sorted(int(v) for v in L.severity.dropna().unique())
    out["in_city"] = int(L.in_city.sum())
    out["outside_city_by_region"] = L[~L.in_city].region_name.value_counts()

    sev = pd.crosstab(L.label_type, L.severity.fillna(0).astype(int)).rename(
        columns={0: "sev_none", 1: "sev_1", 2: "sev_2", 3: "sev_3"})
    ver = pd.crosstab(L.label_type, L.verdict)
    t = pd.concat([L.label_type.value_counts().rename("labels"), sev, ver], axis=1).fillna(0)
    t["in_city"] = L[L.in_city].label_type.value_counts()
    t["mean_severity"] = L.groupby("label_type").severity.mean()
    t["validated_share"] = (t.correct + t.incorrect) / t.labels
    t["correct_share_of_validated"] = t.correct / (t.correct + t.incorrect)
    t["votes_agree"] = L.groupby("label_type").agree_count.sum()
    t["votes_disagree"] = L.groupby("label_type").disagree_count.sum()
    t["votes_unsure"] = L.groupby("label_type").unsure_count.sum()
    out["by_type"] = T("labels_by_type", t.sort_values("labels", ascending=False))

    per_user = L.user_id.value_counts()
    out["users"] = dict(n=per_user.size, top1_share=per_user.iloc[0] / len(L), top10_share=per_user.head(10).sum() / len(L),
                        gini=gini(per_user.values), median_labels_per_user=per_user.median(),
                        high_quality_label_share=truthy(L.high_quality_user).mean())

    created = pd.to_datetime(L.time_created, utc=True, format="ISO8601")
    captured = pd.to_datetime(L.image_capture_date, format="%Y-%m", errors="coerce").dt.tz_localize("UTC")
    age = (created - captured).dt.days
    out["labels_per_year"] = created.dt.year.value_counts().sort_index()
    out["image_capture_year"] = captured.dt.year.value_counts().sort_index()
    out["image_age_at_labeling_days"] = dict(mean=age.mean(), median=age.median(), p90=age.quantile(0.9))
    out["image_age_at_snapshot_years"] = dict(
        median=((snapshot - captured).dt.days / 365.25).median(),
        share_older_than_5y=((snapshot - captured).dt.days > 5 * 365.25).mean())

    per_pano = L.pano_id.value_counts()
    out["panos"] = dict(n=per_pano.size, labels_per_pano_mean=per_pano.mean(), labels_per_pano_max=per_pano.max(),
                        pano_sources=L.pano_source.value_counts())

    tags = L[["label_type", "tags"]].copy()
    tags["tag"] = tags.tags.fillna("[]").map(json.loads)
    tags = tags.explode("tag").dropna(subset=["tag"])
    top = (tags.groupby(["label_type", "tag"]).size().rename("n").reset_index()
           .sort_values(["label_type", "n"], ascending=[True, False]).groupby("label_type").head(8))
    T("label_top_tags", top, index=False)
    out["top_tags"] = {k: dict(zip(g.tag, g.n)) for k, g in top.groupby("label_type")}
    out["share_labels_with_tags"] = (L.tags.fillna("[]") != "[]").mean()
    return out


def section_validations(V, L, T):
    out = {}
    V = V.copy()
    V["year"] = V.start_timestamp.str[:4].astype(int)
    out["n"] = len(V)
    out["by_validator_type_result"] = T("validations_by_type_result", pd.crosstab(V.validator_type, V.validation_result))
    out["by_year_validator_type"] = T("validations_by_year", pd.crosstab(V.year, V.validator_type))
    out["by_source"] = V.source.value_counts()
    out["ai_share"] = (V.validator_type == "AI").mean()

    per = V.pivot_table(index="label_id", columns=["validator_type", "validation_result"], aggfunc="size", fill_value=0)
    per.columns = [f"{a}_{b}" for a, b in per.columns]
    for c in ["Human_Agree", "Human_Disagree", "Human_Unsure", "AI_Agree", "AI_Disagree", "AI_Unsure"]:
        if c not in per:
            per[c] = 0
    m = L.set_index("label_id")[["label_type", "agree_count", "disagree_count", "unsure_count", "correct"]].join(per, how="left").fillna(0)
    # Do the per-label vote counts in rawLabels include the AI's votes? Matters for "evaluate against human disagreement".
    out["rawlabels_agree_count_matches"] = dict(
        human_plus_ai=(m.agree_count == m.Human_Agree + m.AI_Agree).mean(),
        human_only=(m.agree_count == m.Human_Agree).mean())
    human_n = m.Human_Agree + m.Human_Disagree + m.Human_Unsure
    ai_n = m.AI_Agree + m.AI_Disagree + m.AI_Unsure
    out["labels_with_any_human_vote"] = int((human_n > 0).sum())
    out["labels_with_any_ai_vote"] = int((ai_n > 0).sum())
    out["labels_only_ai_votes"] = int(((ai_n > 0) & (human_n == 0)).sum())
    out["labels_no_votes"] = int(((ai_n == 0) & (human_n == 0)).sum())

    H = V[V.validator_type == "Human"]
    ht = pd.crosstab(H.label_type, H.validation_result)
    ht["human_agree_rate_decisive"] = ht.Agree / (ht.Agree + ht.Disagree)
    A = V[V.validator_type == "AI"]
    at = pd.crosstab(A.label_type, A.validation_result)
    at["ai_agree_rate_decisive"] = at.Agree / (at.Agree + at.Disagree)
    at["ai_unsure_rate"] = at.Unsure / at.sum(axis=1)
    out["agreement_by_label_type"] = T("validation_agreement_by_type", ht.join(at, lsuffix="_human", rsuffix="_ai", how="outer"))

    dec = H[H.validation_result.isin(["Agree", "Disagree"])]
    g = dec.groupby("label_id").validation_result.agg(n="size", agree_frac=lambda s: (s == "Agree").mean())
    multi = g[g.n >= 2]
    out["human_unanimity_labels_with_2plus_decisive_votes"] = dict(n_labels=len(multi),
                                                                   unanimous_share=((multi.agree_frac == 0) | (multi.agree_frac == 1)).mean())

    def verdict(agree, disagree, unsure):
        return np.select([agree > disagree, disagree > agree, (agree + disagree) > 0], ["Agree", "Disagree", "Tie"],
                         default=np.where(unsure > 0, "Unsure", "none"))
    m["human_verdict"] = verdict(m.Human_Agree, m.Human_Disagree, m.Human_Unsure)
    m["ai_verdict"] = verdict(m.AI_Agree, m.AI_Disagree, m.AI_Unsure)
    both = m[(m.human_verdict != "none") & (m.ai_verdict != "none")]
    out["human_vs_ai_verdict"] = T("human_vs_ai_verdict", pd.crosstab(both.human_verdict, both.ai_verdict))
    return out


def section_clusters(LC, T):
    t = LC.groupby("label_type").agg(clusters=("label_cluster_id", "size"), labels=("cluster_size", "sum"),
                                    mean_cluster_size=("cluster_size", "mean"), median_severity=("median_severity", "mean"))
    return dict(n_clusters=len(LC), n_labels_in_clusters=int(LC.cluster_size.sum()),
                cluster_size_hist=LC.cluster_size.value_counts().sort_index(),
                at_intersection_share=LC.intersection_id.notna().mean(),
                by_type=T("clusters_by_type", t.sort_values("clusters", ascending=False)))


def section_streets(S, C, regions, hoods, Lp, overall, T):
    out = {}
    # -- the deployment as Project Sidewalk defines it
    op = S[S.status == "open"]
    out["deployment"] = dict(
        explorable_km=op.km.sum(), explorable_km_in_city=op.km[op.in_city].sum(),
        explorable_km_outside_city=op.km[~op.in_city].sum(),
        outside_city_by_region=op[~op.in_city].groupby("region_name").km.sum().sort_values(ascending=False),
        audited_km=S.km[S.audit_count > 0].sum(), audited_km_outside_city=S.km[(S.audit_count > 0) & ~S.in_city].sum(),
        km_by_status_all=S.groupby("status").km.sum(),
        overallStats_api={k: overall.get(k) for k in ["km_explorable", "km_explored", "km_explored_no_overlap",
                                                      "km_needs_reaudit", "km_explored_multiple_users", "km_explored_single_user"]})

    # -- the city's street network
    out["city_network_km"] = C.km.sum()
    out["city_km_by_state"] = C.groupby("state").km.sum().reindex(STATES).fillna(0)
    out["city_audited_share"] = C.km[C.state == "audited"].sum() / C.km.sum()
    out["city_opened_share"] = C.km[C.state.isin(STATES[:2])].sum() / C.km.sum()
    out["city_unaudited_km"] = C.km[C.state != "audited"].sum()
    out["city_audited_outdated_km"] = C.km[(C.state == "audited") & truthy(C.outdated)].sum()
    out["city_multi_audit_km"] = C.km[C.audit_count >= 2].sum()
    out["city_audit_count_km"] = C.groupby(C.audit_count.clip(upper=3)).km.sum()
    last = pd.to_datetime(C.last_label_date, utc=True, errors="coerce")
    aud = C.state == "audited"
    out["city_audited_km_by_last_label_year"] = C[aud].groupby(last[aud].dt.year.astype("Int64")).km.sum()

    wt = C.pivot_table(index="way_type", columns="state", values="km", aggfunc="sum").reindex(columns=STATES).fillna(0)
    wt["total_km"] = wt.sum(axis=1)
    wt["coverage"] = wt.audited / wt.total_km
    out["by_way_type"] = T("city_streets_by_way_type", wt.sort_values("total_km", ascending=False))

    rg = regions[["region_id", "name", "total_distance_m", "audited_distance_m", "completion_rate", "label_count", "user_count"]].copy()
    rg["total_km_api"] = rg.total_distance_m / 1000
    rg["audited_km_api"] = rg.audited_distance_m / 1000
    out["regions_api"] = dict(n=len(rg), total_km=rg.total_km_api.sum(), quality_audited_km=rg.audited_km_api.sum())
    T("regions_coverage", rg.drop(columns=["total_distance_m", "audited_distance_m"]).sort_values("completion_rate"), index=False)

    # -- neighbourhoods, the unit the city plans in
    mid = gpd.GeoDataFrame(C[["km", "state"]], geometry=C.geometry.interpolate(0.5, normalized=True), crs=UTM)
    mid = gpd.sjoin(mid, hoods[["hood", "geometry"]], how="left", predicate="within")
    mid = mid[~mid.index.duplicated()]
    C["hood"] = mid.hood.fillna("(outside neighbourhood polygons)")
    lab = gpd.sjoin(Lp[Lp.in_city], hoods[["hood", "geometry"]], how="left", predicate="within")
    lab = lab[~lab.index.duplicated()]
    hb = lab.assign(barrier=lab.label_type.isin(BARRIERS),
                    severe=lab.label_type.isin(SEVERITY_BARRIERS) & (lab.severity == 3),
                    nosidewalk=lab.label_type == "NoSidewalk").groupby("hood").agg(
        labels=("label_id", "size"), barrier_labels=("barrier", "sum"), severe_barriers=("severe", "sum"),
        nosidewalk_labels=("nosidewalk", "sum"))
    h = C.pivot_table(index="hood", columns="state", values="km", aggfunc="sum").reindex(columns=STATES).fillna(0)
    h.columns = ["audited_km", "open_unaudited_km", "not_opened_km", "no_imagery_km"]
    h["total_km"] = h.sum(axis=1)
    h = h.join(hb, how="left").fillna(0)
    h["coverage"] = h.audited_km / h.total_km
    h["opened_share"] = (h.audited_km + h.open_unaudited_km) / h.total_km
    h["barriers_per_audited_km"] = h.barrier_labels / h.audited_km.where(h.audited_km >= 1)
    h = h[h.index != "(outside neighbourhood polygons)"]
    out["neighborhoods"] = dict(n=len(h), never_opened=int((h.opened_share < 0.01).sum()),
                                opened_but_under_10pct_audited=int(((h.opened_share >= 0.01) & (h.coverage < 0.10)).sum()),
                                coverage_ge_50pct=int((h.coverage >= 0.5).sum()),
                                coverage_quantiles=h.coverage.quantile([0.1, 0.25, 0.5, 0.75, 0.9]))
    out["by_neighborhood"] = T("neighborhood_coverage", h.sort_values("coverage", ascending=False))
    return out, C, h


def section_access_scores(ASS, ASR, T):
    s = ASS.score.dropna()
    ASR = ASR[["region_id", "name", "score", "coverage", "audited_street_count", "total_street_count", "intersection_score"]]
    T("access_score_regions", ASR.sort_values("score"), index=False)
    return dict(streets_scored=len(s), streets_total=len(ASS), street_score_quantiles=s.quantile([0.05, 0.25, 0.5, 0.75, 0.95]),
                street_score_hist=np.histogram(s, bins=10, range=(0, 1))[0].tolist(),
                region_score_vs_coverage_spearman=spearmanr(ASR.score, ASR.coverage, nan_policy="omit")[0])


def section_demand(D, city, C, T):
    out = {}

    stops = gpd.read_file(D / "wprdc/prt_stops.geojson").to_crs(UTM)
    stops = stops[stops.within(city)].copy()
    stops["trips_wd"] = pd.to_numeric(stops.trips_wd, errors="coerce").fillna(0)
    j = nearest_edge(stops, C, 50)
    out["stops"] = dict(n_in_city=len(stops), weekday_trips=stops.trips_wd.sum(),
                        state_share=state_shares(j.state), weekday_trip_state_share=state_shares(j.state, j.trips_wd),
                        modes=stops["mode"].value_counts(), trips_wd_quantiles=stops.trips_wd.quantile([0.25, 0.5, 0.75, 0.95]))

    rides = pd.read_csv(D / "wprdc/prt_monthly_ridership_by_route.csv")
    rides["month"] = pd.to_datetime(rides.Month_Start, format="%m/%d/%Y", errors="coerce")
    latest = rides.month.max()
    wk = rides[(rides.month == latest) & (rides.Day_Type.str.upper() == "WEEKDAY")]
    out["ridership"] = dict(latest_month=latest, first_month=rides.month.min(),
                            system_avg_weekday_riders=wk.Avg_Riders.sum(),
                            top_routes=wk.sort_values("Avg_Riders", ascending=False).head(10)[["Route_Full_Name", "Avg_Riders"]].values.tolist())

    # destinations: where people who most need a sidewalk are going
    assets = pd.read_csv(D / "wprdc/county_assets.csv", low_memory=False)
    keep = ["schools", "nursing_homes", "senior_centers", "health_centers", "doctors_offices", "pharmacies", "libraries",
            "affordable_housing", "supermarkets", "rec_centers", "universities", "child_care_centers", "parks_and_facilities",
            "food_banks", "homeless_shelters", "va_facilities"]
    ap = points(assets[assets.asset_type.isin(keep)], "latitude", "longitude")
    ap = ap[ap.within(city)][["asset_type", "name", "geometry"]].rename(columns={"asset_type": "category"})
    hosp = pd.read_csv(D / "wprdc/hospitals.csv")
    hp = points(hosp, "Y", "X")
    hp = hp[hp.within(city)].assign(category="hospitals (2015)").rename(columns={"Facility": "name"})[["category", "name", "geometry"]]
    poi = pd.concat([ap, hp], ignore_index=True)
    pj = nearest_edge(poi, C, 100)
    dt = pd.crosstab(pj.category, pj.state, normalize="index").reindex(columns=STATES + [NO_STREET]).fillna(0)
    dt.insert(0, "n_in_city", pj.category.value_counts())
    out["destinations"] = T("destinations_street_state", dt.sort_values("n_in_city", ascending=False))
    out["destinations_all"] = dict(n=len(pj), state_share=state_shares(pj.state))

    frames = []
    for y in range(2019, 2026):
        p = D / f"wprdc/crashes_{y}.csv"
        if p.exists():
            frames.append(pd.read_csv(p, low_memory=False, usecols=lambda c: c in {
                "CRASH_YEAR", "PED_COUNT", "PED_DEATH_COUNT", "PED_MAJ_INJ_COUNT", "DEC_LAT", "DEC_LONG"}))
    cr = pd.concat(frames, ignore_index=True)
    ped = points(cr[cr.PED_COUNT > 0], "DEC_LAT", "DEC_LONG")
    ped = ped[ped.within(city)]
    cj = nearest_edge(ped, C, 30)
    out["pedestrian_crashes"] = dict(years=f"{int(cr.CRASH_YEAR.min())}-{int(cr.CRASH_YEAR.max())}", n_in_city=len(ped),
                                     by_year=ped.CRASH_YEAR.value_counts().sort_index(),
                                     ped_deaths=ped.PED_DEATH_COUNT.sum(), ped_major_injuries=ped.PED_MAJ_INJ_COUNT.sum(),
                                     state_share=state_shares(cj.state))

    hin = gpd.read_file(D / "wprdc/high_injury_network.geojson").to_crs(UTM)
    hin_km = hin.length.sum() / 1000

    def near(mask):
        return hin.intersection(C[mask].buffer(20).union_all()).length.sum() / 1000 / hin_km
    out["high_injury_network"] = dict(n_corridors=len(hin), km=hin_km, pedestrian_crashes=hin.Count_PedestrianCrash.sum(),
                                      share_len_near_audited=near(C.state == "audited"),
                                      share_len_near_opened=near(C.state.isin(STATES[:2])),
                                      share_len_near_any_city_street=near(C.km > 0))

    steps = gpd.read_file(D / "wprdc/city_steps.geojson").to_crs(UTM)
    out["city_steps"] = dict(n=len(steps), km=steps.length.sum() / 1000,
                             steps_total=pd.to_numeric(steps.number_of_steps, errors="coerce").sum(),
                             note="public stairways; not street edges, so outside Project Sidewalk's audit network")

    walk = pd.read_csv(D / "wprdc/walkability_blockgroup.csv", dtype={"GEOID": str})
    out["sidewalk_to_street_ratio_blockgroups"] = dict(n=len(walk), median=walk.Ratio.median(),
                                                       share_below_1=(walk.Ratio < 1).mean())

    xw = pd.read_csv(D / "wprdc/crosswalks.csv", low_memory=False)
    out["city_crosswalk_inventory"] = dict(n=len(xw), types=xw["type"].value_counts())
    return out


# The 13 of 147 311 subjects that are about walking. Snow/ice on roads, potholes and signal
# repair are left out: they are street or traffic requests, not sidewalk ones.
PED_311 = {
    "Broken Sidewalk": "sidewalk condition",
    "Sidewalk/Curb/ADA Ramp Maintenance": "sidewalk condition",
    "Sidewalk Repair Program": "sidewalk condition",
    "Blocked or Closed Sidewalks or Trails": "sidewalk obstruction",
    "Sidewalk has Ice or Litter": "sidewalk obstruction",
    "Curb Cuts": "curb ramps / ADA",
    "ADA Ramp Installation": "curb ramps / ADA",
    "Americans with Disabilities Act (ADA) Concerns": "curb ramps / ADA",
    "Accessibility Related Construction Concerns": "curb ramps / ADA",
    "Crosswalk, Curb, and Street Markings - Maintenance": "crosswalks / pedestrian safety",
    "Pedestrian Safety Concerns - Non Signal Related": "crosswalks / pedestrian safety",
    "Blocked City Steps": "city steps",
    "Repair City Steps": "city steps",
}
BARRIER_311_GROUPS = ["sidewalk condition", "sidewalk obstruction", "curb ramps / ADA"]


def section_311(D, city, C, hoods, hood_table, T):
    d = pd.read_csv(D / "wprdc/pgh_311.csv", low_memory=False, usecols=["subject", "created_date_et", "latitude", "longitude"])
    years = d.created_date_et.str[:4]
    out = dict(n_requests=len(d), n_subjects=d.subject.nunique(), first_year=years.min(), last_year=years.max())
    p = d[d.subject.isin(PED_311)].copy()
    p["group"] = p.subject.map(PED_311)
    p["year"] = p.created_date_et.str[:4].astype(int)
    out["pedestrian_requests"] = len(p)
    out["pedestrian_share"] = len(p) / len(d)
    subj = p.groupby(["group", "subject"]).size().rename("requests").reset_index().sort_values("requests", ascending=False)
    out["by_subject"] = T("pgh_311_pedestrian_subjects", subj, index=False)
    out["by_year_group"] = T("pgh_311_pedestrian_by_year", pd.crosstab(p.year, p.group))

    pts = points(p, "latitude", "longitude")
    pts = pts[pts.within(city)]
    j = nearest_edge(pts, C, 30)
    out["geocoded_in_city"] = len(pts)
    out["state_share"] = state_shares(j.state)
    out["barrier_complaints_state_share"] = state_shares(j.state[j.group.isin(BARRIER_311_GROUPS)])

    # Neighbourhood level. Coverage vs complaint density uses the whole network; label density
    # vs complaint density uses audited streets only, so both counts sit on the same streets.
    b = j[j.group.isin(BARRIER_311_GROUPS)]
    b = gpd.sjoin(b.drop(columns=["index_right"], errors="ignore"), hoods[["hood", "geometry"]], how="inner", predicate="within")
    h = hood_table.join(b.groupby("hood").size().rename("barrier_complaints"), how="left")
    h = h.join(b[b.state == "audited"].groupby("hood").size().rename("barrier_complaints_on_audited"), how="left")
    h = h.fillna({"barrier_complaints": 0, "barrier_complaints_on_audited": 0})
    h["complaints_per_km"] = h.barrier_complaints / h.total_km
    h["complaints_per_audited_km"] = h.barrier_complaints_on_audited / h.audited_km.where(h.audited_km >= 5)
    T("pgh_311_by_neighborhood", h[["total_km", "audited_km", "coverage", "barrier_complaints", "complaints_per_km",
                                    "barrier_complaints_on_audited", "complaints_per_audited_km", "barriers_per_audited_km"]]
      .sort_values("complaints_per_km", ascending=False))
    rc = spearmanr(h.coverage, h.complaints_per_km)
    hh = h.dropna(subset=["complaints_per_audited_km", "barriers_per_audited_km"])
    rb = spearmanr(hh.barriers_per_audited_km, hh.complaints_per_audited_km)
    out["neighborhoods"] = dict(n=len(h), coverage_vs_complaints_rho=rc[0], coverage_vs_complaints_p=rc[1],
                                n_with_5km_audited=len(hh), label_density_vs_complaint_density_rho=rb[0],
                                label_density_vs_complaint_density_p=rb[1])
    return out


def section_equity(D, city, C, Lp, T):
    tracts = gpd.read_file(D / "wprdc/census_tracts_2020.geojson").to_crs(UTM)
    tracts = tracts[tracts.representative_point().within(city)][["GEOID", "NAMELSAD", "geometry"]]

    def est(table):
        d = json.load(open(D / f"census/acs5_tract_{table}.json"))
        rows = {g[7:]: v[table]["estimate"] for g, v in d["data"].items()}
        return pd.DataFrame.from_dict(rows, orient="index"), d["release"]["name"]

    b01003, release = est("B01003")
    b01001, _ = est("B01001"); b19013, _ = est("B19013"); b03002, _ = est("B03002"); b17001, _ = est("B17001")
    b18101, _ = est("B18101"); b08201, _ = est("B08201"); b08301, _ = est("B08301")
    acs = pd.DataFrame(index=b01003.index)
    acs["population"] = b01003.B01003001
    acs["median_hh_income"] = b19013.B19013001
    older = [f"B01001{i:03d}" for i in list(range(20, 26)) + list(range(44, 50))]
    acs["pct_65_plus"] = b01001[older].sum(axis=1) / b01001.B01001001
    acs["pct_black_nh"] = b03002.B03002004 / b03002.B03002001
    acs["pct_white_nh"] = b03002.B03002003 / b03002.B03002001
    acs["pct_poverty"] = b17001.B17001002 / b17001.B17001001
    disab = [f"B18101{i:03d}" for i in (4, 7, 10, 13, 16, 19, 23, 26, 29, 32, 35, 38)]
    acs["pct_disability"] = b18101[disab].sum(axis=1) / b18101.B18101001
    acs["pct_zero_vehicle_hh"] = b08201.B08201002 / b08201.B08201001
    acs["pct_walk_to_work"] = b08301.B08301019 / b08301.B08301001
    acs["pct_transit_to_work"] = b08301.B08301010 / b08301.B08301001

    mid = gpd.GeoDataFrame(C[["km", "state"]], geometry=C.geometry.interpolate(0.5, normalized=True), crs=UTM)
    mid = gpd.sjoin(mid, tracts, how="inner", predicate="within")
    cov = mid.pivot_table(index="GEOID", columns="state", values="km", aggfunc="sum").reindex(columns=STATES).fillna(0)
    cov["total_km"] = cov.sum(axis=1)
    cov["coverage"] = cov.audited / cov.total_km
    cov["opened_share"] = (cov.audited + cov["open, not audited"]) / cov.total_km
    lab = gpd.sjoin(Lp[Lp.label_type.isin(BARRIERS) & Lp.in_city], tracts, how="inner", predicate="within")
    cov["barrier_labels"] = lab.groupby("GEOID").size()
    cov = cov.fillna({"barrier_labels": 0})
    cov["barriers_per_audited_km"] = cov.barrier_labels / cov.audited.where(cov.audited >= 1)
    df = tracts.drop(columns="geometry").set_index("GEOID").join(cov, how="inner").join(acs, how="left")
    df = df[(df.total_km >= 2) & (df.population > 0)]
    T("tract_coverage_acs", df)

    vars_ = ["median_hh_income", "pct_black_nh", "pct_white_nh", "pct_poverty", "pct_65_plus", "pct_disability",
             "pct_zero_vehicle_hh", "pct_walk_to_work", "pct_transit_to_work", "population"]
    corr = {}
    for v in vars_:
        row = dict(n=int(df[v].notna().sum()))
        for metric in ["coverage", "opened_share", "barriers_per_audited_km"]:
            d = df[[metric, v]].dropna()
            r = spearmanr(d[metric], d[v])
            row[f"{metric}_rho"], row[f"{metric}_p"], row[f"{metric}_n"] = r[0], r[1], len(d)
        corr[v] = row
    T("equity_spearman", pd.DataFrame(corr).T)

    def by_quartile(col, labels):
        q = pd.qcut(df[col], 4, labels=labels)
        g = df.groupby(q, observed=True).agg(tracts=("total_km", "size"), total_km=("total_km", "sum"), audited_km=("audited", "sum"),
                                             opened_km=("open, not audited", "sum"), barrier_labels=("barrier_labels", "sum"),
                                             population=("population", "sum"))
        g["opened_km"] += g.audited_km
        g["coverage"] = g.audited_km / g.total_km
        g["opened_share"] = g.opened_km / g.total_km
        g["barriers_per_audited_km"] = g.barrier_labels / g.audited_km
        return g
    inc = by_quartile("median_hh_income", ["Q1 lowest income", "Q2", "Q3", "Q4 highest income"])
    blk = by_quartile("pct_black_nh", ["Q1 lowest % Black", "Q2", "Q3", "Q4 highest % Black"])
    T("coverage_by_income_quartile", inc); T("coverage_by_black_share_quartile", blk)
    scatter = df.reset_index()[["GEOID", "coverage", "opened_share", "median_hh_income", "pct_black_nh", "total_km", "population"]]
    return dict(acs_release=release, n_tracts=len(df), spearman=corr, by_income_quartile=inc, by_black_share_quartile=blk,
                scatter=scatter.round(4).to_dict(orient="records"))


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data/datasets/pittsburgh")
    ap.add_argument("--out", default="/data/eda/pittsburgh")
    a = ap.parse_args()
    D, OUT = Path(a.data), Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    T = Tables(OUT)

    manifest = json.load(open(D / "MANIFEST.json"))
    snapshot = pd.Timestamp(min(m["retrieved_at"] for m in manifest.values()))
    city = gpd.read_file(D / "wprdc/city_boundary.geojson").to_crs(UTM).union_all()
    hoods = gpd.read_file(D / "wprdc/neighborhoods.geojson").to_crs(UTM)

    S = gpd.read_file(D / "projectsidewalk/streets.geojson").to_crs(UTM)
    S["km"] = S.length / 1000
    S["in_city"] = S.geometry.interpolate(0.5, normalized=True).within(city)
    S["state"] = np.select([S.audit_count > 0, S.status == "open", S.status == "closed", S.status == "no_imagery"],
                           STATES, default="other")
    C = S[S.in_city].copy()

    L = pd.read_csv(D / "projectsidewalk/rawLabels.csv", low_memory=False)
    Lp = points(L, "latitude", "longitude")
    Lp["in_city"] = Lp.within(city)
    L["in_city"] = Lp.in_city.reindex(L.index).fillna(False).astype(bool)
    V = pd.read_csv(D / "projectsidewalk/validations.csv", low_memory=False)
    LC = gpd.read_file(D / "projectsidewalk/labelClusters.geojson")
    regions = gpd.read_file(D / "projectsidewalk/regions.geojson")
    overall = json.load(open(D / "projectsidewalk/overallStats.json"))

    summary = dict(snapshot_retrieved_at=snapshot, city_area_km2=city.area / 1e6)
    print("labels ...", flush=True);       summary["labels"] = section_labels(L, T, snapshot)
    print("validations ...", flush=True);  summary["validations"] = section_validations(V, L, T)
    print("clusters ...", flush=True);     summary["clusters"] = section_clusters(LC, T)
    print("streets ...", flush=True)
    summary["streets"], C, hood_table = section_streets(S, C, regions, hoods, Lp, overall, T)
    print("access scores ...", flush=True)
    summary["access_scores"] = section_access_scores(gpd.read_file(D / "projectsidewalk/accessScoreStreets.geojson"),
                                                     gpd.read_file(D / "projectsidewalk/accessScoreRegions.geojson"), T)
    print("demand ...", flush=True);       summary["demand"] = section_demand(D, city, C, T)
    print("311 ...", flush=True);          summary["pgh_311"] = section_311(D, city, C, hoods, hood_table, T)
    print("equity ...", flush=True);       summary["equity"] = section_equity(D, city, C, Lp, T)

    hg = hoods[["hood", "geometry"]].merge(hood_table.reset_index(), on="hood", how="left")
    hg["geometry"] = hg.geometry.simplify(15)
    hg.to_crs(4326).to_file(OUT / "neighborhood_coverage.geojson", driver="GeoJSON", COORDINATE_PRECISION=5)

    summary["sources"] = {k: dict(url=v["url"], page=v["page"], bytes=v["bytes"], sha256=v["sha256"],
                                  retrieved_at=v["retrieved_at"], license=v.get("license"), desc=v.get("desc"))
                          for k, v in manifest.items()}
    with open(OUT / "summary.json", "w") as f:
        json.dump(clean(summary), f, indent=1)
    print("wrote", OUT / "summary.json", flush=True)


if __name__ == "__main__":
    main()
