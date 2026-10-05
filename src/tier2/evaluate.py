#!/usr/bin/env python3
"""Tier 2 evaluation. A ranking has no ground truth, so it is judged in three other ways:

1. Robustness. How far does the ranking move when its judgment calls change?
   - demand weights: Dirichlet draws around the configured weights, and leave-one-component-out
   - the demand floor, the walking radius, and Project Sidewalk's other weight presets
   - ignoring human validation altogether
   Each change is reported per list as Kendall's tau against the default ranking, plus top-k overlap.
2. Agreement with Project Sidewalk's own access score, street by street, first on condition alone
   and then after demand weighting. This shows how much the demand layer reorders things.
3. Who the list serves. The top-k by tract quartile of % Black residents, income and disability,
   against the share of all known barriers and of residents. It also checks whether barriers near
   311 sidewalk complaints rank higher (311 is a biased signal, so this is a sanity check).

Writes <out>/evaluation.json and <out>/eval_*.csv.

usage:
  python src/tier2/evaluate.py --config configs/tier2_default.yaml
"""
import argparse, copy, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prioritize as P  # noqa: E402  (also sets PROJ_DATA)

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.stats import kendalltau, spearmanr  # noqa: E402


def ranks_within(lists, prio):
    return pd.Series(prio).groupby(lists).rank(ascending=False, method="first").to_numpy()


def compare(base, alt, lists, ks):
    rb, ra = ranks_within(lists, base), ranks_within(lists, alt)
    out = {}
    for name in pd.unique(lists):
        m = lists == name
        row = dict(kendall_tau=float(kendalltau(base[m], alt[m])[0]))
        for k in ks:
            a, b = set(np.flatnonzero(m & (rb <= k))), set(np.flatnonzero(m & (ra <= k)))
            row[f"top{k}_overlap"] = len(a & b) / min(k, int(m.sum()))
        out[name] = row
    return out


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tier2_default.yaml")
    a = ap.parse_args()
    cfg = P.load_config(a.config)
    D, OUT, ev = Path(cfg["data"]), Path(cfg["out"]), cfg["evaluation"]
    ks = ev["top_k"]

    comp = pd.read_parquet(OUT / "components.parquet")
    ranked = pd.read_csv(OUT / "barriers_ranked.csv", usecols=["label_cluster_id", "lat", "lon", "way_type", "priority"])
    C = comp.merge(ranked, on="label_cluster_id", how="inner").reset_index(drop=True)
    lists = C["list"].to_numpy()
    _, _, base = P.score(C, cfg)
    res = dict(run_id=cfg["run_id"], n=C["list"].value_counts().to_dict(),
               reproduces_stored_priority=bool(np.allclose(base, C.priority.to_numpy(), atol=1e-3)))

    # ---- 1a. Dirichlet draws around the demand weights
    rng = np.random.default_rng(ev["seed"])
    w0 = cfg["demand"]["weights"]
    keys = list(w0)
    alpha = np.array([w0[k] for k in keys]) * ev["dirichlet_concentration"]
    rb = ranks_within(lists, base)
    keep = np.zeros(len(C))
    rows = []
    for _ in range(ev["n_weight_draws"]):
        w = dict(zip(keys, rng.dirichlet(alpha)))
        _, _, p = P.score(C, cfg, weights=w)
        keep += ranks_within(lists, p) <= 100
        c = compare(base, p, lists, ks)
        rows.append({**{f"w_{k}": w[k] for k in keys}, **{f"{n}_{m}": v for n, r in c.items() for m, v in r.items()}})
    draws = pd.DataFrame(rows)
    draws.to_csv(OUT / "eval_weight_draws.csv", index=False)
    dir_sum = {}
    for name in pd.unique(lists):
        m = (lists == name) & (rb <= 100)
        dir_sum[name] = dict(
            kendall_tau_mean=draws[f"{name}_kendall_tau"].mean(), kendall_tau_p5=draws[f"{name}_kendall_tau"].quantile(0.05),
            **{f"top{k}_overlap_mean": draws[f"{name}_top{k}_overlap"].mean() for k in ks},
            **{f"top{k}_overlap_p5": draws[f"{name}_top{k}_overlap"].quantile(0.05) for k in ks},
            top100_items_kept_in_80pct_of_draws=float((keep[m] >= 0.8 * ev["n_weight_draws"]).mean()))
    res["weight_draws"] = dict(n=ev["n_weight_draws"], concentration=ev["dirichlet_concentration"],
                               weight_p5_p95={k: [draws[f"w_{k}"].quantile(0.05), draws[f"w_{k}"].quantile(0.95)] for k in keys},
                               by_list=dir_sum)

    # ---- 1b. single changes
    alt = {}
    for k in keys:
        w = dict(w0); w[k] = 0.0
        alt[f"drop demand component: {k}"] = compare(base, P.score(C, cfg, weights=w)[2], lists, ks)
    for f in ev["floor_alternatives"]:
        alt[f"demand floor {f}"] = compare(base, P.score(C, cfg, floor=f)[2], lists, ks)
    alt["condition only (ignore demand)"] = compare(base, P.score(C, cfg, floor=1.0)[2], lists, ks)
    C_nv = C.copy(); C_nv["validity"] = 1.0
    alt["ignore human validation"] = compare(base, P.score(C_nv, cfg)[2], lists, ks)

    city = gpd.read_file(D / "wprdc/city_boundary.geojson").to_crs(P.UTM).union_all()
    tracts = P.acs_tracts(D, city)
    G = gpd.GeoDataFrame(C.copy(), geometry=gpd.points_from_xy(C.lon, C.lat), crs=4326).to_crs(P.UTM)
    for r in ev["radius_alternatives"]:
        c2 = copy.deepcopy(cfg); c2["demand"]["radius_m"] = r
        Gr = P.demand_components(G.copy(), D, city, tracts, c2)
        alt[f"walking radius {r} m"] = compare(base, P.score(Gr, c2)[2], lists, ks)

    ps = json.load(open(D / "projectsidewalk/accessScoreConfig.json"))
    clusters = P.load_clusters(D, city)
    clusters = clusters[clusters.label_cluster_id.isin(C.label_cluster_id)]
    for preset in [p for p in ps["presets"] if p != cfg["barrier"]["weights_preset"]]:
        c3 = copy.deepcopy(cfg); c3["barrier"]["weights_preset"] = preset
        mag = P.barrier_magnitude(clusters.copy(), ps, c3).set_index("label_cluster_id").magnitude
        C3 = C.copy(); C3["magnitude"] = C3.label_cluster_id.map(mag).to_numpy()
        alt[f"Project Sidewalk preset: {preset}"] = compare(base, P.score(C3, c3)[2], lists, ks)
    res["single_changes"] = alt
    pd.DataFrame([{"change": ch, "list": n, **r} for ch, d in alt.items() for n, r in d.items()]).to_csv(OUT / "eval_single_changes.csv", index=False)

    # ---- 2. agreement with Project Sidewalk's access score, per street
    st = gpd.read_file(D / "projectsidewalk/accessScoreStreets.geojson", columns=["street_edge_id", "score"])
    st = pd.DataFrame(st.drop(columns="geometry")).dropna(subset=["score"])
    cond = C.assign(cond=C.magnitude * C.validity).groupby("street_edge_id").cond.sum()
    prio = C.assign(p=base).groupby("street_edge_id").p.sum()
    df = st.assign(condition=st.street_edge_id.map(cond).fillna(0), priority=st.street_edge_id.map(prio).fillna(0),
                   badness=1 - st.score)
    df = df[df.street_edge_id.isin(set(C.street_edge_id)) | (df.badness > 0)]

    def top_overlap(x, y, k=100):
        return len(set(df.nlargest(k, x).street_edge_id) & set(df.nlargest(k, y).street_edge_id)) / k
    res["vs_project_sidewalk_score"] = dict(
        streets=len(df),
        spearman_condition_vs_ps=spearmanr(df.condition, df.badness)[0], spearman_priority_vs_ps=spearmanr(df.priority, df.badness)[0],
        worst100_overlap_condition_vs_ps=top_overlap("condition", "badness"), worst100_overlap_priority_vs_ps=top_overlap("priority", "badness"),
        top100_overlap_condition_vs_priority=top_overlap("condition", "priority"))

    # ---- 3a. equity of the top-k
    eq_rows = []
    for var, label in [("pct_black_nh", "% Black"), ("median_hh_income", "median income"), ("pct_disability", "% disability")]:
        cuts = tracts[var].quantile([0.25, 0.5, 0.75]).to_numpy()
        tq = np.searchsorted(cuts, tracts[var].to_numpy(), side="right") + 1
        pop = pd.Series(tracts.population.to_numpy()).groupby(tq).sum()
        bq = pd.Series(np.searchsorted(cuts, C[var].to_numpy(), side="right") + 1).where(C[var].notna())
        for name in pd.unique(lists):
            m = lists == name
            base_share = bq[m].value_counts(normalize=True)
            for k in ks + [int(m.sum())]:
                sel = m & (rb <= k)
                share = bq[sel].value_counts(normalize=True)
                for q in [1, 2, 3, 4]:
                    eq_rows.append(dict(variable=label, list=name, top_k="all" if k == int(m.sum()) else k, quartile=q,
                                        share_of_top_k=share.get(q, 0.0), share_of_all_barriers=base_share.get(q, 0.0),
                                        share_of_residents=pop.get(q, 0) / pop.sum()))
    eq = pd.DataFrame(eq_rows)
    eq.to_csv(OUT / "eval_equity.csv", index=False)
    res["equity_top100_Q4"] = {f"{r.variable} | {r.list}": dict(share_of_top_100=r.share_of_top_k, share_of_all_barriers=r.share_of_all_barriers,
                                                               share_of_residents=r.share_of_residents)
                               for r in eq[(eq.top_k == 100) & (eq.quartile == 4)].itertuples()}
    cov_path = Path("/data/eda/pittsburgh/tables/tract_coverage_acs.csv")
    if cov_path.exists():
        tc = pd.read_csv(cov_path, dtype={"GEOID": str})
        low = tc[tc.coverage < 0.10]
        res["blind_spot"] = dict(tracts_under_10pct_audited=len(low), residents=int(low.population.sum()),
                                 share_of_residents=float(low.population.sum() / tc.population.sum()),
                                 note="barriers there are mostly unknown, so they can barely enter either list")

    # ---- 3b. 311 sanity check (repair list)
    c311 = pd.read_csv(D / "wprdc/pgh_311.csv", low_memory=False, usecols=["subject", "latitude", "longitude"])
    c311 = c311[c311.subject.isin(P.BARRIER_311) & c311.latitude.notna() & c311.longitude.notna()]
    c311 = gpd.GeoDataFrame(c311, geometry=gpd.points_from_xy(c311.longitude, c311.latitude), crs=4326).to_crs(P.UTM)
    tree = cKDTree(P.xy(c311))
    n311 = np.array([len(x) for x in tree.query_ball_point(P.xy(G), r=30)])
    rep = lists == "repair"
    dec = pd.qcut(rb[rep], 10, labels=False) + 1
    by_dec = pd.DataFrame(dict(decile=dec, has_complaint=n311[rep] > 0)).groupby("decile").has_complaint.mean()
    by_dec.to_csv(OUT / "eval_311_by_priority_decile.csv")
    res["repair_311_within_30m"] = dict(share_by_priority_decile=by_dec.to_dict(),
                                        spearman_priority_vs_complaints=spearmanr(base[rep], n311[rep])[0],
                                        spearman_condition_vs_complaints=spearmanr(C.magnitude[rep], n311[rep])[0],
                                        note="decile 1 = highest priority. 311 tracks who reports, so read as a sanity check only")

    # ---- composition of the top of each list
    full = pd.read_csv(OUT / "barriers_ranked.csv")
    compo = {}
    for name, g in full.groupby("list"):
        for k in [100, len(g)]:
            t = g.nsmallest(k, "rank")
            compo[f"{name} top {k if k < len(g) else 'all'}"] = dict(
                types=t.label_type.value_counts(normalize=True).round(3).to_dict(),
                validation=t.validation.value_counts(normalize=True).round(3).to_dict(),
                needs_reverification=float(t.needs_reverification.mean()),
                neighborhoods=int(t.neighborhood.nunique()), top5_neighborhood_share=float(t.neighborhood.value_counts().head(5).sum() / k),
                on_high_injury_network=float(t.on_high_injury_network.mean()), median_nearest_stop_m=float(t.nearest_stop_m.median()))
    res["composition"] = compo

    json.dump(clean(res), open(OUT / "evaluation.json", "w"), indent=1)
    print(json.dumps(clean(res), indent=1)[:12000])


if __name__ == "__main__":
    main()
