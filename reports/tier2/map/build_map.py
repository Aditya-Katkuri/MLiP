#!/usr/bin/env python3
"""Build the self-contained Tier 2 map page (one HTML file) from a Tier 2 run's results.

usage:
  python reports/tier2/map/build_map.py --results /data/runs/20260915-tier2-default-claude/results \
      --out /data/runs/20260915-tier2-default-claude/results/pittsburgh-repair-priorities.html
"""
import argparse, json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

HERE = Path(__file__).parent
FIELDS = ["list", "rank", "typeRank", "type", "sev", "tags", "validation", "agree", "disagree", "priority", "magnitude", "demand",
          "transit", "dest_score", "vuln", "safety", "hood", "stop", "stopM", "dest", "hin", "reverify", "imagery", "pano", "heading",
          "pitch", "cluster", "labels", "lon", "lat"]


def pano_parts(url):
    if not isinstance(url, str):
        return "", 0, 0
    q = parse_qs(urlparse(url).query)
    return q.get("pano", [""])[0], round(float(q.get("heading", [0])[0]), 1), round(float(q.get("pitch", [0])[0]), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/data/runs/20260915-tier2-default-claude/results")
    ap.add_argument("--hoods", default="/data/eda/pittsburgh/neighborhood_coverage.geojson")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = Path(a.results)
    info = json.load(open(R / "run_info.json"))
    ev = json.load(open(R / "evaluation.json"))
    b = pd.read_csv(R / "barriers_ranked.csv")

    rows = []
    for r in b.itertuples(index=False):
        pano, heading, pitch = pano_parts(r.gsv_url)
        sev = None if pd.isna(r.severity_bucket) else int(float(r.severity_bucket))
        rows.append([r.list, int(r.rank), int(r.type_rank), r.label_type, sev, r.active_tags if isinstance(r.active_tags, str) else "",
                     r.validation, int(r.votes_agree or 0), int(r.votes_disagree or 0), round(r.priority, 3), round(r.magnitude, 3),
                     round(r.demand, 3), round(r.transit_score, 3), round(r.destinations_score, 3), round(r.vulnerability_score, 3),
                     round(r.safety_score, 3), r.neighborhood if isinstance(r.neighborhood, str) else "",
                     r.nearest_stop.title() if isinstance(r.nearest_stop, str) else "", int(r.nearest_stop_m),
                     r.key_destination if isinstance(r.key_destination, str) else "", bool(r.on_high_injury_network),
                     bool(r.needs_reverification), str(r.avg_image_capture_date)[:7] if isinstance(r.avg_image_capture_date, str) else "",
                     pano, heading, pitch, int(r.label_cluster_id), int(r.cluster_size), round(r.lon, 5), round(r.lat, 5)])

    hg = json.load(open(a.hoods))
    for f in hg["features"]:
        f["properties"] = dict(hood=f["properties"].get("hood"), opened=bool((f["properties"].get("opened_share") or 0) >= 0.01))

    single = ev["single_changes"]
    def sentence(s):
        return s[:1].upper() + s[1:]
    changes = [dict(change=sentence(k.replace("Project Sidewalk preset: ", "Project Sidewalk weight preset: ")),
                    repair=v["repair"]["top100_overlap"], missing_sidewalk=v["missing_sidewalk"]["top100_overlap"]) for k, v in single.items()]
    wd = ev["weight_draws"]["by_list"]
    changes.insert(0, dict(change=f"Random demand weights ({ev['weight_draws']['n']} draws, mean)", repair=wd["repair"]["top100_overlap_mean"],
                           missing_sidewalk=wd["missing_sidewalk"]["top100_overlap_mean"]))
    eq = [dict(label=lbl, top=ev["equity_top100_Q4"][f"{key} | repair"]["share_of_top_100"],
               all=ev["equity_top100_Q4"][f"{key} | repair"]["share_of_all_barriers"], residents=ev["equity_top100_Q4"][f"{key} | repair"]["share_of_residents"])
          for key, lbl in [("% Black", "% Black residents"), ("% disability", "% residents with a disability"), ("median income", "median household income")]]
    ps = ev["vs_project_sidewalk_score"]
    comp = ev["composition"]
    rep100 = comp["repair top 100"]
    rob = (f"The biggest single lever is the walking radius, and for missing sidewalks the demand floor. "
           f"Street by street, ranking on condition alone agrees with Project Sidewalk's access score (Spearman {ps['spearman_condition_vs_ps']:.2f}); "
           f"adding demand keeps that agreement ({ps['spearman_priority_vs_ps']:.2f}) while replacing {1 - ps['top100_overlap_condition_vs_priority']:.0%} of the worst 100 streets.")
    eqn = (f"The vulnerability index uses age, disability and car ownership, never race or income, yet the repair top 100 still leans towards Black and "
           f"lower-income tracts. That's because those tracts carry denser barriers where they were audited. The larger problem is coverage: "
           f"{ev['blind_spot']['residents']:,} residents ({ev['blind_spot']['share_of_residents']:.0%}) live in tracts under 10% audited. Their barriers are "
           f"mostly unknown, so they can barely appear here. Only the Tier 3 sweep can fix that.")
    limits = [
        f"The repair top 100 is entirely curb ramps ({rep100['types'].get('NoCurbRamp', 0):.0%} missing, {rep100['types'].get('CurbRamp', 0):.0%} deficient). Under Project Sidewalk's constants a missing ramp outweighs any obstacle; each barrier's rank within its own type is shown for crews that handle obstacles or surfaces.",
        f"{comp['repair top all']['needs_reverification']:.0%} of ranked repairs sit on streets where newer imagery exists than the audit used; some may already be fixed. Re-check before dispatching anyone.",
        f"Missing-sidewalk labels are almost never validated ({comp['missing_sidewalk top all']['validation'].get('unvalidated', 0):.1%} unvalidated), so that list rests on single volunteers' calls.",
        "Demand uses stop-level service frequency, not boardings, which PRT does not publish per stop. Destinations come from the county asset map, whose dates vary by source.",
        "311 complaints are left out: they track who reports, not where barriers are, and priority shows no relationship with them.",
        "Every weight is a judgment call made without the disability community; the config file and the robustness table exist so those calls can be argued with numbers.",
    ]
    DATA = dict(run_id=info["run_id"], fields=FIELDS, barriers=rows, hoods=hg,
                summary=dict(repair=info["counts"]["ranked_barriers"]["repair"], missing=info["counts"]["ranked_barriers"]["missing_sidewalk"]),
                eval=dict(robust_top100_repair=wd["repair"]["top100_items_kept_in_80pct_of_draws"], blind_spot_share=ev["blind_spot"]["share_of_residents"],
                          changes=changes, robust_note=rob, equity=eq, equity_note=eqn),
                limits=limits,
                footer=(f"Run {info['run_id']} · git {info.get('git_sha', '?')}{' (Tier 2 code uncommitted)' if info.get('tier2_code_or_config_uncommitted') else ''} · "
                        f"inputs: Project Sidewalk, WPRDC and ACS snapshot of 14 Sep 2026 · configs/tier2_default.yaml, src/tier2/prioritize.py, src/tier2/evaluate.py · "
                        f"full tables in {a.results}"))
    html = (HERE / "template.html").read_text()
    # the detail panel's "Draws people" row shows the destination name
    html = html.replace('row("Draws people", r[I.dest]);', 'row("Draws people", r[I.dest] || "");')
    html = html.replace("const DATA = /*__DATA__*/null;", "const DATA = " + json.dumps(DATA, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/") + ";")
    Path(a.out).write_text(html)
    print(f"wrote {a.out} ({Path(a.out).stat().st_size / 1e6:.2f} MB, {len(rows):,} barriers)")


if __name__ == "__main__":
    main()
