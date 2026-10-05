#!/usr/bin/env python3
"""Charts and numbers for the pitch deck, from the CLIP baseline run and the Pittsburgh snapshot.

Writes to reports/presentation/build/: chart_baseline.png, chart_risk.png, metrics.json, worklist.json
Run after src/baseline/clip_baseline.py and reports/presentation/build_assets.py.

usage: /opt/miniconda/envs/sidewalk/bin/python reports/presentation/build_charts.py
"""
import glob, json, os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

for f in glob.glob(os.path.expanduser("~/.local/share/fonts/Carlito-*.ttf")):
    font_manager.fontManager.addfont(f)
plt.rcParams["font.family"] = "Carlito"

B = Path(__file__).parent / "build"
RUN = Path("/data/eda/baseline")
D = Path("/data/datasets/pittsburgh")
INK, MUTED, TEAL, TEAL_L, GREY, RED, SLATE = "#111827", "#6B7280", "#0F766E", "#8CC3BD", "#C4C9D0", "#B91C1C", "#475569"
CLASSES = ["missing_curb_ramp", "obstacle", "surface_problem", "no_barrier"]
TYPES = ["NoCurbRamp", "Obstacle", "SurfaceProblem"]
TYPE_NAMES = ["Missing curb ramp", "Obstacle", "Surface problem"]
# Saha et al., CHI 2019, label-type accuracy against researcher ground truth (Washington DC): recall per type
CHI2019_RECALL = {"NoCurbRamp": 0.693, "Obstacle": 0.399, "SurfaceProblem": 0.271}
WORKLIST = [23869, 8861, 7926, 6533, 14216]
SHORT_TYPE = {"NoCurbRamp": "No ramp", "Obstacle": "Obstacle", "SurfaceProblem": "Surface"}
SHORT_HOOD = {"Squirrel Hill North": "Squirrel Hill N.", "Central Business District": "Downtown"}


def hbars(ax, labels, values, colors, fmt, xmax, err=None):
    y = np.arange(len(values))[::-1]
    ax.barh(y, values, color=colors, height=0.64)
    if err is not None:
        i, lo, hi = err
        ax.errorbar(values[i], y[i], xerr=[[values[i] - lo], [hi - values[i]]], fmt="none", ecolor=INK, capsize=3, lw=1)
    for yi, lab, v, e in zip(y, labels, values, range(len(values))):
        ax.text(-0.015 * xmax, yi, lab, ha="right", va="center", color=INK, fontsize=13)
        has_err = err is not None and e == err[0]
        end = err[2] if has_err else v
        ax.text(end + (0.045 if has_err else 0.02) * xmax, yi, fmt(v), va="center", color=INK, fontsize=14, fontweight="bold")
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, len(values) - 0.4)
    ax.axis("off")


def main():
    R = json.load(open(RUN / "results.json"))
    meta = pd.read_csv(RUN / "crops_meta.csv")
    P = pd.read_csv(RUN / "pittsburgh_predictions.csv").merge(
        pd.read_csv(B / "crop_tiers.csv")[["label_id", "dir", "tier"]], on=["label_id", "dir"], how="left")
    assert P.tier.notna().all()

    rng = np.random.default_rng(0)
    boot = [f1_score(s.y, s.linear_probe, labels=CLASSES, average="macro")
            for s in (P.iloc[rng.integers(0, len(P), len(P))] for _ in range(2000))]
    pgh = meta.city == "pittsburgh"
    M = dict(zero_shot_pgh=R["zero_shot"]["pittsburgh_macro_f1"], zero_shot_other=R["zero_shot"]["other_cities_test_macro_f1"],
             probe_pgh=R["linear_probe"]["pittsburgh_macro_f1"], probe_other=R["linear_probe"]["other_cities_test_macro_f1"],
             probe_pgh_ci=[round(float(x), 3) for x in np.percentile(boot, [2.5, 97.5])],
             n_train=R["n_train"], n_val=R["n_val"], n_other_test=int(((~pgh) & (meta.split == "test")).sum()),
             n_test_pgh=R["n_test_pittsburgh"], n_other=int((~pgh).sum()),
             class_counts_other=meta[~pgh].y.value_counts().to_dict(), class_counts_pgh=meta[pgh].y.value_counts().to_dict())

    P["is_bar"], P["pred_bar"] = P.y != "no_barrier", P.linear_probe != "no_barrier"
    def pr(g):
        tp = int((g.is_bar & g.pred_bar).sum())
        return dict(n=len(g), barriers=int(g.is_bar.sum()), recall=round(tp / max(g.is_bar.sum(), 1), 3),
                    precision=round(tp / max(g.pred_bar.sum(), 1), 3), base_rate=round(float(g.is_bar.mean()), 3))
    M["barrier_vs_none"] = pr(P)
    M["barrier_vs_none_by_tier"] = {int(t): pr(g) for t, g in P.groupby("tier")}
    ex = P[P.label_id == 23869].iloc[0]
    M["example_23869"] = dict(y=ex.y, pred=ex.linear_probe, p=round(float(ex[f"p_{ex.linear_probe}"]), 3), tier=int(ex.tier))

    V = pd.read_csv(D / "projectsidewalk/validations.csv", low_memory=False)
    H = V[V.validator_type == "Human"]
    net = H.assign(v=H.validation_result.map({"Agree": 1, "Disagree": -1}).fillna(0)).groupby(["label_type", "label_id"]).v.sum()
    rej = (net < 0).groupby(level=0).mean()
    M["pgh_labels_voted_down"] = {t: round(float(rej[t]), 3) for t in TYPES}
    M["pgh_human_votes"], M["ai_share_of_validations"] = int(len(H)), round(float((V.validator_type == "AI").mean()), 3)
    M["chi2019_missed"] = {t: round(1 - r, 3) for t, r in CHI2019_RECALL.items()}

    L = pd.read_csv(D / "projectsidewalk/rawLabels.csv", usecols=["label_id", "label_type", "region_name"]).set_index("label_id")
    T = pd.read_csv(B / "crop_tiers.csv").drop_duplicates("label_id").set_index("label_id")
    rows = [dict(label_id=i, tier=int(T.loc[i, "tier"]), type=SHORT_TYPE[L.loc[i, "label_type"]],
                 hood=SHORT_HOOD.get(L.loc[i, "region_name"], L.loc[i, "region_name"])) for i in WORKLIST]
    json.dump(sorted(rows, key=lambda r: r["tier"]), open(B / "worklist.json", "w"), indent=1)

    fig, ax = plt.subplots(figsize=(4.6, 2.2), dpi=300)
    hbars(ax, ["Zero-shot CLIP · Pittsburgh", "Trained · 18 other cities", "Trained · Pittsburgh"],
          [M["zero_shot_pgh"], M["probe_other"], M["probe_pgh"]], [GREY, TEAL_L, TEAL], lambda v: f"{v:.2f}", 1.0,
          err=(2, *M["probe_pgh_ci"]))
    fig.subplots_adjust(left=0.43, right=0.97, top=0.98, bottom=0.02)
    fig.savefig(B / "chart_baseline.png", transparent=True)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(4.9, 3.9), dpi=300)
    panels = [("Labels Pittsburgh validators voted down", [M["pgh_labels_voted_down"][t] * 100 for t in TYPES], RED),
              ("Real barriers that labellers missed (DC study)", [M["chi2019_missed"][t] * 100 for t in TYPES], SLATE)]
    fig.subplots_adjust(left=0.33, right=0.95, top=0.9, bottom=0.03, hspace=0.62)
    for ax, (title, vals, col) in zip(axes, panels):
        hbars(ax, TYPE_NAMES, vals, [col] * 3, lambda v: f"{v:.0f}%", 100)
        pos = ax.get_position()
        fig.text(0.01, pos.y1 + 0.035, title, fontsize=14, fontweight="bold", color=INK, va="bottom")
    fig.savefig(B / "chart_risk.png", transparent=True)
    plt.close(fig)

    json.dump(M, open(B / "metrics.json", "w"), indent=1)
    print(json.dumps(M, indent=1))


if __name__ == "__main__":
    main()
