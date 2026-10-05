#!/usr/bin/env python3
"""Build a self-contained gallery of Project Sidewalk validator-AI crop examples, each with its label.

Pittsburgh crops are joined (by the label id in their filename) to the Pittsburgh label API snapshot,
so each card shows the full label record: type, severity, tags, human vs AI votes, place, imagery date.

usage:
  python reports/eda/examples/build_examples.py --out /tmp/.../project-sidewalk-crop-examples.html
"""
import argparse, base64, glob, html, io, json, os, random

import pandas as pd
from PIL import Image

D = "/data/datasets"
CLASSES = [("nocurbramp", "NoCurbRamp", "Missing curb ramp"), ("obstacle", "Obstacle", "Obstacle"),
           ("surfaceproblem", "SurfaceProblem", "Surface problem")]
CITY = {"sea": "Seattle", "chicago": "Chicago", "taipei": "Taipei", "teaneck": "Teaneck, NJ", "oradell": "Oradell, NJ",
        "mendota": "Mendota, IL", "newberg": "Newberg, OR", "columbus": "Columbus, OH", "pittsburgh": "Pittsburgh",
        "keelung": "Keelung", "amsterdam": "Amsterdam", "new_taipei": "New Taipei", "cdmx": "Mexico City",
        "spgg": "San Pedro Garza García", "st_louis": "St. Louis", "knox": "Knox County", "cliffside_park": "Cliffside Park, NJ",
        "blackhawk_hills": "Blackhawk Hills", "la_piedad": "La Piedad"}
E = html.escape


def crops(d):
    out = []
    for p in glob.glob(f"{D}/sidewalk-validator-ai-dataset-{d}/*/*/*.webp"):
        split, verdict, fn = p.split("/")[-3:]
        city, lid = fn[:-5].rsplit("_", 1)
        out.append(dict(path=p, split=split, verdict=verdict, file=fn, city=city, label_id=int(lid)))
    return sorted(out, key=lambda r: r["path"])


def encode(path, px=420):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    im.thumbnail((px, px))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), (w, h), im.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(10718)

    L = pd.read_csv(f"{D}/pittsburgh/projectsidewalk/rawLabels.csv").set_index("label_id")
    V = pd.read_csv(f"{D}/pittsburgh/projectsidewalk/validations.csv", usecols=["label_id", "validator_type", "validation_result"])
    votes = V.groupby(["label_id", "validator_type", "validation_result"]).size()

    def vote_str(lid, who):
        parts = [f"{votes.get((lid, who, r), 0)} {r.lower()}" for r in ["Agree", "Disagree", "Unsure"]]
        return ", ".join(parts)

    def record(c):
        r = L.loc[c["label_id"]] if c["city"] == "pittsburgh" and c["label_id"] in L.index else None
        rec = None
        if r is not None:
            tags = json.loads(r.tags) if isinstance(r.tags, str) else []
            rec = {
                "Label type": r.label_type,
                "Severity (1–3)": "—" if pd.isna(r.severity) else str(int(r.severity)),
                "Tags": ", ".join(tags) if tags else "—",
                "Description": r.description if isinstance(r.description, str) else "—",
                "Human votes": vote_str(c["label_id"], "Human"),
                "AI votes": vote_str(c["label_id"], "AI"),
                "Neighbourhood": r.region_name,
                "Labelled": str(r.time_created)[:10],
                "Imagery captured": str(r.image_capture_date),
                "Location": f"{r.latitude:.5f}, {r.longitude:.5f}",
            }
            rec["_pano"] = r.pano_url
        return rec

    sections = []
    anatomy = None
    for d, api_type, name in CLASSES:
        allc = crops(d)
        pgh_ok = [c for c in allc if c["city"] == "pittsburgh" and c["verdict"] == "correct"]
        pgh_bad = [c for c in allc if c["city"] == "pittsburgh" and c["verdict"] == "incorrect"]
        # prefer confirmed examples that carry tags, so the label record has something to show
        tagged = [c for c in pgh_ok if c["label_id"] in L.index and isinstance(L.loc[c["label_id"]].tags, str) and L.loc[c["label_id"]].tags != "[]"]
        ok = rng.sample(tagged, min(3, len(tagged))) + rng.sample([c for c in pgh_ok if c not in tagged], 1)
        bad = rng.sample(pgh_bad, 2)
        others = []
        for city in ["sea", "chicago", "taipei", "amsterdam", "teaneck", "oradell"]:
            pool = [c for c in allc if c["city"] == city and c["verdict"] == "correct"]
            if pool and len(others) < 3:
                others.append(rng.choice(pool))
        items = []
        for c in ok + bad + others:
            src, (w, h), (tw, th) = encode(c["path"])
            items.append(dict(c, src=src, w=w, h=h, tw=tw, th=th, rec=record(c)))
        if anatomy is None:
            anatomy = items[0]
        n_ok = sum(1 for c in allc if c["verdict"] == "correct")
        n_bad = len(allc) - n_ok
        sections.append(dict(d=d, api=api_type, name=name, items=items, n_ok=n_ok, n_bad=n_bad,
                             n_pgh=len(pgh_ok) + len(pgh_bad)))

    def card(it):
        verdict = "confirmed" if it["verdict"] == "correct" else "rejected"
        rec = it["rec"]
        rows = ""
        if rec:
            for k in ["Label type", "Severity (1–3)", "Tags", "Human votes", "Neighbourhood", "Imagery captured"]:
                rows += f"<dt>{E(k)}</dt><dd>{E(str(rec[k]))}</dd>"
            link = f'<a href="{E(rec["_pano"])}" target="_blank" rel="noopener">Open in Street View ↗</a>'
        else:
            rows = f"<dt>City</dt><dd>{E(CITY.get(it['city'], it['city']))}</dd><dt>Label record</dt><dd>in that city's own API</dd>"
            link = ""
        return f"""
<article class="card">
  <img src="{it['src']}" width="{it['tw']}" height="{it['th']}" alt="{E(verdict)} crop from {E(CITY.get(it['city'], it['city']))}, label {it['label_id']}" loading="lazy">
  <div class="meta">
    <div class="top"><span class="pill {verdict}">{verdict}</span><span class="split">{E(it['split'])}</span></div>
    <p class="file mono">{E(it['file'])}</p>
    <dl>{rows}</dl>
    {link}
  </div>
</article>"""

    an = anatomy
    an_rows = "".join(f"<tr><th>{E(k)}</th><td>{E(str(v))}</td></tr>" for k, v in an["rec"].items() if not k.startswith("_"))
    body_sections = ""
    for s in sections:
        pgh = [it for it in s["items"] if it["city"] == "pittsburgh"]
        oth = [it for it in s["items"] if it["city"] != "pittsburgh"]
        body_sections += f"""
<section id="{s['d']}">
  <h2>{E(s['name'])} <span class="count">{s['n_ok']:,} confirmed · {s['n_bad']:,} rejected · {s['n_pgh']} from Pittsburgh</span></h2>
  <p class="path mono">sidewalk-validator-ai-dataset-{s['d']}/&lt;train|val|test&gt;/&lt;correct|incorrect&gt;/&lt;city&gt;_&lt;label_id&gt;.webp</p>
  <h3>Pittsburgh, with the matching label record</h3>
  <div class="grid">{''.join(card(it) for it in pgh)}</div>
  <h3>Other cities</h3>
  <div class="grid">{''.join(card(it) for it in oth)}</div>
</section>"""

    page = f"""<title>Project Sidewalk Crop Examples</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Overpass:wght@700;800&family=Overpass+Mono:wght@400;500&family=Public+Sans:wght@400;500;600&display=swap">
<style>
:root {{
  color-scheme: light;
  --ground: #f0f1ee; --surface: #fafaf8; --ink: #16191c; --ink-2: #4b5158; --muted: #6f757c; --rule: #d6d9d4;
  --rule-strong: #b6bab4; --link: #1f5fae; --ok-bg: #e3f1e6; --ok-ink: #1d5e2c; --bad-bg: #f7e6e2; --bad-ink: #8a2f22; --dome: #dfa200;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --ground: #101214; --surface: #171a1d; --ink: #e8ebee; --ink-2: #a8afb6; --muted: #858c93; --rule: #282c31;
    --rule-strong: #3a3f45; --link: #86b9f2; --ok-bg: #173323; --ok-ink: #9fd9ad; --bad-bg: #3a1e19; --bad-ink: #f2a79a; --dome: #edb92b;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --ground: #101214; --surface: #171a1d; --ink: #e8ebee; --ink-2: #a8afb6; --muted: #858c93; --rule: #282c31;
  --rule-strong: #3a3f45; --link: #86b9f2; --ok-bg: #173323; --ok-ink: #9fd9ad; --bad-bg: #3a1e19; --bad-ink: #f2a79a; --dome: #edb92b;
}}
* {{ box-sizing: border-box; }}
html, body {{ background: var(--ground); }}
body {{ color: var(--ink); font: 400 15px/1.55 "Public Sans", system-ui, -apple-system, "Segoe UI", sans-serif; padding-inline: 20px; }}
.page {{ max-width: 1180px; margin-inline: auto; padding-block: 32px 64px; }}
.mono {{ font-family: "Overpass Mono", ui-monospace, Menlo, Consolas, monospace; }}
a {{ color: var(--link); text-underline-offset: 2px; }}
:focus-visible {{ outline: 2px solid var(--link); outline-offset: 2px; }}
.eyebrow {{ font: 500 12px/1.4 "Overpass Mono", ui-monospace, monospace; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); margin: 0 0 10px; }}
h1 {{ font: 800 clamp(30px, 4.6vw, 44px)/1.05 Overpass, "Public Sans", system-ui, sans-serif; margin: 0 0 12px; text-wrap: balance; }}
.dek {{ font-size: 17px; color: var(--ink-2); max-width: 70ch; margin: 0 0 18px; }}
.domes {{ height: 12px; margin: 0 0 26px; background-image: radial-gradient(circle, var(--dome) 0 2.8px, transparent 3.3px); background-size: 10px 10px; }}
h2 {{ font: 700 26px/1.2 Overpass, "Public Sans", system-ui, sans-serif; margin: 44px 0 6px; display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 14px; }}
h2 .count {{ font: 500 13px/1.4 "Public Sans", system-ui, sans-serif; color: var(--muted); }}
h3 {{ font: 600 13px/1.3 "Public Sans", system-ui, sans-serif; text-transform: uppercase; letter-spacing: .05em; color: var(--muted); margin: 20px 0 10px; }}
.path {{ font-size: 12.5px; color: var(--ink-2); margin: 0; overflow-wrap: anywhere; }}
.explain {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 20px; align-items: start; }}
@media (max-width: 860px) {{ .explain {{ grid-template-columns: minmax(0, 1fr); }} }}
.box {{ background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; padding: 16px 18px; }}
.box h2 {{ margin-top: 0; font-size: 19px; }}
pre.tree {{ font: 13px/1.6 "Overpass Mono", ui-monospace, monospace; margin: 8px 0 0; overflow-x: auto; color: var(--ink-2); }}
.anatomy {{ display: grid; grid-template-columns: 300px minmax(0, 1fr); gap: 16px; align-items: start; }}
@media (max-width: 560px) {{ .anatomy {{ grid-template-columns: minmax(0, 1fr); }} }}
.anatomy img {{ width: 100%; height: auto; border-radius: 6px; display: block; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13.5px; }}
th, td {{ text-align: left; vertical-align: top; padding: 5px 10px 5px 0; border-bottom: 1px solid var(--rule); }}
th {{ color: var(--muted); font-weight: 500; white-space: nowrap; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 14px; }}
.card {{ background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; overflow: hidden; display: flex; flex-direction: column; }}
.card img {{ width: 100%; height: auto; aspect-ratio: 1 / 1; object-fit: cover; display: block; background: var(--rule); }}
.meta {{ padding: 10px 12px 12px; display: flex; flex-direction: column; gap: 6px; font-size: 13px; }}
.top {{ display: flex; justify-content: space-between; align-items: center; }}
.pill {{ font: 600 11px/1 "Public Sans", system-ui, sans-serif; text-transform: uppercase; letter-spacing: .05em; padding: 5px 7px; border-radius: 4px; }}
.pill.confirmed {{ background: var(--ok-bg); color: var(--ok-ink); }}
.pill.rejected {{ background: var(--bad-bg); color: var(--bad-ink); }}
.split {{ font: 500 11.5px "Overpass Mono", ui-monospace, monospace; color: var(--muted); }}
.file {{ margin: 0; font-size: 12px; color: var(--ink-2); overflow-wrap: anywhere; }}
dl {{ display: grid; grid-template-columns: auto 1fr; gap: 2px 10px; margin: 0; }}
dt {{ color: var(--muted); }}
dd {{ margin: 0; overflow-wrap: anywhere; }}
ul.notes {{ padding-left: 20px; max-width: 80ch; }}
ul.notes li {{ margin-bottom: 6px; }}
</style>
<div class="page">
  <p class="eyebrow">CMU 10-718 · Sidewalk Scanner · training data</p>
  <h1>Project Sidewalk Crop Examples</h1>
  <p class="dek">Real crops from the three barrier datasets on the VM, with the labels that come with them. Pittsburgh crops are joined by label id to Project Sidewalk's Pittsburgh label data, which is where coordinates, severity, tags and votes live.</p>
  <div class="domes" role="presentation"></div>

  <div class="explain">
    <div class="box">
      <h2>How the dataset is organised</h2>
      <p>The folder a crop sits in <em>is</em> its label. The filename holds the city and the label id; nothing else ships with the image.</p>
      <pre class="tree">sidewalk-validator-ai-dataset-nocurbramp/
├── train/
│   ├── correct/     ← validators confirmed: a missing curb ramp
│   │   └── pittsburgh_{an['label_id']}.webp
│   └── incorrect/   ← validators rejected the label
├── val/   (same two folders)
└── test/  (same two folders)</pre>
      <ul class="notes">
        <li><strong>confirmed</strong> = volunteers agreed the label is right, so the crop shows that barrier.</li>
        <li><strong>rejected</strong> = volunteers disagreed. The crop may show nothing, or a different barrier.</li>
        <li>The <span class="mono">curbramp</span> dataset is a copy of <span class="mono">crosswalk</span>, and there is no missing-sidewalk dataset, so neither is shown.</li>
      </ul>
    </div>
    <div class="box">
      <h2>One example, fully joined</h2>
      <div class="anatomy">
        <img src="{an['src']}" width="{an['tw']}" height="{an['th']}" alt="Confirmed {E(an['rec']['Label type'])} crop from Pittsburgh">
        <div>
          <p class="file mono">{E(an['path'].replace(D + '/', ''))}</p>
          <p class="path">Crop {an['w']}×{an['h']} px · split <b>{E(an['split'])}</b> · folder <b>{E(an['verdict'])}</b></p>
          <table>{an_rows}</table>
          <p><a href="{E(an['rec']['_pano'])}" target="_blank" rel="noopener">Open in Street View ↗</a></p>
        </div>
      </div>
    </div>
  </div>
  {body_sections}
  <p class="path" style="margin-top:40px">Built by reports/eda/examples/build_examples.py from /data/datasets (Hugging Face projectsidewalk/sidewalk-validator-ai-dataset-*) and the 14 Sep 2026 Pittsburgh label snapshot. "Human votes" count only people; "AI votes" are Project Sidewalk's validator model.</p>
</div>
"""
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w").write(page)
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
