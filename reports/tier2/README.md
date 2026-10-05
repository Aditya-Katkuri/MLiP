# Tier 2: Pittsburgh repair priorities from existing labels

Tier 2 ranks every sidewalk barrier Project Sidewalk volunteers have already found inside the city.
The ranking weighs how bad the barrier is, how well human validators back the label, and how many
people, and which people, need that spot. It uses no imagery and no model, so it stands whether or
not Tiers 0, 1 or 3 ever run.

| | |
|---|---|
| Run | `20260915-tier2-default-claude` (results in `/data/runs/20260915-tier2-default-claude/results/`, backed up to blob `results/`) |
| Code | `src/tier2/prioritize.py`, `src/tier2/evaluate.py`, `configs/tier2_default.yaml` |
| Map | `reports/tier2/map/build_map.py` builds a single-file interactive page |
| Inputs | the 14 Sep 2026 snapshot in `/data/datasets/pittsburgh` (sha256s in `run_info.json`) |

```bash
python src/tier2/prioritize.py --config configs/tier2_default.yaml   # ~10 s
python src/tier2/evaluate.py   --config configs/tier2_default.yaml   # ~15 s
python reports/tier2/map/build_map.py --out <file>.html
```

## Method

**Unit.** One Project Sidewalk *label cluster* per physical feature, inside the city boundary. There are
18,662 clusters citywide and 17,306 inside the city.

**Two lists, ranked separately.**
- **Spot repairs:** NoCurbRamp, Obstacle, SurfaceProblem, and CurbRamp when the ramp itself is deficient.
- **Missing sidewalks:** NoSidewalk.

A cracked ramp is maintenance and an absent sidewalk is new construction, which the proposal already
calls "different budget lines". Project Sidewalk's magnitudes for the two are also not on one scale.
In a single list, tagged NoSidewalk clusters (magnitude up to 2.9) took the entire top 10.

```
priority = magnitude × validity × (0.2 + 0.8 × demand)
```

- **Magnitude** is the negative of Project Sidewalk's own access-score term for the cluster, using
  the engine's published constants (`accessScoreConfig.json`): base weight × severity multiplier
  (0.33 / 0.67 / 1.0), plus tag adjustments. Examples:
  - a severity-3 missing ramp tagged "no alternate route" scores 1.5;
  - a severity-3 obstacle or surface problem scores 1.0;
  - a NoSidewalk cluster counts |−2| / 3, plus its tags.

  Curb ramps enter only when their term is negative, i.e. a deficient ramp: 5,774 good-ramp clusters
  score 0 and are excluded. 403 NoSidewalk clusters tagged "street has a sidewalk" also score 0.
  Adopting the engine's constants, rather than inventing new ones, keeps the scale public and
  comparable with Project Sidewalk's own street scores.
- **Validity** comes from human votes only, rebuilt from `validations.csv`:

  | Human votes on the cluster | Validity |
  |---|---|
  | More agree than disagree | 1.0 |
  | No decisive vote, labeller high-quality | 0.8 |
  | No decisive vote, labeller low-quality | 0.6 |
  | More disagree than agree | dropped (86 clusters) |

  SidewalkAI votes are ignored: they are 48% of all validations and abstain on 96% of obstacles.
- **Demand** is a weighted mean of four components, each turned into a 0–1 percentile first:

  | Component | Weight | What it measures |
  |---|---:|---|
  | Transit | 0.30 | PRT weekday trips at stops within 400 m, decaying linearly with distance |
  | Destinations | 0.30 | Schools, hospitals, senior centres, nursing homes, pharmacies, supermarkets, libraries, affordable housing, etc. within 400 m, each weighted |
  | Vulnerability | 0.25 | The tract's percentile on residents 65+, residents with a disability, households without a car, and density |
  | Safety | 0.15 | Pedestrian crashes 2019–25 within 100 m (deaths and major injuries count triple), and the High Injury Network |

  Race and income are deliberately not inputs; they are audited in the output instead. 311 is left
  out because it tracks who reports, not where barriers are (EDA, and confirmed below).
- **The demand floor of 0.2** keeps a severe barrier on a quiet street from scoring zero.

## Results

**10,002 barriers ranked:** 7,715 spot repairs (3,602 obstacles, 3,021 surface problems, 757 missing
ramps, 335 deficient ramps) and 2,287 missing-sidewalk locations. Spot repairs fall on 2,707 street
segments and 139 intersections; missing sidewalks fall on 1,005 segments.

**The top of the repair list is curb ramps.** All of the top 100 are missing (87%) or deficient (13%)
ramps; 99% are human-confirmed and their median distance to a bus stop is 52 m. The first obstacle
is rank 102 (a severity-3 fire hydrant at Penn Ave and Sheridan Ave), and the first surface problem is
rank 124 (Centre Ave, Middle Hill). This follows directly from Project Sidewalk's constants: a missing
ramp with no alternate route outweighs any obstacle. It also matches the ADA's own emphasis, since
Title II transition plans must schedule curb ramps. Every row carries `type_rank`, so a crew or
budget line for obstacles can read its own order.

**Where the work is.**
- **Intersections:** the top one is Robinson St at Allequippa St in Terrace Village, with 11 missing
  or deficient ramps. Next come two Middle Hill intersections and Liberty Ave at Smithfield St.
- **Repair top 100:** Middle Hill 26, the Central Business District 15, West Oakland 10.
- **Missing-sidewalk top 100:** 43 in the CBD. These are mostly "ends abruptly" gaps on busy
  downtown blocks.

**Re-check before dispatch.** 66% of ranked repairs sit on streets flagged `outdated`, where newer
imagery exists than the audit used. Some of these barriers may already be fixed.

## How much to trust it

There is no ground truth for a priority list, so the evaluation measures how much the order moves
when each judgment call changes. The table shows the share of the default top 100 still in the top
100.

| Change | Spot repairs | Missing sidewalks |
|---|---:|---:|
| Random demand weights (300 Dirichlet draws, mean / 5th percentile) | 0.90 / 0.80 | 0.86 / 0.74 |
| Drop transit | 0.83 | 0.81 |
| Drop destinations | 0.81 | 0.61 |
| Drop vulnerability | 0.83 | 0.82 |
| Drop safety | 0.91 | 0.69 |
| Demand floor 0 (pure product) | 0.86 | 0.90 |
| Demand floor 0.5 | 0.90 | 0.69 |
| Ignore demand (condition only) | 0.88 | 0.43 |
| Ignore human validation | 0.90 | 1.00 |
| Walking radius 200 m | **0.74** | **0.52** |
| Walking radius 800 m | 0.87 | 0.85 |
| Project Sidewalk preset "barriers" | 0.76 | 0.96 |
| Project Sidewalk preset "missing_ramps" | 0.86 | 1.00 |

- **Repair list:** 78% of the top 100 stays in the top 100 in at least 80% of the weight draws.
  Kendall's τ against the default averages 0.94.
- **Missing-sidewalk list:** it is more fragile (69% stable). It depends on demand much more,
  because every missing-sidewalk cluster has nearly the same magnitude.
- **The walking radius is the biggest single lever for both lists.** It is the parameter most worth
  settling with the team.

**Agreement with Project Sidewalk.** Street by street, over 4,982 scored streets:
- Ranking on condition alone agrees with Project Sidewalk's access score (Spearman 0.71), so the
  magnitude layer reproduces their ordering.
- Adding demand keeps that agreement (0.69) while replacing 31% of the 100 worst streets. That is
  the demand layer doing its job.

**311 sanity check.** The share of repairs with a barrier-type 311 complaint within 30 m is flat
across priority deciles (Spearman 0.02). This is consistent with the EDA finding that 311 measures
reporting, not barriers.

## Who it serves, and who it can't see

| Tracts in the top quartile of… | Share of repair top 100 | Share of all known repairs | Share of residents |
|---|---:|---:|---:|
| % Black residents | **49%** | 32% | 23% |
| % residents with a disability | 34% | 23% | 22% |
| median household income | 7% | 13% | 28% |

- **The top 100 leans towards Black, disabled and lower-income tracts.** Race and income are not
  inputs; the lean follows from denser barriers where those tracts were audited (EDA finding 6).
- **The blind spot is the bigger equity problem.** 150,313 residents (49%) live in 62 tracts under
  10% audited. Their barriers are mostly unknown, so they can barely enter either list. Tier 2
  cannot fix that; only the Tier 3 sweep can. Any list shown to the city should carry that sentence.

## Limitations

- **Every weight is a judgment call** made without input from disabled pedestrians. Project
  Sidewalk's own docs make the same point about their presets. The config and the robustness table
  exist so those calls can be argued with numbers.
- **Stop-level demand is service frequency** (`trips_wd`), not boardings; PRT doesn't publish
  boardings per stop.
- **Asset-map destination dates vary** by source; hospitals are 2015.
- **Missing sidewalks rest on single volunteers.** 99.7% of those labels were never validated.
- **NoSidewalk is scored per cluster.** Project Sidewalk scores it as a street-level term, so
  clusters on one street are summed here, not saturated.
- **Magnitude is coarse** (0.33 / 0.67 / 1.0 for obstacles and surfaces), so demand breaks most
  ties within a severity level.

## Decisions for the team

1. **The walking radius** (200 / 400 / 800 m) moves the list most; pick it deliberately.
2. **Whether a quota or separate budget lines** for obstacles and surface problems should replace the
   single repair ranking, whose top is all curb ramps.
3. **Whether to re-validate missing-sidewalk labels** before using that list.
4. **Whether to take the weights to disability advocates** before presenting to the city; the Tier 3
   human study already plans that contact.

## Outputs (`results/`)

| File | Contents |
|---|---|
| `repair_ranked.csv`, `missing_sidewalk_ranked.csv`, `barriers_ranked.{csv,geojson}` | one row per barrier: rank, type rank, every score component, validation, nearest stop, key destination, re-verification flag, Street View pano link |
| `intersections_ranked.csv`, `segments_ranked.{csv,geojson}` | barriers rolled up per intersection and per street edge |
| `neighborhoods.csv` | per-neighbourhood barriers, priority and top-100 counts, beside audited and un-audited km |
| `barriers_rejected.csv` | the 86 clusters human validators voted down |
| `evaluation.json`, `eval_*.csv` | robustness draws and single changes, Project Sidewalk agreement, equity, 311 check, top-k composition |
| `run_info.json` | config, git state, input sha256s, counts |
