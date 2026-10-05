# Pittsburgh: labels, audit coverage and demand layers

Snapshot retrieved **2026-09-14 14:12 UTC** by `scripts/fetch_pittsburgh_data.py`. Every number below
is reproduced by `src/eda/pittsburgh_eda.py`, which writes `/data/eda/pittsburgh/summary.json` and the
CSVs in `/data/eda/pittsburgh/tables/`. The raw snapshot lives in `/data/datasets/pittsburgh/`, and
`MANIFEST.json` there records the URL, byte size and sha256 of each of its 68 files.

## Headline findings

**1. The un-audited area is 1,324 km, not 494.** The proposal's "334 of Pittsburgh's 828 walkable km"
uses Project Sidewalk's *explorable* km. That figure counts only streets in regions opened for
auditing, and it includes four boroughs outside the city. Measured against the street network inside
the city boundary:

| | km | share of city network |
|---|---:|---:|
| City street network (Project Sidewalk edges, midpoint inside the city) | **1,595.7** | 100% |
| Audited at least once | **271.4** | **17.0%** |
| Opened but never audited | 402.7 | 25.2% |
| Never opened for auditing | **906.9** | 56.8% |
| No street-view imagery | 14.7 | 0.9% |

- "Explorable" is 827.3 km. Of that, 153.4 km lies outside the city: Wilkinsburg 79.2 km, Swissvale
  39.0, Millvale 18.0, Edgewood 17.2.
- `km_explored` = 334.4 double-counts streets audited more than once. The API's own
  `km_explored_no_overlap` gives 287.5.
- **52 of 90 neighbourhoods have under 1% of their streets opened**, among them Brookline (64.9 km),
  Beechview, Brighton Heights, Sheraden, Lincoln-Lemington-Belmar and Perry South. Only 10
  neighbourhoods are at least half audited.
- **207 of the 271 audited city km (76%) are flagged `outdated`**: the street has newer imagery than
  its audit. Most audits are old, too — 121.6 km were last labelled in 2020–21.

**2. Labels: 25,690 in total, 23,829 inside the city.** 1,861 labels sit in Swissvale (859),
Wilkinsburg (502), Millvale (295) and Edgewood (182), so any "Pittsburgh" statistic should drop them.

| Type | Labels | Severity 1 / 2 / 3 | No severity | Validated | Correct of validated |
|---|---:|---|---:|---:|---:|
| CurbRamp | 9,511 | 8,018 / 834 / 399 | 260 | 89% | 94% |
| Obstacle | 5,529 | 2,391 / 1,109 / 1,628 | 401 | **45%** | **66%** |
| SurfaceProblem | 4,301 | 1,554 / 1,240 / 1,113 | 394 | 84% | 92% |
| NoSidewalk | 3,495 | — | **3,495** | **0.4%** | — |
| NoCurbRamp | 1,255 | 266 / 373 / 529 | 87 | 89% | 72% |
| Crosswalk | 1,016 | 816 / 76 / 83 | 41 | 88% | 79% |
| Occlusion / Signal / Other | 335 / 217 / 31 | — | | | |

- **Severity runs 1–3 in this API, not 1–5** as the proposal states. The proposal's ~0.8 SD figure was
  computed on this 3-point scale, so its framing needs fixing.
- **NoSidewalk carries no severity and is essentially never validated** (13 of 3,495). A severity ×
  demand score needs its own rule for it. Project Sidewalk's access score gives NoSidewalk base
  weight −2 under a "street_condition" rule (`accessScoreConfig.json`).
- There are **18,662 label clusters** for 23,295 clustered labels, so about 20% of labels re-mark a
  physical feature someone else already marked. Rank clusters, not raw labels.
- The labels sit on 11,823 GSV panoramas, all `pano_source = gsv`.

**3. Nearly half of all "validations" are an AI's, and the vote counts mix the two.** 22,141 of 46,416
validations (47.7%) come from `SidewalkAI`, all dated 2025–26. The `agree_count` / `disagree_count`
fields in `rawLabels.csv` equal human + AI votes on 100% of labels, but human-only votes on just 57%.
9,046 labels have only AI votes and 3,948 have none. **The proposal's "evaluate against human
disagreement" must rebuild per-label votes from `validations.csv` using `validator_type == "Human"`.**

| Type | Human agree rate (agree / (agree + disagree)) | AI agree rate | AI "unsure" rate |
|---|---:|---:|---:|
| CurbRamp | 0.914 | 0.918 | 16% |
| SurfaceProblem | 0.846 | 0.952 | 39% |
| Crosswalk | 0.814 | 0.716 | 17% |
| Obstacle | 0.688 | 0.000 | **96%** |
| NoCurbRamp | 0.666 | 0.581 | 24% |

Among the 4,953 labels with at least two decisive human votes, **only 69% are unanimous**. That is the
realistic agreement ceiling for any model scored on these labels.

**4. The imagery is old.** Median image age at snapshot is **5.9 years**, and 70% of labelled
panoramas are over 5 years old. At labelling time the mean image age was 767 days, matching the
proposal, but the median was 473. Capture years peak in 2019 (7,730 labels) and 2020 (4,153).

**5. A few volunteers did most of the work.** 351 users; the top contributor made 12.8% of labels, the
top ten made 41%, and the Gini coefficient is 0.72. Labels by year: 6,084 in 2020, 7,343 in 2021, then
2,000–3,700 a year.

**6. Equity: coverage skews away from Black and disabled residents, and the barriers skew towards
them.** This covers 120 city tracts with at least 2 km of streets; ACS 2020–24 5-year data via Census
Reporter; Spearman ρ values.

| Tract variable | ρ with audited share | ρ with opened share | ρ with barrier labels per audited km |
|---|---:|---:|---:|
| % Black, non-Hispanic | **−0.29** (p=0.001) | **−0.33** (p<0.001) | **+0.42** (p=0.001) |
| % with a disability | **−0.35** (p<0.001) | **−0.32** (p<0.001) | **+0.48** (p<0.001) |
| Median household income | +0.05 (n.s.) | +0.10 (n.s.) | **−0.39** (p=0.004) |
| % who walk to work | **+0.55** (p<0.001) | +0.49 | −0.16 (n.s.) |
| % aged 65+ | −0.13 (n.s.) | −0.06 | +0.24 (p=0.07) |

| Quartile of % Black residents | Audited share | Opened share | Barrier labels per audited km |
|---|---:|---:|---:|
| Q1 (lowest) | 26.3% | 64.3% | 34.7 |
| Q2 | 22.5% | 53.8% | 37.1 |
| Q3 | 11.7% | 32.4% | 64.3 |
| Q4 (highest) | 10.6% | 25.7% | **84.5** |

The coverage gap tracks the *opened* share almost exactly. So it mostly reflects which regions
Project Sidewalk opened, not where volunteers chose to walk. Where streets *were* audited,
high-%-Black tracts carry 2.4× the barrier density. The model's un-audited sweep would therefore
land disproportionately on the neighbourhoods with the most barriers per km. That is the strongest
argument for Tier 3, and the proposal's equity audit must report it per neighbourhood. The strong
walk-to-work correlation points to the student areas: Oakland, Shadyside, the CBD.

**7. Demand layers: most of what pedestrians walk to is on un-audited streets.** Each point is
snapped to the nearest city street edge; stops within 50 m, destinations 100 m, crashes 30 m.

| Layer | n in city | on audited street | open, not audited | never opened |
|---|---:|---:|---:|---:|
| PRT stops | 2,696 | 20.1% | 26.5% | 51.6% |
| …weighted by weekday trips (183,508) | | 30.7% | 33.3% | 34.1% |
| Destinations (schools, clinics, pharmacies, senior centres, parks, …) | 1,772 | 22.3% | 30.1% | 44.6% |
| Pedestrian crashes 2019–2025 (35 deaths, 146 major injuries) | 1,030 | 38.7% | 34.2% | 25.2% |
| 311 pedestrian requests 2015–2026 | 36,883 | 21.9% | 27.6% | 36.0% |

- Senior centres are 8% audited (2 of 24); rec centres 5%; nursing homes 19%; schools 28%.
- The High Injury Network runs 192 km. 33% of its length lies within 20 m of an audited street and 60%
  within 20 m of an opened one.
- Stop-level boardings are not published. `trips_wd` (weekday trips serving the stop) is the only
  stop-level demand proxy. Ridership is per route: 135,585 average weekday riders system-wide in
  April 2026, led by the Red Line (7,892) and the 51 Carrick (6,013).

**8. 311 is a biased demand signal.** 37,442 of 968,914 requests (3.9%) are about walking. The largest
subjects are Broken Sidewalk 8,457, Blocked or Closed Sidewalks 7,246, Sidewalk has Ice or Litter
4,916, Crosswalk/Curb/Markings 4,529 and Sidewalk/Curb/ADA Ramp Maintenance 4,071.

- Across the 90 neighbourhoods, audit coverage correlates with barrier-complaint density
  (ρ = +0.53, p < 1e-7). Volunteers and complainers are concentrated in the same places.
- In the 23 neighbourhoods with at least 5 audited km, measured on audited streets only, label density
  and complaint density are *negatively* related (ρ = −0.36, p = 0.09). Where volunteers find the most
  barriers, residents file the fewest complaints.
- Use 311 as a reporting-propensity layer, not as ground truth or unweighted demand.
- "Curb ramps / ADA" requests jump from about 140 a year to 692 (2023) and 758 (2024), then fall back
  to 169. That looks like a category or intake change, not a real spike.

## Other layers in the snapshot

| Layer | What's there |
|---|---|
| City steps | 1,128 public stairways, 47.6 km, 44,450 steps. Not street edges, so outside Project Sidewalk's network entirely. |
| City crosswalk inventory | 4,648 crosswalks (3,457 two-lined, 703 continental, 284 ladder, 194 brick). |
| Sidewalk-to-street ratio | 2,103 county block groups, median 0.79; 30% under 1.0 (WPRDC, 2021). |
| Access scores | Project Sidewalk's own score for 4,982 of 12,902 explorable streets (median 0.50), 35 regions and 1,622 intersections. A ready-made baseline ranking to compare Tier 2 against. |
| Parcels | 2025 parcel centroids with geographic ids, plus the July 2026 parcel polygons. |

## Caveats & gaps

- **City membership** is by street-edge midpoint inside the city boundary. Tracts count as in the city
  when their representative point is.
- **"Audited"** means `audit_count > 0` on `/v3/api/streets`. The regions API's `audited_distance_m`
  counts only audits by high-quality users, 254.2 km over the whole deployment, so it sits lower.
- **`outdated`** is Project Sidewalk's flag for "audited, but newer imagery now exists". It does not
  prove the labels are wrong.
- **ACS tract estimates** carry wide margins of error, and none are propagated here. Correlations are
  unweighted Spearman across tracts. Barrier density uses only the 54–56 tracts with at least 1 audited km.
- **Point layers are dated inconsistently.** Hospitals are 2015; the county asset map mixes sources;
  2025 crashes may be incomplete; 311 is filtered by subject using the 13 subjects listed in the script.
- **Not checked:** whether `rawLabels.correct` itself is influenced by AI votes. It should be before
  `correct` is used as a label-quality target.
- **Not in the snapshot:** Mapillary / Panoramax coverage of the un-audited streets (the Tier 3 imagery
  question needs a Mapillary API token), OpenStreetMap sidewalk geometry, and the city's own curb-ramp
  inventory. The last does not appear to be published on WPRDC.

## Sources

Every file below is in `/data/datasets/pittsburgh/<file>`; `MANIFEST.json` there carries the sha256 of each.

### Project Sidewalk — Pittsburgh API (CC0)

| File | What | Size | Source | Page |
|---|---|---:|---|---|
| `accessScoreConfig.json` | weights behind the access score | 3 KB | [download](https://sidewalk-pittsburgh.cs.washington.edu/v3/api/accessScoreConfig) | [page](https://sidewalk-pittsburgh.cs.washington.edu/api) |
| `accessScoreIntersections.geojson` | access score per intersection | 1.1 MB | [download](https://sidewalk-pittsburgh.cs.washington.edu/v3/api/accessScoreIntersections?filetype=geojson) | [page](https://sidewalk-pittsburgh.cs.washington.edu/api) |
| `accessScoreRegions.geojson` | access score per neighbourhood | 0.4 MB | [download](https://sidewalk-pittsburgh.cs.washington.edu/v3/api/accessScoreRegions?filetype=geojson) | [page](https://sidewalk-pittsburgh.cs.washington.edu/api) |
| `accessScoreStreets.geojson` | Project Sidewalk access score per street | 14.5 MB | [download](https://sidewalk-pittsburgh.cs.washington.edu/v3/api/accessScoreStreets?filetype=geojson) | [page](https://sidewalk-pittsburgh.cs.washington.edu/api) |
| `cities.json` | all Project Sidewalk deployments | 19 KB | [download](https://sidewalk-pittsburgh.cs.washington.edu/v3/api/cities?filetype=json) | [page](https://sidewalk-pittsburgh.cs.washington.edu/api) |
| `labelClusters.geojson` | labels clustered into de-duplicated physical features | 10.4 MB | [download](https://sidewalk-pittsburgh.cs.washington.edu/v3/api/labelClusters?filetype=geojson) | [page](https://sidewalk-pittsburgh.cs.washington.edu/api) |
| `labelTags.json` | tag vocabulary per label type | 11 KB | [download](https://sidewalk-pittsburgh.cs.washington.edu/v3/api/labelTags) | [page](https://sidewalk-pittsburgh.cs.washington.edu/api) |
| `labelTypes.json` | label type definitions | 4 KB | [download](https://sidewalk-pittsburgh.cs.washington.edu/v3/api/labelTypes) | [page](https://sidewalk-pittsburgh.cs.washington.edu/api) |
| `overallStats.json` | city totals: km audited, label counts, severity stats | 7 KB | [download](https://sidewalk-pittsburgh.cs.washington.edu/v3/api/overallStats) | [page](https://sidewalk-pittsburgh.cs.washington.edu/api) |
| `rawLabels.csv` | every label: type, severity, tags, lat/lon, pano id, pixel coords, validations | 15.7 MB | [download](https://sidewalk-pittsburgh.cs.washington.edu/v3/api/rawLabels?filetype=csv) | [page](https://sidewalk-pittsburgh.cs.washington.edu/api) |
| `regions.geojson` | neighbourhood polygons with audit coverage | 0.4 MB | [download](https://sidewalk-pittsburgh.cs.washington.edu/v3/api/regions?filetype=geojson) | [page](https://sidewalk-pittsburgh.cs.washington.edu/api) |
| `sidewalkPresence.geojson` | per-street sidewalk present/absent inference | 99.9 MB | [download](https://sidewalk-pittsburgh.cs.washington.edu/v3/api/sidewalkPresence?filetype=geojson) | [page](https://sidewalk-pittsburgh.cs.washington.edu/api) |
| `streetTypes.json` | OSM way types | 2 KB | [download](https://sidewalk-pittsburgh.cs.washington.edu/v3/api/streetTypes) | [page](https://sidewalk-pittsburgh.cs.washington.edu/api) |
| `streets.geojson` | street edges with audit counts -- audited vs un-audited km | 42.8 MB | [download](https://sidewalk-pittsburgh.cs.washington.edu/v3/api/streets?filetype=geojson) | [page](https://sidewalk-pittsburgh.cs.washington.edu/api) |
| `userStats.csv` | per-user label counts and accuracy | 54 KB | [download](https://sidewalk-pittsburgh.cs.washington.edu/v3/api/userStats?filetype=csv) | [page](https://sidewalk-pittsburgh.cs.washington.edu/api) |
| `validationResultTypes.json` | validation result codes | 0 KB | [download](https://sidewalk-pittsburgh.cs.washington.edu/v3/api/validationResultTypes) | [page](https://sidewalk-pittsburgh.cs.washington.edu/api) |
| `validations.csv` | every agree/disagree/unsure validation | 9.1 MB | [download](https://sidewalk-pittsburgh.cs.washington.edu/v3/api/validations?filetype=csv) | [page](https://sidewalk-pittsburgh.cs.washington.edu/api) |

### WPRDC / City of Pittsburgh / Allegheny County / PRT

| File | What | Size | Source | Page |
|---|---|---:|---|---|
| `census_tracts_2020.geojson` | Allegheny County 2020 census tracts | 4.3 MB | [download](https://data.wprdc.org/dataset/93634185-1779-45dd-a593-c8e8aba3f95b/resource/83b2018d-70e5-4bad-b99d-89781f7425ed/download/2020_census_tracts.geojson) | [page](https://data.wprdc.org/dataset/allegheny-county-census-tracts-2020) |
| `city_boundary.geojson` | City of Pittsburgh boundary (WGS84) | 0.1 MB | [download](https://services1.arcgis.com/YZCmUqbcsUpOKfj7/arcgis/rest/services/CityBoundary/FeatureServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=geojson) | [page](https://data.wprdc.org/dataset/pittsburgh-city-boundary) |
| `city_steps.geojson` | City of Pittsburgh public steps | 1.4 MB | [download](https://data.wprdc.org/dataset/e9aa627c-cb22-4ba4-9961-56d9620a46af/resource/ff6dcffa-49ba-4431-954e-044ed519a4d7/download/___) | [page](https://data.wprdc.org/dataset/city-steps) |
| `city_steps_dictionary.csv` | steps data dictionary | 0 KB | [download](https://data.wprdc.org/datastore/dump/428b48ff-e9f6-49dd-8402-37c61e907de0) | [page](https://data.wprdc.org/dataset/city-steps) |
| `county_assets.csv` | community asset map: senior centres, libraries, rec centres, food, health ... | 12.4 MB | [download](https://data.wprdc.org/datastore/dump/5c7825d2-6814-40c7-aefe-3d0f3d6f22e7) | [page](https://data.wprdc.org/dataset/allegheny-county-assets) |
| `county_assets_sources.csv` | asset map: data source per asset type | 7 KB | [download](https://data.wprdc.org/dataset/cd2b3e27-ca31-43e0-a8c6-2e6c43b4050a/resource/279da54f-a520-4bb4-afd0-ffbb7e797faf/download/data_sources_by_asset_type.csv) | [page](https://data.wprdc.org/dataset/allegheny-county-assets) |
| `county_private_schools.geojson` | Allegheny County private schools (archived) | 0.2 MB | [download](https://data.wprdc.org/dataset/2a0c1b95-6154-4a67-8074-ad3d5c67ada1/resource/5737e805-1049-4ea9-a055-b3dff7542a4c/download/private_schools.geojson) | [page](https://data.wprdc.org/dataset/allegheny-county-private-schools-locations) |
| `county_public_schools.geojson` | Allegheny County public schools (archived) | 0.6 MB | [download](https://data.wprdc.org/dataset/ed485f75-044f-4614-b35d-65ff73dc46c3/resource/8bda18bd-716e-4b7e-b8fa-d9c5c632e481/download/public_schools.geojson) | [page](https://data.wprdc.org/dataset/allegheny-county-public-schools-local-education-agency-leas-locations) |
| `crash_data_primer.pdf` | crash data dictionary / primer | 0.4 MB | [download](https://data.wprdc.org/dataset/3130f583-9499-472b-bb5a-f63a6ff6059a/resource/c884d6da-588d-45ec-b029-8aaec8018500/download/database-primer-4-15.pdf) | [page](https://data.wprdc.org/dataset/allegheny-county-crash-data) |
| `crashes_2019.csv` | Allegheny County crashes 2019 (PennDOT) | 5.7 MB | [download](https://data.wprdc.org/datastore/dump/cb0a4d8b-2893-4d20-ad1c-47d5fdb7e8d5) | [page](https://data.wprdc.org/dataset/allegheny-county-crash-data) |
| `crashes_2020.csv` | Allegheny County crashes 2020 (PennDOT) | 4.6 MB | [download](https://data.wprdc.org/datastore/dump/514ae074-f42e-4bfb-8869-8d8c461dd824) | [page](https://data.wprdc.org/dataset/allegheny-county-crash-data) |
| `crashes_2021.csv` | Allegheny County crashes 2021 (PennDOT) | 5.4 MB | [download](https://data.wprdc.org/datastore/dump/e3b145c0-41ba-4cc9-9054-8f686ac59643) | [page](https://data.wprdc.org/dataset/allegheny-county-crash-data) |
| `crashes_2022.csv` | Allegheny County crashes 2022 (PennDOT) | 5.4 MB | [download](https://data.wprdc.org/datastore/dump/980f784b-978b-40c6-b649-f92f90b14dae) | [page](https://data.wprdc.org/dataset/allegheny-county-crash-data) |
| `crashes_2023.csv` | Allegheny County crashes 2023 (PennDOT) | 4.8 MB | [download](https://data.wprdc.org/datastore/dump/96777349-57df-48fb-a1d7-8384786fe71a) | [page](https://data.wprdc.org/dataset/allegheny-county-crash-data) |
| `crashes_2024.csv` | Allegheny County crashes 2024 (PennDOT) | 4.7 MB | [download](https://data.wprdc.org/datastore/dump/4c016b4c-59f0-45ca-981c-718c784b3462) | [page](https://data.wprdc.org/dataset/allegheny-county-crash-data) |
| `crashes_2025.csv` | Allegheny County crashes 2025 (PennDOT) | 4.9 MB | [download](https://data.wprdc.org/dataset/3130f583-9499-472b-bb5a-f63a6ff6059a/resource/c6bedb17-2d23-49b1-843a-8f6b41e5e5c3/download/crash_allegheny_2025.csv) | [page](https://data.wprdc.org/dataset/allegheny-county-crash-data) |
| `crosswalks.csv` | city crosswalk inventory | 0.6 MB | [download](https://data.wprdc.org/datastore/dump/632fbb91-c55d-4221-a8ad-91c72902bc61) | [page](https://data.wprdc.org/dataset/city-of-pittsburgh-markings) |
| `high_injury_network.geojson` | city High Injury Network corridors | 0.5 MB | [download](https://data.wprdc.org/dataset/ee23d41a-7c7d-4093-9245-b1691b7b9b4d/resource/f66956f7-0099-40c0-b316-e97cd1360ceb/download/high_injury.geojson) | [page](https://data.wprdc.org/dataset/high-injury-network) |
| `hospitals.csv` | Allegheny County hospitals (2015) | 2 KB | [download](https://data.wprdc.org/dataset/d2289ea6-b1ef-4acb-afa7-028cccb7fb18/resource/2d9db439-8f85-4b6d-ab92-423a2ef9c7d9/download/data-hospitallocations.csv) | [page](https://data.wprdc.org/dataset/hospitals) |
| `intersection_markings.geojson` | city intersection markings (crosswalks, stop bars) | 5.1 MB | [download](https://data.wprdc.org/dataset/31ce085b-87b9-4ffd-adbb-0a9f5b3cf3df/resource/f2f0c299-4f7b-4689-be3c-a2ad38252cf4/download/markings.geojson) | [page](https://data.wprdc.org/dataset/city-of-pittsburgh-markings) |
| `markings_dictionary.csv` | markings data dictionary | 0 KB | [download](https://data.wprdc.org/datastore/dump/4cc6c85a-d7bd-44f6-86bd-d9ade92cd362) | [page](https://data.wprdc.org/dataset/city-of-pittsburgh-markings) |
| `neighborhoods.geojson` | City of Pittsburgh neighbourhood polygons | 1.2 MB | [download](https://data.wprdc.org/dataset/e672f13d-71c4-4a66-8f38-710e75ed80a4/resource/4af8e160-57e9-4ebf-a501-76ca1b42fc99/download/neighborhoods.geojson) | [page](https://data.wprdc.org/dataset/neighborhoods2) |
| `parcel_boundaries_shp.zip` | Allegheny County parcel polygons (shapefile) | 115.3 MB | [download](https://data.wprdc.org/dataset/709e4e52-6f82-4cd0-a848-f3e2b3f5d22b/resource/be216088-d51c-41ce-aa4a-2c315c2c7725/download/alleghenycounty_parcels202607.zip) | [page](https://data.wprdc.org/dataset/allegheny-county-parcel-boundaries1) |
| `parcel_centroids_2025_03.csv` | every parcel centroid with tract / neighbourhood ids | 111.8 MB | [download](https://data.wprdc.org/dataset/2536e5e2-253b-4c58-969d-687828bb94c6/resource/3fab7152-3f11-4788-8372-4c33f86ea813/download/parcel_centroids_2025_march.csv) | [page](https://data.wprdc.org/dataset/parcel-centroids-in-allegheny-county-with-geographic-identifiers) |
| `pedestrian_crashes_2004_2020_shp.zip` | pedestrian crashes 2004-2020 (shapefile) | 1.3 MB | [download](https://data.wprdc.org/dataset/853a077d-0a31-4292-8a1d-5d60b530169b/resource/00715f59-c979-438f-81d8-0a338ddb21b3/download/crashesped.zip) | [page](https://data.wprdc.org/dataset/sidewalk-to-street-walkability-ratio) |
| `pgh_311.csv` | all Pittsburgh 311 requests | 288.4 MB | [download](https://data.wprdc.org/datastore/dump/5202679a-d243-402e-b82a-63189995a942) | [page](https://data.wprdc.org/dataset/pittsburgh-311-data) |
| `pgh_311_codebook.xlsx` | 311 issue / subject codebook | 16 KB | [download](https://data.wprdc.org/dataset/8069a170-92d1-4f5e-bc03-327bcf262545/resource/2de7cb02-b60f-4a54-99b4-bd633b0fa365/download/311-codebook-with-public-access-level-2026.xlsx) | [page](https://data.wprdc.org/dataset/pittsburgh-311-data) |
| `pgh_311_dictionary.xlsx` | 311 data dictionary | 10 KB | [download](https://data.wprdc.org/dataset/8069a170-92d1-4f5e-bc03-327bcf262545/resource/3a1d150e-007c-4126-9f41-43e56fed7695/download/311-data-dictionary-2026.xlsx) | [page](https://data.wprdc.org/dataset/pittsburgh-311-data) |
| `pps_school_locations.csv` | Pittsburgh Public Schools, April 2019 | 9 KB | [download](https://data.wprdc.org/dataset/46fb3ca4-e844-4b42-b034-e87291d34699/resource/06664b02-c673-49d5-8a70-d3cd1c18ac8d/download/pps_schoolsapr2019publish.csv) | [page](https://data.wprdc.org/dataset/pittsburgh-public-school-locations) |
| `primary_care.csv` | primary care facilities (2014) | 59 KB | [download](https://data.wprdc.org/dataset/52f87f59-02ed-4f74-9ab9-2c6b9542234d/resource/a11c31cf-a116-4076-8475-c4f185358c2d/download/data-primary-care-access-facilities.csv) | [page](https://data.wprdc.org/dataset/allegheny-county-primary-care-facilities) |
| `prt_monthly_ridership_by_route.csv` | average weekday ridership per route per month | 1.9 MB | [download](https://data.wprdc.org/dataset/e6c089da-43d1-439b-92fc-e500d6fb5e73/resource/12bb84ed-397e-435c-8d1b-8ce543108698/download/prt-monthly-ridership-through-april-2026.csv) | [page](https://data.wprdc.org/dataset/prt-monthly-average-ridership-by-route) |
| `prt_ridership_dictionary.csv` | ridership data dictionary | 1 KB | [download](https://data.wprdc.org/dataset/e6c089da-43d1-439b-92fc-e500d6fb5e73/resource/911b53eb-65f8-4002-9481-096a31699acd/download/ridershipmonthavg-data-dictionary.csv) | [page](https://data.wprdc.org/dataset/prt-monthly-average-ridership-by-route) |
| `prt_routes.geojson` | PRT current routes | 7.6 MB | [download](https://data.wprdc.org/dataset/6be8185a-7fc3-48c7-9b8d-716602ea03f0/resource/4c3bf8b1-1bcd-42d6-bc7b-4c45be7da085/download/routes.geojson) | [page](https://data.wprdc.org/dataset/prt-current-transit-routes) |
| `prt_stops.geojson` | PRT bus/rail stops | 6.3 MB | [download](https://data.wprdc.org/dataset/33d5f44b-5315-4374-b3e3-e4246e8ad5c9/resource/d6e6ed6e-9220-4a0e-9796-e72d83ce8e7a/download/stops.geojson) | [page](https://data.wprdc.org/dataset/prt-of-allegheny-county-transit-stops) |
| `prt_stops_dictionary.csv` | PRT stops data dictionary | 0 KB | [download](https://data.wprdc.org/dataset/33d5f44b-5315-4374-b3e3-e4246e8ad5c9/resource/0a26e18e-f78b-43d2-8c75-2498ae255ecf/download/stopsdatadictionary.csv) | [page](https://data.wprdc.org/dataset/prt-of-allegheny-county-transit-stops) |
| `sidewalks_and_steps_shp.zip` | sidewalk and step centrelines (shapefile) | 30.8 MB | [download](https://data.wprdc.org/dataset/853a077d-0a31-4292-8a1d-5d60b530169b/resource/8b7f8dbf-3c31-4873-9953-533e4bf88f52/download/sidewalksstepsblockgroup.zip) | [page](https://data.wprdc.org/dataset/sidewalk-to-street-walkability-ratio) |
| `ucsur_neighborhood_profiles_2024.csv` | UCSUR neighbourhood profiles: ACS 2008-12 vs 2018-22 by neighbourhood | 0.2 MB | [download](https://data.wprdc.org/dataset/fd29df0b-07c0-4e97-926e-d781dd1f8a8d/resource/a2d6468e-0229-4c6a-92c5-10814092e580/download/neighborhoodprofiles_data_june2024.csv) | [page](https://data.wprdc.org/dataset/ucsur_neighborhoodprofiles_2024) |
| `ucsur_neighborhood_profiles_dictionary.xlsx` | UCSUR profiles data dictionary | 31 KB | [download](https://data.wprdc.org/dataset/fd29df0b-07c0-4e97-926e-d781dd1f8a8d/resource/345020ef-6941-44a8-9d48-58219283d537/download/pittsburghprofiles_june2024_datadictionary.xlsx) | [page](https://data.wprdc.org/dataset/ucsur_neighborhoodprofiles_2024) |
| `ucsur_neighborhood_tract_index_2022.xlsx` | neighbourhood <-> 2020 tract crosswalk | 12 KB | [download](https://data.wprdc.org/dataset/fd29df0b-07c0-4e97-926e-d781dd1f8a8d/resource/95abf133-32d9-4f73-b262-2b239d1eca2e/download/pittsburghneighborhood_tract_index_2022.xlsx) | [page](https://data.wprdc.org/dataset/ucsur_neighborhoodprofiles_2024) |
| `walk_scores_tract_2014.csv` | Walk Score by census tract (2014) | 4 KB | [download](https://data.wprdc.org/dataset/4d3d4324-b32a-4519-b56f-cee2340057bf/resource/682b1df1-a63b-4413-9362-ba077af63baa/download/walkscorect.xls-walk-score-by-ct.csv) | [page](https://data.wprdc.org/dataset/allegheny-county-walk-scores) |
| `walkability_blockgroup.csv` | sidewalk-to-street length ratio per block group (2021) | 0.1 MB | [download](https://data.wprdc.org/dataset/853a077d-0a31-4292-8a1d-5d60b530169b/resource/b90ccee1-c0aa-43b9-93e2-8a25e690c393/download/sidewalkstreetratioupload.csv) | [page](https://data.wprdc.org/dataset/sidewalk-to-street-walkability-ratio) |
| `walkability_streets_shp.zip` | street centrelines used in the ratio (shapefile) | 15.9 MB | [download](https://data.wprdc.org/dataset/853a077d-0a31-4292-8a1d-5d60b530169b/resource/3754d493-625d-41ff-ba0f-1acead07b0e5/download/streetblockgroupwithexclusions.zip) | [page](https://data.wprdc.org/dataset/sidewalk-to-street-walkability-ratio) |
| `walkability_tract.csv` | sidewalk-to-street length ratio per tract (2021) | 29 KB | [download](https://data.wprdc.org/dataset/853a077d-0a31-4292-8a1d-5d60b530169b/resource/ab8db07e-09ef-45c4-bad2-f7d32bfc5f26/download/sidewalkstreetratiotract.csv) | [page](https://data.wprdc.org/dataset/sidewalk-to-street-walkability-ratio) |

### ACS 2020–24 5-year via Census Reporter

| File | What | Size | Source | Page |
|---|---|---:|---|---|
| `acs5_tract_B01001.json` | sex by age (-> share 65+) | 0.7 MB | [download](https://api.censusreporter.org/1.0/data/show/latest?table_ids=B01001&geo_ids=140%7C05000US42003) | [page](https://censusreporter.org/tables/B01001/) |
| `acs5_tract_B01003.json` | total population | 75 KB | [download](https://api.censusreporter.org/1.0/data/show/latest?table_ids=B01003&geo_ids=140%7C05000US42003) | [page](https://censusreporter.org/tables/B01003/) |
| `acs5_tract_B03002.json` | race by Hispanic origin | 0.3 MB | [download](https://api.censusreporter.org/1.0/data/show/latest?table_ids=B03002&geo_ids=140%7C05000US42003) | [page](https://censusreporter.org/tables/B03002/) |
| `acs5_tract_B08201.json` | vehicles available per household (-> car-free households) | 0.5 MB | [download](https://api.censusreporter.org/1.0/data/show/latest?table_ids=B08201&geo_ids=140%7C05000US42003) | [page](https://censusreporter.org/tables/B08201/) |
| `acs5_tract_B08301.json` | means of transportation to work (-> walk / transit share) | 0.3 MB | [download](https://api.censusreporter.org/1.0/data/show/latest?table_ids=B08301&geo_ids=140%7C05000US42003) | [page](https://censusreporter.org/tables/B08301/) |
| `acs5_tract_B17001.json` | poverty status | 0.9 MB | [download](https://api.censusreporter.org/1.0/data/show/latest?table_ids=B17001&geo_ids=140%7C05000US42003) | [page](https://censusreporter.org/tables/B17001/) |
| `acs5_tract_B18101.json` | disability status by age | 0.6 MB | [download](https://api.censusreporter.org/1.0/data/show/latest?table_ids=B18101&geo_ids=140%7C05000US42003) | [page](https://censusreporter.org/tables/B18101/) |
| `acs5_tract_B19013.json` | median household income | 76 KB | [download](https://api.censusreporter.org/1.0/data/show/latest?table_ids=B19013&geo_ids=140%7C05000US42003) | [page](https://censusreporter.org/tables/B19013/) |

Code: [`scripts/fetch_pittsburgh_data.py`](../../scripts/fetch_pittsburgh_data.py) (fetch), [`src/eda/pittsburgh_eda.py`](../../src/eda/pittsburgh_eda.py) (analysis).
