#!/usr/bin/env python3
"""Snapshot the Pittsburgh-specific public data behind Tier 2 into one directory.

Sources: Project Sidewalk's Pittsburgh API (labels, streets, clusters, validations),
WPRDC (PRT transit, city/county GIS, 311, crashes, UCSUR neighbourhood demographics)
and Census Reporter (ACS 5-year by tract -- the Census Bureau's own API now demands a
key, Census Reporter serves the same tables without one).

These are live APIs that change daily, so every file is recorded in MANIFEST.json with
its source URL, landing page, retrieval time, byte size and sha256. That is what lets an
analysis say exactly which snapshot it ran on.

Stdlib only, so it runs from any env. Idempotent: files already present are skipped
unless --force.

usage:
  python scripts/fetch_pittsburgh_data.py --out /data/datasets/pittsburgh
"""
import argparse, concurrent.futures as cf, hashlib, json, os, sys, threading, time, urllib.request
from datetime import datetime, timezone

UA = {"User-Agent": "cmu-10718-sidewalk/1.0 (research; contact via github.com/Aditya-Katkuri/MLiP)"}

PS = "https://sidewalk-pittsburgh.cs.washington.edu/v3/api"
PS_PAGE = "https://sidewalk-pittsburgh.cs.washington.edu/api"
WPRDC = "https://data.wprdc.org"
CR = "https://api.censusreporter.org/1.0/data/show/latest"
ALLEGHENY_TRACTS = "140|05000US42003"


def wprdc(path, package, url, desc, license="see landing page"):
    return dict(path=path, url=url, page=f"{WPRDC}/dataset/{package}", desc=desc, license=license)


SOURCES = [
    # ---- Project Sidewalk, Pittsburgh (CC0) ----
    *[dict(path=f"projectsidewalk/{name}", url=f"{PS}/{q}", page=PS_PAGE, desc=desc, license="CC0")
      for name, q, desc in [
          ("rawLabels.csv",               "rawLabels?filetype=csv",               "every label: type, severity, tags, lat/lon, pano id, pixel coords, validations"),
          ("streets.geojson",             "streets?filetype=geojson",             "street edges with audit counts -- audited vs un-audited km"),
          ("regions.geojson",             "regions?filetype=geojson",             "neighbourhood polygons with audit coverage"),
          ("labelClusters.geojson",       "labelClusters?filetype=geojson",       "labels clustered into de-duplicated physical features"),
          ("sidewalkPresence.geojson",    "sidewalkPresence?filetype=geojson",    "per-street sidewalk present/absent inference"),
          ("validations.csv",             "validations?filetype=csv",             "every agree/disagree/unsure validation"),
          ("accessScoreStreets.geojson",  "accessScoreStreets?filetype=geojson",  "Project Sidewalk access score per street"),
          ("accessScoreRegions.geojson",  "accessScoreRegions?filetype=geojson",  "access score per neighbourhood"),
          ("accessScoreIntersections.geojson", "accessScoreIntersections?filetype=geojson", "access score per intersection"),
          ("accessScoreConfig.json",      "accessScoreConfig",                    "weights behind the access score"),
          ("overallStats.json",           "overallStats",                         "city totals: km audited, label counts, severity stats"),
          ("userStats.csv",               "userStats?filetype=csv",               "per-user label counts and accuracy"),
          ("labelTypes.json",             "labelTypes",                           "label type definitions"),
          ("labelTags.json",              "labelTags",                            "tag vocabulary per label type"),
          ("streetTypes.json",            "streetTypes",                          "OSM way types"),
          ("validationResultTypes.json",  "validationResultTypes",                "validation result codes"),
          ("cities.json",                 "cities?filetype=json",                 "all Project Sidewalk deployments"),
      ]],

    # ---- pedestrian demand: transit ----
    wprdc("wprdc/prt_stops.geojson", "prt-of-allegheny-county-transit-stops",
          f"{WPRDC}/dataset/33d5f44b-5315-4374-b3e3-e4246e8ad5c9/resource/d6e6ed6e-9220-4a0e-9796-e72d83ce8e7a/download/stops.geojson",
          "PRT bus/rail stops", "CC-BY"),
    wprdc("wprdc/prt_stops_dictionary.csv", "prt-of-allegheny-county-transit-stops",
          f"{WPRDC}/dataset/33d5f44b-5315-4374-b3e3-e4246e8ad5c9/resource/0a26e18e-f78b-43d2-8c75-2498ae255ecf/download/stopsdatadictionary.csv",
          "PRT stops data dictionary", "CC-BY"),
    wprdc("wprdc/prt_routes.geojson", "prt-current-transit-routes",
          f"{WPRDC}/dataset/6be8185a-7fc3-48c7-9b8d-716602ea03f0/resource/4c3bf8b1-1bcd-42d6-bc7b-4c45be7da085/download/routes.geojson",
          "PRT current routes", "CC-BY"),
    wprdc("wprdc/prt_monthly_ridership_by_route.csv", "prt-monthly-average-ridership-by-route",
          f"{WPRDC}/dataset/e6c089da-43d1-439b-92fc-e500d6fb5e73/resource/12bb84ed-397e-435c-8d1b-8ce543108698/download/prt-monthly-ridership-through-april-2026.csv",
          "average weekday ridership per route per month", "CC-BY"),
    wprdc("wprdc/prt_ridership_dictionary.csv", "prt-monthly-average-ridership-by-route",
          f"{WPRDC}/dataset/e6c089da-43d1-439b-92fc-e500d6fb5e73/resource/911b53eb-65f8-4002-9481-096a31699acd/download/ridershipmonthavg-data-dictionary.csv",
          "ridership data dictionary", "CC-BY"),

    # ---- pedestrian demand: destinations ----
    wprdc("wprdc/pps_school_locations.csv", "pittsburgh-public-school-locations",
          f"{WPRDC}/dataset/46fb3ca4-e844-4b42-b034-e87291d34699/resource/06664b02-c673-49d5-8a70-d3cd1c18ac8d/download/pps_schoolsapr2019publish.csv",
          "Pittsburgh Public Schools, April 2019", "public domain"),
    wprdc("wprdc/county_public_schools.geojson", "allegheny-county-public-schools-local-education-agency-leas-locations",
          f"{WPRDC}/dataset/ed485f75-044f-4614-b35d-65ff73dc46c3/resource/8bda18bd-716e-4b7e-b8fa-d9c5c632e481/download/public_schools.geojson",
          "Allegheny County public schools (archived)"),
    wprdc("wprdc/county_private_schools.geojson", "allegheny-county-private-schools-locations",
          f"{WPRDC}/dataset/2a0c1b95-6154-4a67-8074-ad3d5c67ada1/resource/5737e805-1049-4ea9-a055-b3dff7542a4c/download/private_schools.geojson",
          "Allegheny County private schools (archived)"),
    wprdc("wprdc/hospitals.csv", "hospitals",
          f"{WPRDC}/dataset/d2289ea6-b1ef-4acb-afa7-028cccb7fb18/resource/2d9db439-8f85-4b6d-ab92-423a2ef9c7d9/download/data-hospitallocations.csv",
          "Allegheny County hospitals (2015)", "CC0"),
    wprdc("wprdc/primary_care.csv", "allegheny-county-primary-care-facilities",
          f"{WPRDC}/dataset/52f87f59-02ed-4f74-9ab9-2c6b9542234d/resource/a11c31cf-a116-4076-8475-c4f185358c2d/download/data-primary-care-access-facilities.csv",
          "primary care facilities (2014)", "CC0"),
    wprdc("wprdc/county_assets.csv", "allegheny-county-assets",
          f"{WPRDC}/datastore/dump/5c7825d2-6814-40c7-aefe-3d0f3d6f22e7",
          "community asset map: senior centres, libraries, rec centres, food, health ...", "CC0"),
    wprdc("wprdc/county_assets_sources.csv", "allegheny-county-assets",
          f"{WPRDC}/dataset/cd2b3e27-ca31-43e0-a8c6-2e6c43b4050a/resource/279da54f-a520-4bb4-afd0-ffbb7e797faf/download/data_sources_by_asset_type.csv",
          "asset map: data source per asset type", "CC0"),

    # ---- the pedestrian network itself ----
    wprdc("wprdc/city_steps.geojson", "city-steps",
          f"{WPRDC}/dataset/e9aa627c-cb22-4ba4-9961-56d9620a46af/resource/ff6dcffa-49ba-4431-954e-044ed519a4d7/download/___",
          "City of Pittsburgh public steps", "CC-BY"),
    wprdc("wprdc/city_steps_dictionary.csv", "city-steps",
          f"{WPRDC}/datastore/dump/428b48ff-e9f6-49dd-8402-37c61e907de0", "steps data dictionary", "CC-BY"),
    wprdc("wprdc/intersection_markings.geojson", "city-of-pittsburgh-markings",
          f"{WPRDC}/dataset/31ce085b-87b9-4ffd-adbb-0a9f5b3cf3df/resource/f2f0c299-4f7b-4689-be3c-a2ad38252cf4/download/markings.geojson",
          "city intersection markings (crosswalks, stop bars)", "CC-BY"),
    wprdc("wprdc/crosswalks.csv", "city-of-pittsburgh-markings",
          f"{WPRDC}/datastore/dump/632fbb91-c55d-4221-a8ad-91c72902bc61", "city crosswalk inventory", "CC-BY"),
    wprdc("wprdc/markings_dictionary.csv", "city-of-pittsburgh-markings",
          f"{WPRDC}/datastore/dump/4cc6c85a-d7bd-44f6-86bd-d9ade92cd362", "markings data dictionary", "CC-BY"),
    wprdc("wprdc/walkability_blockgroup.csv", "sidewalk-to-street-walkability-ratio",
          f"{WPRDC}/dataset/853a077d-0a31-4292-8a1d-5d60b530169b/resource/b90ccee1-c0aa-43b9-93e2-8a25e690c393/download/sidewalkstreetratioupload.csv",
          "sidewalk-to-street length ratio per block group (2021)", "CC-BY"),
    wprdc("wprdc/walkability_tract.csv", "sidewalk-to-street-walkability-ratio",
          f"{WPRDC}/dataset/853a077d-0a31-4292-8a1d-5d60b530169b/resource/ab8db07e-09ef-45c4-bad2-f7d32bfc5f26/download/sidewalkstreetratiotract.csv",
          "sidewalk-to-street length ratio per tract (2021)", "CC-BY"),
    wprdc("wprdc/sidewalks_and_steps_shp.zip", "sidewalk-to-street-walkability-ratio",
          f"{WPRDC}/dataset/853a077d-0a31-4292-8a1d-5d60b530169b/resource/8b7f8dbf-3c31-4873-9953-533e4bf88f52/download/sidewalksstepsblockgroup.zip",
          "sidewalk and step centrelines (shapefile)", "CC-BY"),
    wprdc("wprdc/walkability_streets_shp.zip", "sidewalk-to-street-walkability-ratio",
          f"{WPRDC}/dataset/853a077d-0a31-4292-8a1d-5d60b530169b/resource/3754d493-625d-41ff-ba0f-1acead07b0e5/download/streetblockgroupwithexclusions.zip",
          "street centrelines used in the ratio (shapefile)", "CC-BY"),
    wprdc("wprdc/pedestrian_crashes_2004_2020_shp.zip", "sidewalk-to-street-walkability-ratio",
          f"{WPRDC}/dataset/853a077d-0a31-4292-8a1d-5d60b530169b/resource/00715f59-c979-438f-81d8-0a338ddb21b3/download/crashesped.zip",
          "pedestrian crashes 2004-2020 (shapefile)", "CC-BY"),
    wprdc("wprdc/high_injury_network.geojson", "high-injury-network",
          f"{WPRDC}/dataset/ee23d41a-7c7d-4093-9245-b1691b7b9b4d/resource/f66956f7-0099-40c0-b316-e97cd1360ceb/download/high_injury.geojson",
          "city High Injury Network corridors"),
    *[wprdc(f"wprdc/crashes_{y}.csv", "allegheny-county-crash-data", u, f"Allegheny County crashes {y} (PennDOT)", "CC0")
      for y, u in [
          (2019, f"{WPRDC}/datastore/dump/cb0a4d8b-2893-4d20-ad1c-47d5fdb7e8d5"),
          (2020, f"{WPRDC}/datastore/dump/514ae074-f42e-4bfb-8869-8d8c461dd824"),
          (2021, f"{WPRDC}/datastore/dump/e3b145c0-41ba-4cc9-9054-8f686ac59643"),
          (2022, f"{WPRDC}/datastore/dump/980f784b-978b-40c6-b649-f92f90b14dae"),
          (2023, f"{WPRDC}/datastore/dump/96777349-57df-48fb-a1d7-8384786fe71a"),
          (2024, f"{WPRDC}/datastore/dump/4c016b4c-59f0-45ca-981c-718c784b3462"),
          (2025, f"{WPRDC}/dataset/3130f583-9499-472b-bb5a-f63a6ff6059a/resource/c6bedb17-2d23-49b1-843a-8f6b41e5e5c3/download/crash_allegheny_2025.csv"),
      ]],
    wprdc("wprdc/crash_data_primer.pdf", "allegheny-county-crash-data",
          f"{WPRDC}/dataset/3130f583-9499-472b-bb5a-f63a6ff6059a/resource/c884d6da-588d-45ec-b029-8aaec8018500/download/database-primer-4-15.pdf",
          "crash data dictionary / primer", "CC0"),

    # ---- 311: what residents actually report ----
    wprdc("wprdc/pgh_311.csv", "pittsburgh-311-data",
          f"{WPRDC}/datastore/dump/5202679a-d243-402e-b82a-63189995a942", "all Pittsburgh 311 requests", "CC-BY"),
    wprdc("wprdc/pgh_311_codebook.xlsx", "pittsburgh-311-data",
          f"{WPRDC}/dataset/8069a170-92d1-4f5e-bc03-327bcf262545/resource/2de7cb02-b60f-4a54-99b4-bd633b0fa365/download/311-codebook-with-public-access-level-2026.xlsx",
          "311 issue / subject codebook", "CC-BY"),
    wprdc("wprdc/pgh_311_dictionary.xlsx", "pittsburgh-311-data",
          f"{WPRDC}/dataset/8069a170-92d1-4f5e-bc03-327bcf262545/resource/3a1d150e-007c-4126-9f41-43e56fed7695/download/311-data-dictionary-2026.xlsx",
          "311 data dictionary", "CC-BY"),

    # ---- geography + demographics (equity audit) ----
    wprdc("wprdc/neighborhoods.geojson", "neighborhoods2",
          f"{WPRDC}/dataset/e672f13d-71c4-4a66-8f38-710e75ed80a4/resource/4af8e160-57e9-4ebf-a501-76ca1b42fc99/download/neighborhoods.geojson",
          "City of Pittsburgh neighbourhood polygons", "CC-BY"),
    dict(path="wprdc/city_boundary.geojson", page=f"{WPRDC}/dataset/pittsburgh-city-boundary",
         url="https://services1.arcgis.com/YZCmUqbcsUpOKfj7/arcgis/rest/services/CityBoundary/FeatureServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=geojson",
         desc="City of Pittsburgh boundary (WGS84)", license="not specified"),
    wprdc("wprdc/census_tracts_2020.geojson", "allegheny-county-census-tracts-2020",
          f"{WPRDC}/dataset/93634185-1779-45dd-a593-c8e8aba3f95b/resource/83b2018d-70e5-4bad-b99d-89781f7425ed/download/2020_census_tracts.geojson",
          "Allegheny County 2020 census tracts"),
    wprdc("wprdc/ucsur_neighborhood_profiles_2024.csv", "ucsur_neighborhoodprofiles_2024",
          f"{WPRDC}/dataset/fd29df0b-07c0-4e97-926e-d781dd1f8a8d/resource/a2d6468e-0229-4c6a-92c5-10814092e580/download/neighborhoodprofiles_data_june2024.csv",
          "UCSUR neighbourhood profiles: ACS 2008-12 vs 2018-22 by neighbourhood", "public domain"),
    wprdc("wprdc/ucsur_neighborhood_profiles_dictionary.xlsx", "ucsur_neighborhoodprofiles_2024",
          f"{WPRDC}/dataset/fd29df0b-07c0-4e97-926e-d781dd1f8a8d/resource/345020ef-6941-44a8-9d48-58219283d537/download/pittsburghprofiles_june2024_datadictionary.xlsx",
          "UCSUR profiles data dictionary", "public domain"),
    wprdc("wprdc/ucsur_neighborhood_tract_index_2022.xlsx", "ucsur_neighborhoodprofiles_2024",
          f"{WPRDC}/dataset/fd29df0b-07c0-4e97-926e-d781dd1f8a8d/resource/95abf133-32d9-4f73-b262-2b239d1eca2e/download/pittsburghneighborhood_tract_index_2022.xlsx",
          "neighbourhood <-> 2020 tract crosswalk", "public domain"),
    wprdc("wprdc/walk_scores_tract_2014.csv", "allegheny-county-walk-scores",
          f"{WPRDC}/dataset/4d3d4324-b32a-4519-b56f-cee2340057bf/resource/682b1df1-a63b-4413-9362-ba077af63baa/download/walkscorect.xls-walk-score-by-ct.csv",
          "Walk Score by census tract (2014)", "CC0"),
    wprdc("wprdc/parcel_centroids_2025_03.csv", "parcel-centroids-in-allegheny-county-with-geographic-identifiers",
          f"{WPRDC}/dataset/2536e5e2-253b-4c58-969d-687828bb94c6/resource/3fab7152-3f11-4788-8372-4c33f86ea813/download/parcel_centroids_2025_march.csv",
          "every parcel centroid with tract / neighbourhood ids", "CC0"),
    wprdc("wprdc/parcel_boundaries_shp.zip", "allegheny-county-parcel-boundaries1",
          f"{WPRDC}/dataset/709e4e52-6f82-4cd0-a848-f3e2b3f5d22b/resource/be216088-d51c-41ce-aa4a-2c315c2c7725/download/alleghenycounty_parcels202607.zip",
          "Allegheny County parcel polygons (shapefile)"),

    # ---- ACS 2020-24 5-year by tract, via Census Reporter ----
    *[dict(path=f"census/acs5_tract_{t}.json", url=f"{CR}?table_ids={t}&geo_ids={ALLEGHENY_TRACTS}",
           page=f"https://censusreporter.org/tables/{t}/", desc=desc, license="public domain (US Census Bureau)")
      for t, desc in [
          ("B01003", "total population"),
          ("B01001", "sex by age (-> share 65+)"),
          ("B19013", "median household income"),
          ("B17001", "poverty status"),
          ("B03002", "race by Hispanic origin"),
          ("B18101", "disability status by age"),
          ("B08201", "vehicles available per household (-> car-free households)"),
          ("B08301", "means of transportation to work (-> walk / transit share)"),
      ]],
]

HOST_LIMIT = {"sidewalk-pittsburgh.cs.washington.edu": 2}   # be gentle with a research server
_sems, _sems_lock = {}, threading.Lock()


def _sem(url):
    host = url.split("/")[2]
    with _sems_lock:
        return _sems.setdefault(host, threading.Semaphore(HOST_LIMIT.get(host, 4)))


def fetch(src, out, force):
    dest = os.path.join(out, src["path"])
    if os.path.exists(dest) and os.path.getsize(dest) > 0 and not force:
        return src["path"], None, "skip"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    last = None
    for attempt in range(4):
        try:
            with _sem(src["url"]):
                t0 = time.time()
                h, n = hashlib.sha256(), 0
                req = urllib.request.Request(src["url"], headers=UA)
                with urllib.request.urlopen(req, timeout=900) as r, open(dest + ".part", "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk); h.update(chunk); n += len(chunk)
            if n == 0:
                raise IOError("empty response")
            os.replace(dest + ".part", dest)
            meta = dict(src, bytes=n, sha256=h.hexdigest(), seconds=round(time.time() - t0, 1),
                        retrieved_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            return src["path"], meta, "ok"
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:200]}"
            time.sleep(10 * (attempt + 1))
    return src["path"], None, f"FAIL {last}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/data/datasets/pittsburgh")
    p.add_argument("--force", action="store_true")
    p.add_argument("--only", default="", help="substring filter on path, e.g. projectsidewalk/")
    a = p.parse_args()

    manifest_path = os.path.join(a.out, "MANIFEST.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    todo = [s for s in SOURCES if a.only in s["path"]]
    print(f"{len(todo)} sources -> {a.out}", flush=True)

    fails = 0
    # More workers than any host limit, so threads parked on the Project Sidewalk semaphore
    # never starve the other hosts; as_completed so a slow endpoint doesn't hold up the log.
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(fetch, s, a.out, a.force) for s in todo]
        for fut in cf.as_completed(futs):
            path, meta, status = fut.result()
            if meta:
                manifest[path] = meta
                print(f"  ok   {path:48s} {meta['bytes']/1e6:9.2f} MB  {meta['seconds']:6.1f}s", flush=True)
            elif status == "skip":
                print(f"  skip {path}", flush=True)
            else:
                fails += 1
                print(f"  {status}  {path}", flush=True)
            os.makedirs(a.out, exist_ok=True)
            with open(manifest_path + ".tmp", "w") as f:
                json.dump(dict(sorted(manifest.items())), f, indent=1)
            os.replace(manifest_path + ".tmp", manifest_path)
    print(f"done, {fails} failed", flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
