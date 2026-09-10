#!/usr/bin/env python3
"""Copy a Hugging Face repo into Azure Blob **server-side** -- no bytes touch this machine.

Why this exists: the runbook's normal path is HF -> VM -> Blob, which is right once
a VM exists (Azure ingress is free, VM->blob in-region is free). Before quota is
approved there is no VM, and routing hundreds of GB through a laptop is both slow
and, for the 462 GB dataset, physically impossible on most disks.

Azure Blob's "copy from URL" makes Azure's own servers fetch the source. One
wrinkle: it refuses HTTP redirects, and every HF `resolve/` URL is a 307 to their
CDN. So we resolve the redirect ourselves and hand Azure the final CDN URL.

usage:
  python scripts/hf_to_blob.py --repo projectsidewalk/rampnet-benchmark \
      --repo-type dataset --dest rampnet-benchmark
"""
import argparse, concurrent.futures as cf, sys, time, urllib.request

from azure.identity import AzureCliCredential
from azure.storage.blob import BlobServiceClient

HF_API = "https://huggingface.co/api"
UA = {"User-Agent": "sidewalk-10718/1.0"}


def list_files(repo, repo_type):
    path = f"{HF_API}/models/{repo}" if repo_type == "model" else f"{HF_API}/datasets/{repo}"
    import json
    with urllib.request.urlopen(urllib.request.Request(path, headers=UA), timeout=60) as r:
        meta = json.load(r)
    return [s["rfilename"] for s in meta.get("siblings", [])]


def source_url(repo, repo_type, rel):
    base = "https://huggingface.co" + ("" if repo_type == "model" else f"/{repo_type}s")
    return f"{base}/{repo}/resolve/main/{rel}"


def resolve(url):
    """Follow HF's 307 to the CDN. Azure will not follow it for us."""
    req = urllib.request.Request(url, headers=UA, method="GET")
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.url


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--repo-type", default="dataset", choices=["model", "dataset"])
    p.add_argument("--account", default="sidewalkdata23770")
    p.add_argument("--container", default="datasets")
    p.add_argument("--dest", required=True, help="blob prefix, e.g. rampnet-benchmark")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="only first N files (for testing)")
    a = p.parse_args()

    svc = BlobServiceClient(f"https://{a.account}.blob.core.windows.net",
                            credential=AzureCliCredential())
    cc = svc.get_container_client(a.container)

    files = list_files(a.repo, a.repo_type)
    files = [f for f in files if not f.startswith(".")]
    if a.limit:
        files = files[:a.limit]
    print(f"{a.repo}: {len(files)} files -> {a.container}/{a.dest}/", flush=True)

    started, skipped, failed = [], 0, []

    def kick(rel):
        blob = cc.get_blob_client(f"{a.dest}/{rel}")
        try:
            if blob.exists():
                return ("skip", rel)
            blob.start_copy_from_url(resolve(source_url(a.repo, a.repo_type, rel)))
            return ("started", rel)
        except Exception as e:
            return ("fail", f"{rel}: {type(e).__name__}: {str(e)[:160]}")

    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for kind, val in ex.map(kick, files):
            if kind == "started":
                started.append(val)
            elif kind == "skip":
                skipped += 1
            else:
                failed.append(val)

    print(f"started {len(started)}, already present {skipped}, failed {len(failed)}", flush=True)
    for f in failed[:10]:
        print("  FAIL", f, flush=True)

    # Azure copies asynchronously; wait for them to settle.
    pending = list(started)
    while pending:
        time.sleep(15)
        still = []
        for rel in pending:
            props = cc.get_blob_client(f"{a.dest}/{rel}").get_blob_properties()
            st = props.copy.status
            if st == "pending":
                still.append(rel)
            elif st not in ("success", None):
                failed.append(f"{rel}: copy {st} {props.copy.status_description}")
        done = len(started) - len(still)
        print(f"  [{time.strftime('%H:%M:%S')}] {done}/{len(started)} copies complete", flush=True)
        pending = still

    total = sum(b.size for b in cc.list_blobs(name_starts_with=f"{a.dest}/"))
    print(f"done. {a.container}/{a.dest}/ now holds {total/1e9:.2f} GB", flush=True)
    if failed:
        print(f"{len(failed)} FAILED", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
