#!/usr/bin/env python3
"""Copy a Hugging Face repo into Azure Blob **server-side** -- no bytes touch this machine.

Why this exists: the runbook's normal path is HF -> VM -> Blob, which is right once
a VM exists (Azure ingress is free, VM->blob in-region is free). Before quota is
approved there is no VM, and routing hundreds of GB through a laptop is both slow
and, for the 462 GB dataset, physically impossible on most disks.

Azure Blob's "copy from URL" makes Azure's own servers fetch the source. Two
wrinkles, both learned the hard way:

1. Azure refuses HTTP redirects, and every HF `resolve/` URL is a 307 to their CDN.
   So we resolve the redirect ourselves and hand Azure the final CDN URL.
2. Those resolved CDN URLs are **signed and short-lived**. Starting hundreds of
   copies at once means Azure is still working through its queue when the later
   signatures expire, and those copies die with
   `403 Forbidden "Copy failed when reading the source."` So we work in batches:
   resolve a batch's URLs, start them, wait for that batch to finish, then move on.
   URLs are never more than one batch old.

A failed copy leaves a **0-byte blob behind**, so presence alone is not proof of a
good copy -- we re-copy anything whose size is 0 or whose copy status is not
"success". Otherwise a retry silently skips exactly the shards that failed.

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
    p.add_argument("--batch", type=int, default=40,
                   help="copies in flight at once; keeps signed source URLs fresh")
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

    # A 0-byte blob, or one whose copy status is not "success", is a failed copy
    # masquerading as a present one. Re-copy those.
    good = set()
    for b in cc.list_blobs(name_starts_with=f"{a.dest}/", include=["copy"]):
        st = b.copy.status
        if b.size and st in ("success", None):
            good.add(b.name[len(a.dest) + 1:])
    todo = [f for f in files if f not in good]
    print(f"already good: {len(good)}   to copy: {len(todo)}", flush=True)

    failed, copied = [], 0

    def kick(rel):
        """Resolve the signed URL and start the copy, as close together as possible."""
        try:
            cc.get_blob_client(f"{a.dest}/{rel}").start_copy_from_url(
                resolve(source_url(a.repo, a.repo_type, rel)))
            return ("started", rel)
        except Exception as e:
            return ("fail", f"{rel}: {type(e).__name__}: {str(e)[:160]}")

    for bi in range(0, len(todo), a.batch):
        batch = todo[bi:bi + a.batch]
        started = []
        with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
            for kind, val in ex.map(kick, batch):
                (started if kind == "started" else failed).append(val)

        # Wait for THIS batch before resolving the next one, so no signed URL
        # sits unused long enough to expire.
        pending = list(started)
        while pending:
            time.sleep(10)
            still = []
            for rel in pending:
                props = cc.get_blob_client(f"{a.dest}/{rel}").get_blob_properties()
                st = props.copy.status
                if st == "pending":
                    still.append(rel)
                elif st == "success":
                    copied += 1
                else:
                    failed.append(f"{rel}: copy {st}: {props.copy.status_description}")
            pending = still
        done = min(bi + a.batch, len(todo))
        print(f"  [{time.strftime('%H:%M:%S')}] batch {bi//a.batch + 1}: "
              f"{done}/{len(todo)} attempted, {copied} copied, {len(failed)} failed",
              flush=True)

    for f in failed[:15]:
        print("  FAIL", f, flush=True)

    total = sum(b.size for b in cc.list_blobs(name_starts_with=f"{a.dest}/"))
    print(f"done. {a.container}/{a.dest}/ now holds {total/1e9:.2f} GB", flush=True)
    if failed:
        print(f"{len(failed)} FAILED", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
