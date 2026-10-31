#!/usr/bin/env python3
"""Make synthetic null images by removing the labelled object from Obstacle frames.

Runs the pilot pipeline (notebooks/lama_obstacle_pilot.ipynb) unattended: Grounding DINO finds the tagged objects, only
those at the click point are kept, SAM outlines them, LaMa fills the exact outline, and SDXL redraws a padded rectangle
around it, so the mask's shape doesn't guide what SDXL paints. A frame is rejected if the mask misses the click, or if
Grounding DINO finds an object inside the mask in the output. No hand-marked shadow points.

Candidates are Obstacle PNG frames from the training splits only, Pittsburgh first, then other cities in a seeded
shuffle. Frames tagged with something removal can't fix are skipped. Frames are read from the public Validated zip,
and `--meta-dir` only supplies the split CSVs, so any sidewalk-data-2 or processed copy works.

--strength and --guidance each take several values, and every pair runs on the same frames and loaded models. Each pair
writes to its own folder, <out>_s<strength>_g<guidance>, referred to as <run> below.

Writes under <run>: nulls/<stem>_null.png (accepted), rejected/<stem>_null.png (failed the output check),
lama/<stem>_lama.png, masks/<stem>_mask.png and log.csv. The log's lama_objects and null_objects count objects found
inside the mask after each stage, to show which stage a failure starts at (lama_objects is blank with --skip-lama).
A rerun skips everything already in log.csv, and a lost runtime only costs the images since the last --hf-repo upload,
which goes to a folder named after <run>. HF_TOKEN must be set to upload.

setup:  pip uninstall -q -y torchao
        pip install -q diffusers==0.41.0 && pip install -q --no-deps simple-lama-inpainting
usage:  python scripts/gen_synthetic_nulls.py --meta-dir DIR --out DIR [--hf-repo USER/REPO] [--max-minutes M]
        python scripts/gen_synthetic_nulls.py --meta-dir DIR --dry-run
"""
import argparse, io, sys, time, zipfile
from itertools import product
from pathlib import Path

import numpy as np, pandas as pd
from PIL import Image

SPLITS = ["pitt-train", "other-cities-train"]  # never pitt-val or pitt-test
ZIP_PATH = ("datasets/projectsidewalk/sidewalk-tagger-ai-validated@"
            "6e3a116a3c228dd35bcd72f6e5fb921f6ebb6a50/Validated/Obstacle.zip")

# Obstacle tag -> Grounding DINO phrases (lowercase; joined with periods).
REMOVABLE_TAGS = {
    "construction": ["construction barrier", "traffic cone"],
    "fire-hydrant": ["fire hydrant"],
    "litter-garbage": ["garbage bag", "litter"],
    "outdoor-dining-area": ["table", "chair"],
    "parked-bike": ["bicycle"],
    "parked-car": ["car", "truck", "van"],
    "parked-scooter-motorcycle": ["scooter", "motorcycle"],
    "pole": ["pole"],
    "sign": ["sign"],
    "street-vendor": ["cart"],
    "trash-recycling-can": ["trash can", "recycling bin"],
    "tree": ["tree trunk"],
    "mailbox": ["mailbox"],
    "seating": ["bench"],
}
# These describe the sidewalk itself, so removing objects can't make the image obstacle-free.
NON_REMOVABLE_TAGS = ["narrow", "vegetation", "height-difference", "stairs", "garage-entrance", "uneven-slanted"]

BOX_THRESHOLD = 0.30        # detection confidence
TEXT_THRESHOLD = 0.25       # phrase-match confidence
LABEL_MAX_DIST_FRAC = 0.02  # a box counts as the labeled obstacle if the click is inside or this close (frac of max(w, h))
CLICK_TOLERANCE_PX = 30     # the mask must reach within this distance of the click, or the labeled obstacle was missed
OBJECT_IN_MASK_FRAC = 0.5   # a box found in an output counts as inside the mask if this much of it is
MAX_MASK_FRAC = 0.40        # skip images where the mask covers more than this
MASK_DILATE_PX = 15         # grow the SAM mask so LaMa doesn't copy the object's edge

DIFFUSION_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
DIFFUSION_PROMPT = ("street-level photo, empty, uncracked pavement with nothing on it, "
                    "seamless continuation of the surrounding scene")
DIFFUSION_NEGATIVE = ("object, box, bag, trash can, debris, car, bicycle, pole, street light, lamp post, sign, "
                      "cone, barrier, fence, construction, person, shadow, text, blurry, smeared")
DIFFUSION_STRENGTH = 0.7    # <1 keeps LaMa's layout and redraws texture on top of it; lower invents less
DIFFUSION_STEPS = 30
DIFFUSION_GUIDANCE = 6.0
DIFFUSION_PAD_PX = 40       # SDXL redraws the mask's bounding box grown by this much, to cover shadows and LaMa's edges
DIFFUSION_CROP_PAD = 64     # context px around the region; the crop is inpainted at 1024x1024 and pasted back
FEATHER_PX = 8              # blend weight ramps from 0 to 1 over this many px inside the region

CHECK_PROMPT = ". ".join(p for phrases in REMOVABLE_TAGS.values() for p in phrases) + "."
LOG_FIELDS = ["filename", "split", "tags", "status", "sec", "n_boxes", "n_kept", "mask_frac", "rect_frac",
              "lama_objects", "null_objects"]


def candidates(meta_dir, seed, only_tags=(), exclude_tags=()):
    parts = []
    for split in SPLITS:
        df = pd.read_csv(Path(meta_dir) / f"{split}.csv", low_memory=False)
        df = df[df["class"].eq("Obstacle") & df.filename.str.endswith(".png") & df.normalized_x.notna()]
        tags = df[list(REMOVABLE_TAGS) + NON_REMOVABLE_TAGS].fillna(0).astype(bool)
        keep = tags[list(REMOVABLE_TAGS)].any(axis=1) & ~tags[NON_REMOVABLE_TAGS + list(exclude_tags)].any(axis=1)
        if only_tags:
            keep &= tags[list(only_tags)].any(axis=1)
        df = df[keep].copy()
        df["tags"] = [[t for t in REMOVABLE_TAGS if tags.at[i, t]] for i in df.index]
        parts.append(df.sample(frac=1, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def disk(r):
    import cv2
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)


def bounding_rect(mask, pad, pad_down=0):
    """Filled rectangle around a uint8 mask, grown by pad, plus pad_down at the bottom, and clipped to the image."""
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    rect = np.zeros_like(mask)
    rect[max(ys.min() - pad, 0):min(ys.max() + pad + pad_down + 1, h), max(xs.min() - pad, 0):min(xs.max() + pad + 1, w)] = 255
    return rect


class Pipeline:
    def __init__(self, max_mask_frac=MAX_MASK_FRAC, skip_lama=False, pad_down=0):
        sys.modules["torchao"] = None  # diffusers needs a newer torchao than Colab's; nothing here uses it
        import torch
        from diffusers import AutoPipelineForInpainting
        from simple_lama_inpainting import SimpleLama
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor

        self.torch, self.device = torch, "cuda"
        self.strength, self.guidance = DIFFUSION_STRENGTH, DIFFUSION_GUIDANCE
        self.max_mask_frac, self.skip_lama, self.pad_down = max_mask_frac, skip_lama, pad_down
        self.processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny")
        self.dino = self.dino.to(self.device).eval()
        self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        self.sam = SamModel.from_pretrained("facebook/sam-vit-base").to(self.device).eval()
        self.lama = SimpleLama()
        self.inpaint = AutoPipelineForInpainting.from_pretrained(DIFFUSION_MODEL, variant="fp16",
                                                                 torch_dtype=torch.float16).to(self.device)
        self.inpaint.set_progress_bar_config(disable=True)

    def detect(self, img, text):
        inputs = self.processor(images=img, text=text, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            out = self.dino(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD,
            target_sizes=[img.size[::-1]])[0]
        return res["boxes"].cpu().numpy()

    def segment(self, img, boxes):
        # One SAM mask per box, (n, h, w) bool.
        inputs = self.sam_processor(img, input_boxes=[[[float(v) for v in b] for b in boxes]],
                                    return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            out = self.sam(**inputs, multimask_output=False)
        masks = self.sam_processor.image_processor.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0]
        return masks[:, 0].numpy()

    def refine(self, img, mask, seed):
        # Redraw the padded rectangle with SDXL, then blend it in only inside that region.
        import cv2
        region = bounding_rect(mask, DIFFUSION_PAD_PX, self.pad_down)
        gen = self.torch.Generator(self.device).manual_seed(seed)
        out = self.inpaint(prompt=DIFFUSION_PROMPT, negative_prompt=DIFFUSION_NEGATIVE, image=img,
                           mask_image=Image.fromarray(region), strength=self.strength,
                           num_inference_steps=DIFFUSION_STEPS, guidance_scale=self.guidance,
                           padding_mask_crop=DIFFUSION_CROP_PAD, generator=gen).images[0]
        if out.size != img.size:
            out = out.resize(img.size, Image.LANCZOS)
        # Erode by FEATHER_PX, then blur with a kernel of the same radius (3 sigma), so the weight is 0 outside.
        alpha = cv2.GaussianBlur(cv2.erode(region, disk(FEATHER_PX)), (0, 0), FEATHER_PX / 3)[..., None] / 255
        blended = np.array(img) * (1 - alpha) + np.array(out) * alpha
        return Image.fromarray(blended.round().astype(np.uint8))

    def objects_in_mask(self, img, mask):
        """How many boxes Grounding DINO finds that sit mostly inside the mask."""
        n = 0
        for x0, y0, x1, y1 in self.detect(img, CHECK_PROMPT).round().astype(int):
            crop = mask[max(y0, 0):y1, max(x0, 0):x1]
            n += crop.size > 0 and (crop > 0).mean() >= OBJECT_IN_MASK_FRAC
        return n

    def remove(self, img, tags, point, seed):
        """Returns (status, stages, record). stages holds the mask, lama and null reached before stopping."""
        import cv2
        w, h = img.size
        text = ". ".join(p for t in tags for p in REMOVABLE_TAGS[t]) + "."
        boxes = self.detect(img, text)
        dists = np.array([np.hypot(max(b[0] - point[0], 0, point[0] - b[2]), max(b[1] - point[1], 0, point[1] - b[3]))
                          for b in boxes])
        keep = dists / max(w, h) <= LABEL_MAX_DIST_FRAC if len(boxes) else np.zeros(0, bool)
        rec = dict(n_boxes=len(boxes), n_kept=int(keep.sum()))
        if not keep.any():
            return "no_detection_at_label", {}, rec
        obj = self.segment(img, boxes[keep]).any(axis=0)
        mask = cv2.dilate(obj.astype(np.uint8) * 255, disk(MASK_DILATE_PX))
        stages = dict(mask=mask)
        rec.update(mask_frac=float((mask > 0).mean()),
                   rect_frac=float((bounding_rect(mask, DIFFUSION_PAD_PX, self.pad_down) > 0).mean()))
        x, y = min(int(point[0]), w - 1), min(int(point[1]), h - 1)
        if not cv2.dilate(mask, disk(CLICK_TOLERANCE_PX))[y, x]:
            return "click_outside_mask", stages, rec
        if rec["mask_frac"] > self.max_mask_frac:
            return "mask_too_large", stages, rec
        if self.skip_lama:
            stages["null"] = self.refine(img, mask, seed)
        else:
            lama = self.lama(img, Image.fromarray(mask))
            if lama.size != img.size:  # LaMa may pad to a multiple of 8
                lama = lama.crop((0, 0, w, h))
            stages["lama"] = lama
            rec["lama_objects"] = int(self.objects_in_mask(lama, mask))
            stages["null"] = self.refine(lama, mask, seed)
        rec["null_objects"] = int(self.objects_in_mask(stages["null"], mask))
        return ("ok" if rec["null_objects"] == 0 else "object_in_output"), stages, rec


def upload(repo, out):
    from huggingface_hub import HfApi
    try:
        HfApi().upload_folder(repo_id=repo, folder_path=out, repo_type="dataset", path_in_repo=out.name,
                              commit_message=f"Add synthetic nulls: {out.name}")
    except Exception as e:  # a failed upload must not stop generation; the next one retries
        print(f"upload failed: {e}", flush=True)


def run(pipe, todo, zf, out, a):
    """Process every frame in todo not yet in out/log.csv with the pipeline's current settings."""
    for name in ["nulls", "rejected", "lama", "masks"]:
        (out / name).mkdir(parents=True, exist_ok=True)
    log_path = out / "log.csv"
    if log_path.exists():
        todo = todo[~todo.filename.isin(pd.read_csv(log_path).filename)]
    print(f"{out.name}: {len(todo)} frames to process", flush=True)
    t0, done, ok = time.perf_counter(), 0, 0
    for i, row in enumerate(todo.itertuples()):
        if (time.perf_counter() - t0) / 60 > a.max_minutes:
            print("time limit reached", flush=True)
            break
        img = Image.open(io.BytesIO(zf.read(f"Obstacle/crops/{row.filename}"))).convert("RGB")
        point = (row.normalized_x * img.width, row.normalized_y * img.height)
        t1 = time.perf_counter()
        status, stages, rec = pipe.remove(img, row.tags, point, a.seed + i)
        stem = Path(row.filename).stem
        if "mask" in stages:
            Image.fromarray(stages["mask"]).save(out / "masks" / f"{stem}_mask.png")
        if "lama" in stages:
            stages["lama"].save(out / "lama" / f"{stem}_lama.png")
        if "null" in stages:
            stages["null"].save(out / ("nulls" if status == "ok" else "rejected") / f"{stem}_null.png")
        ok += status == "ok"
        pd.DataFrame([dict(filename=row.filename, split=row.split, tags=",".join(row.tags), status=status,
                           sec=time.perf_counter() - t1, **rec)], columns=LOG_FIELDS).to_csv(
            log_path, mode="a", header=not log_path.exists(), index=False)
        done += 1
        if done % 10 == 0 or done == len(todo):
            print(f"{done}/{len(todo)}, {ok} ok, {(time.perf_counter() - t0) / done:.1f} s/frame", flush=True)
        if a.hf_repo and done % a.upload_every == 0:
            upload(a.hf_repo, out)
    if a.hf_repo:
        upload(a.hf_repo, out)
    print(f"{done} frames, {ok} nulls in {(time.perf_counter() - t0) / 60:.1f} min -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--hf-repo", help="private dataset repo to back results up to")
    ap.add_argument("--upload-every", type=int, default=25)
    ap.add_argument("--max-minutes", type=float, default=float("inf"))
    ap.add_argument("--limit", type=int, help="stop after this many frames, e.g. 20 to check speed and quality")
    ap.add_argument("--strength", type=float, nargs="+", default=[DIFFUSION_STRENGTH], help="SDXL strengths to sweep")
    ap.add_argument("--guidance", type=float, nargs="+", default=[DIFFUSION_GUIDANCE],
                    help="SDXL guidance scales to sweep")
    ap.add_argument("--max-mask-frac", type=float, default=MAX_MASK_FRAC, help="reject larger masks")
    ap.add_argument("--only-tags", nargs="*", default=[], help="keep only frames with one of these tags")
    ap.add_argument("--exclude-tags", nargs="*", default=[], help="drop frames with any of these tags")
    ap.add_argument("--skip-lama", action="store_true", help="inpaint the original directly with SDXL")
    ap.add_argument("--pad-down", type=int, default=0, help="extend the SDXL rectangle down by this many px")
    ap.add_argument("--zip-file", type=Path, help="local copy of Obstacle.zip; faster than reading it remotely")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="print the candidate counts and exit")
    a = ap.parse_args()

    todo = candidates(a.meta_dir, a.seed, a.only_tags, a.exclude_tags)
    print(todo.groupby("split").size().to_string())
    if a.dry_run:
        return
    if a.limit:
        todo = todo.head(a.limit)

    from huggingface_hub import HfApi, HfFileSystem
    if a.hf_repo:
        HfApi().create_repo(a.hf_repo, repo_type="dataset", private=True, exist_ok=True)
    pipe = Pipeline(a.max_mask_frac, a.skip_lama, a.pad_down)
    source = open(a.zip_file, "rb") if a.zip_file else HfFileSystem().open(ZIP_PATH, "rb", block_size=4 * 2**20)
    with source as f, zipfile.ZipFile(f) as zf:
        for strength, guidance in product(a.strength, a.guidance):
            pipe.strength, pipe.guidance = strength, guidance
            out = a.out.with_name(f"{a.out.name}_s{strength}_g{guidance}")
            run(pipe, todo, zf, out, a)


if __name__ == "__main__":
    main()
