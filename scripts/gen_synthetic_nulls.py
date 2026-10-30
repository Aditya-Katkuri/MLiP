#!/usr/bin/env python3
"""Make synthetic null images by removing the labelled object from Obstacle frames.

Runs the reviewed pilot pipeline (notebooks/lama_obstacle_pilot.ipynb) unattended: Grounding DINO finds the tagged
objects, the click point checks one is the labelled obstacle, SAM outlines them, LaMa fills them, and SDXL redraws the
padded region to hide LaMa's blur and nearby shadows. No hand-marked shadow points. Settings match the pilot.

Candidates are Obstacle PNG frames from the training splits only, Pittsburgh first, then other cities in a seeded
shuffle. Frames tagged with something removal can't fix are skipped. Frames are read from the public Validated zip,
and `--meta-dir` only supplies the split CSVs, so any sidewalk-data-2 or processed copy works.

Writes <out>/nulls/<stem>_null.png, <out>/masks/<stem>_mask.png and <out>/log.csv. A rerun skips everything already
in log.csv, so a lost runtime only costs the images since the last --hf-repo upload. HF_TOKEN must be set to upload.

setup:  pip uninstall -q -y torchao
        pip install -q diffusers==0.41.0 && pip install -q --no-deps simple-lama-inpainting
usage:  python scripts/gen_synthetic_nulls.py --meta-dir DIR --out DIR [--hf-repo USER/REPO] [--max-minutes M]
        python scripts/gen_synthetic_nulls.py --meta-dir DIR --dry-run
"""
import argparse, io, sys, time, zipfile
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
LABEL_MAX_DIST_FRAC = 0.10  # a box counts as the labeled obstacle if the click is inside or this close (frac of max(w, h))
MAX_MASK_FRAC = 0.40        # skip images where the mask covers more than this
MASK_DILATE_PX = 15         # grow the SAM mask so LaMa doesn't copy the object's edge

DIFFUSION_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
DIFFUSION_PROMPT = ("street-level photo, empty, uncracked pavement with nothing on it, "
                    "seamless continuation of the surrounding scene")
DIFFUSION_NEGATIVE = ("object, box, bag, trash can, debris, car, bicycle, pole, street light, lamp post, sign, "
                      "cone, barrier, fence, construction, person, shadow, text, blurry, smeared")
DIFFUSION_STRENGTH = 0.7    # <1 keeps LaMa's layout and redraws texture on top of it
DIFFUSION_STEPS = 30
DIFFUSION_GUIDANCE = 6.0
DIFFUSION_PAD_PX = 40       # SDXL redraws the LaMa mask grown by this much, to cover shadows and LaMa's edges
DIFFUSION_CROP_PAD = 64     # context px around the region; the crop is inpainted at 1024x1024 and pasted back
FEATHER_PX = 8              # blend weight ramps from 0 to 1 over this many px inside the region


def candidates(meta_dir, allow_narrow, seed):
    """Eligible frames, Pittsburgh first so a time limit still covers the target domain."""
    blocked = [t for t in NON_REMOVABLE_TAGS if not (allow_narrow and t == "narrow")]
    parts = []
    for split in SPLITS:
        df = pd.read_csv(Path(meta_dir) / f"{split}.csv", low_memory=False)
        df = df[df["class"].eq("Obstacle") & df.filename.str.endswith(".png") & df.normalized_x.notna()]
        tags = df[list(REMOVABLE_TAGS) + NON_REMOVABLE_TAGS].fillna(0).astype(bool)
        df = df[tags[list(REMOVABLE_TAGS)].any(axis=1) & ~tags[blocked].any(axis=1)].copy()
        df["tags"] = [[t for t in REMOVABLE_TAGS if tags.at[i, t]] for i in df.index]
        parts.append(df.sample(frac=1, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def disk(r):
    import cv2
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2)


class Pipeline:
    def __init__(self):
        sys.modules["torchao"] = None  # diffusers needs a newer torchao than Colab's; nothing here uses it
        import torch
        from diffusers import AutoPipelineForInpainting
        from simple_lama_inpainting import SimpleLama
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel, SamProcessor

        self.torch, self.device = torch, "cuda"
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
        # Redraw the padded mask with SDXL, then blend it in only inside that region.
        import cv2
        region = cv2.dilate(mask, disk(DIFFUSION_PAD_PX))
        gen = self.torch.Generator(self.device).manual_seed(seed)
        out = self.inpaint(prompt=DIFFUSION_PROMPT, negative_prompt=DIFFUSION_NEGATIVE, image=img,
                           mask_image=Image.fromarray(region), strength=DIFFUSION_STRENGTH,
                           num_inference_steps=DIFFUSION_STEPS, guidance_scale=DIFFUSION_GUIDANCE,
                           padding_mask_crop=DIFFUSION_CROP_PAD, generator=gen).images[0]
        if out.size != img.size:
            out = out.resize(img.size, Image.LANCZOS)
        # Erode by FEATHER_PX, then blur with a kernel of the same radius (3 sigma), so the weight is 0 outside.
        alpha = cv2.GaussianBlur(cv2.erode(region, disk(FEATHER_PX)), (0, 0), FEATHER_PX / 3)[..., None] / 255
        blended = np.array(img) * (1 - alpha) + np.array(out) * alpha
        return Image.fromarray(blended.round().astype(np.uint8))

    def remove(self, img, tags, point, seed):
        """Returns (status, null, mask, record)."""
        import cv2
        w, h = img.size
        text = ". ".join(p for t in tags for p in REMOVABLE_TAGS[t]) + "."
        boxes = self.detect(img, text)
        dists = [np.hypot(max(b[0] - point[0], 0, point[0] - b[2]), max(b[1] - point[1], 0, point[1] - b[3]))
                 for b in boxes]
        if not len(boxes) or min(dists) / max(w, h) > LABEL_MAX_DIST_FRAC:
            return "no_detection_at_label", None, None, dict(n_boxes=len(boxes))
        obj = self.segment(img, boxes).any(axis=0)
        mask = cv2.dilate(obj.astype(np.uint8) * 255, disk(MASK_DILATE_PX))
        rec = dict(n_boxes=len(boxes), mask_frac=float(mask.mean() / 255))
        if rec["mask_frac"] > MAX_MASK_FRAC:
            return "mask_too_large", None, mask, rec
        lama = self.lama(img, Image.fromarray(mask))
        if lama.size != img.size:  # LaMa may pad to a multiple of 8
            lama = lama.crop((0, 0, w, h))
        return "ok", self.refine(lama, mask, seed), mask, rec


def upload(repo, out):
    from huggingface_hub import HfApi
    try:
        HfApi().upload_folder(repo_id=repo, folder_path=out, repo_type="dataset", path_in_repo="synthetic_nulls",
                              commit_message="Add synthetic nulls")
    except Exception as e:  # a failed upload must not stop generation; the next one retries
        print(f"upload failed: {e}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--hf-repo", help="private dataset repo to back results up to")
    ap.add_argument("--upload-every", type=int, default=25)
    ap.add_argument("--max-minutes", type=float, default=float("inf"))
    ap.add_argument("--limit", type=int, help="stop after this many frames, e.g. 20 to check speed and quality")
    ap.add_argument("--allow-narrow", action="store_true", help="also use frames tagged narrow")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="print the candidate counts and exit")
    a = ap.parse_args()

    todo = candidates(a.meta_dir, a.allow_narrow, a.seed)
    print(todo.groupby("split").size().to_string())
    if a.dry_run:
        return
    (a.out / "nulls").mkdir(parents=True, exist_ok=True)
    (a.out / "masks").mkdir(exist_ok=True)
    log_path = a.out / "log.csv"
    if log_path.exists():
        todo = todo[~todo.filename.isin(pd.read_csv(log_path).filename)]
    if a.limit:
        todo = todo.head(a.limit)
    print(f"{len(todo)} frames to process", flush=True)

    from huggingface_hub import HfApi, HfFileSystem
    if a.hf_repo:
        HfApi().create_repo(a.hf_repo, repo_type="dataset", private=True, exist_ok=True)
    pipe = Pipeline()
    t0, done, ok = time.perf_counter(), 0, 0
    with HfFileSystem().open(ZIP_PATH, "rb", block_size=4 * 2**20) as f, zipfile.ZipFile(f) as zf:
        for i, row in enumerate(todo.itertuples()):
            if (time.perf_counter() - t0) / 60 > a.max_minutes:
                print("time limit reached", flush=True)
                break
            img = Image.open(io.BytesIO(zf.read(f"Obstacle/crops/{row.filename}"))).convert("RGB")
            point = (row.normalized_x * img.width, row.normalized_y * img.height)
            t1 = time.perf_counter()
            status, null, mask, rec = pipe.remove(img, row.tags, point, a.seed + i)
            stem = Path(row.filename).stem
            if status == "ok":
                null.save(a.out / "nulls" / f"{stem}_null.png")
                Image.fromarray(mask).save(a.out / "masks" / f"{stem}_mask.png")
                ok += 1
            pd.DataFrame([dict(filename=row.filename, split=row.split, tags=",".join(row.tags), status=status,
                               sec=time.perf_counter() - t1, **rec)]).to_csv(
                log_path, mode="a", header=not log_path.exists(), index=False)
            done += 1
            if done % 10 == 0 or done == len(todo):
                print(f"{done}/{len(todo)}, {ok} ok, {(time.perf_counter() - t0) / done:.1f} s/frame", flush=True)
            if a.hf_repo and done % a.upload_every == 0:
                upload(a.hf_repo, a.out)
    if a.hf_repo:
        upload(a.hf_repo, a.out)
    print(f"{done} frames, {ok} nulls in {(time.perf_counter() - t0) / 60:.1f} min -> {a.out}")


if __name__ == "__main__":
    main()
