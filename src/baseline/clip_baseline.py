#!/usr/bin/env python3
"""Zero-shot CLIP and frozen-CLIP linear-probe baselines for barrier classification.

Classes: missing_curb_ramp, obstacle, surface_problem (validator-confirmed crops) and no_barrier
(validator-rejected crops of those three types). Trained on the 18 non-Pittsburgh cities; tested on the
held-out Pittsburgh crops. CPU only.

usage: python src/baseline/clip_baseline.py --out /data/eda/baseline
"""
import argparse, glob, json, os, time
import numpy as np, pandas as pd, torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import CLIPModel, CLIPProcessor

D = "/data/datasets"
DIRS = [("nocurbramp", "missing_curb_ramp"), ("obstacle", "obstacle"), ("surfaceproblem", "surface_problem")]
CLASSES = ["missing_curb_ramp", "obstacle", "surface_problem", "no_barrier"]
PROMPTS = {
    "missing_curb_ramp": ["a street photo of a sidewalk corner where the curb has no ramp for wheelchairs",
                          "a raised curb at a street crossing with no curb ramp"],
    "obstacle": ["a street photo of a pole, sign, trash can or parked car blocking the sidewalk",
                 "an object obstructing a sidewalk"],
    "surface_problem": ["a street photo of a cracked, broken or uneven sidewalk",
                        "a damaged sidewalk surface with cracks, bumps or grass"],
    "no_barrier": ["a street photo of a clear sidewalk with no problems",
                   "a sidewalk corner with a curb ramp and nothing blocking the path"],
}


class Crops(torch.utils.data.Dataset):
    def __init__(self, paths, proc):
        self.paths, self.proc = paths, proc
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        return self.proc(images=Image.open(self.paths[i]).convert("RGB"), return_tensors="pt")["pixel_values"][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/eda/baseline")
    ap.add_argument("--model", default="openai/clip-vit-base-patch16")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(48)
    rows = []
    for d, cls in DIRS:
        for p in sorted(glob.glob(f"{D}/sidewalk-validator-ai-dataset-{d}/*/*/*.webp")):
            split, verdict, fn = p.split("/")[-3:]
            city, lid = fn[:-5].rsplit("_", 1)
            rows.append(dict(path=p, dir=d, split=split, verdict=verdict, city=city, label_id=int(lid),
                             y=cls if verdict == "correct" else "no_barrier"))
    M = pd.DataFrame(rows)
    print(len(M), "crops", M.y.value_counts().to_dict(), flush=True)

    model = CLIPModel.from_pretrained(a.model).eval()
    proc = CLIPProcessor.from_pretrained(a.model)
    feat_path = f"{a.out}/clip_features.npy"
    if os.path.exists(feat_path):
        F = np.load(feat_path)
    else:
        dl = torch.utils.data.DataLoader(Crops(M.path.tolist(), proc.image_processor), batch_size=96, num_workers=16)
        feats, t0 = [], time.time()
        with torch.inference_mode():
            for i, x in enumerate(dl):
                f = model.get_image_features(pixel_values=x)
                f = getattr(f, "pooler_output", f)  # transformers 5 returns an output object
                feats.append(torch.nn.functional.normalize(f, dim=-1).numpy())
                if i % 20 == 0:
                    print(f"  batch {i}/{len(dl)}  {(i + 1) * 96 / (time.time() - t0):.0f} img/s", flush=True)
        F = np.concatenate(feats).astype(np.float32)
        print("features", F.shape, flush=True)
        np.save(feat_path, F)
    M.drop(columns="path").to_csv(f"{a.out}/crops_meta.csv", index=False)

    pgh = (M.city == "pittsburgh").to_numpy()
    tr = (~pgh) & (M.split == "train").to_numpy()
    va = (~pgh) & (M.split == "val").to_numpy()
    y = M.y.to_numpy()
    res = dict(model=a.model, n_train=int(tr.sum()), n_val=int(va.sum()), n_test_pittsburgh=int(pgh.sum()),
               test_class_counts=M[pgh].y.value_counts().to_dict())

    # zero-shot
    with torch.inference_mode():
        tok = proc.tokenizer([p for c in CLASSES for p in PROMPTS[c]], padding=True, return_tensors="pt")
        T = model.get_text_features(**tok)
        T = torch.nn.functional.normalize(getattr(T, "pooler_output", T), dim=-1).numpy()
    T = T.reshape(len(CLASSES), -1, T.shape[-1]).mean(1)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    zs = np.array(CLASSES)[np.argmax(F @ T.T, axis=1)]

    # linear probe: pick C on non-Pittsburgh val, report on held-out Pittsburgh
    best = None
    for C in [0.1, 0.3, 1, 3, 10]:
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(F[tr], y[tr])
        f1 = f1_score(y[va], clf.predict(F[va]), average="macro")
        print(f"  C={C}: val macro-F1 {f1:.3f}", flush=True)
        if best is None or f1 > best[0]:
            best = (f1, C, clf)
    res["probe_C"], res["probe_val_macro_f1"] = best[1], best[0]
    lp = best[2].predict(F)
    proba = best[2].predict_proba(F)

    for name, pred in [("zero_shot", zs), ("linear_probe", lp)]:
        rep = classification_report(y[pgh], pred[pgh], labels=CLASSES, output_dict=True, zero_division=0)
        is_bar, pred_bar = y[pgh] != "no_barrier", pred[pgh] != "no_barrier"
        tp = int((is_bar & pred_bar).sum())
        res[name] = dict(
            pittsburgh_per_class={c: {k: round(rep[c][k], 3) for k in ["precision", "recall", "f1-score", "support"]} for c in CLASSES},
            pittsburgh_accuracy=round(rep["accuracy"], 3), pittsburgh_macro_f1=round(rep["macro avg"]["f1-score"], 3),
            pittsburgh_barrier_vs_none=dict(precision=round(tp / max(pred_bar.sum(), 1), 3), recall=round(tp / max(is_bar.sum(), 1), 3)),
            confusion=dict(labels=CLASSES, matrix=confusion_matrix(y[pgh], pred[pgh], labels=CLASSES).tolist()))
        other_test = (~pgh) & (M.split == "test").to_numpy()
        res[name]["other_cities_test_macro_f1"] = round(f1_score(y[other_test], pred[other_test], average="macro"), 3)
    out = M[pgh].drop(columns="path").assign(zero_shot=zs[pgh], linear_probe=lp[pgh])
    for i, c in enumerate(best[2].classes_):
        out[f"p_{c}"] = proba[pgh, i].round(4)
    out.to_csv(f"{a.out}/pittsburgh_predictions.csv", index=False)
    json.dump(res, open(f"{a.out}/results.json", "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
