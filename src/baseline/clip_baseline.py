#!/usr/bin/env python3
"""Zero-shot CLIP and frozen-CLIP linear-probe baselines on sidewalk-data.

Known classes CurbRamp, Obstacle and SurfaceProblem appear in every split; no_obstacles is a test-only proxy
class (Pittsburgh validator-AI crops of crowd-rejected labels) with no training examples. Crosswalk is excluded.
Splits are by city: test = pittsburgh, val = columbus + amsterdam, train = every other city.

Input: a 640x640 window of the 1440x960 Street View frame centred on the label point (shifted to stay inside
the frame, no padding), as in the dataset paper. no_obstacles images already are label-centred crops, so the
whole image is used, letterboxed to a square. An ablation feeds the full frame letterboxed to a square.

Zero-shot scores all 4 labels from two prompts each. The linear probe is fit on the 3 known classes (C chosen
on val macro F1) and predicts no_obstacles when its max probability is below tau, the 5th percentile of its
max probability on val. Macro precision/recall/F1 and balanced accuracy average over the classes with support in
the split. CPU only.

Writes predictions.csv (every image, every split), results.json, feature caches and run.log to --out.
Cached features are reused per file path; only images missing from the cache are embedded.

usage: python src/baseline/clip_baseline.py [--out /data/eda/clip_sidewalk_data]
"""
import argparse, json, os, time
os.environ["HF_HOME"] = "/data/hf_cache"  # the model is cached here, not under the shell default
import numpy as np, pandas as pd, torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from transformers import CLIPModel, CLIPProcessor

D = "/data/datasets/sidewalk-data"
KNOWN = ["CurbRamp", "Obstacle", "SurfaceProblem"]
NOVEL = "no_obstacles"
CLASSES = KNOWN + [NOVEL]
EXCLUDED = ["Crosswalk"]
SPLIT_CITIES = {"test": ["pittsburgh"], "val": ["columbus", "amsterdam"]}  # train: every other city
SPLITS = ["test", "val", "train"]
EXPECTED = {"test": {"CurbRamp": 761, "Obstacle": 59, "SurfaceProblem": 609, NOVEL: 204},
            "val": {"CurbRamp": 1772, "Obstacle": 300, "SurfaceProblem": 1500},
            "train": {"CurbRamp": 8324, "Obstacle": 2073, "SurfaceProblem": 6983}}
N_TOTAL = 22585
VIEWS = ["crop", "full"]
W, H, S = 1440, 960, 640  # frame size, window side
GREY = (128, 128, 128)
C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]
TAU_PCT = 5
PROMPTS = {
    "CurbRamp": ["a street photo of a concrete curb ramp sloping from the sidewalk down to the street at a corner",
                 "a curb cut ramp with a tactile warning strip at a crosswalk"],
    "Obstacle": ["a street photo of a pole, sign, trash can or parked car blocking the sidewalk",
                 "an object obstructing a sidewalk"],
    "SurfaceProblem": ["a street photo of a cracked, broken or uneven sidewalk",
                       "a damaged sidewalk surface with cracks, bumps or grass"],
    NOVEL: ["a street photo of a clear, smooth sidewalk with nothing blocking it",
            "an undamaged sidewalk or street corner with a clear, usable path"],
}
INPUT = (f"{S}x{S} window of the {W}x{H} frame centred on the label point (normalized_x*W, normalized_y*H), "
         "shifted to stay inside the frame (no padding), RGB, resized to 224x224 by CLIPProcessor; "
         f"{NOVEL} images already are label-centred crops, so the whole image is letterboxed to a square with grey {GREY}")
FULL = (f"full {W}x{H} frame letterboxed to {W}x{W} with neutral grey {GREY} bars, resized to 224x224 by "
        f"CLIPProcessor (its centre-crop drops nothing); {NOVEL} images have no frame, so they use the same whole "
        "letterboxed image as the primary input; same prompts and probe procedure (C and tau chosen on val)")
NOTE = ("Pittsburgh crops of labels the crowd rejected, from the archived validator-AI datasets; a rejection does not "
        "verify the scene is free of obstacles or issues. Test only, with no training or validation examples.")


def window(nx, ny):
    """Pixel box (x0, y0, x1, y1) of the SxS window centred on the label point, kept inside the frame."""
    x0 = min(max(int(round(nx * W)) - S // 2, 0), W - S)
    y0 = min(max(int(round(ny * H)) - S // 2, 0), H - S)
    return x0, y0, x0 + S, y0 + S


def letterbox(im):
    s = max(im.size)
    sq = Image.new("RGB", (s, s), GREY)
    sq.paste(im, ((s - im.width) // 2, (s - im.height) // 2))
    return sq


class Frames(torch.utils.data.Dataset):
    """Each item stacks one 3x224x224 tensor per requested view; each image is decoded once.
    A box of None marks a no_obstacles crop: there is no frame, so every view is the whole letterboxed image."""
    def __init__(self, paths, boxes, views, proc):
        self.paths, self.boxes, self.views, self.proc = paths, boxes, views, proc
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        im, box = Image.open(self.paths[i]).convert("RGB"), self.boxes[i]
        if box is None:
            ims = [letterbox(im)] * len(self.views)
        else:
            assert im.size == (W, H), (self.paths[i], im.size)
            ims = [im.crop(box) if v == "crop" else letterbox(im) for v in self.views]
        return self.proc(images=ims, return_tensors="pt")["pixel_values"]


def load():
    M = pd.concat([pd.read_csv(f"{D}/{s}.csv") for s in ["train", "val", "test"]], ignore_index=True)
    M = M[~M["class"].isin(EXCLUDED)].copy()
    parts = M.filename.str.extract(r"^gsv-(.+)-(\d+)-([A-Za-z_]+)\.(?:png|webp)$")
    assert parts.notna().all().all() and (parts[2] == M["class"]).all()
    M["city"], M["label_id"] = parts[0], parts[1].astype(int)
    city_split = {c: s for s, cs in SPLIT_CITIES.items() for c in cs}
    csv_split, M["split"] = M.split, M.city.map(city_split).fillna("train")
    assert (csv_split == M.split).all() and (M.image_path.str.split("/").str[0] == M.split).all()
    assert set(M["class"]) == set(CLASSES) and (M[M["class"] == NOVEL].split == "test").all()
    M["split"] = pd.Categorical(M.split, SPLITS, ordered=True)
    M["class"] = pd.Categorical(M["class"], CLASSES, ordered=True)
    M = M.sort_values(["split", "class", "city", "label_id"]).reset_index(drop=True)
    M["split"], M["class"] = M.split.astype(str), M["class"].astype(str)
    for s, counts in EXPECTED.items():
        assert M[M.split == s]["class"].value_counts().to_dict() == counts, (s, M[M.split == s]["class"].value_counts())
    assert M.image_path.is_unique and len(M) == N_TOTAL
    novel = (M["class"] == NOVEL).to_numpy()
    assert M.loc[~novel, ["normalized_x", "normalized_y"]].notna().all().all()
    boxes = [None if nv else window(x, y) for nv, x, y in zip(novel, M.normalized_x, M.normalized_y)]
    B = np.array([(0, 0, 1, 1) if b is None else np.divide(b, [W, H, W, H]) for b in boxes], dtype=float)
    M["crop_x0"], M["crop_y0"], M["crop_x1"], M["crop_y1"] = B.T
    return M, boxes


def save_atomic(path, write):
    tmp = os.path.join(os.path.dirname(path), f".{os.path.basename(path)}.tmp")
    with open(tmp, "wb") as fh:
        write(fh)
    os.replace(tmp, path)


def features(M, boxes, model, proc, out, log):
    """L2-normalised image embeddings per view. The .npy rows follow the file list saved beside it; cached rows
    are reused by file path and only images missing from a view's cache are embedded."""
    files, dim = M.image_path.tolist(), model.config.projection_dim
    F, need = {}, {}
    for v in VIEWS:
        F[v], need[v] = np.zeros((len(files), dim), np.float32), np.ones(len(files), bool)
        npy, lst = f"{out}/features_{v}.npy", f"{out}/features_{v}_files.txt"
        if os.path.exists(npy) and os.path.exists(lst):
            head = open(lst).read().split("\n")
            X = np.load(npy)
            if head[:2] == [model.name_or_path, v] and len(X) == len(head) - 2:
                pos = {p: i for i, p in enumerate(head[2:])}
                idx = np.array([pos.get(p, -1) for p in files])
                need[v] = idx < 0
                F[v][~need[v]] = X[idx[~need[v]]]
        log(f"{v} cache: reusing {(~need[v]).sum()} rows, embedding {need[v].sum()}")
    rows = np.flatnonzero(np.any([need[v] for v in VIEWS], axis=0))
    todo = [v for v in VIEWS if need[v].any()]
    if len(rows):
        dl = torch.utils.data.DataLoader(Frames([f"{D}/{files[i]}" for i in rows], [boxes[i] for i in rows], todo,
                                                proc.image_processor), batch_size=64, num_workers=16)
        feats, t0, done = [], time.time(), 0
        log(f"embedding {len(rows)} images, views {todo}, {len(dl)} batches")
        with torch.inference_mode():
            for i, x in enumerate(dl):
                b, nv = x.shape[:2]
                f = model.get_image_features(pixel_values=x.flatten(0, 1))
                f = getattr(f, "pooler_output", f)  # transformers 5 returns an output object
                feats.append(torch.nn.functional.normalize(f, dim=-1).reshape(b, nv, -1).numpy())
                done += b
                if i % 20 == 0 or i == len(dl) - 1:
                    rate = done / (time.time() - t0)
                    log(f"  batch {i + 1}/{len(dl)}  {rate:.1f} img/s ({rate * nv:.1f} views/s)  "
                        f"eta {(len(rows) - done) / rate / 60:.1f} min")
        E = np.concatenate(feats).astype(np.float32)
        for j, v in enumerate(todo):
            m = need[v][rows]
            F[v][rows[m]] = E[m, j]
        log(f"features done in {(time.time() - t0) / 60:.1f} min")
    for v in VIEWS:
        save_atomic(f"{out}/features_{v}.npy", lambda fh: np.save(fh, F[v]))
        save_atomic(f"{out}/features_{v}_files.txt", lambda fh: fh.write("\n".join([model.name_or_path, v] + files).encode()))
    return F


def r3(v):
    return round(float(v), 3)


def metrics(y, p):
    """Confusion and per-class scores over all 4 labels (precision of a 0-support class still comes from its
    predictions); macro scores and balanced accuracy average over the labels with support in y."""
    P, R, F1, N = precision_recall_fscore_support(y, p, labels=CLASSES, zero_division=0)
    sup = N > 0
    return dict(accuracy=r3(accuracy_score(y, p)), balanced_accuracy=r3(R[sup].mean()),
                macro_precision=r3(P[sup].mean()), macro_recall=r3(R[sup].mean()), macro_f1=r3(F1[sup].mean()),
                weighted_f1=r3((F1 * N).sum() / N.sum()),
                per_class={c: dict(precision=r3(P[i]), recall=r3(R[i]), f1=r3(F1[i]), support=int(N[i]))
                           for i, c in enumerate(CLASSES)},
                confusion=dict(labels=CLASSES, matrix=confusion_matrix(y, p, labels=CLASSES).tolist()))


def test_metrics(y, p, novelty):
    """Test M plus the AUROC of the no_obstacles score against everything else."""
    m = metrics(y, p)
    m["no_obstacles_auroc"] = r3(roc_auc_score(y == NOVEL, novelty))
    return m


def zero_shot(F, T, scale):
    z = scale * F @ T.T
    z = np.exp(z - z.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def probe(F, y, tr, va, log, tag):
    by_C, best = {}, None
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=20000, class_weight="balanced").fit(F[tr], y[tr])
        assert list(clf.classes_) == KNOWN
        by_C[str(C)] = r3(f1_score(y[va], clf.predict(F[va]), labels=KNOWN, average="macro"))
        log(f"  [{tag}] C={C}: val macro F1 {by_C[str(C)]:.3f}  (n_iter {int(clf.n_iter_[0])})")
        if best is None or by_C[str(C)] > by_C[str(best[0])]:
            best = (C, clf)
    log(f"  [{tag}] chosen C={best[0]}")
    return best[0], best[1], by_C


def run_view(F, y, sp, T, scale, log, v):
    """Zero-shot and the probe with its reject rule on one view's features; returns predictions and metrics."""
    tr, va, te = sp == "train", sp == "val", sp == "test"
    assert not (y[tr | va] == NOVEL).any()
    zs_p = zero_shot(F, T, scale)
    C, clf, by_C = probe(F, y, tr, va, log, v)
    lp_p = clf.predict_proba(F)
    lp_max = lp_p.max(1).round(4)  # the 4-dp value written to predictions.csv, so lp_max < tau reproduces the rule
    tau = round(float(np.percentile(lp_max[va], TAU_PCT)), 4)
    lp3 = np.array(KNOWN)[lp_p.argmax(1)]
    lp = np.where(lp_max < tau, NOVEL, lp3)
    zs = np.array(CLASSES)[zs_p.argmax(1)]
    log(f"  [{v}] tau={tau}  rejected: val {(lp_max[va] < tau).mean():.3f}  test {(lp_max[te] < tau).mean():.3f}  "
        f"test {NOVEL} {(lp_max[te & (y == NOVEL)] < tau).mean():.3f}  test known {(lp_max[te & (y != NOVEL)] < tau).mean():.3f}")
    o = dict(zs_p=zs_p, lp_p=lp_p, lp_max=lp_max, zs=zs, lp=lp, C=C, by_C=by_C, tau=tau)
    for name, pred, nov, splits in [("zero_shot", zs, zs_p[:, CLASSES.index(NOVEL)], ["val", "test"]),
                                    ("linear_probe", lp, 1 - lp_max, ["train", "val", "test"])]:
        m = {s: metrics(y[sp == s], pred[sp == s]) for s in splits if s != "test"}
        m["test"] = test_metrics(y[te], pred[te], nov[te])
        t = m["test"]
        for s in ["test", "val"]:
            log(f"  [{v}] {name} {s}: acc {m[s]['accuracy']:.3f} bal {m[s]['balanced_accuracy']:.3f} macro P/R/F1 "
                f"{m[s]['macro_precision']:.3f}/{m[s]['macro_recall']:.3f}/{m[s]['macro_f1']:.3f} wF1 {m[s]['weighted_f1']:.3f}  "
                f"per-class P/R {({c: (q['precision'], q['recall']) for c, q in m[s]['per_class'].items()})}")
        log(f"  [{v}] {name} test AUROC {t['no_obstacles_auroc']}")
        log(f"  [{v}] {name} test confusion (rows true {CLASSES}) {t['confusion']['matrix']}")
        o[name] = {s: m[s] for s in splits}
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/data/eda/clip_sidewalk_data")
    ap.add_argument("--model", default="openai/clip-vit-base-patch16")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    logf = open(f"{a.out}/run.log", "w")  # each run's log is self-contained
    def log(s):
        line = f"{time.strftime('%H:%M:%S')} {s}"
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()
    t_start = time.time()
    torch.set_num_threads(48)
    log(f"=== clip_baseline.py  model {a.model}  out {a.out}  classes {CLASSES}")

    M, boxes = load()
    for s in SPLITS:
        log(f"{s}: {(M.split == s).sum()} images {M[M.split == s]['class'].value_counts().to_dict()}")
    model = CLIPModel.from_pretrained(a.model).eval()
    proc = CLIPProcessor.from_pretrained(a.model)
    F = features(M, boxes, model, proc, a.out, log)

    with torch.inference_mode():
        tok = proc.tokenizer([p for c in CLASSES for p in PROMPTS[c]], padding=True, return_tensors="pt")
        T = model.get_text_features(**tok)
        T = torch.nn.functional.normalize(getattr(T, "pooler_output", T), dim=-1).numpy()
        scale = float(model.logit_scale.exp())
    T = T.reshape(len(CLASSES), -1, T.shape[-1]).mean(1)
    T /= np.linalg.norm(T, axis=1, keepdims=True)

    y, sp = M["class"].to_numpy(), M.split.to_numpy()
    out = {v: run_view(F[v], y, sp, T, scale, log, v) for v in VIEWS}

    c = out["crop"]
    P = M[["image_path", "class", "split", "city", "label_id", "crop_x0", "crop_y0", "crop_x1", "crop_y1"]]
    P = P.rename(columns={"image_path": "file"}).assign(zero_shot=c["zs"])
    for i, k in enumerate(CLASSES):
        P[f"zs_p_{k}"] = c["zs_p"][:, i]
    P["linear_probe"] = c["lp"]
    for i, k in enumerate(KNOWN):
        P[f"lp_p_{k}"] = c["lp_p"][:, i]
    P["lp_max"] = c["lp_max"]
    assert len(P) == N_TOTAL
    save_atomic(f"{a.out}/predictions.csv", lambda fh: P.to_csv(fh, index=False, float_format="%.4f"))

    te = sp == "test"
    maj = M[sp == "train"]["class"].mode()[0]
    mm = {s: metrics(y[sp == s], np.full((sp == s).sum(), maj)) for s in ["val", "test"]}
    mm["test"] = test_metrics(y[te], np.full(te.sum(), maj), np.zeros(te.sum()))  # constant score: AUROC 0.5
    f = out["full"]
    ablation = dict(description=FULL, **{n: dict(test_accuracy=f[n]["test"]["accuracy"], test_macro_f1=f[n]["test"]["macro_f1"],
                                                 test_no_obstacles_auroc=f[n]["test"]["no_obstacles_auroc"])
                                         for n in ["zero_shot", "linear_probe"]})
    res = dict(
        model=a.model, dataset=D, classes=CLASSES, known_classes=KNOWN, excluded_classes=EXCLUDED,
        class_notes={NOVEL: NOTE},
        splits={s: dict(cities=sorted(M[M.split == s].city.unique()), n=int((M.split == s).sum()),
                        class_counts={k: int(((M.split == s) & (M["class"] == k)).sum()) for k in CLASSES})
                for s in ["train", "val", "test"]},
        input=INPUT, prompts=PROMPTS,
        probe=dict(C_grid=C_GRID, val_macro_f1_by_C=c["by_C"], C=c["C"], class_weight="balanced",
                   reject=dict(tau=c["tau"], rule="no_obstacles if max prob < tau",
                               tau_source="5th percentile of val max prob")),
        majority_baseline=dict(**{"class": maj}, test_accuracy=mm["test"]["accuracy"], test_macro_f1=mm["test"]["macro_f1"],
                               val=mm["val"], test=mm["test"]),
        models={n: c[n] for n in ["zero_shot", "linear_probe"]},
        ablation_full_frame=ablation)
    save_atomic(f"{a.out}/results.json", lambda fh: fh.write(json.dumps(res, indent=1).encode()))
    log(f"majority ({maj}): " + "  ".join(f"{s} acc {mm[s]['accuracy']} bal {mm[s]['balanced_accuracy']} macro P/R/F1 "
                                          f"{mm[s]['macro_precision']}/{mm[s]['macro_recall']}/{mm[s]['macro_f1']}" for s in mm))
    log(f"ablation full frame: {json.dumps(ablation)}  (probe C={f['C']}, tau={f['tau']})")
    log(f"wrote predictions.csv ({len(P)} rows) and results.json; wall time {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
