"""Browser-pipeline metrics on DFF val/test, plus a browser-specific Platt fit.

    python analyse.py <repo> <items.json> <scores.jsonl> <out.json>
"""
import json
import sys

import numpy as np

repo, items_path, scores_path, out_path = sys.argv[1:5]
sys.path.insert(0, repo)
from training.metrics import balanced_indices, bootstrap_auc_ci, summary  # noqa: E402
from training.model import fit_platt, logit  # noqa: E402

items = {it["id"]: it for it in json.load(open(items_path))}
scores = [json.loads(line) for line in open(scores_path) if line.strip()]
rows = [{**items[s["id"]], **s} for s in scores]


def arrays(split, with_face=True):
    sel = [r for r in rows if r["split"] == split and (r["prob"] is not None or not with_face)]
    y = np.array([r["label"] for r in sel])
    p = np.array([r["prob"] for r in sel], dtype=float)
    g = np.array([r["group"] for r in sel])
    src = np.array([r["source"] for r in sel])
    return y, p, g, src


report = {}
for split in ["val", "test"]:
    total = [r for r in rows if r["split"] == split]
    report[f"{split}_no_face_rate"] = sum(r["prob"] is None for r in total) / len(total)
    report[f"{split}_n"] = len(total)

yv, pv, _, _ = arrays("val")
iv = balanced_indices(yv)
a, b = fit_platt(pv[iv], yv[iv])
report["browser_platt"] = {"a": a, "b": b}

yt, pt, gt, st = arrays("test")
pt_cal = 1 / (1 + np.exp(-(a * logit(pt) + b)))
it = balanced_indices(yt)
for tag, p in [("as_shipped", pt), ("browser_calibrated", pt_cal)]:
    s = summary(yt, p)
    s["auc_ci95"] = bootstrap_auc_ci(yt, p, gt)
    sb = summary(yt[it], p[it])
    report[tag] = {"overall": s, "balanced": sb,
                   "per_generator_auc": {src: summary(yt[(st == "wiki") | (st == src)],
                                                      p[(st == "wiki") | (st == src)])["auc"]
                                         for src in ["inpainting", "insight", "text2img"]}}

json.dump(report, open(out_path, "w"), indent=2)
print(f"no-face rate: val {report['val_no_face_rate']:.3%}, test {report['test_no_face_rate']:.3%}")
print(f"browser Platt (on top of shipped model): a={a:.3f} b={b:.3f}")
for tag in ["as_shipped", "browser_calibrated"]:
    o, bl = report[tag]["overall"], report[tag]["balanced"]
    print(f"{tag:19s} test AUC {o['auc']:.4f} {tuple(round(x, 4) for x in o['auc_ci95'])}  "
          f"EER {o['eer']:.3f}  TPR@1% {o['tpr_at_fpr_1pct']:.3f}  bal-acc {bl['balanced_accuracy']:.3f}  "
          f"ECE(bal) {bl['ece']:.3f}  per-gen {report[tag]['per_generator_auc']}")
