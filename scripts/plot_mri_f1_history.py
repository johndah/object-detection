#!/usr/bin/env python3
"""Plot the own MRI model's F1 score over the training runs saved in the git history.

Each run's notebook output is read from its commit, so the scores are the printed ones. The first
runs printed a per-image F1 score on the validation set, later runs the F1 score on the test set.
Each label names the main change of that run.

Run: python scripts/plot_mri_f1_history.py
"""
import json
import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt

repo_root = Path(__file__).resolve().parents[1]
figure_path = repo_root / "docs" / "mri" / "f1_history.png"

validation_metric_color = "#3d8fd1"
test_metric_color = "#c27a06"

runs = [
    ("c559d74", "1.8M parameters"),
    ("b46d123", "7.0M parameters"),
    ("3769edb", "15.8M parameters"),
    ("70d943f", "test F1 from here"),
    ("6da9c4f", "70% of data for training"),
    ("04c9f51", "mosaic in every image"),
    ("27caa1d", "scale and shift after mosaic, lower learning rate"),
    ("8aaf83f", "loss decoding fixed, peak learning rate 0.03"),
    ("4fcbb95", "validation folder added"),
    ("21b1624", "copies removed, split by patient group"),
    ("61ef4bd", "box scaling and clipping fixed, evaluation mode"),
]


def printed_f1_score(commit):
    notebook_json = subprocess.run(["git", "show", f"{commit}:mri-detection.ipynb"], cwd=repo_root,
                                   capture_output=True, text=True, check=True).stdout
    notebook = json.loads(notebook_json)
    output_text = "".join("".join(output.get("text", "")) for cell in notebook["cells"] for output in cell.get("outputs", []))
    match = re.search(r"(Mean )?F1 score: ([0-9.]+)", output_text)
    is_validation_metric = match.group(1) is not None
    return float(match.group(2)), is_validation_metric


scores = [printed_f1_score(commit) for commit, _ in runs]

plt.style.use("dark_background")
fig, ax = plt.subplots(figsize=(12, 7))
run_numbers = list(range(1, len(runs) + 1))

for is_validation_metric, color, marker, legend_label in (
        (True, validation_metric_color, "s", "Per-image F1 score on the validation set (older metric)"),
        (False, test_metric_color, "o", "F1 score on the test set")):
    points = [(number, score) for number, (score, validation) in zip(run_numbers, scores) if validation == is_validation_metric]
    ax.plot(*zip(*points), color=color, marker=marker, markersize=8, linewidth=2, label=legend_label)

ax.set_xticks(run_numbers)
ax.set_xticklabels([f"{number}. {change}" for number, (_, change) in zip(run_numbers, runs)], rotation=35, ha="right")
ax.set_xlabel("Training run, in order, with its main change")
ax.set_ylabel("F1 score of the own model")
ax.set_ylim(0.55, 0.95)
ax.set_xlim(0.5, len(runs) + 0.5)
ax.grid(axis="y", color="gray", alpha=0.3)
ax.legend(loc="upper left")
ax.set_title("F1 score of the own model over the training runs")
fig.tight_layout()
fig.savefig(figure_path, dpi=100)
print(f"Wrote {figure_path}")
for (commit, change), (score, validation) in zip(runs, scores):
    print(commit, score, "validation per image" if validation else "test", change.replace("\n", " "))
