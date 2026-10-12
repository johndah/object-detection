#!/usr/bin/env python3
"""Show what the pasting of the notebook produces on our own camera pictures.

The notebook does not draw these samples, because the camera pictures are private. This script runs the
pasting code of the notebook itself on a few empty camera pictures and saves one figure in the local
nordic_animals folder. It needs the prepared data in datasets/, as the notebook leaves it.

Run: python scripts/show_pasted_samples.py
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics.utils.ops import xywhn2xyxy

repo_root = Path(__file__).resolve().parents[1]


def read_cell_sources(notebook_path):
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    return [
        cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        for cell in notebook["cells"]
    ]


def find_cell(cell_sources, marker):
    return next(source for source in cell_sources if marker in source)


def read_classes(cell_sources, dataset_root):
    """The class names of the notebook, without bird when the local labels are from after the bird drop."""
    class_cell = {}
    exec(find_cell(cell_sources, "DEER_CLASSES = ["), class_cell)
    classes = class_cell["CLASSES"]

    highest_class_id = max(
        int(line.split()[0])
        for label_path in (dataset_root / "train" / "labels").glob("*.txt")
        for line in label_path.read_text().splitlines()
    )

    return classes if highest_class_id == len(classes) - 1 else [name for name in classes if name != "bird"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--notebook", type=Path, default=repo_root / "nordic-animal-detection.ipynb")
    parser.add_argument("--frames", type=Path, default=repo_root / "nordic_animals" / "empty")
    parser.add_argument("--output", type=Path, default=repo_root / "nordic_animals" / "pasted_samples.jpg")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    cell_sources = read_cell_sources(arguments.notebook)
    dataset_root = repo_root / "datasets"
    classes = read_classes(cell_sources, dataset_root)

    plt.style.use("dark_background")

    # The names that the notebook cells expect from the cells before them
    notebook_names = dict(
        np=np, plt=plt, Image=Image, random=random, Path=Path, defaultdict=defaultdict, Counter=Counter,
        xywhn2xyxy=xywhn2xyxy, seed=arguments.seed, dataset_root=dataset_root,
        splits=["train", "validation", "test"], CLASSES=classes, class_indices=np.arange(len(classes)),
        class_colors=plt.colormaps["viridis"](np.linspace(0, 1, len(classes))),
        min_box_side=int(re.search(r"^min_box_side = (\d+)", find_cell(cell_sources, "min_box_side = "), re.M).group(1)),
        garden_prefix="garden_",
    )

    exec(find_cell(cell_sources, "def plot_boxes"), notebook_names)

    # Only the definitions of the pasting cell. Its second half draws the figure of the notebook
    pasting_cell = find_cell(cell_sources, "def paste_animal")
    exec(pasting_cell[:pasting_cell.index("segmenter = SAM(")], notebook_names)
    notebook_names["segmenter"] = notebook_names["SAM"](str(repo_root / "mobile_sam.pt"))

    find_source_animals = notebook_names["find_source_animals"]
    cut_out_animal = notebook_names["cut_out_animal"]
    paste_animal = notebook_names["paste_animal"]
    plot_boxes = notebook_names["plot_boxes"]
    camera_frame_size = notebook_names["camera_frame_size"]

    print("Looking for animals to cut out of the training images ...")
    source_animals = find_source_animals("train")
    frame_paths = sorted(
        path for path in arguments.frames.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    rng = np.random.default_rng(arguments.seed)

    class_ids = [class_id for class_id in range(len(classes)) if source_animals[class_id]][-6:]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for ax in axes.flatten():
        ax.axis("off")

    for ax, class_id in zip(axes.flatten(), class_ids):
        frame_path = frame_paths[rng.integers(len(frame_paths))]
        frame = Image.open(frame_path).convert("RGB").resize(camera_frame_size, Image.LANCZOS)
        box = None

        for _ in range(30):
            image_path, source_box = source_animals[class_id][rng.integers(len(source_animals[class_id]))]
            cut_out = cut_out_animal(image_path, source_box)

            if cut_out is None:
                continue

            animal, mask, _ = cut_out
            box = paste_animal(frame, animal, mask, [], rng, classes[class_id])

            if box is not None:
                break

        if box is None:
            continue

        ax.axis("on")
        plot_boxes(ax, frame, np.array([box], dtype=float), [class_id])
        ax.legend(loc="upper right")

    fig.suptitle("Animals from the training set, pasted into our own camera pictures", fontsize=14)
    fig.tight_layout()
    fig.savefig(arguments.output, dpi=110)
    print(f"Saved {arguments.output}")


if __name__ == "__main__":
    main()
