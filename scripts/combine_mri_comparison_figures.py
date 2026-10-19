#!/usr/bin/env python3
"""Join each figure of the own MRI model with the same figure of Ultralytics YOLOv5, side by side.

The two figures of a pair can differ by a few pixels, so the smaller one is padded with black to the size
of the larger one. Both halves then show at exactly the same size, wherever the page is rendered.

Run: python scripts/combine_mri_comparison_figures.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

figure_folder = Path(__file__).resolve().parents[1] / "docs" / "mri"
pairs = [
    ("evaluation_scores.png", "yolov5_evaluation_scores.png", "comparison_scores.png"),
    ("confusion_matrix.png", "yolov5_confusion_matrix.png", "comparison_confusion_matrices.png"),
    ("test_predictions.jpg", "yolov5_test_predictions.jpg", "comparison_predictions.jpg"),
]
model_names = ("Own model", "Ultralytics YOLOv5")


def padded_to(image, width, height):
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return canvas


plt.style.use("dark_background")
for own_name, yolov5_name, combined_name in pairs:
    images = [Image.open(figure_folder / name).convert("RGB") for name in (own_name, yolov5_name)]
    width, height = max(image.width for image in images), max(image.height for image in images)
    images = [padded_to(image, width, height) for image in images]

    fig, axes = plt.subplots(1, 2, figsize=(2 * width / 100, height / 100 + .6), dpi=100)
    for ax, image, model_name in zip(axes, images, model_names):
        ax.imshow(np.asarray(image))
        ax.set_title(model_name, fontsize=16)
        ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1 - .6 / (height / 100 + .6), wspace=0.02)
    fig.savefig(figure_folder / combined_name, dpi=100, pil_kwargs={"quality": 90} if combined_name.endswith(".jpg") else {})
    plt.close(fig)
    print(f"Wrote {figure_folder / combined_name}")
