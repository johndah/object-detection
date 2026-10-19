#!/usr/bin/env python3
"""Show how the earlier split by image leaked: test images next to their most similar training image.

Rebuilds the split of the MRI notebook before the group split (commit e36d5fc): the data set's Train and
Val folders merged, then shuffled image by image into 3672 training, 787 validation and 786 test images
(the notebook printed the planned 3674 and 788, but two images without labels are never moved). Its test set
holds the same 201 glioma, 248 meningioma and 301 pituitary boxes as the run's confusion matrix.
Similarity is the correlation of 48 px grayscale copies, as in the notebook's group split. Writes
docs/mri/leaked_pairs.jpg and prints the share of test images with an almost identical training image.

Run: python scripts/plot_mri_leaked_pairs.py
"""
import random
from pathlib import Path

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

repo_root = Path(__file__).resolve().parents[1]
figure_path = repo_root / "docs" / "mri" / "leaked_pairs.jpg"
seed = 1
training_split = .7
validation_split = (1. - training_split) / 2.
exact_copy = .999
almost_identical = .97
neighboring_slices = (.93, almost_identical)
very_similar = .9

data_set_path = Path(kagglehub.dataset_download("ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes"))
# Like the notebook's cp -n: sorted, Train before Val, and the first file of a name wins
image_paths = {}
for split in ("Train", "Val"):
    for path in sorted(data_set_path.glob(f"{split}/*/images/*")):
        image_paths.setdefault(path.name, path)
labeled_names = {path.stem for split in ("Train", "Val") for path in data_set_path.glob(f"{split}/*/labels/*")}


def split_by_image():
    data_size = len(image_paths)
    sizes = [int(data_size * training_split), int(data_size * validation_split)]
    sizes.append(data_size - sum(sizes))
    random.seed(seed)
    np.random.seed(seed)
    remaining = sorted(image_paths)
    splits = []
    for size in sizes:
        shuffled = sorted(remaining)
        np.random.shuffle(shuffled)
        chosen = shuffled[:size]
        moved = [name for name in chosen if Path(name).stem in labeled_names]
        splits.append(moved)
        moved_names = set(moved)
        remaining = [name for name in remaining if name not in moved_names]
    return splits


def thumbnail(name, size=48):
    pixels = np.asarray(Image.open(image_paths[name]).convert("L").resize((size, size)), dtype=np.float32).ravel()
    pixels -= pixels.mean()
    return pixels / (np.linalg.norm(pixels) + 1e-6)


training_names, validation_names, test_names = split_by_image()
assert (len(training_names), len(validation_names), len(test_names)) == (3672, 787, 786)

training_thumbnails = np.stack([thumbnail(name) for name in training_names])
test_thumbnails = np.stack([thumbnail(name) for name in test_names])
similarities = test_thumbnails @ training_thumbnails.T
closest_training_index = similarities.argmax(axis=1)
closest_similarity = similarities.max(axis=1)
print(f"Test images with a training image at similarity {almost_identical} or more: {np.mean(closest_similarity >= almost_identical):.0%}")
print(f"Test images with a training image at similarity {very_similar} or more: {np.mean(closest_similarity >= very_similar):.0%}")

rng = np.random.default_rng(seed)
copies = np.flatnonzero(closest_similarity >= exact_copy)
neighbors = np.flatnonzero((closest_similarity >= neighboring_slices[0]) & (closest_similarity < neighboring_slices[1]))
examples = list(rng.choice(copies, 2, replace=False)) + list(rng.choice(neighbors, 2, replace=False))

plt.style.use("dark_background")
fig, axes = plt.subplots(2, len(examples), figsize=(3 * len(examples), 6.4))
for column, test_index in enumerate(examples):
    training_name = training_names[closest_training_index[test_index]]
    for row, (name, role) in enumerate(((test_names[test_index], "Test image"), (training_name, "Training image"))):
        axes[row, column].imshow(Image.open(image_paths[name]).convert("RGB").resize((256, 256)))
        axes[row, column].set_title(role if row == 0 else f"{role}\nsimilarity {closest_similarity[test_index]:.2f}", fontsize=10)
        axes[row, column].axis("off")
plt.suptitle("Split by image: test images and their most similar training image")
fig.tight_layout()
fig.savefig(figure_path, dpi=100, pil_kwargs={"quality": 90})
print(f"Wrote {figure_path}")
