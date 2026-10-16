#!/usr/bin/env python3
"""Draw the sample figures: predictions on chosen test pictures, examples of the pasting, and predictions on pictures
from our own camera by a pretrained COCO model, the first model, trained before the pasting, and the current one.

The pictures are read from the prepared data in datasets/, which has to be in the state of the seed set in the
notebook (the notebook run up to just before the pasting of camera pictures). They are copied into
docs/nordic/test_samples, difficult_conditions and pasting, next to the figures. Boxes and pasting use the notebook's own code.
The camera pictures, with their labels in a labels folder beside them, only exist locally in nordic_animals. Only their
figures are written, into docs/nordic/own_camera.

Run: python scripts/sample_figures.py path/to/best.pt
"""
import argparse
import json
import re
import shutil
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.ops import xywhn2xyxy

repo_root = Path(__file__).resolve().parents[1]
dataset_root = repo_root / "datasets"
camera_root = repo_root / "nordic_animals"
output_root = repo_root / "docs" / "nordic"

# Test pictures to show predictions on. Training pictures do not belong here, the model has learned them
test_samples = [
    "wildlife_wildlife5_train_0240_jpg.rf.5b5b1e9c34c29f86f65edc0701766ea5.jpg",
    "wildlife_moose_detection_test_file_00020_png.rf.0e916bf2f2a4593ac9083446e05b6704.jpg",
    "wildlife_wild_deer6_test_PICT0621_JPG.rf.0a5d3fc60a49dab586ba4b3a4bf758d8.jpg",
    "wildlife_italian_cameras_test_vlcsnap-2024-03-07-11h55m55s868_png.rf.c7fb21b49f2c39731a44a839470b839a.jpg",
    "wildlife_animal_detection3_test_783450880b2a146b_jpg.rf.367b36726a7a2cb5369107fe53c4e708.jpg",
    "coco_000000219474.jpg",
]
# Successes in the first row, failures in the second
difficult_conditions = [
    "wildlife_wild_deer6_test_PICT0659_JPG.rf.0a027dfc762fbae983e2bad00465ce58.jpg",
    "wildlife_wild_deer6_test_PICT0395_JPG.rf.0ccdd6e85cbd54c8a5bc28e8648e625e.jpg",
    "wildlife_animal_detection3_test_FY08PRVLLYQT_jpg.rf.466dfd8f7904819ac495c6c459e63b5d.jpg",
    "wildlife_wild_deer6_test_PICT0171_JPG.rf.4ed93553223ebf020bbf9fbed4efda28.jpg",
]
picture_titles = {
    "wildlife_wild_deer6_test_PICT0659_JPG.rf.0a027dfc762fbae983e2bad00465ce58.jpg": "Occlusion: captured both sides of the tree",
    "wildlife_wild_deer6_test_PICT0395_JPG.rf.0ccdd6e85cbd54c8a5bc28e8648e625e.jpg":
        "Darkness: detected elk, low confidence but\nabove our threshold",
    "wildlife_animal_detection3_test_FY08PRVLLYQT_jpg.rf.466dfd8f7904819ac495c6c459e63b5d.jpg": "Low resolution: detected fox but also a false boar",
    "wildlife_wild_deer6_test_PICT0171_JPG.rf.4ed93553223ebf020bbf9fbed4efda28.jpg": "Hidden and overlapping: detected two deer as one",
}
# Pixel values are multiplied by this factor before the model sees the picture
brightness_factors = {
    "wildlife_wild_deer6_test_PICT0395_JPG.rf.0ccdd6e85cbd54c8a5bc28e8648e625e.jpg": 0.5,
}
# Pictures of animals from our own camera
camera_animals = [
    "cropped_animals/twilight_fox_in_a_rural_meadow.png",
    "cropped_animals/twilight_deer_in_snowy_field.png",
    "cropped_animals/moose_crossing_a_sunlit_meadow.png",
]
# One of the two moose the final model fails on, which it calls a roe deer
camera_failure = "cropped_animals/dawn_moose_crossing_the_meadow.png"
# Source picture, its class, target picture, its class, both from the training set, a factor on the usual height
# of the source species, and the brightness relative to the ground (None draws it as in training). The roe deer
# source is a fawn, smaller than a grown roe deer
pasting_examples = [
    (
        "wildlife_italian_cameras_train_vlcsnap-2024-03-06-12h50m25s819_png.rf.a0e1477bc9d439383c8e1668f3ab1d97.jpg", "deer",
        "wildlife_wild_deer6_train_PICT0333_JPG.rf.76adeea0e0ba989ab4251a99f20108ac.jpg", "moose",
        0.5, 0.75,
    ),
    (
        "wildlife_animal_detection3_train_2a95dcb240b0ac29_jpg.rf.e21acc73f561577238b4617d43e40b0a.jpg", "fox",
        "wildlife_wild_deer6_train_PICT0918_JPG.rf.f69e1553bb225b7997a92875d8d12d4d.jpg", "elk",
        0.7, None,
    ),
]


def read_cell_sources():
    notebook = json.loads((repo_root / "nordic-animal-detection.ipynb").read_text(encoding="utf-8"))

    return [
        cell["source"] if isinstance(cell["source"], str) else "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def find_cell(cell_sources, marker):
    return next(source for source in cell_sources if marker in source)


def find_picture(file_name):
    """The picture in whichever split the seed has put it."""
    return next(
        dataset_root / split / "images" / file_name
        for split in ("train", "validation", "test")
        if (dataset_root / split / "images" / file_name).exists()
    )


def read_labeled_boxes(image_path, labels_folder=None):
    """Class ids and pixel boxes (x_min, y_min, x_max, y_max) of a picture, by default a prepared one."""
    labels_folder = labels_folder or image_path.parents[1] / "labels"
    image = Image.open(image_path)
    label_rows = [line.split() for line in (labels_folder / f"{image_path.stem}.txt").read_text().splitlines()]
    boxes = xywhn2xyxy(np.array([row[1:] for row in label_rows], dtype=float).reshape(-1, 4), w=image.width, h=image.height)

    return [int(row[0]) for row in label_rows], boxes


def draw_predictions(notebook_names, model, file_names, columns, title, figure_path, confidence):
    plot_boxes = notebook_names["plot_boxes"]
    rows = (len(file_names) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4.4 * rows), squeeze=False)

    for ax in axes.flatten():
        ax.axis("off")

    for ax, file_name in zip(axes.flatten(), file_names):
        image_path = find_picture(file_name)
        image = Image.open(image_path).convert("RGB")
        image = Image.fromarray((np.asarray(image) * brightness_factors.get(file_name, 1.0)).astype(np.uint8))
        class_ids, boxes = read_labeled_boxes(image_path)
        prediction = model.predict(image, imgsz=640, conf=confidence, verbose=False)[0]

        ax.axis("on")
        plot_boxes(ax, image, boxes, class_ids)
        plot_boxes(
            ax,
            image,
            prediction.boxes.xyxy.cpu().numpy(),
            prediction.boxes.cls.cpu().numpy(),
            prediction.boxes.conf.cpu().numpy(),
        )

        if file_name in picture_titles:
            ax.set_title(picture_titles[file_name])

        notebook_names["add_picture_legend"](ax)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=110)
    plt.close(fig)
    print(f"Saved {figure_path}")


def draw_before_and_after(notebook_names, model_rows, image_paths, title, figure_path, confidence, picture_titles=()):
    """One row of predictions per model and one column per picture."""
    plot_boxes = notebook_names["plot_boxes"]
    our_classes, our_colors = notebook_names["CLASSES"], notebook_names["class_colors"]
    fig, axes = plt.subplots(
        len(model_rows), len(image_paths), figsize=(4.2 * len(image_paths), 4.4 * len(model_rows)), squeeze=False
    )

    for row_axes, (row_title, model) in zip(axes, model_rows):
        for ax, image_path, picture_title in zip(row_axes, image_paths, picture_titles or [None] * len(image_paths)):
            image = Image.open(image_path).convert("RGB")
            class_ids, boxes = read_labeled_boxes(image_path, image_path.parent / "labels")
            prediction = model.predict(image, imgsz=640, conf=confidence, verbose=False)[0]
            predicted_names = [model.names[int(class_id)] for class_id in prediction.boxes.cls.cpu().numpy()]

            # plot_boxes knows only our classes. Other classes of the model, like horse in COCO, are added in grey
            extra_classes = [name for name in dict.fromkeys(predicted_names) if name not in our_classes]
            notebook_names["CLASSES"] = our_classes + extra_classes
            notebook_names["class_colors"] = np.vstack([our_colors] + [(0.6, 0.6, 0.6, 1.0)] * len(extra_classes))

            plot_boxes(ax, image, boxes, class_ids)
            plot_boxes(
                ax,
                image,
                prediction.boxes.xyxy.cpu().numpy(),
                [notebook_names["CLASSES"].index(name) for name in predicted_names],
                prediction.boxes.conf.cpu().numpy(),
            )
            notebook_names["add_picture_legend"](ax)

            notebook_names["CLASSES"], notebook_names["class_colors"] = our_classes, our_colors

            if picture_title:
                ax.set_title(picture_title)

        if len(model_rows) > 1:
            row_axes[0].set_ylabel(row_title, fontsize=13)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=110)
    plt.close(fig)
    print(f"Saved {figure_path}")


class BrightnessSettingRandom:
    """Passes every random draw through, except the brightness of the animal when a fixed value is given."""

    def __init__(self, rng, fixed_brightness, brightness_ranges):
        self.rng = rng
        self.fixed_brightness = fixed_brightness
        self.brightness_ranges = brightness_ranges
        self.used_brightness = float("nan")

    def uniform(self, low=0.0, high=1.0, size=None):
        value = self.rng.uniform(low, high, size)

        if size is None and (low, high) in self.brightness_ranges:
            value = value if self.fixed_brightness is None else self.fixed_brightness
            self.used_brightness = value

        return value

    def __getattr__(self, name):
        return getattr(self.rng, name)


def draw_pasting(notebook_names, cell_sources, figure_path):
    classes = notebook_names["CLASSES"]
    picked_animals = defaultdict(list)
    animals_per_pair = {}
    pairs = []

    height_factors = {}

    brightness_settings = {}

    for source_name, source_class, target_name, target_class, height_factor, relative_brightness in pasting_examples:
        source_path = find_picture(source_name)
        class_ids, boxes = read_labeled_boxes(source_path)

        # A picture with several animals of the class: the largest one is cut out
        candidates = [box for class_id, box in zip(class_ids, boxes) if classes[class_id] == source_class]
        largest = max(candidates, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
        source = (source_path, tuple(float(value) for value in largest))
        target = (find_picture(target_name), None)

        picked_animals[classes.index(source_class)].append(source)
        picked_animals[classes.index(target_class)].append(target)
        animals_per_pair[classes.index(source_class), classes.index(target_class)] = {
            classes.index(source_class): [source],
            classes.index(target_class): [target],
        }
        pairs.append((source_class, target_class))
        height_factors[classes.index(source_class), classes.index(target_class)] = height_factor
        brightness_settings[classes.index(source_class), classes.index(target_class)] = relative_brightness

    # Darker than the ground only. Training also pastes brighter animals, which look artificial in a figure
    notebook_names["darker_animal_probability"] = 1.0
    brightness_ranges = {
        tuple(notebook_names["darker_brightness_range"]),
        tuple(notebook_names["brighter_brightness_range"]),
    }
    animal_heights = notebook_names["relative_animal_heights"]

    # Each example draws its source and target only from its own pair, also when two pairs share a class
    create_pasting_example = notebook_names["create_pasting_example"]

    def create_picked_example(source_class_id, target_class_id, source_animals, rng):
        source_class = classes[source_class_id]
        usual_heights = animal_heights[source_class]
        height_factor = height_factors[source_class_id, target_class_id]
        animal_heights[source_class] = (usual_heights[0] * height_factor, usual_heights[1] * height_factor)

        try:
            pair_rng = BrightnessSettingRandom(rng, brightness_settings[source_class_id, target_class_id], brightness_ranges)
            example = create_pasting_example(source_class_id, target_class_id, animals_per_pair[source_class_id, target_class_id], pair_rng)
            print(f"  {source_class} into {classes[target_class_id]}: relative brightness {pair_rng.used_brightness:.3f}")
            return example
        finally:
            animal_heights[source_class] = usual_heights

    notebook_names["create_pasting_example"] = create_picked_example

    # The figure part of the notebook's pasting cell, with the picked pictures in place of random ones
    pasting_cell = find_cell(cell_sources, "def create_pasting_example")
    figure_code = pasting_cell[pasting_cell.index("segmenter = SAM("):]
    figure_code = re.sub(r"training_source_animals = find_source_animals\(\"train\"\)", "training_source_animals = picked_animals", figure_code)
    figure_code = re.sub(r"example_class_names = \[.*\]", "example_class_names = picked_pairs", figure_code)
    notebook_names.update(picked_animals=picked_animals, picked_pairs=pairs)
    notebook_names["plt"].show = lambda: (plt.savefig(figure_path, dpi=110), plt.close("all"))
    exec(figure_code, notebook_names)

    print(f"Saved {figure_path} with {len(notebook_names['examples'])} of {len(pairs)} examples")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("weights", type=Path, nargs="?", default=repo_root / "animals_v8.pt")
    parser.add_argument("--confidence", type=float, default=0.40)
    parser.add_argument("--first-weights", type=Path, default=repo_root / "yolo26n_nordic_animals.pt")
    parser.add_argument("--pretrained-weights", type=Path, default=repo_root / "yolo26n.pt")
    arguments = parser.parse_args()

    cell_sources = read_cell_sources()
    seed = int(re.search(r"^seed = (\d+)$", find_cell(cell_sources, "DEER_CLASSES = ["), re.M).group(1))
    model = YOLO(str(arguments.weights))
    classes = [model.names[class_id] for class_id in sorted(model.names)]

    plt.style.use("dark_background")
    notebook_names = dict(
        np=np, plt=plt, Image=Image, Path=Path, defaultdict=defaultdict, Counter=Counter, OrderedDict=OrderedDict,
        xywhn2xyxy=xywhn2xyxy, seed=seed, CLASSES=classes, dataset_root=dataset_root,
        class_colors=plt.colormaps["viridis"](np.linspace(0, 1, len(classes))), class_indices=np.arange(len(classes)),
        min_box_side=32, garden_prefix="garden_", image_counts=Counter(),
    )
    exec(find_cell(cell_sources, "def plot_boxes"), notebook_names)
    pasting_cell = find_cell(cell_sources, "def create_pasting_example")
    exec(pasting_cell[:pasting_cell.index("segmenter = SAM(")], notebook_names)

    # The figures promise test pictures and training pictures, so a pick from the wrong split is refused
    for file_name in test_samples + difficult_conditions:
        if find_picture(file_name).parts[-3] != "test":
            raise SystemExit(f"Not a test picture with seed {seed}: {file_name}")

    for example in pasting_examples:
        for file_name in (example[0], example[2]):
            if find_picture(file_name).parts[-3] != "train":
                raise SystemExit(f"Not a training picture with seed {seed}: {file_name}")

    for folder_name, file_names in (
        ("test_samples", test_samples),
        ("difficult_conditions", difficult_conditions),
        ("pasting", [name for example in pasting_examples for name in (example[0], example[2])]),
    ):
        # Pictures of earlier picks are removed, so that each folder holds exactly the current picks
        folder = output_root / folder_name
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)

        for file_name in file_names:
            shutil.copy2(find_picture(file_name), folder / file_name)

    draw_predictions(
        notebook_names, model, test_samples, 3, "Predictions on samples of the test set",
        output_root / "test_samples" / "predictions.jpg", arguments.confidence,
    )
    draw_predictions(
        notebook_names, model, difficult_conditions, 2, "Predictions in difficult conditions",
        output_root / "difficult_conditions" / "predictions.jpg", arguments.confidence,
    )
    draw_pasting(notebook_names, cell_sources, output_root / "pasting" / "how_pasting_works.jpg")

    # Our camera pictures are private and only exist locally, so their figure is kept when the pictures are missing
    if camera_root.is_dir():
        model_rows = [
            ("Pretrained on COCO", YOLO(str(arguments.pretrained_weights))),
            ("Before pasting augmentation", YOLO(str(arguments.first_weights))),
            ("After pasting augmentation", model),
        ]
        (output_root / "own_camera").mkdir(exist_ok=True)
        animal_paths = [camera_root / name for name in camera_animals]
        draw_before_and_after(
            notebook_names, model_rows, animal_paths,
            "Predictions on animals from our own camera, before and after the segmentation pasting augmentation",
            output_root / "own_camera" / "animals.jpg", arguments.confidence,
        )

        # One row at a time for the summary page, where each model has its own section
        for model_row, image_paths, picture_titles, file_name, title in zip(
            model_rows,
            [animal_paths, animal_paths, animal_paths + [camera_root / camera_failure]],
            [(), (), [None] * len(animal_paths) + ["Fail case 1 of 2: moose named roe deer"]],
            ["pretrained.jpg", "before_pasting.jpg", "after_pasting.jpg"],
            [
                "Downloaded pretrained YOLO26n as it is, on animals from our own camera",
                "First model, trained before the pasting augmentation, on animals from our own camera",
                "Model trained with the pasting augmentation on animals from our own camera",
            ],
        ):
            draw_before_and_after(
                notebook_names, [model_row], image_paths, title, output_root / "own_camera" / file_name,
                arguments.confidence, picture_titles,
            )


if __name__ == "__main__":
    main()
