#!/usr/bin/env python3
"""Score a trained model on the labeled frames of our own camera.

The frames in nordic_animals/animals are never used for training. Each has a label file in
nordic_animals/labels with at most one box. An empty label file means the frame shows an animal
that is none of our classes, where the right answer is no detection.

Run: python scripts/evaluate_camera_frames.py path/to/best.pt
"""
import argparse
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO

repo_root = Path(__file__).resolve().parents[1]
min_iou = 0.3


def intersection_over_union(first, second):
    overlap_width = max(0, min(first[2], second[2]) - max(first[0], second[0]))
    overlap_height = max(0, min(first[3], second[3]) - max(first[1], second[1]))
    overlap = overlap_width * overlap_height
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])

    return overlap / (first_area + second_area - overlap)


def read_label(label_path, class_names, image_width, image_height):
    """Returns class name and pixel box, or None for a frame without an animal of our classes."""
    values = label_path.read_text().split()

    if not values:
        return None

    x_center, y_center, box_width, box_height = (float(value) for value in values[1:])
    box = (
        (x_center - box_width / 2) * image_width,
        (y_center - box_height / 2) * image_height,
        (x_center + box_width / 2) * image_width,
        (y_center + box_height / 2) * image_height,
    )

    return class_names[int(values[0])], box


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("weights", type=Path, nargs="?", default=repo_root / "yolo26n_nordic_animals.pt")
    parser.add_argument("--confidence", type=float, default=0.55)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--frames", type=Path, default=repo_root / "nordic_animals" / "animals")
    parser.add_argument("--labels", type=Path, default=repo_root / "nordic_animals" / "labels")
    arguments = parser.parse_args()

    model = YOLO(str(arguments.weights))
    label_class_names = (arguments.labels / "classes.txt").read_text().split()

    if label_class_names != [model.names[class_id] for class_id in sorted(model.names)]:
        raise SystemExit(f"The classes of the labels {label_class_names} differ from the model's {model.names}")

    outcomes = Counter()
    outcomes_per_class = Counter()

    for frame_path in sorted(arguments.frames.iterdir()):
        image = cv2.imread(str(frame_path))
        image_height, image_width = image.shape[:2]
        label = read_label(arguments.labels / f"{frame_path.stem}.txt", label_class_names, image_width, image_height)

        result = model.predict(image, imgsz=arguments.image_size, conf=arguments.confidence, verbose=False)[0]
        detections = sorted(
            (
                (float(confidence), model.names[int(class_id)], [float(value) for value in box])
                for class_id, confidence, box in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy)
            ),
            reverse=True,
        )
        detected_text = ", ".join(f"{name} {confidence:.2f}" for confidence, name, _ in detections) or "nothing"

        if label is None:
            outcome = "false detection" if detections else "correctly nothing"
            outcomes[f"other animal: {outcome}"] += 1
            print(f"{frame_path.name:55s} other animal  -> {detected_text}")
            continue

        true_class, true_box = label
        names_on_animal = [name for _, name, box in detections if intersection_over_union(box, true_box) > min_iou]

        if not names_on_animal:
            outcome = "missed"
        elif names_on_animal[0] == true_class:
            outcome = "right class"
        else:
            outcome = "wrong class"

        outcomes[outcome] += 1
        outcomes_per_class[true_class, outcome] += 1
        print(f"{frame_path.name:55s} {true_class:12s} -> {detected_text:28s} {outcome}")

    n_labeled = sum(outcomes[outcome] for outcome in ("right class", "wrong class", "missed"))
    print(f"\n{arguments.weights.name} at {arguments.image_size} px, confidence {arguments.confidence}")
    print(f"Right class: {outcomes['right class']} of {n_labeled} frames with an animal of our classes")

    for class_name in label_class_names:
        counts = [outcomes_per_class[class_name, outcome] for outcome in ("right class", "wrong class", "missed")]

        if sum(counts):
            print(f"  {class_name:6s} right {counts[0]}, wrong class {counts[1]}, missed {counts[2]}")

    n_other = outcomes["other animal: correctly nothing"] + outcomes["other animal: false detection"]
    print(f"Other animals: no detection on {outcomes['other animal: correctly nothing']} of {n_other} frames")


if __name__ == "__main__":
    main()
