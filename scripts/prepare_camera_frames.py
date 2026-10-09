#!/usr/bin/env python3
"""Turn the saved detections of the garden camera into clean images.

The camera saves a frame whenever its detector fires and draws the box and label into the frame.
Most of these detections are false positives on fixed objects, so the frames are in fact empty
scenes. This script finds the drawn overlay, removes it and writes

    nordic_animals/empty/    frames without animals, as backgrounds for training
    nordic_animals/animals/  frames with real animals, as the camera test set

Run: python scripts/prepare_camera_frames.py
"""
import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timedelta
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

# Box colors of the Ultralytics plot for the COCO classes the camera has saved, in BGR
overlay_colors = {
    "person": (255, 42, 4),
    "car": (243, 243, 243),
    "truck": (0, 237, 204),
    "cat": (255, 36, 125),
    "dog": (104, 0, 123),
    "horse": (108, 27, 255),
}
color_tolerance = 60
label_min_size = (15, 40)  # rows, columns
label_max_height = 45
line_width = 7
inpaint_margin = 3
blend_margin = 8
excluded_dates = {"2025-09-13", "2026-05-14"}
spot_cell = 60
static_spot_min_days = 5
donor_time_window = timedelta(days=5)
donor_max_residual = 20
donor_max_tries = 12
frames_per_day_and_hour = 3
timestamp_format = "%Y-%m-%d  %H:%M:%S"


def parse_name(path):
    timestamp_text, _, class_text = path.stem.rpartition("-")
    return datetime.strptime(timestamp_text, timestamp_format), class_text.split("_")


def find_overlays(image):
    """Find the drawn boxes with their labels.

    A box and its label have the same flat color and touch, so they form one connected region of that color.
    The label is the solid rectangle at the top left of the region, the box is what lies below it. Returns one
    dict per overlay with class, box and label, both as x1, y1, x2, y2.
    """
    overlays = []
    for class_name, color in overlay_colors.items():
        color = np.array(color, dtype=np.int16)
        lower, upper = np.clip(color - color_tolerance, 0, 255), np.clip(color + color_tolerance, 0, 255)
        color_mask = cv2.inRange(image, lower.astype(np.uint8), upper.astype(np.uint8))
        closed_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        n_regions, regions, stats, _ = cv2.connectedComponentsWithStats(closed_mask)

        for region_id in range(1, n_regions):
            left, top, width, height, _ = stats[region_id]
            if width < 8 or height < 8 or width > image.shape[1] // 2:
                continue
            region = (regions[top:top + height, left:left + width] == region_id).astype(np.uint8)
            solid = cv2.morphologyEx(region, cv2.MORPH_OPEN, np.ones(label_min_size, np.uint8))
            _, _, solid_stats, _ = cv2.connectedComponentsWithStats(solid)
            labels_at_top_left = [stat for stat in solid_stats[1:] if stat[0] <= 4 and stat[1] <= 4]
            if not labels_at_top_left:
                continue
            label_left, label_top, label_width, label_height, label_area = (int(value) for value in labels_at_top_left[0])
            if label_area < 0.7 * label_width * label_height or label_height > label_max_height:
                continue

            below_label = region[label_top + label_height + 2:]
            if below_label.any():
                box_columns = np.flatnonzero(below_label.any(axis=0))
                box_x1, box_x2, box_y1 = int(box_columns.min()), int(box_columns.max()) + 1, label_top + label_height - line_width
                beside_label = region[label_top + 1:label_top + line_width, label_left + label_width + 3:box_x2]
                if beside_label.size and beside_label.mean() > 0.6:
                    box_y1 = label_top
            else:
                box_x1, box_x2, box_y1 = 0, int(width), label_top
            overlays.append({
                "class": class_name,
                "box": [int(left) + box_x1, int(top) + max(box_y1, 0), int(left) + box_x2, int(top + height)],
                "label": [int(left) + label_left, int(top) + label_top, int(left) + label_left + label_width, int(top) + label_top + label_height],
            })
    return overlays


def overlay_mask(shape, overlays, margin, lines=True, labels=True):
    """Mask of the box lines and labels. The inside of a box is left out, so an animal in it is never touched.

    JPEG compression smears the strong overlay colors a few pixels outwards, which the margin covers.
    """
    mask = np.zeros(shape[:2], np.uint8)
    for overlay in overlays:
        x1, y1, x2, y2 = overlay["box"]
        inset = line_width // 2
        if lines:
            cv2.rectangle(mask, (x1 + inset, y1 + inset), (x2 - 1 - inset, y2 - 1 - inset), 1, line_width + 2 * margin)
        label_x1, label_y1, label_x2, label_y2 = overlay["label"]
        if labels:
            cv2.rectangle(mask, (label_x1 - margin, label_y1 - margin), (label_x2 + margin, label_y2 + margin), 1, -1)
    return mask


def overlay_rectangles(overlays, margin):
    """Outer rectangles of boxes and labels, for a quick test whether an overlay covers some area."""
    rectangles = []
    for overlay in overlays:
        for x1, y1, x2, y2 in (overlay["box"], overlay["label"]):
            rectangles.append((max(x1 - margin, 0), max(y1 - margin, 0), x2 + margin, y2 + margin))
    return rectangles


def index_frame(path_text):
    path = Path(path_text)
    timestamp, class_names = parse_name(path)
    overlays = [overlay for overlay in find_overlays(cv2.imread(path_text)) if overlay["class"] in class_names]
    found_classes = {overlay["class"] for overlay in overlays}
    return {
        "path": path_text,
        "time": timestamp.isoformat(),
        "overlays": overlays,
        "complete": all(name in found_classes for name in class_names),
    }


def build_index(folders, cache_path):
    if cache_path and cache_path.exists():
        return json.loads(cache_path.read_text())
    paths = sorted(str(path) for folder in folders for path in folder.glob("*.jpg"))
    with Pool() as pool:
        index = pool.map(index_frame, paths, chunksize=32)
    if cache_path:
        cache_path.write_text(json.dumps(index))
    return index


def spot_of(overlay):
    x1, y1, x2, y2 = overlay["box"]
    return overlay["class"], (x1 + x2) // 2 // spot_cell, (y1 + y2) // 2 // spot_cell, (x2 - x1) // spot_cell, (y2 - y1) // spot_cell


def find_static_spots(index):
    """A detection that returns at the same place and size on many days is a fixed object, not an animal."""
    days_per_spot = defaultdict(set)
    for entry in index:
        for overlay in entry["overlays"]:
            days_per_spot[spot_of(overlay)].add(entry["time"][:10])
    return {spot for spot, days in days_per_spot.items() if len(days) >= static_spot_min_days}


def seconds_of_day_apart(first, second):
    difference = abs((first.hour * 3600 + first.minute * 60 + first.second) - (second.hour * 3600 + second.minute * 60 + second.second))
    return min(difference, 86400 - difference)


def fill_from_other_frame(image, mask, entry, index_by_time):
    """Fill the masked pixels from another frame in which that part of the scene is free.

    The camera does not move, so a frame from the same days and a similar time of day shows the same scene
    in similar light. Of a few such frames the one that fits best on a ring around the mask is blended in
    (gradients of the other frame, colors of this frame at the border). Returns None when no frame fits.
    """
    time = datetime.fromisoformat(entry["time"])
    mask = mask.copy()
    mask[[0, -1], :] = 0
    mask[:, [0, -1]] = 0
    if not mask.any():
        return image
    wide_mask = cv2.dilate(mask, np.ones((19, 19), np.uint8))
    ring = (wide_mask > 0) & (mask == 0)
    neighbors = sorted(
        (
            other for other in index_by_time
            if other["complete"] and other["path"] != entry["path"] and abs(other["datetime"] - time) <= donor_time_window
        ),
        key=lambda other: seconds_of_day_apart(other["datetime"], time),
    )
    best_residual, best_donor = donor_max_residual, None
    n_tried = 0
    for other in neighbors:
        if any(wide_mask[y1:y2, x1:x2].any() for x1, y1, x2, y2 in overlay_rectangles(other["overlays"], blend_margin)):
            continue
        donor = cv2.imread(other["path"])
        gain = image[ring].mean(axis=0) / (donor[ring].mean(axis=0) + 1)
        residual = np.abs(np.clip(donor[ring] * gain, 0, 255) - image[ring]).mean()
        if residual < best_residual:
            best_residual, best_donor = residual, donor
        n_tried += 1
        if n_tried == donor_max_tries:
            break
    if best_donor is None:
        return None
    x, y, width, height = cv2.boundingRect(mask)
    return cv2.seamlessClone(best_donor, image, mask * 255, (x + width // 2, y + height // 2), cv2.NORMAL_CLONE)


def remove_overlay(entry, index_by_time, keep_box_content):
    """Returns the cleaned image, or None for an empty frame that no other frame can fill."""
    image = cv2.imread(entry["path"])
    if not keep_box_content:
        return fill_from_other_frame(image, overlay_mask(image.shape, entry["overlays"], blend_margin), entry, index_by_time)

    # The box lines touch the animal, which no other frame shows at the same place, so they are painted over
    # from their surroundings. Only the label, which lies outside the box, is taken from another frame.
    label_mask = overlay_mask(image.shape, entry["overlays"], blend_margin, lines=False)
    line_mask = overlay_mask(image.shape, entry["overlays"], inpaint_margin, labels=False)
    image = cv2.inpaint(image, line_mask, 5, cv2.INPAINT_TELEA)
    filled = fill_from_other_frame(image, label_mask, entry, index_by_time)
    if filled is None:
        filled = cv2.inpaint(image, label_mask, 5, cv2.INPAINT_TELEA)
    return filled


def sample_empty_frames(empty_entries, seed):
    """At most a few frames per day and hour, so that light and season are spread and bursts are not repeated."""
    by_day_and_hour = defaultdict(list)
    for entry in empty_entries:
        by_day_and_hour[entry["time"][:13]].append(entry)
    generator = random.Random(seed)
    sampled = []
    for _, entries in sorted(by_day_and_hour.items()):
        sampled += generator.sample(entries, min(frames_per_day_and_hour, len(entries)))
    return sampled


def output_name(entry):
    return datetime.fromisoformat(entry["time"]).strftime("%Y-%m-%d_%H-%M-%S") + ".jpg"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    pictures = Path.home() / "Pictures"
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--detections", type=Path, default=pictures / "detections")
    parser.add_argument("--true-positives", type=Path, default=pictures / "Summer Place Winter 26")
    parser.add_argument("--output", type=Path, default=repo_root / "nordic_animals")
    parser.add_argument("--index-cache", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    index = build_index([arguments.detections, arguments.true_positives], arguments.index_cache)
    detections = [entry for entry in index if Path(entry["path"]).parent == arguments.detections]
    animal_entries = [entry for entry in index if Path(entry["path"]).parent == arguments.true_positives]
    animal_times = {entry["time"] for entry in animal_entries}
    static_spots = find_static_spots(detections)

    empty_entries = [
        entry for entry in detections
        if entry["complete"]
        and entry["time"][:10] not in excluded_dates
        and entry["time"] not in animal_times
        and all(spot_of(overlay) in static_spots for overlay in entry["overlays"])
    ]
    sampled = sample_empty_frames(empty_entries, arguments.seed)
    print(f"{len(detections)} saved detections, {len(empty_entries)} are false positives on fixed objects, {len(sampled)} sampled")

    index_by_time = [dict(entry, datetime=datetime.fromisoformat(entry["time"])) for entry in detections]
    for folder_name, entries, keep_box_content in (("empty", sampled, False), ("animals", animal_entries, True)):
        folder = arguments.output / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        n_written = 0
        for entry in entries:
            cleaned = remove_overlay(entry, index_by_time, keep_box_content) if entry["complete"] else None
            if cleaned is None:
                if keep_box_content:
                    print(f"  overlay not removed, skipped: {Path(entry['path']).name}")
                continue
            cv2.imwrite(str(folder / output_name(entry)), cleaned, [cv2.IMWRITE_JPEG_QUALITY, 95])
            n_written += 1
        print(f"  {folder_name}: {n_written} of {len(entries)} written")


if __name__ == "__main__":
    main()
