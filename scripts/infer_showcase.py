#!/usr/bin/env python3
"""Run a trained model on our own camera frames and save the pictures with its detections.

The boxes are drawn like in the notebook: labeled animals as filled boxes, detections as outlines with
their confidence, in the class colors of the notebook. The frames are private, so the pictures are
written into the local nordic_animals folder.

Run: python scripts/infer_showcase.py path/to/best.pt
"""
import argparse
from collections import OrderedDict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Polygon
from matplotlib.transforms import ScaledTranslation
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.ops import xywhn2xyxy

repo_root = Path(__file__).resolve().parents[1]


def keep_label_inside_image(ax, label_text):
    """Moves a label that reaches outside the image back inside, where it would otherwise be cut off."""
    renderer = ax.figure.canvas.get_renderer()
    label_text.update_bbox_position_size(renderer)
    label_extent = label_text.get_bbox_patch().get_window_extent(renderer).transformed(ax.transData.inverted())

    (x_left, x_right), (y_top, y_bottom) = sorted(ax.get_xlim()), sorted(ax.get_ylim())
    x_shift = max(x_left - label_extent.xmin, 0) + min(x_right - label_extent.xmax, 0)
    y_shift = max(y_top - label_extent.ymin, 0) + min(y_bottom - label_extent.ymax, 0)

    x, y = label_text.get_position()
    label_text.set_position((x + x_shift, y + y_shift))


def attach_label_above_box(ax, label_text, x_min, y_min):
    """Places a label so that it stands on the top line of its box, flush with the left corner."""
    label_text.set_position((x_min, y_min))
    label_text.set_horizontalalignment('left')
    label_text.set_verticalalignment('bottom')

    # The label gets a padding around its text. Moving the text by that padding puts the corner of the
    # padded label on the corner of the box, whatever the size of the figure
    padding = label_text.get_bbox_patch().get_boxstyle().pad * label_text.get_fontsize() / 72
    label_text.set_transform(ax.transData + ScaledTranslation(padding, padding, ax.figure.dpi_scale_trans))


def attach_label_below_box(ax, label_text, x_max, y_max):
    """Places a label so that it hangs from the bottom line of its box, flush with the right corner."""
    label_text.set_position((x_max, y_max))
    label_text.set_horizontalalignment('right')
    label_text.set_verticalalignment('top')

    padding = label_text.get_bbox_patch().get_boxstyle().pad * label_text.get_fontsize() / 72
    label_text.set_transform(ax.transData + ScaledTranslation(-padding, -padding, ax.figure.dpi_scale_trans))


class BoxWithLabel(Polygon):
    """A box and the label standing on its top line as one shape, so that outline and fill run around both."""

    def __init__(self, box, label_text, **kwargs):
        self.box = box
        self.label_text = label_text
        x_min, y_min, x_max, y_max = box
        super().__init__([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)], **kwargs)

    # The label has a fixed size on the page, so its size in the image is only known when the figure is drawn
    def draw(self, renderer):
        x_min, y_min, x_max, y_max = self.box
        box_corners = [(x_max, y_min), (x_max, y_max), (x_min, y_max), (x_min, y_min)]

        text_extent = self.label_text.get_window_extent(renderer)
        padding = renderer.points_to_pixels(
            self.label_text.get_bbox_patch().get_boxstyle().pad * self.label_text.get_fontsize()
        )
        label_corners = self.axes.transData.inverted().transform([
            (text_extent.x0 - padding, text_extent.y0 - padding),
            (text_extent.x1 + padding, text_extent.y1 + padding),
        ])
        label_left, label_right = sorted(label_corners[:, 0])
        label_top, label_bottom = sorted(label_corners[:, 1])

        # A label that was moved into the image no longer stands on the box, and the box is drawn alone
        stands_on_box = abs(label_bottom - y_min) < 0.5 * (label_bottom - label_top)
        label_outline = [
            (label_left, y_min), (label_left, label_top), (label_right, label_top), (label_right, y_min)
        ]

        self.set_xy(label_outline + box_corners if stands_on_box else box_corners)
        super().draw(renderer)


# Same drawing as plot_boxes in the notebook, with the class names and colors passed in
def plot_boxes(ax, image, boxes, labels, class_names, class_colors, prediction_confidences=None):
    ax.imshow(image)

    for i, label in enumerate(labels):
        x_min, y_min, x_max, y_max = boxes[i]
        width, height = x_max - x_min, y_max - y_min

        class_label = class_names[int(label)]
        class_color = class_colors[int(label)]

        legend = class_label
        transparency = .4
        # Faint enough that a small animal stays visible under it
        ground_truth_fill = .2
        line_width = 3
        bounding_box = plt.Rectangle((x_min, y_min),
                                     width, height,
                                     linewidth=line_width,
                                     label=legend)
        label_text = ax.text(x_min+4, y_min-5, class_label, clip_on=False)
        label_text.set_bbox(dict(facecolor=to_rgba(class_color, alpha=transparency)))

        if prediction_confidences is not None:

            confidence_label = f' {prediction_confidences[i]:.0%}'

            bounding_box.set_edgecolor(class_color)
            bounding_box.set_facecolor('none')
            bounding_box.set_label('Predicted ' + legend)

            label_text.set_text(class_label + confidence_label)

            label_box = label_text.get_bbox_patch()
            label_box.set_edgecolor(class_color)
            label_box.set_facecolor('none')
            label_box.set_linewidth(line_width)
            attach_label_below_box(ax, label_text, x_max, y_max)
        else:
            # Box and label are drawn as one shape: the dashed line runs around both and the fill is unbroken
            label_box = label_text.get_bbox_patch()
            label_box.set_edgecolor('none')
            label_box.set_facecolor('none')
            attach_label_above_box(ax, label_text, x_min, y_min)

            bounding_box = BoxWithLabel(
                boxes[i],
                label_text,
                linewidth=line_width,
                linestyle='--',
                edgecolor=class_color,
                facecolor=to_rgba(class_color, alpha=ground_truth_fill),
                label='Ground truth ' + legend,
            )

        keep_label_inside_image(ax, label_text)

        ax.add_patch(bounding_box)

    ax.set_xticks([])
    ax.set_yticks([])


def add_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    legend_entries = OrderedDict(sorted(zip(labels, handles), key=lambda entry: entry[0]))

    if legend_entries:
        ax.legend(legend_entries.values(), legend_entries.keys(), loc="upper right")


def read_labeled_boxes(label_path, label_class_names, model_class_ids, image_size):
    """Class ids of the model and pixel boxes of the labeled animals. Empty when the frame has no label."""
    if not label_path.is_file():
        return [], np.zeros((0, 4))

    label_rows = [line.split() for line in label_path.read_text().splitlines()]
    boxes = xywhn2xyxy(
        np.array([row[1:] for row in label_rows], dtype=float).reshape(-1, 4),
        w=image_size[0],
        h=image_size[1],
    )

    return [model_class_ids[label_class_names[int(row[0])]] for row in label_rows], boxes


def draw_frame(ax, frame_path, model, labels_folder, label_class_names, class_colors, confidence, image_size):
    image = Image.open(frame_path).convert("RGB")
    class_names = [model.names[class_id] for class_id in sorted(model.names)]
    model_class_ids = {name: class_id for class_id, name in model.names.items()}

    labeled_class_ids, labeled_boxes = read_labeled_boxes(
        labels_folder / f"{frame_path.stem}.txt", label_class_names, model_class_ids, image.size
    )
    plot_boxes(ax, image, labeled_boxes, labeled_class_ids, class_names, class_colors)

    prediction = model.predict(image, imgsz=image_size, conf=confidence, verbose=False)[0]
    plot_boxes(
        ax,
        image,
        prediction.boxes.xyxy.cpu().numpy(),
        prediction.boxes.cls.cpu().numpy(),
        class_names,
        class_colors,
        prediction.boxes.conf.cpu().numpy(),
    )

    add_legend(ax)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("weights", type=Path, nargs="?", default=repo_root / "animals.pt")
    parser.add_argument("--confidence", type=float, default=0.40)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--frames", type=Path, default=repo_root / "nordic_animals" / "animals")
    parser.add_argument("--labels", type=Path, default=repo_root / "nordic_animals" / "labels")
    parser.add_argument("--output", type=Path, default=repo_root / "nordic_animals" / "predictions")
    arguments = parser.parse_args()

    model = YOLO(str(arguments.weights))
    class_colors = plt.colormaps["viridis"](np.linspace(0, 1, len(model.names)))
    classes_file = arguments.labels / "classes.txt"
    label_class_names = classes_file.read_text().split() if classes_file.is_file() else []

    plt.style.use("dark_background")
    arguments.output.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(
        path for path in arguments.frames.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    for frame_path in frame_paths:
        fig, ax = plt.subplots(figsize=(10, 7.5))
        draw_frame(
            ax, frame_path, model, arguments.labels, label_class_names, class_colors,
            arguments.confidence, arguments.image_size,
        )
        ax.set_title(frame_path.stem)
        fig.tight_layout()
        fig.savefig(arguments.output / f"{frame_path.stem}.jpg", dpi=110)
        plt.close(fig)

    print(f"Saved {len(frame_paths)} pictures to {arguments.output}")


if __name__ == "__main__":
    main()
