#!/usr/bin/env python3
"""Run quick inference with the local `yolo_nordic_animals.pt` model
and save annotated images plus a simple label-distribution plot.
"""
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    model_path = root / "yolo_nordic_animals.pt"
    images_dir = root / "nordic_animals"
    out_dir = root / "docs" / "infer_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_images = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])[:3]

    if not sample_images:
        print("No sample images found in nordic_animals/", file=sys.stderr)
        return

    try:
        from ultralytics import YOLO
    except Exception:
        print("The 'ultralytics' package is required. Install with: pip install ultralytics", file=sys.stderr)
        raise

    model = YOLO(str(model_path))

    saved = []
    for img_path in sample_images:
        print(f"Running inference on {img_path.name}")
        results = model(str(img_path), imgsz=640, conf=0.25)
        # results[0].plot() returns an annotated image array
        annotated = results[0].plot()

        from PIL import Image
        im = Image.fromarray(annotated)
        out_path = out_dir / img_path.name
        im.save(out_path)
        saved.append(out_path)

    # Attempt a simple distribution plot using label files under datasets
    try:
        import matplotlib.pyplot as plt
        import collections
        from hey import classes as CLASS_NAMES

        counts = collections.Counter()

        labels_root = root / "datasets"
        for txt in labels_root.rglob("*.txt"):
            for line in txt.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                try:
                    cls = int(float(parts[0]))
                except Exception:
                    continue
                counts[cls] += 1

        if counts:
            ids = sorted(counts)
            names = [CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i) for i in ids]
            values = [counts[i] for i in ids]

            plt.style.use("dark_background")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(range(len(ids)), values, color="tab:cyan")
            ax.set_xticks(range(len(ids)))
            ax.set_xticklabels([f"{i}: {n}" for i, n in zip(ids, names)], rotation=45, ha="right")
            ax.set_title("Label instance counts (datasets/**/labels)")
            fig.tight_layout()
            dist_path = out_dir / "dataset_distribution.png"
            fig.savefig(dist_path)
            saved.append(dist_path)

    except Exception as exc:
        print("Skipping distribution plot (missing data or libs):", exc)

    print("Saved:")
    for p in saved:
        print(p)


if __name__ == "__main__":
    main()
