**Inference Showcase**

This folder contains a quick inference showcase using the local `yolo_nordic_animals.pt` model.

Run the script to generate annotated images and a simple distribution plot:

```bash
python3 scripts/infer_showcase.py
```

Generated outputs are placed in `docs/infer_outputs/`.

**Overview:**

- **Model:** `yolo_nordic_animals.pt` (pretrained on a combined COCO subset + nordic animals)
- **Script:** `scripts/infer_showcase.py` — runs inference on a few sample images in `nordic_animals/` and writes annotated images plus a dataset distribution plot to `docs/infer_outputs/`.

**Run:**

```bash
pip install ultralytics pillow matplotlib
python3 scripts/infer_showcase.py
```

Outputs: `docs/infer_outputs/` contains annotated images and `dataset_distribution.png` when available.