**Inference Showcase**

This folder contains a quick inference showcase using the local `yolo_nordic_animals.pt` model. The checkpoint was trained in the Kaggle workflow and then copied into this repo for a lightweight local validation pass.

The goal is to keep the repo browseable and easy to review without publishing the full Kaggle training notebook yet. The local check focuses on a few sample images from `nordic_animals/`, a simple distribution summary built from the existing labels, and an easy way to inspect predicted boxes on real wildlife images.

Run the script to generate annotated images and a label-distribution summary:

```bash
python3 scripts/infer_showcase.py
```

Generated outputs go to `docs/infer_outputs/`.

**Overview:**

- **Model:** `yolo_nordic_animals.pt` (trained on a combined COCO subset + nordic animals / wildlife classes)
- **Script:** `scripts/infer_showcase.py` — runs inference on a few sample images in `nordic_animals/` and saves annotated images plus a dataset distribution plot to `docs/infer_outputs/`.
- **Current focus:** validation of local inference quality, class coverage, and repo-ready documentation before the Kaggle notebook is cleaned up and published.

**Run:**

```bash
pip install ultralytics pillow matplotlib
python3 scripts/infer_showcase.py
```

**Example outputs:**

![Sample 1](infer_outputs/2025-09-26%20%2011:43:32--.png)
![Sample 2](infer_outputs/2025-09-27%20%2008:32:10--.png)
![Sample 3](infer_outputs/2026-02-15%20%2013:03:08--.png)

**Dataset distribution:**

![Distribution plot](infer_outputs/dataset_distribution.png)

The distribution plot is intentionally lightweight and based on labels already present in `datasets/**`. It can be expanded later to reflect the full Kaggle preprocessing pipeline once the final training code and class remapping are finalized for publication.