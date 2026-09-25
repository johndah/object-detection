# Nordic animal detection

This repo is using a frozen pretrained YOLO26 model as the baseline for Nordic wildlife detection. The next step is to fine-tune on deer classes and related Nordic animal classes.

The current model is useful even when it makes imperfect predictions: false positives can still land on the right animal region, which helps us identify what the network already understands before the deer-specific training step, and to get some detections.

## Current frozen YOLO26 examples

![Dog detection](docs/infer_outputs/2025-09-26%20%2011:43:32-dog.jpg)
![Nordic wildlife detection](docs/infer_outputs/2026-02-15%20%2013:03:08--.png)
![Late-season wildlife inference](docs/infer_outputs/2026-04-27%20%2001:45:59--.png)

## Nordic animal data starting point

These are some data distribution, samples and mappings as preparation for the next fine-tuning pass for Nordic-animal detection.

![Training samples](docs/training_samples.png)
![Validation samples](docs/validation_samples.png)
![Dataset distribution](docs/distribution.png)
![Wild deer mapping](docs/wild_deer_mapping.png)

