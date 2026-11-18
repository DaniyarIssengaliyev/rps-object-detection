# Rock Paper Scissors Object Detection — YOLO26n

**Course:** CSCI1409 Deep Learning — Bonus Project: Train an Object Detector

Fine-tune a pretrained Ultralytics YOLO26n detector to identify three hand gestures: **Paper**, **Rock**, and **Scissors**. Compare a 25-epoch baseline against a 35-epoch controlled experiment, choose the final checkpoint by validation performance, then evaluate on a held-out test set.

## Dataset

- **Source:** [Roboflow Universe — Rock Paper Scissors, version 44](https://universe.roboflow.com/team-roboflow/rock-paper-scissors-detection/dataset/44)
- **Classes (IDs 0, 1, 2):** `Paper`, `Rock`, `Scissors`
- **Format:** YOLO annotations (Roboflow YOLO26 export), one `.txt` per image, each row `class_id x_center y_center width height` (normalized coordinates)
- **Images:** 825 total
- **Original Roboflow split:** Train 593 / Valid 154 / Test 78
- **After moving frames of two shared source videos into Train:** Train **601** / Valid **148** / Test **76**
- **Annotated objects:** 599 total; empty annotation files also occur and can represent background examples.
- **Dataset terms:** The original export's `data.yaml` declares `license: Private`. **Do not publicly upload the ZIP or images unless redistribution is permitted.** Obtain access to the source dataset and export your own version 44 copy.

For split integrity, frames from `Cool-Video-of-Ocean-Fish-Swimming` and `GMT20210709-195148_Recording_640x360` are kept in Train. SHA-256 did not find exact cross-split duplicates after adjustment. A dHash scan identified **199 candidate** cross-split near-duplicate pairs, which were *not* fully verified; possible data leakage is a limitation. The split correction is deterministic. Model and sampling seed: `42`.

## Files

```text
rps-detection/
├── RPS_Detection_Colab_Ready.ipynb   # Full notebook
├── README.md
├── requirements.txt
├── .gitignore
├── checkpoints/                   # Created after training (upload these)
│   ├── baseline_best.pt
│   └── experiment_35epochs_best.pt
├── runs/detect/results/training/  # Training logs/plots, generated locally
├── results/                       # Plots, validation and test scores
└── dataset/                       # Obtain separately; NOT in public GitHub
    └── Rock Paper Scissors/
        ├── data.yaml
        ├── train/images + train/labels
        ├── valid/images + valid/labels
        └── test/images + test/labels
```

## Run in Google Colab (clean runtime)

1. Obtain an **authorized copy** of the original Roboflow version 44 dataset ZIP in **YOLO26 format**. Keep its original `Rock Paper Scissors/` root folder and `data.yaml`.
2. Open `RPS_Detection_Colab_Ready.ipynb` in [Google Colab](https://colab.research.google.com/). Choose **Runtime → Change runtime type → GPU**.
3. Click **Runtime → Run all**. In the first setup cell, **upload your original ZIP** when prompted. The notebook installs fixed Python library versions, extracts the dataset, corrects paths, moves eight video frames, and checks its contents.
4. The 25-epoch and 35-epoch runs train from the **same `yolo26n.pt` pretrained weights**. Training happens if checkpoints are missing; running cells again in the same session reuses existing checkpoints.
5. It creates validation comparison graphs, selects the best model by validation **mAP@0.50**, then evaluates the selected model on Test (`conf=0.001`, NMS `iou=0.7`). Five test predictions and qualitative failure examples follow.
6. The last cell creates `checkpoints/` and downloads `rps_results_and_checkpoints.zip`. Download it **before** ending the Colab session. You can also view all results in notebook output.

**Note:** Colab uses CUDA if enabled; the original run used Apple MPS. Scores can vary slightly even with `seed=42`. In a fresh Colab run, the final choice is determined from its own validation results, not hard-coded to the original winning model.

## Run locally (VS Code / Mac)

1. Set up Python and install the dependencies: `python -m pip install -r requirements.txt`. PyTorch must be installed for your platform (the original run used PyTorch 2.14.0 and MPS).
2. Put your own downloaded `Rock Paper Scissors.zip` beside the notebook, **or** retain `dataset/Rock Paper Scissors/` from your existing project.
3. Open the notebook from the project root in VS Code and run the setup and exploration cells. It repairs paths only when needed and moves the same known video frames.
4. The notebook does **not** retrain existing models. If no checkpoints are present, set `TRAIN_MISSING_CHECKPOINTS = True` intentionally in the baseline training cell, and run both training cells; otherwise restore saved weights under `runs/detect/results/training/`.
5. Continue with validation, test, prediction, failure analysis, and the final checkpoint export cell.

## Controlled experiment

| Parameter | Baseline | Experiment 2 |
|---|---|---|
| Detector | YOLO26n pretrained | YOLO26n pretrained |
| Optimizer | SGD | SGD |
| Learning rate | 0.01 | 0.01 |
| Epochs | **25** | **35** |
| Image size | 640 | 640 |
| Batch size | 8 | 8 |
| Random seed | 42 | 42 |
| Device (original run) | Apple M5 / MPS | Apple M5 / MPS |
| Other training settings | Same | Same |

**Hypothesis:** Ten additional epochs may improve detection and bounding-box localization. Ultralytics' default learning-rate schedule also extends with training duration; that interaction is part of changing `epochs`.

## Original local results

| Validation metric | Baseline (25 epochs) | Experiment (35 epochs) |
|---|---:|---:|
| Precision | 98.9% | 98.5% |
| Recall | 95.3% | 96.7% |
| mAP@0.50 | 97.8% | 98.8% |
| mAP@0.50:0.95 | 66.2% | 66.5% |
| Training time | 14.79 min | 20.60 min |

**Decision:** The 35-epoch checkpoint performed slightly better on validation mAP@0.50 and Recall; the bounding-box localization improvement was small and training took longer.

The 35-epoch checkpoint was evaluated **once** on the held-out Test split:

| Test metric | Score |
|---|---:|
| Precision | 99.23% |
| Recall | 91.15% |
| mAP@0.50 | 95.97% |
| mAP@0.50:0.95 | 61.41% |

Test class Recall: Paper **95.0%**, Rock **95.2%**, Scissors **83.2%**. Three qualitative Scissors misses at `conf=0.25` are illustrated in the notebook. Potential reasons (hand orientation, pose, background and scale) are hypotheses, not confirmed causes. A useful next step is to collect more diverse Scissors examples, inspect potential near-duplicates, and improve bounding-box annotation quality.

## Settings, Output and Reproducibility

### Training Environment

The project was originally developed on a MacBook and then reproduced in Google Colab using a Tesla T4 GPU.

| Component | MacBook | Google Colab |
|---|---|---|
| Python | 3.14.7 | 3.13.16 |
| PyTorch | 2.14.0 | 2.14.1+cu130 |
| Ultralytics | 8.4.212 | 8.4.212 |
| Device | Apple MPS | NVIDIA Tesla T4 (CUDA) |

The same dataset, model, and training settings were used on both platforms. Small differences in results are expected because of different hardware.

### Training Configuration

- Model: YOLO26n (pretrained weights)
- Image size: 640 × 640
- Batch size: 8
- Optimizer: SGD
- Initial learning rate: 0.01
- Random seed: 42
- Baseline: 25 epochs
- Experiment: 35 epochs

Only the number of training epochs was changed between the two experiments.

### Validation Results (Google Colab)

| Metric | 25 Epochs | 35 Epochs |
|---|---:|---:|
| Precision | 97.44% | 96.43% |
| Recall | 94.44% | 93.93% |
| mAP@0.50 | 98.36% | 98.95% |
| mAP@0.50:0.95 | 66.58% | 65.86% |
| Training Time | 6.96 min | 9.90 min |

The 35-epoch model achieved the highest validation mAP@0.50, which was our primary selection metric. Although the baseline performed slightly better on some other metrics, we selected the 35-epoch model for final testing.

### Final Test Results

The selected 35-epoch model was evaluated on the test dataset once.

| Metric | Test Result |
|---|---:|
| Precision | 99.29% |
| Recall | 87.28% |
| mAP@0.50 | 94.72% |
| mAP@0.50:0.95 | 59.45% |

The test set contains 76 images and 64 annotated objects.

### Saved Outputs

Training results are saved in:

- `runs/detect/results/training/baseline/`
- `runs/detect/results/training/experiment_35epochs/`

Each training folder contains model weights, `args.yaml`, `results.csv`, and training plots.

Both best model checkpoints are exported to the `checkpoints/` folder.

Test predictions, evaluation results, and failure analysis images are saved in `results/evaluation/final_test/`.

### Evaluation Settings

- Validation and test confidence threshold: `0.001`
- Non-Maximum Suppression (NMS) IoU threshold: `0.7`
- Prediction visualization confidence threshold: `0.25`

The NMS IoU threshold is different from the IoU thresholds used to calculate mAP.

### Reproducibility

The Google Colab notebook provides the complete workflow, including dataset preparation, training, validation, model selection, testing, and visualization.

To reproduce the project, open the notebook in Google Colab, enable GPU acceleration, and upload the original Rock Paper Scissors dataset ZIP (version 44) when requested.

The dataset is marked as Private in the downloaded metadata, so it is not included in this public repository. Dataset usage and redistribution must follow the dataset owner's permissions.

The trained checkpoints can be included separately in the repository or provided through a GitHub Release. No API keys or private credentials are required in the published code.

### References

- [Ultralytics training documentation](https://docs.ultralytics.com/modes/train/)
- [Ultralytics configuration](https://docs.ultralytics.com/usage/cfg/)
- [Original dataset version 44 (Roboflow Universe)](https://universe.roboflow.com/team-roboflow/rock-paper-scissors-detection/dataset/44)
