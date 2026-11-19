# Rock Paper Scissors Object Detection — YOLO26n

**Course:** CSCI1409 Deep Learning  
**Project:** Bonus Project — Train an Object Detector

## 1. Project Overview

This project uses a pretrained YOLO26n model to detect three hand gestures: **Paper, Rock, and Scissors**.

The main goal is to fine-tune an object detection model, compare two training experiments, and evaluate the selected model on unseen test images.

Two experiments were conducted:

- **Baseline:** 25 training epochs
- **Experiment:** 35 training epochs

Both models were trained using the same dataset, pretrained weights, and hyperparameters. The only intentional change was the number of training epochs.

The model was selected using validation mAP@0.50 and evaluated on the test dataset.

The project was originally developed on a MacBook and reproduced in Google Colab using an NVIDIA Tesla T4 GPU.

## 2. Dataset

**Source:** [Rock Paper Scissors Detection — Roboflow Universe, Version 44](https://universe.roboflow.com/team-roboflow/rock-paper-scissors-detection/dataset/44)

The dataset contains images of people showing hand gestures in different environments.

### Classes

| Class ID | Class Name |
|---|---|
| 0 | Paper |
| 1 | Rock |
| 2 | Scissors |

### Dataset Statistics

The original dataset contains 825 images. After checking the data splits, eight images from two shared video sources were moved into the training set to reduce possible data leakage.

| Split | Images | Annotated Objects |
|---|---:|---:|
| Train | 601 | 430 |
| Validation | 148 | 105 |
| Test | 76 | 64 |
| **Total** | **825** | **599** |

Some images have empty annotation files and are treated as background examples.

### Annotation Format

The dataset uses the YOLO bounding-box annotation format.

Each annotation is stored in a `.txt` file with the following structure:

`class_id x_center y_center width height`

The bounding-box coordinates are normalized between 0 and 1.

The notebook includes examples of images with ground-truth bounding boxes and checks for missing or incorrect annotations.

### Data Split and Quality Checks

The dataset was checked for missing labels, corrupted files, and possible duplicate images.

Frames from the following video sources were moved into the training split:

- `Cool-Video-of-Ocean-Fish-Swimming`
- `GMT20210709-195148_Recording_640x360`

After this adjustment, SHA-256 checks did not identify exact duplicates across splits.

A dHash analysis identified 199 potential near-duplicate pairs across different splits. These candidates were not fully verified manually, so possible similarity between training and evaluation images remains a limitation.

The split correction is deterministic, and the training random seed was set to `42`.

### Dataset Usage

The downloaded Roboflow export contains `license: Private` in its `data.yaml`.

For this reason, the dataset images and ZIP archive are not included in the public GitHub repository. Users must obtain an authorized copy of version 44 from the original source. Redistribution requires permission from the dataset owner.

## 3. Project Structure

The main project files are:

- `RPS_Detection_Colab.ipynb` — complete Google Colab notebook
- `README.md` — project documentation
- `requirements.txt` — required Python libraries
- `.gitignore` — files excluded from Git
- `checkpoints/` — trained model weights
- `results/` — evaluation results, plots, and prediction examples

The `checkpoints/` folder contains:

- `baseline_best.pt` — model trained for 25 epochs
- `experiment_35epochs_best.pt` — model trained for 35 epochs

Training also generates configuration files, logs, and learning curves in:

- `runs/detect/results/training/baseline/`
- `runs/detect/results/training/experiment_35epochs/`

The original dataset must be downloaded separately.

## 4. How to Run in Google Colab

[**Open in Google Colab**](https://colab.research.google.com/github/DaniyarIssengaliyev/rps-object-detection/blob/main/RPS_Detection_Colab.ipynb)

### Instructions

1. Open the notebook using the link above.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Obtain the original Rock Paper Scissors dataset ZIP, version 44, in Roboflow YOLO26 format.
4. Click **Runtime → Run all**.
5. Upload the original dataset ZIP when the notebook requests it.
6. The notebook will install dependencies, extract the dataset, correct paths, and prepare the data splits.
7. It will train the baseline and experimental models if their checkpoints are missing.
8. The models will be compared on the validation set, and the better model by mAP@0.50 will be selected for testing.
9. The notebook will display predictions, analyze failure cases, and export the model checkpoints and results.

At the end, download the generated `rps_results_and_checkpoints.zip` archive before the Colab session ends.

The notebook uses existing checkpoints when available instead of retraining them unnecessarily.

The final test evaluation should only be performed after choosing the model using validation results. Test results should not be used to change training settings.

### Dependencies

The project uses Python, PyTorch, Ultralytics YOLO, pandas, matplotlib, Pillow, and PyYAML.

The exact versions of the main Python libraries are specified in `requirements.txt`.

Google Colab uses its installed CUDA-compatible PyTorch version.

## 5. Training Configuration

The pretrained **YOLO26n** detector was selected because it is lightweight and suitable for a small object detection dataset.

| Parameter | Baseline | Experiment |
|---|---|---|
| Model | YOLO26n | YOLO26n |
| Initial weights | `yolo26n.pt` | `yolo26n.pt` |
| Optimizer | SGD | SGD |
| Learning rate | 0.01 | 0.01 |
| Image size | 640 × 640 | 640 × 640 |
| Batch size | 8 | 8 |
| Epochs | **25** | **35** |
| Random seed | 42 | 42 |
| Device | Tesla T4 | Tesla T4 |

**Hypothesis:** Increasing the number of epochs from 25 to 35 may improve object detection accuracy by allowing the model to learn more from the training data.

All other explicitly configured training settings were kept the same. The training learning-rate schedule also depends on the total number of epochs.

## 6. Training Environment

| Component | MacBook | Google Colab |
|---|---|---|
| Python | 3.14.7 | 3.13.16 |
| PyTorch | 2.14.0 | 2.14.1+cu130 |
| Ultralytics | 8.4.212 | 8.4.212 |
| Device | Apple MPS | NVIDIA Tesla T4 (CUDA) |

The experiments were first performed locally and then repeated in Google Colab.

The final results presented below are from the **Google Colab experiments**.

Small differences between MacBook and Colab results are expected because the hardware and computation backends are different.

## 7. Validation Results

Both models were evaluated on the same validation dataset containing 148 images and 105 annotated objects.

| Metric | Baseline (25 Epochs) | Experiment (35 Epochs) |
|---|---:|---:|
| Precision | **97.44%** | 96.43% |
| Recall | **94.44%** | 93.93% |
| mAP@0.50 | 98.36% | **98.95%** |
| mAP@0.50:0.95 | **66.58%** | 65.86% |
| Training Time | **6.96 min** | 9.90 min |

### Comparison and Model Selection

The 35-epoch model achieved a higher validation mAP@0.50:

- Baseline: 98.36%
- Experiment: 98.95%
- Improvement: **0.59 percentage points**

However, the baseline achieved slightly higher Precision, Recall, and mAP@0.50:0.95.

The longer experiment also required more training time.

We selected the **35-epoch model** because mAP@0.50 was our primary model selection metric.

The improvement was relatively small, which suggests that additional epochs did not provide a large overall performance increase.

## 8. Final Test Evaluation

After selecting the 35-epoch model using validation results, its best checkpoint was evaluated once on the held-out test dataset.

The test set contains **76 images and 64 annotated objects**.

### Final Test Results

| Metric | Result |
|---|---:|
| Precision | **99.29%** |
| Recall | **87.28%** |
| mAP@0.50 | **94.72%** |
| mAP@0.50:0.95 | **59.45%** |

### Class Recall

| Class | Recall |
|---|---:|
| Paper | 93.6% |
| Rock | 90.5% |
| Scissors | 77.8% |

### Metric Explanation

**Precision** measures how many predicted detections are correct.

**Recall** measures how many actual objects are successfully detected.

**mAP@0.50** measures detection performance using an Intersection over Union (IoU) threshold of 0.50.

**mAP@0.50:0.95** measures detection performance across several IoU thresholds from 0.50 to 0.95. It is a stricter evaluation metric.

### Evaluation Settings

- Validation and test confidence threshold: `0.001`
- Non-Maximum Suppression (NMS) IoU threshold: `0.7`
- Prediction visualization confidence threshold: `0.25`

The NMS IoU threshold is different from the IoU thresholds used to calculate mAP.

### Test Result Interpretation

The model achieved very high Precision, indicating that most of its predicted detections were correct.

However, Recall was lower, especially for the Scissors class.

This means the model sometimes failed to detect hand gestures that were present in the images.

## 9. Prediction Visualization

The notebook includes five test images showing:

- Predicted bounding boxes
- Predicted class names
- Confidence scores

The selected examples include all three classes: Paper, Rock, and Scissors.

The visualization is saved as:

`results/evaluation/final_test/test_predictions_5.png`

These examples demonstrate how the trained model detects different hand gestures in unseen images.

## 10. Failure Analysis

The notebook analyzes cases where the model failed to detect an annotated object.

At a prediction confidence threshold of `0.25`, four test images contained missed detections. Three examples were selected for visual analysis.

### Failure Case 1 — Rock

The model failed to detect a closed fist.

A possible reason is the hand orientation and lighting conditions. The position of the fist may also make detection more difficult.

### Failure Case 2 — Scissors

The model failed to detect a Scissors gesture.

The hand occupies a relatively small area of the image, making the gesture harder to recognize.

### Failure Case 3 — Scissors

The model missed another Scissors gesture.

Possible reasons include the viewing angle, background, and variation in hand appearance.

These are possible explanations based on visual inspection, not experimentally confirmed causes.

The failure analysis image is saved as:

`results/evaluation/final_test/failure_analysis.png`

### Possible Improvements

Future improvements could include collecting more examples of Scissors gestures, increasing the variety of hand positions and backgrounds, and improving bounding-box annotation quality.

It would also be useful to review potential near-duplicate images across data splits.

## 11. Limitations

The project has several limitations.

**Small dataset:** The model was trained on only 825 images, which may limit its ability to generalize to different environments.

**Class differences:** Scissors had the lowest Recall in the final test evaluation.

**Missed detections:** The model sometimes failed to recognize gestures, especially when hands were small or appeared in unusual positions.

**Possible data leakage:** The near-duplicate analysis identified candidate similar images across data splits that were not all manually verified.

**Hardware differences:** Training results may vary between Apple MPS and NVIDIA CUDA, even when the same training settings and random seed are used.

**Dataset access:** The downloaded export is marked Private, so the original images are not redistributed through this repository.

## 12. Saved Results and Reproducibility

The project saves both trained checkpoints, evaluation results, and prediction examples.

The training output directories contain the Ultralytics configuration and logs generated during training, including:

- `args.yaml` — training configuration
- `results.csv` — training and validation metrics by epoch
- `results.png` — learning curves
- `weights/best.pt` — selected validation checkpoint

The notebook also saves validation comparison plots and final test evaluation results.

To reproduce the experiment, use the Google Colab notebook, download an authorized copy of the original dataset, and follow the notebook instructions.

No private credentials or API keys are included in the project.

## 13. Use of AI Tools

ChatGPT was used to assist with planning the workflow, writing and debugging parts of the Python notebook, explaining evaluation metrics, and improving the project documentation.

The experiments were executed and their outputs reviewed by the student. The final project and its results are presented as part of the individual Deep Learning assignment.

## 14. Conclusion

This project successfully fine-tuned a pretrained YOLO26n object detector for three Rock Paper Scissors hand gestures.

Two training configurations were compared using 25 and 35 epochs.

The 35-epoch model achieved the highest validation mAP@0.50 of **98.95%** and was selected for final testing.

On the held-out test dataset, it achieved:

- **Precision:** 99.29%
- **Recall:** 87.28%
- **mAP@0.50:** 94.72%
- **mAP@0.50:0.95:** 59.45%

The results show that YOLO26n can detect the three hand gestures with high accuracy on this dataset, although some objects are still missed.

## References

- [Ultralytics Training Documentation](https://docs.ultralytics.com/modes/train/)
- [Ultralytics Configuration](https://docs.ultralytics.com/usage/cfg/)
- [Rock Paper Scissors Dataset — Roboflow Version 44](https://universe.roboflow.com/team-roboflow/rock-paper-scissors-detection/dataset/44)
