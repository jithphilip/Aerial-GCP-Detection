---

# Aerial Ground Control Point (GCP) Detection

An end-to-end Computer Vision pipeline designed to locate and classify Ground Control Point (GCP) markers in high-resolution aerial drone imagery.

This project implements a **Multi-Task Learning (MTL)** neural network in PyTorch to simultaneously perform continuous coordinate regression (finding the exact X, Y pixel location) and categorical classification (identifying the marker's shape).

## 1. Architecture & Methodology

To maximize computational efficiency and leverage shared visual context, this pipeline utilizes a unified multi-task architecture rather than training two distinct models.

* **Backbone:** A pre-trained **ResNet18** model acts as the primary feature extractor, compressing the high-dimensional pixel space into a dense 512-dimensional feature vector.
* **Regression Head:** A custom Multi-Layer Perceptron (MLP) maps the 512 features to a continuous spatial domain. A Sigmoid activation bounds the output between 0.0 and 1.0, representing normalized image coordinates.
* **Classification Head:** A parallel MLP takes the exact same feature vector and outputs raw logits corresponding to three expected marker shapes (Cross, Square, L-Shaped).

## 2. Training Strategy & Loss Balancing

The network optimizes a combined loss metric using the Adam optimizer:

* **Coordinate Regression:** Mean Squared Error (MSE) Loss.
* **Shape Classification:** Cross-Entropy Loss.

**Gradient Balancing:** Because MSE outputs tiny fractional errors (e.g., 0.01) while Cross-Entropy outputs larger logarithmic errors (e.g., 1.5), a static scalar multiplier of `10x` was applied to the regression loss during training. This prevents "Gradient Domination" by the classification head and forces the ResNet backbone to learn geometric localization features.

## 3. Data Engineering & EDA Mitigations

Exploratory Data Analysis (EDA) on the provided dataset revealed several real-world inconsistencies that required active mitigation within the pipeline:

* **Dirty Data & Missing Labels:** Several entries in the ground-truth JSON lacked required keys (e.g., missing `verified_shape`). I engineered a robust PyTorch `Dataset` class that actively evaluates and filters out corrupted annotation records during initialization, preventing fatal runtime exceptions.
* **Severe Class Imbalance:** Statistical analysis showed an extreme imbalance: ~89% Square, ~11% Cross, and 0% L-Shaped. To prevent the model from blindly predicting the majority class, I applied inverse-frequency **Class Weights** (Cross: 9.5, Square: 1.1, L-Shaped: 0.0) directly to the Cross-Entropy loss function.
* **Undocumented Resolution Variance:** Visual EDA proved that image dimensions varied significantly, frequently exceeding the documented 2048x1365 baseline. The data loader dynamically queries each image's exact dimensions via OpenCV, mathematically scaling targets into a normalized 0.0 to 1.0 space. This allows the pipeline to natively adapt to variable resolutions without crashing.

## 4. Limitations & Future Improvements

* **Information Compression:** Because the physical GCP markers take up an incredibly small percentage of the total image area, aggressively resizing 4000x3000 resolution images down to 512x512 results in severe visual signal loss.
* **Next Iteration:** A future iteration of this pipeline would abandon full-image resizing in favor of **Random Jitter Cropping** during training, or a **Coarse-to-Fine (Two-Stage)** detection pipeline, to preserve the high-resolution geometric features of the markers.

## 5. Instructions for Inference

To reproduce the `predictions.json` file on the hidden test set, run the final cell in **"main.ipynb"** jupyter notebook under the **"Inference & Deliverable Generation"** section or use the 'prediction_helper.py' for generating a prediction file.

**Prerequisites:**

1. Ensure the trained weights file (`gcp_detector_weights_balanced.pth`) is in the root directory.
2. Place the unlabelled test images inside a directory named `test_dataset`.

**Running the Script:**
Run the inference script/cell. The script utilizes Python's `glob` module with `recursive=True` to automatically traverse any deeply nested subdirectories. It processes the images, mathematically un-scales the coordinate predictions back to their native resolutions, handles OS-specific file duplication quirks, and outputs a formatted `predictions.json` dictionary.

---
