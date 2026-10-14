# computer-vision-project
# Road Scene Segmentation

A semantic segmentation project for understanding road scenes using deep learning. The model assigns a class label to each pixel, separating regions such as roads, sidewalks, buildings, vehicles, pedestrians, and other objects in a street image.

The project covers the full segmentation workflow, from dataset preparation and mask processing to model training, evaluation, and inference on custom images.

## Problem & Approach

The task is **11-class semantic segmentation** using the CamVid dataset. Given an RGB image, the model predicts a pixel-level segmentation mask.

The implementation uses U-Net with a ResNet34 encoder pretrained on ImageNet. The encoder extracts visual features, while the decoder produces spatial predictions for the 11 classes.

## Technical Implementation

- **Framework:** PyTorch
- **Model:** U-Net with an ImageNet-pretrained ResNet34 encoder
- **Dataset:** CamVid
- **Input:** RGB road-scene images
- **Output:** Pixel-wise class predictions for 11 semantic classes

### Data preparation

The dataset is divided into 367 training images, 101 validation images, and 233 test images. Images and masks are resized to 512 × 384 pixels.

Training augmentation includes random horizontal flips, rotations up to ±15°, brightness adjustments, and contrast adjustments. Images are normalized using ImageNet mean and standard deviation. Masks use nearest-neighbor interpolation to preserve discrete class labels, with label 255 excluded from the loss and pixel-accuracy calculations.

### Training

The training objective combines Cross-Entropy Loss and Dice Loss. Adam is used with separate learning rates for the encoder (1e-4) and decoder and segmentation head (1e-3), along with weight decay of 1e-4.

The training loop evaluates the model on the validation set, tracks mIoU and pixel accuracy, adjusts the learning rate using `ReduceLROnPlateau`, and saves a checkpoint when validation mIoU improves.

### Evaluation

The model is evaluated using mean Intersection over Union (mIoU), pixel accuracy, and per-class IoU. Qualitative predictions are also compared with ground-truth masks, and a confusion matrix is generated to inspect pixel-level classification errors.

## Results & Analysis

### Test performance

| Metric | Result |
|---|---:|
| Test mIoU | 57.64% |
| Test Pixel Accuracy | 88.42% |
| Best recorded validation mIoU | 64.48% |

Pixel accuracy is relatively high, but mIoU is notably lower. This difference suggests that correct predictions on large image regions can contribute substantially to pixel accuracy while performance remains weaker across some semantic classes. The two metrics therefore provide complementary views of segmentation quality.

### Per-class IoU

| Class | IoU |
|---|---:|
| Sky | 92.36% |
| Road | 91.72% |
| Building | 76.84% |
| Car | 75.07% |
| Sidewalk | 74.57% |
| Tree | 71.53% |
| Pedestrian | 40.53% |
| Bicyclist | 37.39% |
| SignSymbol | 34.80% |
| Fence | 31.91% |
| Pole | 7.36% |

The results show a clear difference between large scene regions and smaller objects. Sky and Road achieve IoU above 91%, while Building, Car, Sidewalk, and Tree also exceed 70%. These results indicate that the model can identify several major scene components reasonably well on the test set.

Performance drops on Pedestrian, Bicyclist, SignSymbol, and Fence. Pole is the most difficult class, with an IoU of only 7.36%. This is an important limitation for road-scene understanding: high overall pixel accuracy does not guarantee reliable segmentation of small objects that may still matter for downstream perception tasks.

The next step should be to inspect class-specific errors and verify image-mask alignment before making further architectural changes. The current results identify where the model struggles, but they do not establish the cause of those errors.

## Limitations & Next Steps

- **Uneven class performance:** Small or visually ambiguous objects remain difficult to segment, particularly poles, fences, and traffic-related signs.
- **Label alignment:** Visual inspection in the notebook identified cases where labels appear misaligned with objects. The image-mask pairs and resizing pipeline should be checked before drawing stronger conclusions about model quality.
- **Limited experimental evidence:** The notebook reports one set of test metrics. There is no controlled comparison against another architecture or a baseline, so the results do not establish that this model is the best choice.
- **Further evaluation:** After validating the annotations, useful next steps include inspecting the confusion matrix, reviewing per-class prediction errors, and comparing the current model with an alternative training configuration.

## Inference

The notebook includes an inference workflow for uploading a custom RGB image, applying the same preprocessing used during evaluation, predicting a segmentation mask, and resizing the output to the original image dimensions.

The visualization displays the original image alongside the predicted segmentation map, making it possible to inspect the model's output on images outside the test set. Since these custom images do not have ground-truth masks, the visual results are qualitative and should not be interpreted as quantitative accuracy measurements.

## How to Run

The project was developed in Google Colab and uses Google Drive to store the dataset, training history, and model checkpoint.

1. Open the notebook in Google Colab.
2. Install the required packages:

   ```bash
   pip install torchmetrics segmentation-models-pytorch opencv-python
   ```

3. Mount Google Drive and configure `PROJECT_DIR` to point to your project directory.
4. Run the dataset preparation and preprocessing cells. The notebook downloads CamVid from the repository configured in the code if the dataset directory does not already exist.
5. Run the model setup and training cells. To train from scratch, ensure that the resume setting and checkpoint handling are configured accordingly.
6. Run the evaluation cells to obtain test metrics, per-class IoU, qualitative comparisons, and the confusion matrix.
7. Run the final inference cell to upload an image and visualize its predicted segmentation mask.

A CUDA-enabled runtime is recommended for training. The notebook automatically selects CUDA when available and otherwise falls back to CPU.

## Summary

This project implements a complete semantic segmentation workflow and evaluates performance beyond a single aggregate metric. The results show strong segmentation of large road-scene regions, but substantially weaker performance on smaller objects. Addressing annotation alignment and class-specific errors is the most useful next step before claiming broader reliability for real-world road scenes.
