# UE24CS352A Artist Identification

This project predicts one of 15 artists from a painting image. It contains the original classical image-feature pipeline and an ImageNet-pretrained CNN pipeline based on the Chen & Deng comparison.

## Dataset and fixed split

The dataset contains 7,462 already resized 256x256 images in `data/images/<artist>/`. The split is always the stratified 80/10/10 split implemented by `src/common.py::split_data`, with `random_state=42` and 747 test images. The test set is reserved for the final evaluation.

## Final results

| Model | Test top-1 | Test top-3 | Macro-F1 |
|---|---:|---:|---:|
| Classical SVM | 64.7% | 85.1% | 0.646 |
| Frozen ResNet-50 + LogReg C=0.1 | 79.786% | 93.039% | 0.797715 |
| Frozen ConvNeXt-Tiny + LogReg C=0.1 | 83.133% | 94.913% | 0.830005 |
| Fine-tuned ResNet-50, epoch 13 | **87.952%** | **95.582%** | **0.879041** |
| Paper SVM | 68.1% | - | - |
| Paper CNN | 74.7% | - | - |

The fine-tuned ResNet-50 is the approved headline model. Its bootstrap 95% test-accuracy confidence interval is **85.676%–90.228%**. The model beats both paper reference values on this fixed held-out test set.

Footnote: argmax of predict_proba gives 66.0% for the same SVM; we report the standard predict() accuracy.

## Setup

Use the existing environment:

```powershell
.\venv\Scripts\Activate.ps1
```

The installed CUDA build is PyTorch 2.11.0+cu128 and torchvision 0.26.0+cu128. The CUDA wheels come from `https://download.pytorch.org/whl/cu128`.

## Reproduction commands

From the project root:

```powershell
python .\src\cnn_features.py --backbone resnet50 --batch-size 32
python .\src\cnn_features.py --backbone convnext_tiny --batch-size 32
python .\src\cnn_frozen_train.py --backbone resnet50
python .\src\cnn_frozen_train.py --backbone convnext_tiny
python .\src\cnn_finetune.py --backbone resnet50 --batch-size 16 --num-workers 0 --head-epochs 4 --finetune-epochs 12
python .\src\cnn_final_evaluate.py
python .\src\cnn_roc_plot.py
```

The final evaluator is intended to be run once after the configuration has been fixed. It evaluates the approved CNN checkpoint and pre-declared comparison baselines without test-set model selection.

## Prediction

The original SVM remains the default:

```powershell
python .\src\predict.py path\to\painting.jpg
```

Use the fine-tuned CNN explicitly:

```powershell
python .\src\predict.py --model cnn path\to\painting.jpg
```

The command prints the predicted artist and the top three artists with probabilities.

## Project structure

- `src/common.py`: canonical data loading and fixed stratified split
- `src/features.py`, `src/spatial_features.py`: classical features
- `src/train_svm.py`, `src/final_train.py`: classical models
- `src/cnn_features.py`: frozen ImageNet feature extraction
- `src/cnn_frozen_train.py`: validation-only frozen classifier tuning
- `src/cnn_finetune.py`: two-phase ResNet/EfficientNet/ConvNeXt fine-tuning
- `src/cnn_final_evaluate.py`: one-time held-out evaluation and metrics
- `src/cnn_roc_plot.py`: ROC plot from saved final predictions
- `results/FINAL_SUMMARY.md`: complete run record and hyperparameters

## Honest limitations

The CNN uses pretrained ImageNet weights, so it is not trained from random initialization. The test set has 747 images, so accuracy uncertainty is roughly a few percentage points. Images were downsized to 256x256, and training was performed on an RTX 3050 laptop GPU. The final checkpoint is approximately 94.5 MB and is excluded by `.gitignore`.
