# Final Summary

## Approved final configuration

The final configuration was fixed before test access:

- Single fine-tuned ResNet-50
- Checkpoint: `models/cnn_best_resnet50.pt`
- Best validation epoch: 13 of 16
- Validation accuracy: 87.9518%
- No ensemble
- No test-time augmentation
- Fixed stratified split from `src/common.py::split_data`, seed 42

## Final held-out test results

The final test pass used 747 images and was completed once for the approved configurations. The headline model produced:

- Top-1 accuracy: **87.9518%**
- Top-3 accuracy: **95.5823%**
- Macro-F1: **0.879041**
- Macro one-vs-rest ROC AUC: **0.986367**
- Bootstrap 95% accuracy CI: **85.6760%–90.2276%**

Comparison results are in `results/cnn_final_comparison.csv`:

| Model | Top-1 | Top-3 | Macro-F1 |
|---|---:|---:|---:|
| Classical SVM | 64.7% | 85.1% | 0.646 |
| Frozen ResNet-50 + LogReg C=0.1 | 79.7858% | 93.0388% | 0.797715 |
| Frozen ConvNeXt-Tiny + LogReg C=0.1 | 83.1325% | 94.9130% | 0.830005 |
| Fine-tuned ResNet-50 | **87.9518%** | **95.5823%** | **0.879041** |
| Chen & Deng paper SVM | 68.1% | - | - |
| Chen & Deng paper CNN | 74.7% | - | - |

The fine-tuned CNN beats the current classical SVM by 23.2518 percentage points, the paper SVM by 19.8518 points, and the paper CNN by 13.2518 points. The confidence interval remains above both paper reference values.

Footnote: argmax of predict_proba gives 66.0% for the same SVM; we report the standard predict() accuracy.

## Full pipeline and exact run order

1. Verify the existing `venv` and CUDA GPU.
2. Back up the pre-CNN `src/`, `results/`, `models/`, and `data/*.npz` into `_archive/backup_before_cnn/`.
3. Extract frozen ResNet-50 features with `src/cnn_features.py`.
4. Extract frozen ConvNeXt-Tiny features with `src/cnn_features.py`.
5. Tune frozen Logistic Regression and RBF SVM candidates using validation accuracy only with `src/cnn_frozen_train.py`.
6. Fine-tune ResNet-50 with `src/cnn_finetune.py`; select the checkpoint by validation accuracy only.
7. Evaluate the pre-approved configurations once with `src/cnn_final_evaluate.py`.
8. Generate the ROC curves from saved predictions with `src/cnn_roc_plot.py`.

No test result was used for model selection. The first evaluator invocation stopped during report serialization because of an output-field name bug before final artifacts were written; the corrected evaluator then completed the one successful final evaluation. No configuration was changed.

## Model hyperparameters

### Classical SVM

- Features: 1,795 base features plus 240 spatial HSV/LBP features
- StandardScaler fitted on train only
- PCA with 300 components fitted on train only
- RBF SVC, `C=100`, `gamma="scale"`, `probability=True`

### Frozen CNN baselines

- ImageNet preprocessing: resize 256, center crop 224, ImageNet normalization
- Frozen pooled features, no trainable CNN layers
- StandardScaler fitted on train only
- Logistic Regression, `C=0.1`, `max_iter=1000`
- Selected separately for ResNet-50 and ConvNeXt-Tiny by validation accuracy

### Fine-tuned CNN

- Backbone: torchvision ResNet-50 with `IMAGENET1K_V2`
- Classifier: `Dropout(0.4)` followed by `Linear(2048, 15)`
- Phase A: head only, 4 epochs, AdamW, learning rate 1e-3
- Phase B: head plus ResNet `layer3` and `layer4`, 12 epochs
- Phase B learning rates: 1e-3 for head and 1e-4 for backbone
- Weight decay: 1e-4
- Cosine annealing schedule
- Cross-entropy label smoothing: 0.1
- Training augmentation: random resized crop 224 with scale 0.5–1.0, horizontal flip, light color jitter
- Validation/test preprocessing: resize 256 plus center crop 224 only
- Batch size: 16
- Mixed precision: CUDA autocast and GradScaler
- Fixed seed: 42
- Best checkpoint: epoch 13, validation accuracy 87.9518%

## Output files

- `results/cnn_final_results.txt`
- `results/cnn_final_comparison.csv`
- `results/cnn_test_metrics.csv`
- `results/cnn_per_artist.csv`
- `results/cnn_confusion_matrix.png`
- `results/cnn_roc_auc.png`
- `results/cnn_most_confused_pairs.csv`
- `results/cnn_misclassified_grid.png`
- `results/cnn_test_predictions.npz`
- `models/cnn_best_resnet50.pt`
- `results/cnn_training_log_resnet50.csv`
- `results/cnn_training_curves_resnet50.png`

## Limitations and leakage review

The CNN uses external ImageNet pretraining. The 747-image test set is small enough that the confidence interval is several percentage points. Images were downsized to 256x256 and training was performed on a laptop RTX 3050. Training augmentation is only used by the training dataset; validation and test use deterministic center crops. Scalers and PCA are fitted only on training data. No test-based selection, ensemble choice, or test-time augmentation choice was performed.
