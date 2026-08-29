# Garbage Classification with Deep Learning

A 6-class image classification project to identify types of garbage using CNN and Transfer Learning.

**Classes**: cardboard, glass, metal, paper, plastic, trash

## Dataset
- Source: Garbage Classification Dataset ~2024 images
- Note: Dataset is not included in this repo due to size. Place it in `Garbage classification_dataset/` folder

## Results
| Custom CNN + BatchNorm + Dropout | 76.14% | Overfitting. Hit ceiling |
| Transfer Learning: MobileNetV2 Frozen | **77.53%** | Best model. Saved as `garbage_model.h5` |

