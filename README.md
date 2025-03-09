# Gesture Recognition using Deep Learning

This project implements and compares two deep learning architectures (S3D and VideoResNet) for gesture recognition using the Jester dataset. This project was made as part of an university computer vision course, but since it was a free-choice project, it also counts as a personal project to me, since it is unique.

## Project Overview

The project focuses on recognizing hand gestures from video sequences using two different deep learning models:
- S3D
- VideoResNet

The goal was to analyze the trade-off between model complexity and dataset size:

- The VideoResNet model is more complex but trained on a smaller dataset due to computational constraints.
- The S3D model is simpler but benefits from training on a larger dataset.

Both models were trained on the Jester dataset, which contains labeled video sequences of common hand gestures.

## Results

The training results for both models are visualized in the included plots:
- Accuracy comparison: 
  - [S3D Accuracy](accuracy_S3D.png)
  - [VideoResNet Accuracy](accuracy_VideoResNet.png)
- Loss progression:
  - [S3D Loss](loss_S3D.png)
  - [VideoResNet Loss](loss_VideoResNet.png)
 
The results show that:
- The S3D model achieved higher validation accuracy than VideoResNet, even though it is a simpler model.
- This shows that a less complex model is usually preferred when computational resources are limited.

## Model Weights

The trained weights are included for both models:
- [VideoResNet weights](moded_VideoResNet_10.pth)
- [S3D weights](model_S3D_10.pth)

## Dataset

The project uses the Jester dataset with the following dataset split files:
- [Training set](jester-v1-train.csv)
- [Validation set](jester-v1-validation.csv)
- [Label mappings](jester-v1-labels.csv)

## Limitations

- The models were trained with limited computational resources, which affected the final accuracy.
- Training time constraints meant it was not trained to be perfect.
- The dataset size used was small due to computational constraints, which affected accuracy.

## How to use
Install requirements:
```
pip install -r requirements.txt
```

Run the Jupyter notebook.
