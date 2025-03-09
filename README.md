# Gesture Recognition using Deep Learning

This project implements and compares two different deep learning architectures (S3D and VideoResNet) for gesture recognition using the Jester dataset. The project was made as part of a university computer vision course, but since this project was a free-choice project, I do count it partly as a personal project, since it is unique.

## Project Overview

The project focuses on recognizing hand gestures from video sequences using two different deep learning models:
- S3D
- VideoResNet

The goal of this project was to identify if sacrificing model complexity over training time was a good choice or not. The VideoResNet model is the complex one and S3D is the simpler one.

Both models were trained on the Jester dataset, which contains labeled video sequences of common hand gestures.

## Results

The training results for both models are visualized in the included plots:
- Accuracy comparison: 
  - [S3D Accuracy](accuracy_S3D.png)
  - [VideoResNet Accuracy](accuracy_VideoResNet.png)
- Loss progression:
  - [S3D Loss](loss_S3D.png)
  - [VideoResNet Loss](loss_VideoResNet.png)

## Model Weights

The trained weights are included for both models:
- [VideoResNet weights](moded_VideoResNet_10.pth)
- [S3D weights](model_S3D_10.pth)

## Dataset

The project uses the Jester dataset with the following split files:
- [Training set](jester-v1-train.csv)
- [Validation set](jester-v1-validation.csv)
- [Label mappings](jester-v1-labels.csv)

## Limitations

- The models were trained with limited computational resources, which affected the final accuracy
- Training time constraints meant it was not trained to be perfect
- The dataset size used was small due to computational constraints, which affected accuracy.

## How to use
Install requirements:
```
pip install -r requirements.txt
```

Run the notebook.