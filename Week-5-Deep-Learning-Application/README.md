# Week 5 — Deep Learning Application in Data Science

## Project
Handwritten Digit Classification using a PyTorch Neural Network.

## Objective
Apply deep-learning concepts to a supervised multiclass classification problem, covering dataset selection, preprocessing, neural-network architecture design, training, validation, evaluation, and critical analysis.

## Dataset
The scikit-learn Digits dataset contains 1,797 grayscale handwritten-digit images. Each image is 8×8 pixels, represented as 64 numerical features, with 10 target classes (0–9).

## Model Architecture
- Input: 64 features
- Dense layer: 128 neurons + ReLU
- Dropout: 0.25
- Dense layer: 64 neurons + ReLU
- Dropout: 0.20
- Output: 10 class logits
- Loss: Cross-Entropy Loss
- Optimizer: Adam
- Learning rate: 0.001
- Weight decay: 0.0001
- Batch size: 64
- Maximum epochs: 60
- Early-stopping patience: 8

## Evaluation
The data were split using stratification into training, validation, and a held-out test set. Features were standardized using training data only. The final model achieved approximately **98.61% test accuracy** and **98.60% macro F1-score** in the documented run.

## Key Learning Outcomes
- Understanding neural-network architecture and nonlinear activation functions
- Implementing a PyTorch MLP
- Applying scaling without data leakage
- Using dropout and weight decay for regularization
- Monitoring validation loss and applying early stopping
- Interpreting confusion matrices and per-class metrics
- Critically evaluating model limitations and future improvements

## Files
- `README.md` — project overview and results
- `deep_learning_digits.py` — reproducible PyTorch implementation
- `requirements.txt` — required Python packages
- `report/README.md` — report submission note

## Future Improvements
A convolutional neural network (CNN), data augmentation, systematic hyperparameter tuning, multiple random seeds, and external validation would provide a stronger and more realistic handwriting-recognition solution.
