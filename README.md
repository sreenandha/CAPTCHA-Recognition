# CAPTCHA-Recognition-using-Deep-Learning
This project demonstrates a deep learning–based solution for solving text-based CAPTCHAs using a custom Convolutional Neural Network (CNN) implemented in PyTorch. The model is trained to recognize 5-character alphanumeric CAPTCHAs, even when they contain distortions such as rotations, overlapping characters, and noise.

The goal is to explore how deep learning can automate CAPTCHA recognition, a traditionally challenging computer vision task, and to highlight potential vulnerabilities of CAPTCHA systems.

## Features
- Custom CNN Architecture designed for multi-label classification of 5-character CAPTCHAs.
- Character-level encoding to map alphanumeric labels to integer sequences.
- Data preprocessing pipeline with resizing, normalization, and augmentation (rotations, noise, color jitter).
- Training with PyTorch using Adam optimizer and CrossEntropy loss.
- Evaluation metrics include training/validation accuracy and loss tracking across epochs.
- Achieved up to ~95% accuracy on validation data during experiments.
- Visualization of sample CAPTCHA images and model predictions.

## Dataset
- Training images: ~800 synthetic CAPTCHA samples.
- Validation images: ~200 samples.
- Properties: 5-character alphanumeric strings with distortions, overlapping characters, and noise.
- Preprocessing steps:
    - Resizing to 128x128 pixels.
    - Normalization of pixel values.
    - Augmentation with random rotations, noise, and font variations.
- Encoding: Each character mapped to an integer (0–35).

## Model Architecture
- Convolutional Layers: Extract features such as edges, textures, and shapes.
- Pooling Layers: Reduce dimensionality while retaining key features.
- Fully Connected Layers: Flatten and map extracted features to sequence predictions.
- Output: Probability distribution for each character position (5 characters).

## Training Process
- Optimizer: Adam (learning rate = 1e-3).
- Loss function: CrossEntropyLoss (per character position).
- Epochs: 50.
- Batch size: 32.
- Tracking: Training/validation loss and accuracy logged after each epoch.
- Checkpointing: Best model weights saved to prevent overfitting.

## Results
- Training accuracy peaked at ~95%, with validation accuracy slightly lower due to dataset noise.
- Smooth convergence of training/validation loss over epochs.
- Predictions demonstrate robustness against rotated/noisy CAPTCHAs.

## Future Enhancements
- Implement Connectionist Temporal Classification (CTC) loss for variable-length CAPTCHA decoding.
- Add t-SNE visualizations for interpretability of feature embeddings.
- Deploy trained model with Flask/FastAPI for real-time CAPTCHA solving.
- Expand dataset to 30k+ images for better generalization.

## References
1. Edward Raff — Inside Deep Learning.
2. Von Ahn et al. (2003). CAPTCHA: Using Hard AI Problems for Security. EUROCRYPT.
3. Kingma & Ba (2014). Adam: A Method for Stochastic Optimization. arXiv:1412.6980.
4. He et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
5. Goodfellow, Bengio, Courville (2016). Deep Learning. MIT Press.
