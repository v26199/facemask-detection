# Facemask Detection using Transfer Learning

## Overview
This repository contains code for a facemask detection system implemented using transfer learning with various pretrained models. The system is designed to accurately detect whether individuals are wearing facemasks in images or real-time video streams, contributing to public health safety measures.

## Features
- Preprocessing Techniques: Implemented resizing, data augmentation, and normalization to enhance dataset quality.
- Transfer Learning: Utilized transfer learning to fine-tune pretrained models including ResNet152V2, Xception, and MobileNetV2 for facemask detection.
- Model Selection: Evaluated models based on accuracy, training time, and complexity, selecting MobileNetV2 as the optimal performer.
- Real-Time Detection: Integrated the Multi-Task Cascaded Convolutional Neural Network (MTCNN) for real-time facemask detection.
- Performance Analysis: Conducted comprehensive analysis of each model's performance to inform deployment decisions.

## Dependencies
- Python 3.x
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib

**Results:** Here are the key findings:
1. **High Accuracy with Moderate Training Time:** Models like InceptionV3 and DenseNet121 offer high accuracy (close to 1.0) with reasonable training times (85-125 seconds).
2. **Faster Training with Slightly Lower Accuracy:** Xception and MobileNetV2 provide good performance with validation accuracies above 0.97 and faster training times (60-102 seconds).
3. **Balanced Performance with Lower Complexity:** ResNet152V2 and VGG16 achieve decent accuracies (around 0.91 to 0.99) with lower complexity.
4. **Resource-Constrained Environments:** MobileNetV2 and VGG16 are suitable for environments with limited computational resources due to their lower complexity.
5. **High Performance with High Complexity:** DenseNet121 and InceptionV3 offer excellent accuracy but come with higher complexity and longer training times.

**Conclusion:** The choice of model depends on the specific needs of your application, considering factors like accuracy, training time, model complexity, and available computational resources.
