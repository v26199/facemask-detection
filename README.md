# Facemask Detection Using Transfer Learning: Balancing Accuracy and Efficiency

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
  
## Models and Results
- InceptionV3: Accuracy: 99.56%, Validation: 98.67%, Time: 438.11s
- DenseNet121: Accuracy: 99.70%, Validation: 99.56%, Time: 508.63s
- Xception: Accuracy: 99.70%, Validation: 99.56%, Time: 439.86s
- MobileNetV2: Accuracy: 99.66%, Validation: 99.56%, Time: 435.51s
- ResNet152V2: Accuracy: 99.51%, Validation: 99.56%, Time: 833.77s
- VGG16: Accuracy: 93.56%, Validation: 54.87%, Time: 262.43s

## Insights
1. Accuracy: DenseNet121 and Xception achieved the highest accuracy (99.70%). InceptionV3 was also highly accurate (99.56%).
2. Time: VGG16 was the fastest (262.43s), but with much lower validation accuracy. MobileNetV2 was efficient (435.51s) with high accuracy.
3. Model Performance: Xception, DenseNet121, and MobileNetV2 had low validation losses, indicating reliable performance.
4. Complexity vs. Performance: MobileNetV2 and VGG16 are ideal for resource-constrained environments. DenseNet121 and InceptionV3 offer higher accuracy with more computational demand.

## Learning Outcomes
1. Transfer Learning: Gained a deeper understanding of using pre-trained models to boost performance and reduce training time.
2. Data Prep: Learned the importance of preprocessing for optimizing model accuracy.
3. Model Evaluation: Developed skills in assessing models using accuracy, time, and loss metrics.
4. Efficiency Balance: Learned to balance accuracy, complexity, and training time for practical uses.

## Conclusion
The facemask detection project showcased my ability to implement and optimize machine learning solutions effectively. DenseNet121 emerged as the best model, with the highest validation accuracy (99.56%), though MobileNetV2 also stood out for its efficient training time (435.51s) and high validation accuracy (99.56%).


