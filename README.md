# Insightful Identity Analysis: Detecting Age, Gender, and Ethnicity

This repository contains facial classification models developed for predicting demographics such as age, gender, and ethnicity. Utilizing a range of machine learning techniques and model architectures, we achieved notable performance, including **87% accuracy for gender classification** using Convolutional Neural Networks (CNN).

## Key Highlights

- Leveraged **Logistic Regression**, **K-Nearest Neighbors (KNN)**, **Naive Bayes**, **SVM**, **Random Forests**, and **CNN** for gender, ethnicity, and age prediction.
- Achieved **84.41% accuracy for gender prediction using Logistic Regression** on the UTKFace dataset, which consists of over 20,000 images.
- Best performance for **gender prediction: 87% accuracy (CNN)**.
- Ethnicity classification reached **72% accuracy (CNN)**, with lower performance for other models.
- Age prediction remained challenging, with the highest accuracy of **47% (CNN)**.

## Dataset

We used the **UTKFace dataset**, a large-scale dataset containing over 20,000 images annotated for age, gender, and ethnicity.

- Dataset link: [Age, Gender, and Ethnicity Face Data (Kaggle)](https://www.kaggle.com/datasets/nipunarora8/age-gender-and-ethnicity-face-data-csv/data)
- Diverse subjects ranging from 0 to 116 years across four ethnicities.

## Tech Stack

- **Python**
- **PyTorch**
- **Machine Learning**: Scikit-learn, Logistic Regression, KNN, SVM, Random Forests, Naive Bayes
- **Deep Learning**: Convolutional Neural Networks (CNN)
- **Image Processing**

## Results Summary

| Task                | Model         | Accuracy |
|---------------------|---------------|----------|
| Gender Prediction   | Logistic Regression | 84.41%  |
| Gender Prediction   | CNN           | 87%      |
| Ethnicity Prediction| CNN           | 72%      |
| Age Prediction      | CNN           | 47%      |

## Preprocessing and Methodology

1. Images resized from 128x128 to 28x28 pixels.
2. Converted RGB images to grayscale (except CNN).
3. Applied PCA for dimensionality reduction for Naive Bayes.
4. Hyperparameter tuning and optimization for each model.
