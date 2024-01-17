# KDT - K-Digital Training

Welcome to the KDT GitHub repository! This repository contains code for various activities related to health challenges and hackathon projects. Feel free to explore the different projects listed below.

## Projects

### 1. Insurance Prediction
#### Project Period: March 23, 2023, to March 23, 2023

#### Dataset
- **Dataset Source:** Medical Cost Personal Datasets (Kaggle)
  - [Link to Kaggle Dataset] : https://www.kaggle.com/datasets/mirichoi0218/insurance

#### Project Description
The Insurance Prediction project aims to predict medical costs using the "Medical Cost Personal Datasets" obtained from Kaggle. The analysis involves performing multiple linear regression based on variables selected through Exploratory Data Analysis (EDA).

#### Key Steps:
1. **Data Collection:**
   - Acquired the dataset from Kaggle, which includes information on various factors affecting medical costs.

2. **Exploratory Data Analysis (EDA):**
   - Conducted thorough EDA to understand the distribution, relationships, and patterns within the dataset.
   - Selected relevant variables for the multiple linear regression analysis.

3. **Multiple Linear Regression:**
   - Implemented a multiple linear regression model to predict medical costs based on the selected variables.
   - Evaluated the model's performance and assessed the significance of predictors.


### 2. Brain Tumor Classification
#### Project Period: April 7, 2023, to April 10, 2023

Model Overview:
Two models were built during this project with distinct goals:

1. **Model1: Brain Tumor Yes/No Classification**
  - **Architecture:** VGG16
  - **Details:**
    - Dropout-layer + Global Average Pooling (GAP) + Batch Normalization
    - Activation Function: Sigmoid
    - Optimizer: Adam
    - Loss Function: Binary Crossentropy
  - **Performance:**
      - Accuracy: 90.28%
      - Loss: 0.2259
      - Area Under the Curve (AUC): 0.9707
  - **Hyperparameters:**
      - Learning Rate: 0.000001
      - Epochs: 50
      - Batch Size: 32
  - **Validation Dataset Tuning:**
      - Parameters were fine-tuned based on the accuracy and loss of the validation dataset.



### 3. Heart Attack Classification
#### Project Period: April 19, 2023, to April 24, 2023


#### Project Description
This project was conducted during the period 230419-230424 with the aim of comparing the performance of heart disease classification models.

#### Dataset

- **Dataset Source:** Heart Attack Analysis & Prediction Dataset (Kaggle)

#### Data Preprocessing

- Applied 3-sigma outlier removal.

#### Model Comparison

1. **Logistic Regression**
    - One-hot encoding, GridSearchCV
    - RMSE: 0.416, Accuracy: 0.8267, AUROC: 0.8164.

2. **Decision Tree**
    - Cross-validation Accuracy: 0.76.

3. **Random Forest**
    - Cross-validation Accuracy: 0.773.

4. **GradientBoosting Classifier**
    - Cross-validation Accuracy: 0.83.
    - RMSE: 0.476.

5. **XGBoost**
    - Cross-validation Accuracy: 0.81.
    - RMSE: 0.476.

6. **AdaBoost**
    - Cross-validation Accuracy: 0.80.
    - RMSE: 0.529.

7. **LGBMClassifier**
    - Cross-validation Accuracy: 0.80.
    - RMSE: 0.476.

8. **CatBoostClassifier**
    - Accuracy: 0.7733.

#### Ensemble Methods

- Applied the following ensemble techniques:
    - Voting (Soft & Hard).
    - Stacking.


### 4. Diabetes Classification
#### Project Period: May 3, 2023, to May 24, 2023

#### Project Description
This diabetes classification model project was conducted from 2023-05-03 to 2023-05-24.

#### Dataset

- **Dataset Source:** Diabetes Dataset (Kaggle)

#### Data Preprocessing

- Baseline Characteristics:
  - Excluded the variable "BloodPressure."
  - T-test:
    - Conducted normality test (Q-Q plot), assumption of homoscedasticity, and assumption of independence for "BloodPressure."
    - Found that the difference in averages between the normal group and the patient group is not significant.

- Exploratory Data Analysis (EDA):
  - Removed outliers through Boxplot.

- Handling Missing Values:
  - Replaced missing values in "BMI" & "SkinThickness" with the median value.

#### Machine Learning Models

- **Support Vector Classifier:**
  - Precision: 0.73
  - Recall: 0.46
  - Accuracy: 0.77
  - F1-Score: 0.56
  - AUC: 0.69

- **Linear Regression:**
  - Precision: 0.65
  - Recall: 0.47
  - Accuracy: 0.74
  - F1-Score: 0.55
  - AUC: 0.68

- **XGBoost Classifier:**
  - Precision: 0.64
  - Recall: 0.58
  - Accuracy: 0.75
  - F1-Score: 0.61
  - AUC: 0.71

#### Hyperparameter Tuning

- **Optuna (AutoML):**
  - Precision: 0.61
  - Recall: 0.61
  - Accuracy: 0.74
  - F1-Score: 0.61
  - AUC: 0.71

#### Deep Learning Model

- **Multi Layer Perceptron (MLP):**
  - Loss Function: Binary Cross Entropy
  - Optimizer: Adam
  - Metrics: Binary Accuracy

  - Precision: 0.77
  - Recall: 0.71
  - Accuracy: 0.84
  - F1-Score: 0.74
  - AUC: 0.80

### 5. Sepsis Prediction

#### Dataset

- **Dataset Source:** Sepsis Hospital Provided Data (Patient Information De-identified and Randomly Dropped)

#### Data Preprocessing

- **Setting and Reclassifying Target Variable "DEATH":**
  - The target variable "DEATH" was set and reclassified during the data preprocessing phase.

- **Missing Value Handling:**
  - Addressed missing values in the dataset.

- **Outlier Processing (3-Sigma):**
  - Identified and processed outliers using the 3-sigma method.

#### Feature Selection

- **Chi-Square Test (Pearson):**
  - Applied the Chi-square test to select relevant features.

#### Algorithms

- **XGBoost:**
- **CatBoost:**
- **ADABoost:**

- **MLP (Multi Layer Perceptron):**

### 6. Hackathon Projects


