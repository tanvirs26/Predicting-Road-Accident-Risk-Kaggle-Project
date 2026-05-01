# Predicting Road Accident Risk

This repository holds an attempt to apply binary classification models to 
road accident risk prediction using data from the Kaggle Playground Series 
Season 5 Episode 10 challenge (https://www.kaggle.com/competitions/playground-series-s5e10).

## Overview
The task, as defined by the Kaggle challenge, is to predict the likelihood 
of accidents on different types of roads based on road conditions such as 
curvature, speed limit, lighting, and weather. The original challenge was 
framed as a regression task predicting a continuous accident_risk score 
between 0 and 1. This project reformulates the problem as a binary 
classification task, categorizing roads as Low Risk (accident_risk < 0.5) 
or High Risk (accident_risk >= 0.5). Three classification models were 
trained and compared: Logistic Regression, Decision Tree, and Random Forest. 
Our best model, Logistic Regression, achieved an AUC-ROC of 0.9770 on the 
validation set.

## Summary of Work Done

### Data
- Type: CSV file of road condition features, output: binary risk classification
- Size: Train — 517,754 rows, 13 features, 1 target. Test — 172,585 rows, 
  13 features
- Split: 80% training (414,203 rows), 20% validation (103,551 rows). 
  Test set has no labels and was used for generating predictions only.

### Preprocessing / Clean Up
- Converted continuous accident_risk target to binary classification using 
  a 0.5 threshold
- Created binary flags is_highway (speed_limit >= 60) and is_accident_prone 
  (num_reported_accidents >= 3) based on threshold effects identified in EDA
- One-hot encoded categorical features lighting and weather using drop_first=True
- Created 4 interaction terms: speed_x_curvature, night_x_curvature, 
  highway_x_night, fog_x_highway
- Applied StandardScaler to numeric features for Logistic Regression only

<img width="908" height="500" alt="image" src="https://github.com/user-attachments/assets/209eae49-a4fd-4ad6-b77f-7ac5b15d77b1" />

<img width="1389" height="985" alt="image" src="https://github.com/user-attachments/assets/e6345475-d0ce-46a6-ab72-bb77d196147d" />



### Data Visualization
EDA included correlation heatmaps, histograms, countplots, and probability 
tables comparing feature distributions between Low Risk and High Risk roads. 
Key findings:

<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/13976d9a-32ba-4ff2-98d0-612ea92382b8" />

- curvature was the strongest predictor of accident risk

<img width="1790" height="490" alt="image" src="https://github.com/user-attachments/assets/41377996-24de-4a14-83b0-c87eff88fa43" />

- speed_limit and num_reported_accidents were also strong predictors, though less gradual

<img width="2189" height="989" alt="image" src="https://github.com/user-attachments/assets/c8fe66f4-ca45-41b5-89a6-dd64ed008602" />
  
- night lighting conditions showed substantially higher accident risk 
  compared to daylight and dim conditions
- foggy weather was associated with higher accident risk compared to clear 
  and rainy conditions
- num_lanes, road_type, road_signs_present, public_road, time_of_day, 
  holiday, and school_season showed very little variation in accident risk 
  and were excluded from modeling

### Problem Formulation
- Input: 12 engineered features including curvature, speed_limit, 
  is_highway, is_accident_prone, one-hot encoded lighting and weather, 
  and 4 interaction terms
- Output: binary classification — 0 (Low Risk) or 1 (High Risk)
- Models:
  - Logistic Regression: chosen as a simple interpretable baseline. 
    Requires scaling and explicit interaction terms to capture nonlinear 
    relationships
  - Decision Tree: chosen to capture nonlinear patterns without requiring 
    scaling or interaction terms
  - Random Forest: chosen as an ensemble extension of the Decision Tree to 
    reduce overfitting and improve generalization
- Hyperparameters:
  - Logistic Regression: class_weight='balanced', max_iter=1000
  - Decision Tree: class_weight='balanced'
  - Random Forest: class_weight='balanced_subsample', n_jobs=-1

### Training
- Software: Python 3.13, scikit-learn, pandas, numpy
- Hardware: local machine (MacOS)
- Logistic Regression and Decision Tree trained in seconds
- Random Forest took longer due to the large dataset size — 
  n_jobs=-1 was used to split training across all CPU cores
- Training was stopped automatically when the model converged or 
  reached max_iter=1000 for Logistic Regression
  
### Performance Comparison
Key metric: AUC-ROC — measures the model's ability to distinguish between 
Low Risk and High Risk roads regardless of classification threshold.

| Model | AUC-ROC | Accuracy | High Risk Precision | High Risk Recall |
|---|---|---|---|---|
| Logistic Regression | 0.9770 | 0.91 | 0.69 | 0.93 |
| Decision Tree | 0.9752 | 0.90 | 0.67 | 0.95 |
| Random Forest | 0.9755 | 0.91 | 0.68 | 0.95 |

Logistic Regression achieved the best AUC-ROC despite being the simplest 
model. ROC curves and confusion matrices for all three models are available in the notebook.

### Conclusions
- Logistic Regression and Random Forest both achieved an accuracy of 0.91, 
  while Decision Tree achieved 0.90. Logistic Regression showed the best 
  overall performance with the highest AUC-ROC of 0.9770
- The explicit feature engineering steps — binary flags and interaction 
  terms — effectively captured non-linear relationships, reducing the 
  advantage that tree based models typically have
- Decision Tree showed the highest recall for High Risk roads (0.95) but 
  at the cost of more false positives, indicating a trade off between 
  sensitivity and precision
- Random Forest performed similarly to Decision Tree, suggesting that 
  ensemble methods did not provide significant improvement over a single 
  tree for this dataset

### Future Work
- Tune hyperparameters such as max_depth for Decision Tree and 
  n_estimators for Random Forest to reduce overfitting
- Try gradient boosting models such as XGBoost which 
  typically perform better on tabular data
- Explore additional interaction terms for 
  Logistic Regression
- Investigate whether treating this as a regression problem rather than 
  classification would yield better performance results

## How to Reproduce Results
1. Download the dataset from the Kaggle competition page:
   https://www.kaggle.com/competitions/playground-series-s5e10/data
2. Update the file paths in the notebook to point to your local dataset
3. Run all cells in order from top to bottom
4. The notebook will produce all EDA visualizations, train all three 
   models, and generate predictions on the test set

## Overview of Files in Repository
- README.md: project overview and results summary
- PredictingRoadAccidentRisk.ipynb: full analysis including data loading, EDA, feature 
  engineering, modeling, and evaluation

## Software Setup
Required packages:
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tabulate

Install all packages with:
pip install numpy pandas matplotlib seaborn scikit-learn tabulate

## Data
Download the dataset from:
https://www.kaggle.com/competitions/playground-series-s5e10/data

Two files are required:
- train.csv — training data with labels
- test.csv — test data without labels

## Citations
- Kaggle Playground Series Season 5 Episode 10:
  https://www.kaggle.com/competitions/playground-series-s5e10
