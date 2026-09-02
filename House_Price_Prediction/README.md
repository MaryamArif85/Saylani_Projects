# House Price Prediction

Machine learning project to predict house prices using regression models.

## Project Overview
This project implements a full ML pipeline on a house price dataset, including data cleaning, feature encoding, scaling, and model evaluation. The goal was to compare multiple regression algorithms on their ability to predict house prices.

## Files
- `House_Price_Prediction.ipynb` - Main notebook with data cleaning, EDA, model training, and evaluation
  

## Tools & Libraries
- Python
- Pandas, NumPy - Data manipulation
- Matplotlib, Seaborn - Data visualization
- Scikit-learn - Machine learning models and metrics
- XGBoost - Gradient boosting model

## Models Evaluated
- Linear Regression
- Random Forest Regressor
- Decision Tree Regressor
- XGBoost Regressor

## Results & Key Findings

All 3 models achieved negative R² scores on the test set:

- **Linear Regression**: -0.0067
- **Random Forest**: -0.0975  
- **Decision Tree**: -1.0805
- **XGBoost**: Negative R² score

A negative R² means the models perform worse than simply predicting the mean price for every house.

### Reason
Correlation analysis shows all features have very weak correlation with Price, with the highest correlation being only 0.056 for `Floors`. This indicates the features do not have a meaningful linear or non-linear relationship with the target variable.

### Implication
The dataset appears to be synthetic or randomly generated. While the ML pipeline was implemented correctly including data cleaning, encoding, scaling, and model evaluation, no algorithm can predict the target without predictive features.

This exercise demonstrates the importance of **feature relevance** in machine learning.

## How to Run
1. Open `House_Price_Prediction.ipynb` in Google Colab or Jupyter Notebook
2. Upload `House Price Prediction Dataset.csv` to the same folder
3. Run all cells sequentially to reproduce the analysis

## Author
MaryamArif85  
Saylani SMIT - Python Course

## Conclusion
The project highlights that a correct ML pipeline cannot compensate for a lack of predictive features. Feature engineering or a dataset with relevant variables would be required to build a useful model.
