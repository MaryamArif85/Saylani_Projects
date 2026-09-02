# Saylani_Projects

## 📂 Projects

### 1. [House Price Prediction](./House_Price_Prediction)
A regression project to predict house prices using multiple ML models.

**Models Evaluated**:
- Linear Regression
- Random Forest Regressor 
- Decision Tree Regressor
- XGBoost Regressor

**Key Findings**:
All models achieved negative R² scores on the test set, meaning they performed worse than predicting the mean price.
- Linear Regression: -0.0067
- Random Forest: -0.0975  
- Decision Tree: -1.0805
- XGBoost: Negative R² score

**Reason**: Features have very weak correlation with Price. Highest correlation was only 0.056 for `Floors`.  
**Implication**: The dataset appears to be synthetic/randomly generated. This project demonstrates the importance of feature relevance in ML.

---

### 2. [Library Management System](./Library_Management/README.md)
A console-based library management system for Saylani SMIT assignment.

**Features**:
- Add Book, Issue Book, Return Book, Search Book, View Reports
- CSV file handling to save/load data
- Menu-driven interface

**Files**:
- `Library_BookManagement.ipynb` - Main notebook with all code
- `library_data.csv` - Sample data file

**How to Run**:
1. Open `Library_BookManagement.ipynb` in Google Colab
2. Upload `library_data.csv` to the same folder in Colab
3. Run all cells and use menu options 1-6

---

### 3. [Wine Classification](./Wine_Classification)
A FastAPI web app that predicts wine class using RandomForest. Trained on `wine_dataset.csv` with 100% accuracy.

**Features**:
- Train model with comparison: LogisticRegression vs RandomForest
- FastAPI backend with `/predict` endpoint
- Responsive HTML/CSS/JS frontend with wine theme
- Load Sample, Fill Averages, Clear All buttons

**How to Run**:
```bash
pip install -r requirements.txt
uvicorn main:app --reload

---

###4. Garbage Classification with Deep LearningA 6-class image classification project to identify types of garbage using CNN and Transfer Learning
.
##Classes: cardboard, glass, metal, paper, plastic, trash

**Results**:
~ Best Val Accuracy: 77.53% at Epoch 7
~ Training Accuracy: 94.71%
~ Used Early Stopping to prevent overfitting
~ Best model saved to: models/best_garbage_model.h5

---

###Spam Email Detection

##An NLP project to classify emails as Spam or Ham using Machine Learning.

## Model Performance at Threshold = 0.7
#Raising the threshold to 0.7 made the model more conservative about calling something spam.

## Report
At threshold 0.7, the model prioritized **precision over recall**. 
Precision for spam increased to 0.77, meaning fewer legitimate emails are incorrectly marked as spam. 
Recall dropped to 0.63, so some spam emails are missed. 
This tradeoff is suitable for email filtering, where false positives are more costly than missing some spam.



