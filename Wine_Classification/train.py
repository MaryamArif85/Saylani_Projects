from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import joblib

MODEL_PATH = Path("wine_classification_model.pkl")

def train_model() -> None:
    # 1. LOAD CSV
    df = pd.read_csv("wine_dataset.csv")
    
    # 2. RENAME ONLY THIS COLUMN
    df = df.rename(columns={
        "od280/od315_of_diluted_wines": "od280_od315_of_diluted_wines"
    })

    # 3. SPLIT INTO X and y
    X = df.drop("target", axis=1) 
    y = df["target"]              

    # 4. TRAIN TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
       X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    
    # 5. Define models to compare - INDENTED NOW
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42)
    }

    best_model = None
    best_score = 0
    best_name = "" # add this

    print("="*40)
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Check overfitting: Train vs Test
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        
        # Cross Validation
        cv_scores = cross_val_score(model, X, y, cv=5)
        
        print(f"\n{name}")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy:  {test_acc:.4f}")
        print(f"CV Accuracy:    {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
        print(f"Overfit Gap:    {train_acc - test_acc:.4f}")

        # Pick best model
        if test_acc > best_score:
            best_score = test_acc
            best_model = model
            best_name = name

    # 6. Save the best model - INDENTED NOW
    print("="*40)
    print(f"\nBest Model: {best_name} with Test Accuracy: {best_score:.4f}")

    model_bundle = {
        "model": best_model,
        "columns": X.columns.tolist(),
        "target_names": ["Class_0", "Class_1", "Class_2"],
        "accuracy": best_score,
        "version": "1.0.0"
    }
    joblib.dump(model_bundle, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train_model()