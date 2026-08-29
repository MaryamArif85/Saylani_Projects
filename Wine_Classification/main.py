from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
print(f"BASE_DIR: {BASE_DIR}")
MODEL_PATH = BASE_DIR / "wine_classification_model.pkl"

if not MODEL_PATH.exists():
    raise RuntimeError("wine_classification_model.pkl was not found. Run `python train.py` first.")

model_bundle = joblib.load(MODEL_PATH)
model = model_bundle["model"]
columns = model_bundle["columns"] # we saved this in train.py
accuracy = model_bundle["accuracy"]
version = model_bundle["version"]

# Wine has 3 classes: 0, 1, 2. Since we didn't save target_names, make them
target_names = ["Class_0", "Class_1", "Class_2"]

app = FastAPI(
    title="Wine Classification Web App",
    version=version,
)

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)

templates = Jinja2Templates(directory=BASE_DIR / "templates")

class WineInput(BaseModel):
    alcohol: float = Field(gt=0)
    malic_acid: float = Field(gt=0)
    ash: float = Field(gt=0) 
    alcalinity_of_ash: float = Field(gt=0) 
    magnesium: float = Field(gt=0)
    total_phenols: float = Field(gt=0)
    flavanoids: float = Field(gt=0)
    nonflavanoid_phenols: float = Field(gt=0)
    proanthocyanins: float = Field(gt=0)
    color_intensity: float = Field(gt=0)
    hue: float = Field(gt=0)
    od280_od315_of_diluted_wines: float = Field(gt=0)
    proline: float = Field(gt=0) 


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "model_version": version,
            "model_accuracy": f"{accuracy*100:.1f}" 
        },
    ) 

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_status": "loaded",
        "model_version": version,
    }


@app.post("/predict")
def predict(data: WineInput):
    # build features in the EXACT same order as training
    input_dict = data.dict()
    features = np.array([[input_dict[col] for col in columns]])
    
    predicted_class = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    confidence = float(probabilities[predicted_class])

    return {
        "predicted_class": predicted_class,
        "predicted_label": target_names[predicted_class],
        "confidence": round(confidence * 100, 2),
        "model_version": version,
    }
