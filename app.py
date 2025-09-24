from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from sentence_transformers import SentenceTransformer
from typing import Dict

# -----------------------------
# Load only available models
# -----------------------------
available_models = {
    "Kinara": joblib.load("/Users/mazamessomeba/Desktop/Projects/Soulprint_snapshot/kinara-regression/model/Kinara_xgb_model.pkl"),
    "Ubuntu": joblib.load("/Users/mazamessomeba/Desktop/Projects/Soulprint_snapshot/Unbuntu-regression/model/Ubuntu_xgb_model.pkl"),
    "Jali": joblib.load("/Users/mazamessomeba/Desktop/Projects/Soulprint_snapshot/Jali-regression/model/Jali_xgb_model.pkl"),
}

# Archetype list (15 total)
all_archetypes = [
    "Griot", "Kinara", "Ubuntu", "Jali", "Sankofa", "Imani", "Maji",
    "Nzinga", "Bisa", "Zamani", "Tamu", "Shujaa", "Ayo", "Ujamaa", "Kuumba"
]

# Shared embedder
embedder = SentenceTransformer("all-mpnet-base-v2")

# FastAPI app
app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/soulprint_snapshot")
def soulprint_snapshot(input: TextInput):
    # Convert text into embedding
    embedding = embedder.encode([input.text]).reshape(1, -1)

    # Build snapshot
    snapshot = {}
    for name in all_archetypes:
        if name in available_models:
            score = available_models[name].predict(embedding)[0]
            snapshot[name] = float(score)
        else:
            snapshot[name] = 0.0  # filler value for now

    return {"soulprint_snapshot": snapshot}
