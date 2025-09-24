from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download

# -----------------------------
# Download models from Hugging Face Hub
# (replace repo_ids with your actual repos)
# -----------------------------
kinara_path = hf_hub_download(repo_id="mjpsm/Kinara-xgb-model", filename="Kinara_xgb_model.pkl")
ubuntu_path = hf_hub_download(repo_id="mjpsm/Ubuntu-xgb-model", filename="Ubuntu_xgb_model.pkl")
jali_path   = hf_hub_download(repo_id="mjpsm/Jali-xgb-model",   filename="Jali_xgb_model.pkl")

available_models = {
    "Kinara": joblib.load(kinara_path),
    "Ubuntu": joblib.load(ubuntu_path),
    "Jali": joblib.load(jali_path),
}

# Archetype list (15 total, 12 fillers for now)
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
            snapshot[name] = 0.0  # filler until model is ready

    return {"soulprint_snapshot": snapshot}
