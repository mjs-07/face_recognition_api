import os
import numpy as np
import cv2
from fastapi import FastAPI, File, UploadFile, Form
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from starlette.responses import JSONResponse

# Initialize FastAPI
app = FastAPI()

# Paths
DATA_DIR = "face_database"
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "face_embeddings.npy")
LABELS_FILE = os.path.join(DATA_DIR, "face_labels.npy")

os.makedirs(DATA_DIR, exist_ok=True)

# Face recognition settings
MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "retinaface"
SIMILARITY_THRESHOLD = 0.7

def get_face_embedding(img_path):
    """Extracts face embedding from an image file."""
    try:
        embedding = DeepFace.represent(
            img_path=img_path, model_name=MODEL_NAME, detector_backend=DETECTOR_BACKEND, enforce_detection=True
        )[0]["embedding"]
        return np.array(embedding)
    except:
        return None

@app.post("/register")
async def register_user(name: str = Form(...), file: UploadFile = File(...)):
    """Registers a new user."""
    img_path = os.path.join(DATA_DIR, f"{name}.jpg")
    
    with open(img_path, "wb") as buffer:
        buffer.write(file.file.read())

    embedding = get_face_embedding(img_path)
    if embedding is None:
        return JSONResponse(content={"error": "No face detected. Try again."}, status_code=400)

    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(LABELS_FILE):
        embeddings = np.load(EMBEDDINGS_FILE)
        labels = np.load(LABELS_FILE)
    else:
        embeddings = np.empty((0, len(embedding)))
        labels = np.array([])

    embeddings = np.vstack((embeddings, embedding))
    labels = np.append(labels, name)

    np.save(EMBEDDINGS_FILE, embeddings)
    np.save(LABELS_FILE, labels)

    return {"message": f"User {name} registered successfully!"}

@app.post("/authenticate")
async def authenticate_user(file: UploadFile = File(...)):
    """Authenticates a user."""
    if not os.path.exists(EMBEDDINGS_FILE) or not os.path.exists(LABELS_FILE):
        return JSONResponse(content={"error": "No registered users. Please register first."}, status_code=400)

    img_path = os.path.join(DATA_DIR, "captured_face.jpg")
    with open(img_path, "wb") as buffer:
        buffer.write(file.file.read())

    embedding = get_face_embedding(img_path)
    if embedding is None:
        return JSONResponse(content={"error": "No face detected. Try again."}, status_code=400)

    embeddings = np.load(EMBEDDINGS_FILE)
    labels = np.load(LABELS_FILE)

    similarities = cosine_similarity([embedding], embeddings)[0]
    max_sim_index = np.argmax(similarities)
    max_similarity = similarities[max_sim_index]

    if max_similarity > SIMILARITY_THRESHOLD:
        return {"message": f"Authentication Successful! Welcome, {labels[max_sim_index]}."}
    else:
        return JSONResponse(content={"error": "Authentication Failed. Face Not Recognized."}, status_code=401)
