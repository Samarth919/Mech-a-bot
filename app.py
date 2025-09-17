import os
import io
import json
import traceback
import pdfplumber
import pandas as pd
import requests
import numpy as np
import re
import cv2
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
import faiss

# 🔹 New imports for thermal image analysis
from PIL import Image
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
ROOT = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(ROOT, "datasets")
INDEX_DIR = os.path.join(ROOT, "index_store")
UPLOAD_DIR = os.path.join(ROOT, "uploads")
os.makedirs(INDEX_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

FAISS_PATH = os.path.join(INDEX_DIR, "faiss.index")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")

EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_BATCH = 32

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY",
    "ADD_YOUR_API_KEY"
)
GROQ_MODEL = "llama-3.3-70b-versatile"

RELEVANCE_THRESHOLD = 0.28

# ---------------- APP INIT ----------------
app = Flask(__name__, template_folder="templates", static_folder="static")
embedder = SentenceTransformer(EMBED_MODEL)
D = embedder.get_sentence_embedding_dimension()

# ---------------- HELPERS ----------------
def chunk_text(text, max_words=200):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]


def normalize_embeddings(emb):
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return (emb / (norms + 1e-10)).astype(np.float32)


def save_metadata(metadata_list):
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, ensure_ascii=False, indent=2)


def load_metadata():
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


# ---------------- INDEX ----------------
def make_faiss_index(dim):
    return faiss.IndexFlatIP(int(dim))


def build_index_from_datasets():
    metadata = []
    texts = []

    for fname in sorted(os.listdir(DATASETS_DIR)):
        path = os.path.join(DATASETS_DIR, fname)
        ext = os.path.splitext(fname)[1].lower()

        try:
            if ext == ".pdf":
                with pdfplumber.open(path) as pdf:
                    for page in pdf.pages:
                        t = page.extract_text()
                        if t:
                            for chunk in chunk_text(t):
                                texts.append(chunk)
                                metadata.append({"text": chunk, "source": fname})

            elif ext in (".csv", ".xls", ".xlsx"):
                if ext == ".csv":
                    df = pd.read_csv(path)
                else:
                    df = pd.read_excel(path)

                df.columns = [c.strip() for c in df.columns]

                for _, row in df.iterrows():
                    parts = [f"{c}: {row[c]}" for c in df.columns if pd.notna(row[c])]
                    if parts:
                        record = " ; ".join(parts)
                        for chunk in chunk_text(record):
                            texts.append(chunk)
                            metadata.append({"text": chunk, "source": fname})

            else:
                continue

        except Exception as e:
            print(f"Error ingesting {fname}: {e}")

    if not texts:
        index = make_faiss_index(D)
        save_metadata([])
        faiss.write_index(index, FAISS_PATH)
        return index, []

    all_embs = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i:i+EMBED_BATCH]
        emb = embedder.encode(batch, convert_to_numpy=True)
        emb = normalize_embeddings(emb)
        all_embs.append(emb)

    all_embs = np.vstack(all_embs).astype(np.float32)
    index = make_faiss_index(D)
    index.add(all_embs)

    faiss.write_index(index, FAISS_PATH)
    save_metadata(metadata)

    return index, metadata


def load_or_build_index():
    if os.path.exists(FAISS_PATH) and os.path.exists(METADATA_PATH):
        try:
            idx = faiss.read_index(FAISS_PATH)
            meta = load_metadata()
            if idx.d != D:
                print("Embedding dimension mismatch; rebuilding index.")
                return build_index_from_datasets()
            return idx, meta
        except Exception as e:
            print("Failed to load index:", e)
            return build_index_from_datasets()
    else:
        return build_index_from_datasets()


index, metadata = load_or_build_index()


# ---------------- RETRIEVAL ----------------
def retrieve(query, k=5):
    if index.ntotal == 0:
        return []

    q_emb = embedder.encode([query], convert_to_numpy=True)
    q_emb = normalize_embeddings(q_emb)
    k_search = min(int(index.ntotal), k)

    D_scores, I = index.search(q_emb, k_search)
    results = []

    for pos, idx in enumerate(I[0]):
        if idx < len(metadata):
            results.append({"meta": metadata[idx], "score": float(D_scores[0][pos])})

    return results


# ---------------- GROQ ----------------
def groq_chat(messages):
    if not GROQ_API_KEY:
        return "(Groq API key not configured)"

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": GROQ_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 400
    }

    try:
        r = requests.post(url, headers=headers, json=data, timeout=30)
        r.raise_for_status()
        resp = r.json()
        return resp["choices"][0]["message"]["content"]
    except Exception as e:
        return f"(Groq error) {e}"


# ---------------- ENHANCED THERMAL IMAGE ANALYSIS ----------------
def analyze_thermal_image(image_path, show_map=True, as_json=True):
    """
    Analyze a thermal image (false-color infrared) and return
    both quantitative stats and an adaptive descriptive summary.
    """
    # Load and convert to RGB
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Compute average intensities
    avg_intensity = image_np.mean(axis=(0, 1))
    avg_r, avg_g, avg_b = avg_intensity

    # Convert RGB to grayscale intensity (proxy for heat)
    gray_intensity = np.mean(image_np, axis=2)

    # Normalize 0–1
    denom = (gray_intensity.max() - gray_intensity.min())
    if denom == 0:
        norm_gray = np.zeros_like(gray_intensity, dtype=np.float32)
    else:
        norm_gray = (gray_intensity - gray_intensity.min()) / denom

    # Classify zones
    total_pixels = norm_gray.size
    zones = {
        "hot": int(np.sum(norm_gray > 0.75)),
        "warm": int(np.sum((norm_gray > 0.5) & (norm_gray <= 0.75))),
        "cool": int(np.sum((norm_gray > 0.25) & (norm_gray <= 0.5))),
        "cold": int(np.sum(norm_gray <= 0.25)),
    }

    # Convert to percentages
    zone_percent = {k: round((v / total_pixels) * 100, 2) for k, v in zones.items()}

    # Generate adaptive description
    description = []
    if zone_percent["hot"] < 10:
        description.append("The image shows a strong hot core (bright yellow-white) indicating significant heating in the central region.")
    elif zone_percent["warm"] > 30:
        description.append("Most of the image falls into warm zones (red–orange), suggesting elevated but distributed heating.")
    else:
        description.append("The image shows limited hot areas, with most regions in cooler ranges.")

    if zone_percent["cool"] > 25:
        description.append("Cooler zones (green) cover a noticeable portion, representing moderate temperature regions.")
    if zone_percent["cold"] > 20:
        description.append("Cold regions (blue/black) dominate parts of the image, likely casing or ambient areas.")

    # Visualization (disabled by default in server)
    '''if show_map:
        plt.figure(figsize=(8, 6))
        plt.imshow(norm_gray, cmap="inferno")
        plt.colorbar(label="Relative Heat Intensity")
        plt.title("Thermal Intensity Map")
        plt.axis("off")
        plt.show()'''

    # Pack results
    summary = {
        "average_rgb": {
            "red": round(float(avg_r), 2),
            "green": round(float(avg_g), 2),
            "blue": round(float(avg_b), 2)
        },
        "zones_pixel_percent": zone_percent,
        "description": " ".join(description)
    }

    return json.dumps(summary, indent=4) if as_json else summary


# ---------------- NEW (FIXED) FUNCTION: thermal_pattern_insight ----------------
def thermal_pattern_insight(image_path, show_map=True):
    """
    Analyze heat distribution in different regions of the thermal image
    and return an adaptive human-readable insight.
    """
    # Load and convert to RGB
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Convert to grayscale intensity (proxy for heat)
    gray_intensity = np.mean(image_np, axis=2)
    norm_gray = (gray_intensity - gray_intensity.min()) / (gray_intensity.max() - gray_intensity.min())

    # Define thresholds
    hot_mask = norm_gray > 0.75
    warm_mask = (norm_gray > 0.5) & (norm_gray <= 0.75)

    # Split into vertical thirds: left, center, right
    h, w = norm_gray.shape
    thirds = {
        "left": norm_gray[:, : w // 3],
        "center": norm_gray[:, w // 3 : 2 * w // 3],
        "right": norm_gray[:, 2 * w // 3 :]
    }

    # Count hot pixel percentages in each region
    region_hot = {k: np.sum(v > 0.75) / v.size * 100 for k, v in thirds.items()}

    # Find dominant hot region
    dominant_region = max(region_hot, key=region_hot.get)

    # Generate adaptive insight
    if dominant_region == "center":
        insight = "The heat is concentrated in the central section, likely the core windings or stator body, where energy losses and friction are highest."
    elif dominant_region == "left":
        insight = "The heat is concentrated on the left side, suggesting localized stress or inefficient cooling in that area."
    else:
        insight = "The heat is concentrated on the right side, which may indicate asymmetric loading or ventilation issues."

    # Add cool region context
    cool_pixels = np.sum(norm_gray < 0.3) / norm_gray.size * 100
    if cool_pixels > 25:
        insight += " Cooler zones along the edges indicate effective heat dissipation to the casing or ambient surroundings."

    # Visualization (optional)
    '''if show_map:
        plt.imshow(norm_gray, cmap="inferno")
        plt.title("Thermal Heat Map (Normalized)")
        plt.axis("off")
        plt.colorbar(label="Relative Heat Intensity")
        plt.show()'''

    return insight


# ---------------- ANSWER ----------------
def generate_answer(query):
    retrieved = retrieve(query, k=6)

    if not retrieved:
        sys = "Be concise."
        return groq_chat([
            {"role": "system", "content": sys},
            {"role": "user", "content": query}
        ])

    dataset_context = "\n".join([r["meta"]["text"] for r in retrieved[:5]])
    max_score = max(r["score"] for r in retrieved) if retrieved else 0.0

    if max_score >= RELEVANCE_THRESHOLD:
        sys = (
            "Use only dataset context if sufficient; "
            "otherwise add minimal general info. Keep it short."
        )
        user_prompt = f"Dataset:\n{dataset_context}\n\nUser:\n{query}"

        return groq_chat([
            {"role": "system", "content": sys},
            {"role": "user", "content": user_prompt}
        ])
    else:
        sys = "Be concise."
        return groq_chat([
            {"role": "system", "content": sys},
            {"role": "user", "content": query}
        ])


# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    data = request.json or {}
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"answer": "Please provide a question."})

    try:
        # 🔍 If user asks about an image filename → run analyzer
        match = re.search(r"(.*\.(png|jpg|jpeg|bmp|tif|tiff))", query, re.IGNORECASE)
        if match:
            fname = match.group(1)
            path = os.path.join(DATASETS_DIR, fname)

            if os.path.exists(path):
                analysis = analyze_thermal_image(path, show_map=False)
                extra_insight = thermal_pattern_insight(path, show_map=False)
                return jsonify({"answer": f"Thermal analysis of {fname}: {analysis}\n\nPattern Insight: {extra_insight}"})
            else:
                return jsonify({"answer": f"Image file not found: {fname}"})

        # Otherwise → normal text answer
        ans = generate_answer(query)
        return jsonify({"answer": ans})

    except Exception as e:
        return jsonify({"answer": f"Error: {e}\n{traceback.format_exc()}"})
    

# ---------------- IMAGE UPLOAD ----------------
@app.route("/upload_image", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded."})

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "error": "Empty filename."})

    fname = secure_filename(file.filename)
    path = os.path.join(UPLOAD_DIR, fname)
    file.save(path)

    # Analyze with new logic
    analysis = analyze_thermal_image(path, show_map=False)
    extra_insight = thermal_pattern_insight(path, show_map=False)
    return jsonify({"success": True, "file": fname, "analysis": analysis, "pattern_insight": extra_insight})


@app.route("/rebuild_index", methods=["POST"])
def rebuild_index():
    try:
        global index, metadata
        index, metadata = build_index_from_datasets()
        return jsonify({"status": "ok", "message": f"Rebuilt index with {len(metadata)} chunks."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
