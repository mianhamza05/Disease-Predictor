"""
main.py - FastAPI backend for Disease Prediction
Run with: uvicorn main:app --reload --port 8000
"""
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Disease Prediction API", version="1.0")

# Allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model artifacts at startup
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)
with open("feature_cols.pkl", "rb") as f:
    feature_cols = pickle.load(f)


# ── Health check and root endpoints ────────────────────────────────────────────
@app.get("/health")
def health():
    """Health check endpoint that returns available symptoms."""
    return {
        "status": "online",
        "symptoms": feature_cols
    }


@app.get("/")
def root():
    return {"message": "Disease Prediction API is running"}


# ── Disease info (description + precautions) ──────────────────────────────────
DISEASE_INFO = {
    "Fungal infection": {
        "description": "A fungal infection caused by pathogenic fungi affecting skin, nails, or internal organs.",
        "precautions": ["Keep skin clean and dry", "Use antifungal creams", "Avoid sharing personal items", "Wear breathable clothing"],
        "severity": "mild"
    },
    "Allergy": {
        "description": "An immune system reaction to a foreign substance such as pollen, food, or pet dander.",
        "precautions": ["Avoid known allergens", "Take antihistamines", "Keep windows closed during pollen season", "Consult an allergist"],
        "severity": "mild"
    },
    "GERD": {
        "description": "Gastroesophageal Reflux Disease — stomach acid frequently flows back into the esophagus.",
        "precautions": ["Avoid spicy and fatty foods", "Eat smaller meals", "Don't lie down after eating", "Elevate head while sleeping"],
        "severity": "moderate"
    },
    "Chronic cholestasis": {
        "description": "A condition where bile flow from the liver is reduced or blocked.",
        "precautions": ["Avoid alcohol", "Follow a low-fat diet", "Take prescribed medications", "Regular liver function tests"],
        "severity": "severe"
    },
    "Drug Reaction": {
        "description": "An adverse reaction to a drug or medication, ranging from mild rash to severe anaphylaxis.",
        "precautions": ["Stop the triggering drug immediately", "Consult your doctor", "Carry a medical alert", "Keep a list of allergies"],
        "severity": "moderate"
    },
    "Peptic ulcer disease": {
        "description": "Open sores that develop on the inner lining of the stomach and the upper small intestine.",
        "precautions": ["Avoid NSAIDs", "Quit smoking", "Limit alcohol", "Eat smaller and more frequent meals"],
        "severity": "moderate"
    },
    "AIDS": {
        "description": "Acquired Immunodeficiency Syndrome — a chronic condition caused by HIV that damages the immune system.",
        "precautions": ["Take antiretroviral therapy", "Practice safe sex", "Avoid sharing needles", "Regular medical checkups"],
        "severity": "severe"
    },
    "Diabetes": {
        "description": "A metabolic disease that causes high blood sugar levels due to insulin issues.",
        "precautions": ["Monitor blood sugar regularly", "Follow a diabetic diet", "Exercise regularly", "Take prescribed medication"],
        "severity": "severe"
    },
    "Gastroenteritis": {
        "description": "Inflammation of the stomach and intestines, typically caused by viral or bacterial infection.",
        "precautions": ["Stay hydrated", "Rest", "Avoid solid food initially", "Practice good hand hygiene"],
        "severity": "moderate"
    },
    "Bronchial Asthma": {
        "description": "A chronic disease causing airways to become inflamed and narrow, making breathing difficult.",
        "precautions": ["Use inhalers as prescribed", "Avoid triggers like smoke", "Monitor peak flow", "Get flu vaccine"],
        "severity": "moderate"
    },
    "Hypertension": {
        "description": "High blood pressure — a long-term medical condition where blood pressure is persistently elevated.",
        "precautions": ["Reduce salt intake", "Exercise regularly", "Limit alcohol", "Take prescribed medication"],
        "severity": "severe"
    },
    "Migraine": {
        "description": "A neurological condition causing intense, debilitating headaches often with nausea and sensitivity to light.",
        "precautions": ["Identify and avoid triggers", "Maintain regular sleep", "Stay hydrated", "Use prescribed medications"],
        "severity": "moderate"
    },
    "Cervical spondylosis": {
        "description": "Age-related wear and tear affecting the spinal disks in the neck.",
        "precautions": ["Do neck exercises", "Maintain good posture", "Use ergonomic furniture", "Avoid heavy lifting"],
        "severity": "moderate"
    },
    "Paralysis (brain hemorrhage)": {
        "description": "Loss of muscle function caused by bleeding in or around the brain.",
        "precautions": ["Immediate emergency care required", "Rehabilitation therapy", "Blood pressure management", "Regular neurological checkups"],
        "severity": "critical"
    },
    "Jaundice": {
        "description": "A yellowing of the skin and eyes caused by excess bilirubin in the blood.",
        "precautions": ["Rest adequately", "Stay hydrated", "Avoid alcohol", "Follow a liver-friendly diet"],
        "severity": "moderate"
    },
    "Malaria": {
        "description": "A mosquito-borne infectious disease caused by Plasmodium parasites.",
        "precautions": ["Use mosquito nets", "Take antimalarial drugs", "Use insect repellent", "Wear protective clothing"],
        "severity": "severe"
    },
    "Chicken pox": {
        "description": "A highly contagious viral infection causing an itchy, blister-like rash.",
        "precautions": ["Avoid scratching", "Stay isolated", "Take antiviral medication", "Use calamine lotion"],
        "severity": "mild"
    },
    "Dengue": {
        "description": "A mosquito-borne viral disease causing high fever, severe headache, and joint pain.",
        "precautions": ["Rest and hydrate", "Use mosquito repellent", "Monitor platelet count", "Seek medical care immediately"],
        "severity": "severe"
    },
    "Typhoid": {
        "description": "A bacterial infection caused by Salmonella typhi, spread through contaminated food or water.",
        "precautions": ["Drink safe water", "Eat properly cooked food", "Get vaccinated", "Practice hand hygiene"],
        "severity": "severe"
    },
    "hepatitis A": {
        "description": "A viral liver infection spread through contaminated food and water.",
        "precautions": ["Get vaccinated", "Practice hand hygiene", "Avoid contaminated food/water", "Rest adequately"],
        "severity": "moderate"
    },
    "Hepatitis B": {
        "description": "A serious liver infection caused by the Hepatitis B virus, spread through body fluids.",
        "precautions": ["Get vaccinated", "Use protection during sex", "Avoid sharing needles", "Regular liver function tests"],
        "severity": "severe"
    },
    "Hepatitis C": {
        "description": "A viral infection causing liver inflammation, sometimes leading to cirrhosis.",
        "precautions": ["Avoid sharing needles", "Practice safe sex", "Avoid alcohol", "Antiviral treatment"],
        "severity": "severe"
    },
    "Hepatitis D": {
        "description": "A liver infection that only occurs alongside Hepatitis B.",
        "precautions": ["Hepatitis B vaccination", "Avoid sharing needles", "Practice safe sex", "Regular monitoring"],
        "severity": "severe"
    },
    "Hepatitis E": {
        "description": "A liver disease caused by the Hepatitis E virus, mainly spread through contaminated water.",
        "precautions": ["Drink clean water", "Practice food safety", "Rest", "Avoid alcohol"],
        "severity": "severe"
    },
    "Alcoholic hepatitis": {
        "description": "Liver inflammation caused by excessive alcohol consumption over time.",
        "precautions": ["Stop drinking alcohol", "Follow a nutritious diet", "Seek support groups", "Medical supervision"],
        "severity": "severe"
    },
    "Tuberculosis": {
        "description": "A serious infectious disease that mainly affects the lungs, caused by Mycobacterium tuberculosis.",
        "precautions": ["Complete full antibiotic course", "Wear a mask", "Ensure good ventilation", "Regular checkups"],
        "severity": "severe"
    },
    "Common Cold": {
        "description": "A viral infection of the upper respiratory tract, causing runny nose and sore throat.",
        "precautions": ["Rest", "Stay hydrated", "Use over-the-counter remedies", "Wash hands frequently"],
        "severity": "mild"
    },
    "Pneumonia": {
        "description": "An infection that inflames air sacs in one or both lungs, which may fill with fluid.",
        "precautions": ["Get vaccinated", "Quit smoking", "Practice hand hygiene", "Seek immediate medical care"],
        "severity": "severe"
    },
    "Dimorphic hemmorhoids(piles)": {
        "description": "Swollen veins in the lower rectum or anus causing pain, bleeding, and discomfort.",
        "precautions": ["Eat high-fiber diet", "Stay hydrated", "Avoid straining", "Take sitz baths"],
        "severity": "moderate"
    },
    "Heart attack": {
        "description": "A blockage of blood flow to the heart muscle, causing permanent damage if not treated immediately.",
        "precautions": ["Call emergency services immediately", "Chew aspirin", "Rest", "CPR if unconscious"],
        "severity": "critical"
    },
    "Varicose veins": {
        "description": "Enlarged, twisted veins that commonly appear in the legs and feet.",
        "precautions": ["Exercise regularly", "Elevate legs when resting", "Avoid long standing periods", "Wear compression stockings"],
        "severity": "mild"
    },
    "Hypothyroidism": {
        "description": "A condition where the thyroid gland doesn't produce enough thyroid hormone.",
        "precautions": ["Take prescribed medication", "Regular thyroid function tests", "Eat iodine-rich foods", "Manage stress"],
        "severity": "moderate"
    },
    "Hyperthyroidism": {
        "description": "A condition where the thyroid gland produces too much thyroid hormone.",
        "precautions": ["Take antithyroid medication", "Avoid iodine-rich foods", "Limit caffeine", "Regular checkups"],
        "severity": "moderate"
    },
    "Hypoglycemia": {
        "description": "Abnormally low blood sugar levels, which can cause serious complications if untreated.",
        "precautions": ["Eat regular meals", "Carry glucose tablets", "Monitor blood sugar", "Avoid alcohol on empty stomach"],
        "severity": "moderate"
    },
    "Osteoarthritis": {
        "description": "A degenerative joint disease causing cartilage to break down, leading to pain and stiffness.",
        "precautions": ["Exercise gently", "Maintain healthy weight", "Use joint supports", "Take prescribed pain relief"],
        "severity": "moderate"
    },
    "Arthritis": {
        "description": "Inflammation of one or more joints causing pain, swelling, and reduced range of motion.",
        "precautions": ["Physical therapy", "Anti-inflammatory medications", "Hot/cold therapy", "Stay active but rest when needed"],
        "severity": "moderate"
    },
    "(vertigo) Paroxysmal Positional Vertigo": {
        "description": "A condition causing brief episodes of dizziness from changes in head position.",
        "precautions": ["Epley maneuver", "Move slowly and carefully", "Avoid sudden head movements", "Consult a specialist"],
        "severity": "mild"
    },
    "Acne": {
        "description": "A skin condition occurring when hair follicles become plugged with oil and dead skin cells.",
        "precautions": ["Keep skin clean", "Use non-comedogenic products", "Avoid touching face", "Consult a dermatologist"],
        "severity": "mild"
    },
    "Urinary tract infection": {
        "description": "An infection in any part of the urinary system — kidneys, bladder, ureters, or urethra.",
        "precautions": ["Drink plenty of water", "Urinate frequently", "Wipe from front to back", "Take prescribed antibiotics"],
        "severity": "moderate"
    },
    "Psoriasis": {
        "description": "A chronic skin disease causing rapid buildup of skin cells resulting in scaling on skin surface.",
        "precautions": ["Moisturize skin regularly", "Avoid triggers like stress", "Use prescribed topical treatments", "Get sunlight exposure in moderation"],
        "severity": "moderate"
    },
    "Impetigo": {
        "description": "A highly contagious bacterial skin infection causing red sores on the face, nose, and mouth.",
        "precautions": ["Keep sores clean and covered", "Don't touch sores", "Wash hands frequently", "Use prescribed antibiotics"],
        "severity": "mild"
    },
}


# ── Pydantic models ─────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    symptoms: List[str]


class PredictResponse(BaseModel):
    disease: str
    confidence: float
    description: str
    precautions: List[str]
    severity: str
    top_predictions: List[dict]


# ── Routes ──────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Disease Prediction API is running"}


@app.get("/symptoms")
def get_symptoms():
    """Return list of all valid symptom names."""
    return {"symptoms": feature_cols}


@app.get("/diseases")
def get_diseases():
    """Return list of all predictable diseases."""
    return {"diseases": list(label_encoder.classes_)}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.symptoms:
        raise HTTPException(status_code=400, detail="At least one symptom is required")

    # Validate symptoms
    invalid = [s for s in request.symptoms if s not in feature_cols]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Unknown symptoms: {invalid}")

    # Build feature vector
    input_vec = np.zeros(len(feature_cols))
    for symptom in request.symptoms:
        idx = feature_cols.index(symptom)
        input_vec[idx] = 1

    # Predict
    proba = model.predict_proba([input_vec])[0]
    top_indices = np.argsort(proba)[::-1][:5]

    top_predictions = [
        {"disease": label_encoder.classes_[i], "confidence": round(float(proba[i]) * 100, 1)}
        for i in top_indices if proba[i] > 0
    ]

    best_idx = top_indices[0]
    disease_name = label_encoder.classes_[best_idx]
    confidence = float(proba[best_idx])

    info = DISEASE_INFO.get(disease_name, {
        "description": "A medical condition requiring professional evaluation.",
        "precautions": ["Consult a doctor immediately"],
        "severity": "unknown"
    })

    return PredictResponse(
        disease=disease_name,
        confidence=round(confidence * 100, 1),
        description=info["description"],
        precautions=info["precautions"],
        severity=info["severity"],
        top_predictions=top_predictions
    )
