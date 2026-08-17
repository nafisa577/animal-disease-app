import glob
import os
import urllib.parse
from datetime import datetime

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---- 7 disease classes (SAME order the model was trained in) ----
CLASS_NAMES = [
    "cow_foot_and_mouth", "cow_healthy", "cow_lumpy",
    "poultry_cocci", "poultry_healthy", "poultry_ncd", "poultry_salmo",
]

# readable names shown to the farmer
NICE_NAMES = {
    "cow_foot_and_mouth": "Foot & Mouth Disease (Cow)",
    "cow_healthy":        "Healthy Cow",
    "cow_lumpy":          "Lumpy Skin Disease (Cow)",
    "poultry_cocci":      "Coccidiosis (Poultry)",
    "poultry_healthy":    "Healthy Poultry",
    "poultry_ncd":        "Newcastle Disease (Poultry)",
    "poultry_salmo":      "Salmonellosis (Poultry)",
}

# ---- treatment + prevention guidance for each disease ----
TREATMENT = {
    "cow_foot_and_mouth": {
        "about": "Highly contagious viral disease causing blisters on the mouth, tongue and feet.",
        "treat": [
            "Isolate the infected animal immediately from the herd.",
            "Clean mouth and foot sores with a mild antiseptic solution.",
            "Give soft, easy-to-eat feed and plenty of clean water.",
            "Vet may prescribe antibiotics to prevent secondary infection.",
        ],
        "prevent": [
            "Vaccinate the herd regularly.",
            "Maintain farm hygiene and control animal movement.",
            "Report to local veterinary authority (often a notifiable disease).",
        ],
    },
    "cow_lumpy": {
        "about": "Viral skin disease spread by biting insects; causes firm lumps/nodules on the skin.",
        "treat": [
            "Isolate the animal and control flies/mosquitoes around it.",
            "Clean skin nodules; vet may prescribe antibiotics for secondary infection.",
            "Provide anti-inflammatory / pain relief as advised by a vet.",
            "Ensure good nutrition and clean water to support recovery.",
        ],
        "prevent": [
            "Vaccinate cattle against Lumpy Skin Disease.",
            "Control biting insects (vectors) with sprays/nets.",
            "Isolate new or sick animals before mixing with the herd.",
        ],
    },
    "poultry_cocci": {
        "about": "Parasitic gut disease (Eimeria); causes bloody droppings, weakness and weight loss.",
        "treat": [
            "Vet-prescribed anticoccidial medicine (e.g. amprolium) in drinking water.",
            "Keep litter clean and DRY — wet litter spreads the parasite.",
            "Provide vitamins/electrolytes to help recovery.",
            "Separate sick birds from healthy ones.",
        ],
        "prevent": [
            "Good sanitation and dry bedding; avoid overcrowding.",
            "Use coccidiosis vaccine or medicated feed where advised.",
            "Clean and disinfect feeders and drinkers regularly.",
        ],
    },
    "poultry_ncd": {
        "about": "Newcastle Disease — a very contagious, often fatal viral disease of poultry.",
        "treat": [
            "There is no specific cure — provide supportive care only.",
            "Isolate infected birds immediately; strict biosecurity.",
            "Disinfect the shed, feeders and drinkers thoroughly.",
            "Follow the vet's advice on managing or removing affected birds.",
        ],
        "prevent": [
            "Vaccination is the MOST important prevention.",
            "Strong biosecurity: control visitors, new birds and equipment.",
            "Report outbreaks to veterinary authorities (notifiable disease).",
        ],
    },
    "poultry_salmo": {
        "about": "Bacterial infection (Salmonella); causes diarrhoea, weakness and drop in production. Can infect humans too.",
        "treat": [
            "Vet-prescribed antibiotics (correct dose is important).",
            "Provide clean water and probiotics/electrolytes.",
            "Remove contaminated feed and water; deep-clean the shed.",
            "Handle birds carefully and wash hands — it is zoonotic.",
        ],
        "prevent": [
            "Clean water, clean feed and good shed hygiene.",
            "Rodent and pest control (they spread Salmonella).",
            "Buy chicks from certified Salmonella-free sources.",
        ],
    },
}

HEALTHY_CLASSES = {"cow_healthy", "poultry_healthy"}


# find and load the .keras model automatically (any filename works)
@st.cache_resource
def load_model():
    files = [f for f in glob.glob("*.keras") if os.path.isfile(f)]
    if not files:
        st.error("No .keras model file found in this folder. "
                 "Put your model file (e.g. MobileNetV3Large_BEST.keras) next to app.py.")
        st.stop()
    return tf.keras.models.load_model(files[0])


model = load_model()

# ---- page ----
st.title("🐄 Animal Disease Detection")
st.write("Upload an animal image to detect possible disease and get guidance.")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    # preprocess: resize to 224x224, keep pixels in [0..255]
    # (the model normalizes internally, so we do NOT divide by 255)
    img = image.resize((224, 224))
    arr = np.array(img, dtype="float32")
    arr = np.expand_dims(arr, axis=0)          # shape -> (1, 224, 224, 3)

    # predict
    preds = model.predict(arr)[0]
    idx = int(np.argmax(preds))
    key = CLASS_NAMES[idx]
    confidence = float(preds[idx]) * 100
    result = NICE_NAMES.get(key, key)

    st.subheader(f"Result: {result}")
    st.write(f"Confidence: **{confidence:.2f}%**")

    if confidence < 60:
        st.warning("⚠️ The model is not very confident. Please verify with a veterinarian.")

    # ---------- HEALTHY ----------
    if key in HEALTHY_CLASSES:
        st.success("✅ No disease detected. The animal appears healthy.")
        st.write("**Keep it healthy:** clean housing, balanced feed, clean water, "
                 "routine vaccination and deworming.")

    # ---------- DISEASE: treatment + vet alert ----------
    else:
        info = TREATMENT.get(key)
        if info:
            st.error(f"🚨 Disease detected: **{result}**")

            st.markdown("### 🩺 Treatment Suggestion")
            st.write(f"*{info['about']}*")
            st.write("**What to do:**")
            for t in info["treat"]:
                st.write(f"- {t}")
            st.write("**Prevention:**")
            for p in info["prevent"]:
                st.write(f"- {p}")

            # ---------- Vet alert ----------
            st.markdown("### 📣 Notify a Veterinarian")
            vet_email = st.text_input("Veterinarian's email (optional)")

            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            message = (
                f"Animal Disease Alert\n"
                f"Date/Time: {now}\n"
                f"Detected: {result}\n"
                f"Confidence: {confidence:.2f}%\n"
                f"Please advise on treatment."
            )

            if st.button("🚨 Send Alert to Veterinarian"):
                # save a local record of the alert
                with open("vet_alerts.log", "a", encoding="utf-8") as f:
                    f.write(message + "\n----\n")
                st.success("Alert prepared and saved! ✅")
                st.text(message)

                # if an email is given, offer a ready-to-send email link
                if vet_email:
                    subject = urllib.parse.quote("Animal Disease Alert")
                    body = urllib.parse.quote(message)
                    mailto = f"mailto:{vet_email}?subject={subject}&body={body}"
                    st.markdown(f"[📧 Click here to email the vet]({mailto})")

    # ---------- all probabilities ----------
    st.write("---")
    st.write("All class probabilities:")
    order = np.argsort(preds)[::-1]            # highest first
    for i in order:
        name = NICE_NAMES.get(CLASS_NAMES[i], CLASS_NAMES[i])
        st.write(f"- {name}: {float(preds[i]) * 100:.2f}%")

    st.write("---")
    st.caption("⚠️ This is AI-generated guidance for information only. "
               "Always consult a licensed veterinarian for diagnosis and treatment.")
