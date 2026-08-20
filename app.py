import glob
import os
import urllib.parse
from datetime import datetime

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ============================================================
#  BRANDING  (change these two lines to rename the app)
# ============================================================
BRAND_NAME = "PashuScan"
BRAND_TAGLINE = "AI livestock health screening"

# ---- 7 disease classes (SAME order the model was trained in) ----
CLASS_NAMES = [
    "cow_foot_and_mouth", "cow_healthy", "cow_lumpy",
    "poultry_cocci", "poultry_healthy", "poultry_ncd", "poultry_salmo",
]

NICE_NAMES = {
    "cow_foot_and_mouth": "Foot & Mouth Disease (Cow)",
    "cow_healthy":        "Healthy Cow",
    "cow_lumpy":          "Lumpy Skin Disease (Cow)",
    "poultry_cocci":      "Coccidiosis (Poultry)",
    "poultry_healthy":    "Healthy Poultry",
    "poultry_ncd":        "Newcastle Disease (Poultry)",
    "poultry_salmo":      "Salmonellosis (Poultry)",
}

TREATMENT = {
    "cow_foot_and_mouth": {
        "about": "Highly contagious viral disease causing blisters on the mouth, tongue and feet.",
        "treat": [
            "Isolate the infected animal immediately from the herd.",
            "Clean mouth and foot sores with a mild antiseptic solution.",
            "Give soft, easy-to-eat feed and plenty of clean water.",
            "A vet may prescribe antibiotics to prevent secondary infection.",
        ],
        "prevent": [
            "Vaccinate the herd regularly.",
            "Maintain farm hygiene and control animal movement.",
            "Report to the local veterinary authority (often notifiable).",
        ],
    },
    "cow_lumpy": {
        "about": "Viral skin disease spread by biting insects; causes firm lumps on the skin.",
        "treat": [
            "Isolate the animal and control flies and mosquitoes around it.",
            "Clean skin nodules; a vet may prescribe antibiotics for infection.",
            "Give anti-inflammatory or pain relief as advised by a vet.",
            "Ensure good nutrition and clean water to support recovery.",
        ],
        "prevent": [
            "Vaccinate cattle against Lumpy Skin Disease.",
            "Control biting insects with sprays or nets.",
            "Isolate new or sick animals before mixing with the herd.",
        ],
    },
    "poultry_cocci": {
        "about": "Parasitic gut disease (Eimeria); causes bloody droppings, weakness and weight loss.",
        "treat": [
            "Give a vet-prescribed anticoccidial (e.g. amprolium) in drinking water.",
            "Keep litter clean and dry — wet litter spreads the parasite.",
            "Provide vitamins and electrolytes to aid recovery.",
            "Separate sick birds from healthy ones.",
        ],
        "prevent": [
            "Good sanitation and dry bedding; avoid overcrowding.",
            "Use a coccidiosis vaccine or medicated feed where advised.",
            "Clean and disinfect feeders and drinkers regularly.",
        ],
    },
    "poultry_ncd": {
        "about": "Newcastle Disease — a very contagious, often fatal viral disease of poultry.",
        "treat": [
            "There is no specific cure — give supportive care only.",
            "Isolate infected birds immediately; keep strict biosecurity.",
            "Disinfect the shed, feeders and drinkers thoroughly.",
            "Follow the vet's advice on managing affected birds.",
        ],
        "prevent": [
            "Vaccination is the most important prevention.",
            "Strong biosecurity: control visitors, new birds and equipment.",
            "Report outbreaks to veterinary authorities (notifiable).",
        ],
    },
    "poultry_salmo": {
        "about": "Bacterial infection (Salmonella); causes diarrhoea and weakness. Can infect humans.",
        "treat": [
            "Give vet-prescribed antibiotics at the correct dose.",
            "Provide clean water and probiotics or electrolytes.",
            "Remove contaminated feed and water; deep-clean the shed.",
            "Handle birds carefully and wash hands — it is zoonotic.",
        ],
        "prevent": [
            "Clean water, clean feed and good shed hygiene.",
            "Control rodents and pests (they spread Salmonella).",
            "Buy chicks from certified Salmonella-free sources.",
        ],
    },
}

HEALTHY_CLASSES = {"cow_healthy", "poultry_healthy"}

st.set_page_config(page_title=f"{BRAND_NAME} — Animal Disease Detection",
                   page_icon="🩺", layout="centered")

# ============================================================
#  STYLE
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

:root{
  --ink:#14231c; --pine:#0f5137; --pine-2:#0b3d29;
  --bg:#eef3ee; --card:#ffffff; --line:#e0e8e0; --muted:#5d6c63;
  --healthy:#1f8a53; --amber:#bd7c1c; --alert:#c0392b;
}

/* hide streamlit chrome */
#MainMenu, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], footer {visibility:hidden; height:0;}
[data-testid="stHeader"]{background:transparent;}

.stApp{background:var(--bg);}
html, body, .stApp, p, div, span, label, input, button, li{
  font-family:'Inter', sans-serif; color:var(--ink);
}
.block-container{max-width:760px; padding-top:1rem; padding-bottom:3rem;}

/* brand bar */
.ad-brand{display:flex; align-items:center; gap:.7rem; margin-bottom:1.4rem;}
.ad-brand-name{font-family:'Sora'; font-weight:700; font-size:1.15rem; letter-spacing:-.01em;}
.ad-brand-sub{font-size:.78rem; color:var(--muted); margin-top:-2px;}

/* hero */
.ad-hero{background:var(--card); border:1px solid var(--line); border-radius:18px;
  padding:1.6rem 1.6rem 1.7rem; box-shadow:0 1px 2px rgba(20,35,28,.04);}
.ad-eyebrow{font-family:'Sora'; font-weight:600; font-size:.72rem; letter-spacing:.14em;
  color:var(--pine); text-transform:uppercase;}
.ad-title{font-family:'Sora'; font-weight:700; font-size:1.9rem; line-height:1.1;
  margin:.4rem 0 .5rem; letter-spacing:-.02em;}
.ad-lede{color:var(--muted); font-size:.98rem; margin:0; max-width:48ch;}

.ad-h{font-family:'Sora'; font-weight:600; font-size:.8rem; letter-spacing:.12em;
  text-transform:uppercase; color:var(--muted); margin:1.8rem 0 .7rem;}

/* result readout */
.ad-result{background:var(--card); border:1px solid var(--line); border-radius:18px;
  padding:1.4rem 1.5rem; box-shadow:0 1px 2px rgba(20,35,28,.04); border-left:6px solid var(--pine);}
.ad-chip{display:inline-block; font-family:'Sora'; font-weight:600; font-size:.72rem;
  letter-spacing:.1em; text-transform:uppercase; padding:.32rem .7rem; border-radius:999px; color:#fff;}
.ad-dx{font-family:'Sora'; font-weight:700; font-size:1.5rem; margin:.7rem 0 .1rem; letter-spacing:-.01em;}
.ad-conf-row{display:flex; align-items:baseline; justify-content:space-between; margin-top:1rem;}
.ad-conf-lab{font-size:.78rem; color:var(--muted); letter-spacing:.04em;}
.ad-conf-num{font-family:'IBM Plex Mono'; font-weight:600; font-size:1.05rem;}
.ad-meter{height:9px; background:#eef1ee; border-radius:99px; overflow:hidden; margin-top:.45rem;}
.ad-meter-fill{height:100%; border-radius:99px;}
.ad-note{font-size:.82rem; color:var(--amber); margin-top:.7rem;}

/* treatment card */
.ad-card{background:var(--card); border:1px solid var(--line); border-radius:16px;
  padding:1.2rem 1.4rem; margin-top:1rem;}
.ad-about{font-size:.92rem; color:var(--muted); font-style:italic; margin:0 0 .9rem;}
.ad-sub{font-family:'Sora'; font-weight:600; font-size:.9rem; margin:.9rem 0 .4rem;}
.ad-list{margin:0; padding-left:1.1rem;}
.ad-list li{margin:.28rem 0; font-size:.93rem; line-height:1.45;}

/* probability bars */
.ad-pb{margin:.5rem 0;}
.ad-pb-top{display:flex; justify-content:space-between; font-size:.85rem; margin-bottom:.25rem;}
.ad-pb-num{font-family:'IBM Plex Mono'; color:var(--muted);}
.ad-pb-track{height:7px; background:#eef1ee; border-radius:99px; overflow:hidden;}
.ad-pb-fill{height:100%; border-radius:99px; background:var(--line);}
.ad-pb-fill.lead{background:var(--pine);}

/* streamlit widgets */
.stButton>button{background:var(--pine); color:#fff; border:none; border-radius:11px;
  padding:.6rem 1.1rem; font-family:'Sora'; font-weight:600; width:100%; transition:background .15s;}
.stButton>button:hover{background:var(--pine-2); color:#fff;}
.stTextInput input{border-radius:11px; border:1px solid var(--line);}
[data-testid="stFileUploaderDropzone"]{background:var(--card); border:1.5px dashed #b9ccbe; border-radius:14px;}
[data-testid="stImage"] img{border-radius:14px;}

/* footer */
.ad-footer{margin-top:2.4rem; padding-top:1.2rem; border-top:1px solid var(--line);
  font-size:.8rem; color:var(--muted); line-height:1.6;}
.ad-footer strong{color:var(--ink); font-weight:600;}
</style>
""", unsafe_allow_html=True)

LOGO = """
<svg width="36" height="36" viewBox="0 0 36 36" fill="none">
  <rect width="36" height="36" rx="10" fill="#0f5137"/>
  <path d="M6 19 h5 l2.6 -8 l4.2 15 l3.1 -9 l2 3 h5"
        fill="none" stroke="#7fe0ab" stroke-width="2.3"
        stroke-linecap="round" stroke-linejoin="round"/>
</svg>
"""

# ---- brand + hero ----
st.markdown(f"""
<div class="ad-brand">
  {LOGO}
  <div>
    <div class="ad-brand-name">{BRAND_NAME}</div>
    <div class="ad-brand-sub">{BRAND_TAGLINE}</div>
  </div>
</div>
<div class="ad-hero">
  <div class="ad-eyebrow">Cattle · Poultry</div>
  <div class="ad-title">Animal Disease Detection</div>
  <p class="ad-lede">Take or upload a clear photo of your animal to screen for common
  diseases and get first-response guidance.</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    files = [f for f in glob.glob("*.keras") if os.path.isfile(f)]
    if not files:
        st.error("No .keras model file found in this folder. "
                 "Place the model file (e.g. MobileNetV3Large_BEST.keras) next to app.py.")
        st.stop()
    return tf.keras.models.load_model(files[0])


model = load_model()

st.markdown('<div class="ad-h">Upload a photo</div>', unsafe_allow_html=True)
uploaded = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"],
                            label_visibility="collapsed")

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, use_container_width=True)

    img = image.resize((224, 224))
    arr = np.expand_dims(np.array(img, dtype="float32"), axis=0)

    preds = model.predict(arr)[0]
    idx = int(np.argmax(preds))
    key = CLASS_NAMES[idx]
    confidence = float(preds[idx]) * 100
    result = NICE_NAMES.get(key, key)

    healthy = key in HEALTHY_CLASSES
    color = "var(--healthy)" if healthy else "var(--alert)"
    status = "Healthy" if healthy else "Disease detected"

    low = '<div class="ad-note">Low confidence — please verify with a veterinarian.</div>' \
          if confidence < 60 else ""

    st.markdown(f"""
    <div class="ad-result" style="border-left-color:{color}">
      <span class="ad-chip" style="background:{color}">{status}</span>
      <div class="ad-dx">{result}</div>
      <div class="ad-conf-row">
        <span class="ad-conf-lab">Confidence</span>
        <span class="ad-conf-num">{confidence:.1f}%</span>
      </div>
      <div class="ad-meter"><div class="ad-meter-fill"
           style="width:{confidence:.1f}%; background:{color}"></div></div>
      {low}
    </div>
    """, unsafe_allow_html=True)

    if healthy:
        st.markdown("""
        <div class="ad-card">
          <p class="ad-about">No disease detected — the animal appears healthy.</p>
          <div class="ad-sub">Keep it healthy</div>
          <ul class="ad-list">
            <li>Clean housing and dry bedding.</li>
            <li>Balanced feed and clean drinking water.</li>
            <li>Routine vaccination and deworming.</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        info = TREATMENT.get(key, {})
        treat = "".join(f"<li>{t}</li>" for t in info.get("treat", []))
        prevent = "".join(f"<li>{p}</li>" for p in info.get("prevent", []))
        st.markdown(f"""
        <div class="ad-card">
          <p class="ad-about">{info.get('about','')}</p>
          <div class="ad-sub">What to do</div>
          <ul class="ad-list">{treat}</ul>
          <div class="ad-sub">Prevention</div>
          <ul class="ad-list">{prevent}</ul>
        </div>
        """, unsafe_allow_html=True)

        # ---- vet alert ----
        st.markdown('<div class="ad-h">Notify a veterinarian</div>', unsafe_allow_html=True)
        vet_email = st.text_input("Veterinarian's email (optional)",
                                  label_visibility="collapsed",
                                  placeholder="Veterinarian's email (optional)")
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        message = (f"Animal Disease Alert\nDate/Time: {now}\nDetected: {result}\n"
                   f"Confidence: {confidence:.1f}%\nPlease advise on treatment.")

        if st.button("Send alert to veterinarian"):
            with open("vet_alerts.log", "a", encoding="utf-8") as f:
                f.write(message + "\n----\n")
            st.success("Alert prepared and saved.")
            st.code(message, language=None)
            if vet_email:
                mailto = (f"mailto:{vet_email}"
                          f"?subject={urllib.parse.quote('Animal Disease Alert')}"
                          f"&body={urllib.parse.quote(message)}")
                st.markdown(f"[Open a ready-to-send email →]({mailto})")

    # ---- probability breakdown ----
    st.markdown('<div class="ad-h">Full breakdown</div>', unsafe_allow_html=True)
    bars = ""
    for i in np.argsort(preds)[::-1]:
        name = NICE_NAMES.get(CLASS_NAMES[i], CLASS_NAMES[i])
        pct = float(preds[i]) * 100
        lead = "lead" if i == idx else ""
        bars += f"""
        <div class="ad-pb">
          <div class="ad-pb-top"><span>{name}</span>
               <span class="ad-pb-num">{pct:.1f}%</span></div>
          <div class="ad-pb-track"><div class="ad-pb-fill {lead}"
               style="width:{max(pct,1):.1f}%"></div></div>
        </div>"""
    st.markdown(f'<div class="ad-card">{bars}</div>', unsafe_allow_html=True)

# ---- footer ----
st.markdown(f"""
<div class="ad-footer">
  <strong>{BRAND_NAME}</strong> · International Islamic University Chittagong — Dept. of CSE<br>
  Undergraduate thesis project. For informational screening only —
  always consult a licensed veterinarian for diagnosis and treatment.
</div>
""", unsafe_allow_html=True)
