import streamlit as st
import easyocr
import numpy as np
from PIL import Image
import spacy
import pandas as pd
import torch

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="FABET - EasyOCR", layout="wide")

# --- CHARGEMENT DES MODÈLES (Caching pour la vitesse) ---
@st.cache_resource
def load_models():
    # Chargement d'EasyOCR (mode CPU pour le cloud)
    reader = easyocr.Reader(['fr'], gpu=False)
    
    # Chargement de spaCy (le modèle sera déjà installé via requirements.txt)
    try:
        nlp = spacy.load("fr_core_news_sm")
    except OSError:
        # Solution de secours au cas où
        import os
        os.system("python -m spacy download fr_core_news_sm")
        nlp = spacy.load("fr_core_news_sm")
        
    return reader, nlp

reader, nlp = load_models()

# --- STYLE CSS (Dark Mode Minimaliste) ---
st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: white; }
    .reportview-container .main { background: #0E1117; }
    .card {
        background: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333;
        margin-bottom: 20px;
    }
    h1 { color: #00FFCC !important; font-family: 'Inter', sans-serif; }
    .stMetric { background: #262730; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# --- LOGIQUE DE RECONSTRUCTION DE TEXTE ---
def reconstruct_text(results):
    """
    EasyOCR renvoie : [[box], texte, confiance]
    Cette fonction trie les blocs par position verticale pour recréer des lignes.
    """
    # Trier par coordonnée Y (haut de la boîte)
    results.sort(key=lambda x: x[0][0][1])
    
    full_text = ""
    current_y = results[0][0][0][1] if results else 0
    line_threshold = 15 # Tolérance pour considérer que c'est la même ligne
    
    for (bbox, text, prob) in results:
        if abs(bbox[0][1] - current_y) > line_threshold:
            full_text += "\n" # Nouvelle ligne
        else:
            full_text += " " # Même ligne, on ajoute un espace
        
        full_text += text
        current_y = bbox[0][1]
        
    return full_text.strip()

# --- INTERFACE ---
st.title("FABET TRANSCRIPT")
st.markdown("#### Moteur Deep Learning : EasyOCR 🚀")

uploaded_file = st.file_uploader("Importer une photo du document", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_np = np.array(image)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(image, caption="Document Source", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.spinner("L'IA analyse le document..."):
        # 1. Extraction avec EasyOCR
        raw_results = reader.readtext(img_np)
        
        # 2. Reconstruction structurée
        if raw_results:
            text_extracted = reconstruct_text(raw_results)
            avg_conf = np.mean([res[2] for res in raw_results]) * 100
        else:
            text_extracted = ""
            avg_conf = 0

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        # Métriques
        m1, m2 = st.columns(2)
        m1.metric("Confiance IA", f"{avg_conf:.1f}%")
        m2.metric("Mots détectés", len(text_extracted.split()))
        
        st.markdown("---")
        st.subheader("Texte Extrait")
        final_text = st.text_area("", text_extracted, height=300)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- SECTION DATA MINING ---
    if text_extracted:
        st.markdown("### 📊 Data Mining & Intelligence")
        doc = nlp(text_extracted)
        
        tab1, tab2, tab3 = st.tabs(["💎 Entités", "🔍 Mots-clés", "📂 Classification"])
        
        with tab1:
            # Extraction des noms, lieux, dates
            entities = [[ent.text, ent.label_] for ent in doc.ents]
            if entities:
                df_ent = pd.DataFrame(entities, columns=["Texte", "Type"])
                st.dataframe(df_ent, use_container_width=True)
            else:
                st.info("Aucune entité (nom, date) détectée.")

        with tab2:
            # Analyse des noms communs les plus importants
            keywords = [token.lemma_ for token in doc if token.pos_ == "NOUN" and not token.is_stop]
            if keywords:
                df_keys = pd.Series(keywords).value_counts().head(10)
                st.bar_chart(df_keys)
            else:
                st.write("Pas assez de contenu pour extraire des mots-clés.")

        with tab3:
            # Logique de classification simple
            category = "Général"
            low_text = text_extracted.lower()
            if any(x in low_text for x in ["facture", "total", "ttc", "montant"]):
                category = "Finance / Facturation"
            elif any(x in low_text for x in ["contrat", "article", "loi", "signé"]):
                category = "Juridique / Administratif"
            
            st.success(f"Document classifié comme : **{category}**")

    # Téléchargement

    st.download_button("Télécharger la transcription", final_text, file_name="fabet_ocr.txt")
