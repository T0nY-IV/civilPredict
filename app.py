import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Configurer la page Streamlit
st.set_page_config(
    page_title="BétonPredict - Résistance du Béton",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injection de styles CSS personnalisés pour un look premium et moderne (mode clair)
st.markdown("""
    <style>
        /* Importation d'une police moderne */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }
        
        /* Personnalisation du conteneur principal */
        .main-header {
            background: linear-gradient(135deg, #2E7D32, #1B5E20);
            padding: 30px;
            border-radius: 16px;
            color: white;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 4px 20px rgba(46, 125, 50, 0.15);
        }
        
        .main-header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            color: #ffffff !important;
        }
        
        .main-header p {
            font-size: 1.1rem;
            margin-top: 10px;
            margin-bottom: 0;
            opacity: 0.9;
        }
        
        /* Cartes de résultats */
        .result-card {
            background-color: #ffffff;
            border-radius: 16px;
            padding: 30px;
            border-left: 6px solid #2E7D32;
            box-shadow: 0 10px 30px rgba(0,0,0,0.05);
            text-align: center;
            margin-top: 20px;
            animation: fadeIn 0.6s ease-out;
        }
        
        .result-value {
            font-size: 3.5rem;
            font-weight: 700;
            color: #1B5E20;
            margin: 10px 0;
        }
        
        .result-label {
            font-size: 1.2rem;
            color: #555;
            text-transform: uppercase;
            letter-spacing: 1.5px;
        }

        .category-badge {
            display: inline-block;
            padding: 6px 16px;
            border-radius: 50px;
            font-size: 0.95rem;
            font-weight: 600;
            margin-top: 10px;
            color: white;
        }
        
        /* Styles pour les badges de catégorie */
        .badge-faible { background-color: #E53935; }
        .badge-standard { background-color: #1E88E5; }
        .badge-haute { background-color: #43A047; }
        .badge-tres-haute { background-color: #8E24AA; }

        /* Animation */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Amélioration de l'affichage des metrics */
        div[data-testid="stMetric"] {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 12px;
            border: 1px solid #E0E0E0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.02);
        }
    </style>
""", unsafe_allow_html=True)

# Charger le modèle exporté
@st.cache_resource
def load_model():
    model_path = Path("concrete_strength_gradient_boosting.pkl")
    if model_path.exists():
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

# --- En-tête Principal ---
st.markdown("""
    <div class="main-header">
        <h1>🛠️ BétonPredict</h1>
        <p>Prédisez la résistance à la compression du béton (à 28 jours ou autres) en fonction de sa composition chimique et de sa formulation.</p>
    </div>
""", unsafe_allow_html=True)

# --- Sidebar : Informations et Performances du modèle ---
with st.sidebar:
    st.markdown("### 📊 Informations du Modèle")
    st.info("""
        **Modèle :** Gradient Boosting Regressor  
        Ce modèle de Machine Learning a été entraîné et validé sur un dataset de **302 formulations** expérimentales de béton.
    """)
    
    st.markdown("#### 🎯 Métriques de Performance")
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric(label="Précision (R²)", value="93.8 %")
        st.metric(label="Erreur MAE", value="2.52 MPa")
    with col_m2:
        st.metric(label="Erreur RMSE", value="3.44 MPa")
        
    st.markdown("---")
    st.markdown("💡 *Développé pour l'aide à la formulation en laboratoire et sur chantier.*")

if model is None:
    st.error("⚠️ Le fichier du modèle `concrete_strength_gradient_boosting.pkl` est introuvable. Veuillez d'abord exécuter le script d'entraînement.")
else:
    # --- Formulaire d'entrées ---
    st.subheader("🧪 Paramètres de formulation du béton")
    st.write("Ajustez les composants pour simuler votre formulation :")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 🧱 Liants & Eau")
        ciment = st.number_input(
            "Ciment (kg/m³)",
            min_value=200.0,
            max_value=540.0,
            value=350.0,
            step=10.0,
            help="Dosage en ciment par mètre cube de béton."
        )
        eau = st.number_input(
            "Eau (kg/m³)",
            min_value=140.0,
            max_value=300.0,
            value=190.0,
            step=5.0,
            help="Quantité d'eau totale de gâchage."
        )
        superplastifiant = st.number_input(
            "Superplastifiant (kg/m³)",
            min_value=0.0,
            max_value=30.0,
            value=3.0,
            step=0.5,
            help="Adjuvant réducteur d'eau hautement actif."
        )
        
    with col2:
        st.markdown("##### 🪨 Granulats & Âge")
        granulats_grossiers = st.number_input(
            "Granulats Grossiers / Gravier (kg/m³)",
            min_value=420.0,
            max_value=1375.0,
            value=1030.0,
            step=10.0,
            help="Dosage en gravier."
        )
        granulats_fins = st.number_input(
            "Granulats Fins / Sable (kg/m³)",
            min_value=420.0,
            max_value=945.0,
            value=760.0,
            step=10.0,
            help="Dosage en sable fin."
        )
        age = st.number_input(
            "Âge du béton (jours)",
            min_value=1,
            max_value=365,
            value=28,
            step=1,
            help="Âge de maturation du béton avant essai d'écrasement."
        )

    # --- Ratios calculés en temps réel ---
    st.markdown("##### 📈 Indicateurs et Ratios de Formulation")
    
    # Calculs
    ec_ratio = eau / ciment if ciment > 0 else 0
    gs_ratio = granulats_grossiers / granulats_fins if granulats_fins > 0 else 0
    
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.metric(
            label="Rapport Eau / Ciment (E/C)",
            value=f"{ec_ratio:.3f}",
            help="Le rapport E/C est le principal indicateur de la porosité et de la résistance théorique du béton."
        )
    with col_r2:
        st.metric(
            label="Rapport Gravier / Sable (G/S)",
            value=f"{gs_ratio:.3f}",
            help="Le rapport G/S définit le squelette granulaire du béton."
        )
    with col_r3:
        somme_poids = ciment + eau + superplastifiant + granulats_grossiers + granulats_fins
        st.metric(
            label="Masse Volumique Estimée",
            value=f"{somme_poids:.0f} kg/m³",
            help="Masse théorique d'un mètre cube de béton fraîchement préparé."
        )

    st.markdown("---")
    
    # Bouton de prédiction centré
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    predict_btn = st.button("🚀 Prédire la Résistance à la Compression", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    if predict_btn:
        # Préparation des données pour le modèle
        input_data = {
            'Ciment': [ciment],
            'Eau': [eau],
            'Superplastifiant': [superplastifiant],
            'Granulats_grossiers': [granulats_grossiers],
            'Granulats_fins': [granulats_fins],
            'G/S': [gs_ratio],
            'E/C': [ec_ratio],
            'Age': [age]
        }
        input_df = pd.DataFrame(input_data)
        
        # Faire la prédiction
        with st.spinner("Calcul de la résistance en cours..."):
            predicted_strength = model.predict(input_df)[0]
            
        # Catégorisation du béton selon sa résistance
        if predicted_strength < 20:
            cat_label = "Béton de faible résistance (Non structurel)"
            badge_class = "badge-faible"
            advice = "Ce béton convient aux travaux légers (dalles de propreté, remplissage). Ne convient pas aux éléments porteurs."
        elif 20 <= predicted_strength < 40:
            cat_label = "Béton standard de structure"
            badge_class = "badge-standard"
            advice = "Recommandé pour la construction courante de bâtiments (poteaux, poutres, dalles de fondations, etc.)."
        elif 40 <= predicted_strength < 60:
            cat_label = "Béton de haute performance (BHP)"
            badge_class = "badge-haute"
            advice = "Parfaitement adapté aux ouvrages d'art, ponts, et structures fortement sollicitées."
        else:
            cat_label = "Béton de très haute performance (BTHP)"
            badge_class = "badge-tres-haute"
            advice = "Conçu pour des applications industrielles ou de génie civil spécifiques nécessitant une résistance extrême."

        # Affichage du résultat dans une carte stylisée
        st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Résistance à la compression estimée</div>
                <div class="result-value">{predicted_strength:.2f} MPa</div>
                <div class="category-badge {badge_class}">{cat_label}</div>
                <p style="margin-top: 15px; font-style: italic; color: #666;">{advice}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Info supplémentaire sur la physique
        st.markdown("<br>", unsafe_allow_html=True)
        st.info(f"💡 **Note technique :** Avec un rapport E/C de **{ec_ratio:.2f}** et un âge de maturation de **{age} jours**, le modèle estime la résistance maximale probable à **{predicted_strength:.2f} MPa** sous réserve d'une mise en œuvre optimale en laboratoire.")
