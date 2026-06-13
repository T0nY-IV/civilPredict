import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BétonPredict",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Session state: default page ─────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "direct"

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }

/* ── Top Navbar ── */
.navbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 32px;
    height: 64px;
    background: linear-gradient(90deg, #1B2A3B 0%, #243447 100%);
    border-radius: 14px;
    margin-bottom: 28px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.18);
}
.navbar-brand {
    font-size: 1.45rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 0.5px;
    white-space: nowrap;
}
.navbar-brand span { color: #64B5F6; }
.navbar-links {
    display: flex;
    gap: 8px;
}
.nav-btn {
    background: transparent;
    border: 2px solid transparent;
    border-radius: 8px;
    color: #B0BEC5;
    font-family: 'Outfit', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    padding: 7px 20px;
    cursor: pointer;
    transition: all 0.2s;
    white-space: nowrap;
}
.nav-btn:hover  { background: rgba(255,255,255,0.08); color: #ffffff; }
.nav-btn.active { background: rgba(100,181,246,0.15); border-color: #64B5F6; color: #64B5F6; }

/* Hide default Streamlit button borders inside the navbar container */
div[data-testid="stHorizontalBlock"] button {
    background: transparent !important;
    border: 2px solid transparent !important;
    border-radius: 8px !important;
    color: #B0BEC5 !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    padding: 7px 20px !important;
    transition: all 0.2s !important;
}

/* ── Page headers ── */
.page-header {
    padding: 32px 36px;
    border-radius: 14px;
    color: white;
    margin-bottom: 28px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.10);
}
.page-header h1 { font-size: 2.2rem; font-weight: 700; margin: 0; color: #fff !important; }
.page-header p  { font-size: 1.05rem; margin: 10px 0 0; opacity: 0.88; }

/* ── Result card ── */
.result-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 30px;
    box-shadow: 0 10px 32px rgba(0,0,0,0.06);
    text-align: center;
    margin-top: 18px;
    animation: fadeIn 0.5s ease-out;
}
.result-value { font-size: 3.4rem; font-weight: 700; margin: 8px 0; }
.result-label { font-size: 1.1rem; color: #666; text-transform: uppercase; letter-spacing: 1.4px; }
.category-badge {
    display: inline-block;
    padding: 6px 18px;
    border-radius: 50px;
    font-size: 0.92rem;
    font-weight: 600;
    margin-top: 10px;
    color: white;
}
.badge-faible    { background: #E53935; }
.badge-standard  { background: #1E88E5; }
.badge-haute     { background: #43A047; }
.badge-tres-haute{ background: #8E24AA; }

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: #ffffff;
    padding: 14px;
    border-radius: 12px;
    border: 1px solid #E0E0E0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
}

/* ── Section divider ── */
hr { border: none; border-top: 1px solid #E8ECF0; margin: 22px 0; }
</style>
""", unsafe_allow_html=True)


# ── Model loaders ────────────────────────────────────────────────────────────
@st.cache_resource
def load_direct_model():
    p = Path("concrete_strength_gradient_boosting.pkl")
    return pickle.load(open(p, "rb")) if p.exists() else None

@st.cache_resource
def load_inverse_model():
    p = Path("concrete_composition_knn_inverse.pkl")
    return pickle.load(open(p, "rb")) if p.exists() else None

direct_model  = load_direct_model()
inverse_model = load_inverse_model()


# ── Navbar (header) ──────────────────────────────────────────────────────────
nav_col1, nav_col2, nav_col3 = st.columns([3, 1, 1])

with nav_col1:
    st.markdown(
        '<div style="display:flex;align-items:center;height:52px;">'
        '<span style="font-size:1.5rem;font-weight:700;color:#1B2A3B;">'
        '🛠️ Béton<span style="color:#1565C0;">Predict</span></span></div>',
        unsafe_allow_html=True
    )

with nav_col2:
    active_direct = "primary" if st.session_state.page == "direct" else "secondary"
    if st.button(
        "📊 Modèle Direct",
        key="nav_direct",
        type=active_direct,
        use_container_width=True
    ):
        st.session_state.page = "direct"
        st.rerun()

with nav_col3:
    active_inverse = "primary" if st.session_state.page == "inverse" else "secondary"
    if st.button(
        "🧪 Modèle Inverse",
        key="nav_inverse",
        type=active_inverse,
        use_container_width=True
    ):
        st.session_state.page = "inverse"
        st.rerun()

st.markdown("<hr>", unsafe_allow_html=True)


# ── Sidebar: contextual model info ───────────────────────────────────────────
with st.sidebar:
    if st.session_state.page == "direct":
        st.markdown("### 📊 Modèle Direct")
        st.info("**Gradient Boosting Regressor**  \nEntraîné sur **302 formulations** expérimentales.")
        st.markdown("#### 🎯 Métriques (test set)")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("R²",   "93.8 %")
            st.metric("MAE",  "2.52 MPa")
        with c2:
            st.metric("RMSE", "3.44 MPa")
    else:
        st.markdown("### 🧪 Modèle Inverse")
        st.info("**K-Nearest Neighbors (KNN)**  \nPrédit la composition à partir des performances cibles.")
        st.markdown("#### 🎯 Métriques globales (test set)")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("MAE",  "35.85")
        with c2:
            st.metric("RMSE", "85.09")

    st.markdown("---")
    st.caption("💡 Développé pour l'aide à la formulation en laboratoire et sur chantier.")


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — MODÈLE DIRECT  (Composition ➔ Résistance)
# ════════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "direct":

    st.markdown("""
    <div class="page-header" style="background: linear-gradient(135deg,#2E7D32,#1B5E20);">
        <h1>📊 Modèle Direct — Composition ➔ Résistance</h1>
        <p>Prédisez la résistance à la compression du béton à partir de sa formulation de gâchage.</p>
    </div>""", unsafe_allow_html=True)

    if direct_model is None:
        st.error("⚠️ Fichier `concrete_strength_gradient_boosting.pkl` introuvable.")
    else:
        st.subheader("🧪 Paramètres de formulation")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### 🧱 Liants & Eau")
            ciment = st.number_input("Ciment (kg/m³)",
                min_value=200.0, max_value=540.0, value=350.0, step=10.0,
                help="Dosage en ciment par m³ de béton.")
            eau = st.number_input("Eau (kg/m³)",
                min_value=140.0, max_value=300.0, value=190.0, step=5.0,
                help="Eau totale de gâchage.")
            superplastifiant = st.number_input("Superplastifiant (kg/m³)",
                min_value=0.0, max_value=30.0, value=3.0, step=0.5,
                help="Adjuvant réducteur d'eau.")

        with col2:
            st.markdown("##### 🪨 Granulats & Âge")
            granulats_grossiers = st.number_input("Gravier (kg/m³)",
                min_value=420.0, max_value=1375.0, value=1030.0, step=10.0,
                help="Dosage en gravier.")
            granulats_fins = st.number_input("Sable (kg/m³)",
                min_value=420.0, max_value=945.0, value=760.0, step=10.0,
                help="Dosage en sable fin.")
            age = st.number_input("Âge du béton (jours)",
                min_value=1, max_value=365, value=28, step=1,
                help="Âge de maturation avant essai d'écrasement.")

        # Computed ratios
        ec_ratio = eau / ciment if ciment > 0 else 0.0
        gs_ratio = granulats_grossiers / granulats_fins if granulats_fins > 0 else 0.0
        masse    = ciment + eau + superplastifiant + granulats_grossiers + granulats_fins

        st.markdown("##### 📈 Ratios & Indicateurs")
        r1, r2, r3 = st.columns(3)
        with r1: st.metric("E/C",  f"{ec_ratio:.3f}", help="Rapport Eau/Ciment.")
        with r2: st.metric("G/S",  f"{gs_ratio:.3f}", help="Rapport Gravier/Sable.")
        with r3: st.metric("Masse volumique", f"{masse:.0f} kg/m³")

        st.markdown("---")

        if st.button("🚀 Prédire la Résistance", use_container_width=True, type="primary"):
            input_df = pd.DataFrame([{
                "Ciment": ciment, "Eau": eau, "Superplastifiant": superplastifiant,
                "Granulats_grossiers": granulats_grossiers, "Granulats_fins": granulats_fins,
                "G/S": gs_ratio, "E/C": ec_ratio, "Age": age
            }])
            with st.spinner("Calcul en cours…"):
                strength = float(direct_model.predict(input_df)[0])

            if strength < 20:
                label, badge, advice = "Béton de faible résistance (Non structurel)", "badge-faible", \
                    "Convient aux travaux légers. Ne pas utiliser pour les éléments porteurs."
            elif strength < 40:
                label, badge, advice = "Béton standard de structure", "badge-standard", \
                    "Adapté à la construction courante (poteaux, poutres, dalles)."
            elif strength < 60:
                label, badge, advice = "Béton de haute performance (BHP)", "badge-haute", \
                    "Recommandé pour les ouvrages d'art et structures fortement sollicitées."
            else:
                label, badge, advice = "Béton de très haute performance (BTHP)", "badge-tres-haute", \
                    "Pour applications industrielles nécessitant une résistance extrême."

            st.markdown(f"""
            <div class="result-card" style="border-left:6px solid #2E7D32;">
                <div class="result-label">Résistance à la compression estimée</div>
                <div class="result-value" style="color:#1B5E20;">{strength:.2f} MPa</div>
                <div class="category-badge {badge}">{label}</div>
                <p style="margin-top:14px;font-style:italic;color:#666;">{advice}</p>
            </div>""", unsafe_allow_html=True)

            st.info(f"💡 E/C = **{ec_ratio:.2f}** — âge = **{age} jours** — "
                    f"résistance estimée : **{strength:.2f} MPa**.")


# ════════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — MODÈLE INVERSE  (Résistance ➔ Composition)
# ════════════════════════════════════════════════════════════════════════════════
else:

    st.markdown("""
    <div class="page-header" style="background: linear-gradient(135deg,#1565C0,#0D47A1);">
        <h1>🧪 Modèle Inverse — Résistance ➔ Composition</h1>
        <p>Estimez une formulation de gâchage à partir d'une résistance cible et d'un âge de maturité.</p>
    </div>""", unsafe_allow_html=True)

    if inverse_model is None:
        st.error("⚠️ Fichier `concrete_composition_knn_inverse.pkl` introuvable.")
    else:
        st.subheader("🎯 Performances ciblées")
        st.write("Saisissez les exigences de performance du béton :")

        col1, col2 = st.columns(2)

        with col1:
            target_strength = st.number_input(
                "Résistance à la compression cible (MPa)",
                min_value=5.0, max_value=120.0, value=40.0, step=0.5,
                help="Résistance mécanique souhaitée en MPa."
            )

        with col2:
            target_age = st.number_input(
                "Âge de maturation (jours)",
                min_value=1, max_value=365, value=28, step=1,
                help="Durée de cure avant l'essai de compression."
            )

        st.markdown("---")

        if st.button("🚀 Proposer la Formulation", use_container_width=True, type="primary"):
            input_inv = pd.DataFrame([[target_strength, target_age]], columns=["Resistance", "Age"])

            with st.spinner("Calcul de la composition optimale…"):
                pred = inverse_model.predict(input_inv)[0]

            # Extract predictions (column order from training script)
            ciment_p     = max(0.0, float(pred[0]))
            eau_p        = max(0.0, float(pred[1]))
            sp_p         = max(0.0, float(pred[2]))
            gravier_p    = max(0.0, float(pred[3]))
            sable_p      = max(0.0, float(pred[4]))
            gs_p         = max(0.0, float(pred[5]))
            ec_p         = max(0.0, float(pred[6]))
            masse_p      = ciment_p + eau_p + sp_p + gravier_p + sable_p

            # ── Composition grid ──────────────────────────────────────────
            st.subheader("📐 Formulation proposée (par m³)")

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("🧱 Ciment",           f"{ciment_p:.1f} kg/m³")
                st.metric("💧 Eau",               f"{eau_p:.1f} kg/m³")
            with c2:
                st.metric("🧪 Superplastifiant",  f"{sp_p:.2f} kg/m³")
                st.metric("🪨 Gravier",           f"{gravier_p:.1f} kg/m³")
            with c3:
                st.metric("🏖️ Sable",             f"{sable_p:.1f} kg/m³")
                st.metric("⚖️ Masse volumique",   f"{masse_p:.0f} kg/m³")

            st.markdown("##### 📈 Ratios théoriques")
            r1, r2 = st.columns(2)
            with r1: st.metric("Rapport E/C", f"{ec_p:.3f}",
                               delta=f"{ec_p - (eau_p/ciment_p if ciment_p>0 else 0):.3f}",
                               delta_color="off")
            with r2: st.metric("Rapport G/S", f"{gs_p:.3f}",
                               delta=f"{gs_p - (gravier_p/sable_p if sable_p>0 else 0):.3f}",
                               delta_color="off")

            # ── Double validation ─────────────────────────────────────────
            st.markdown("---")
            st.subheader("🔍 Validation par le Modèle Direct")

            if direct_model is None:
                st.warning("⚠️ Modèle direct introuvable — validation impossible.")
            else:
                val_df = pd.DataFrame([{
                    "Ciment": ciment_p, "Eau": eau_p, "Superplastifiant": sp_p,
                    "Granulats_grossiers": gravier_p, "Granulats_fins": sable_p,
                    "G/S": gs_p, "E/C": ec_p, "Age": target_age
                }])
                with st.spinner("Validation croisée en cours…"):
                    est = float(direct_model.predict(val_df)[0])

                diff     = est - target_strength
                diff_pct = diff / target_strength * 100

                if est < 20:   cat, badge = "Béton de faible résistance",         "badge-faible"
                elif est < 40: cat, badge = "Béton standard de structure",         "badge-standard"
                elif est < 60: cat, badge = "Béton de haute performance (BHP)",    "badge-haute"
                else:          cat, badge = "Béton de très haute performance",     "badge-tres-haute"

                color_diff = "#2E7D32" if abs(diff) < 3 else "#C62828"

                st.markdown(f"""
                <div class="result-card" style="border-left:6px solid #1565C0;">
                    <div class="result-label">Résistance simulée par le modèle direct</div>
                    <div class="result-value" style="color:#0D47A1;">{est:.2f} MPa</div>
                    <div class="category-badge {badge}">{cat}</div>
                    <p style="margin-top:14px;font-size:1.02rem;color:#333;">
                        Cible : <b>{target_strength:.1f} MPa</b> à <b>{target_age} j</b>&nbsp;&nbsp;|&nbsp;&nbsp;
                        Écart : <b style="color:{color_diff};">{diff:+.2f} MPa ({diff_pct:+.1f}%)</b>
                    </p>
                </div>""", unsafe_allow_html=True)

                st.info(
                    "💡 **Interprétation :** La formulation proposée est basée sur les 5 voisins les plus "
                    "proches dans les données historiques. La validation croisée (Gradient Boosting, R² = 93.8 %) "
                    "confirme sa cohérence physique. Un écart < 3 MPa indique une excellente cohérence."
                )
