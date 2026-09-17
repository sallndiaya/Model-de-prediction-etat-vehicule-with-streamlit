"""
Application Streamlit — Prédiction de l'état d'un vehicule
Conversion directe de l'application Gradio d'origine.

Lancement en local :  streamlit run app.py
"""

import numpy as np
import pandas as pd
import joblib as jb
import streamlit as st


# Configuration de la page
st.set_page_config(
    page_title="Prédiction de l'état d'un Vehicule",
    page_icon="📱",
    layout="centered",
)

DESCRIPTION = (
    "Ce modèle de machine permet de prédire l'état du vehicule en partant "
    "du marque , de l'anee, de la trasmission du vehicule, le quartier et du prix "

)


# Chargement des artefacts (mis en cache : chargés une seule fois)

@st.cache_resource
def load_artifacts():
    encoders = jb.load("encoders.joblib")   # encodeurs (Marque, Quartier, Transmission)
    uniques = jb.load("uniques.joblib")     # valeurs uniques
    scaler = jb.load("scaler.joblib")       # normaliseur
    svm = jb.load("svm_model.joblib")       # modèle
    return encoders, uniques, scaler, svm


encoders, uniques, scaler, svm = load_artifacts()
clasnames = uniques[2]  # noms des classes



# Fonction de prédiction simple

def Pred_func(Marque, Année, Transmission, Quartier, Prix):
    # Encoder   marque,transmission et Quartier
    Marque = encoders[0].transform([Marque])[0]
    Transmission = encoders[1].transform([Transmission])[0]
    Quartier = encoders[2].transform([Quartier])[0]
    # Vecteur des valeurs numériques
    x_new = np.array([Marque, Année, Transmission, Quartier, Prix])
    x_new = x_new.reshape(1, -1)  # conversion en un tableau 2D
    # Normaliser les données
    x_new = scaler.transform(x_new)
    # Prédire
    y_pred = svm.predict(x_new)
    return clasnames[y_pred[0]]



# Fonction de prédiction multiple

def Pred_func_csv(file):
    # Lire le fichier csv
    df = pd.read_csv(file)
    predictions = []
    # Boucle sur les lignes du dataframe
    for row in df.iloc[:, :].values:
        # prédiction simple
        y_pred = Pred_func(row[0], row[1], row[2], row[3], row[4])
        predictions.append(y_pred)
    df["Etat"] = predictions
    return df



# Interface

st.title("📱 Prédiction de l'état d'un vehicule")

onglet1, onglet2 = st.tabs(["Prédiction simple", "Prédiction multiple"])

# ----------------------------- Onglet 1 -------------------------------
with onglet1:
    st.subheader("Prédire l'état d'un vehicule avec une entrée")
    st.write(DESCRIPTION)

    with st.form("formulaire_simple"):
        col1, col2 = st.columns(2)
        with col1:
            Marque = st.selectbox("Marque", options=list(uniques[0]))
            Année = st.number_input("Année", value=0, step=1, format="%d")
            Transmission = st.selectbox("Transmission", options=list(uniques[1]))
        with col2:
            Quartier = st.selectbox("Quartier", options=list(uniques[2]))
            Prix = st.number_input("Prix", value=0.0, step=1000.0, format="%.2f")

        soumettre = st.form_submit_button("Prédire", type="primary")

    if soumettre:
        try:
            resultat = Pred_func(Marque, Année, Transmission, Quartier, Prix)
            st.success(f"**État du vehicule :** {resultat}")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

# ----------------------------- Onglet 2 -------------------------------
with onglet2:
    st.subheader("Prédire l'état du vehicule avec plusieurs entrées")
    st.write(DESCRIPTION)
    st.caption(
        "Le fichier CSV doit contenir, dans cet ordre, les colonnes : "
        "Marque, Année, Transmission, Quartier, Prix."
    )

    fichier = st.file_uploader("Importer un fichier CSV", type=["csv"])

    if fichier is not None:
        try:
            with st.spinner("Prédictions en cours…"):
                df_resultat = Pred_func_csv(fichier)

            st.success(f"{len(df_resultat)} prédiction(s) effectuée(s).")
            st.dataframe(df_resultat, use_container_width=True)

            st.download_button(
                label="⬇️ Télécharger le fichier CSV",
                data=df_resultat.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
                type="primary",
            )
        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier : {e}")
