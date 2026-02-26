import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# -----------------------------
# Configuración inicial
# -----------------------------
st.set_page_config(page_title="Clasificador IRIS Pedagógico", layout="wide")

st.title("🌸 Clasificador IRIS Interactivo")
st.markdown("""
Esta app permite:
- Entrenar diferentes modelos
- Visualizar métricas y gráficas
- Probar predicciones manuales
""")

# -----------------------------
# Cargar dataset
# -----------------------------
@st.cache_data
def load_data():
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    df["species"] = df["target"].map(dict(enumerate(data.target_names)))
    return df, data.target_names


df, class_names = load_data()

# -----------------------------
# Sidebar pedagógica
# -----------------------------
st.sidebar.header("⚙️ Configuración del Modelo")

model_option = st.sidebar.selectbox(
    "Selecciona el modelo",

