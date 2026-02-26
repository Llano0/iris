# =============================
# main_app.py
# =============================

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
    ["KNN", "Decision Tree", "Random Forest", "SVM"]
)

test_size = st.sidebar.slider("Porcentaje de prueba", 0.1, 0.5, 0.3)
random_state = st.sidebar.slider("Random State", 0, 100, 42)

# Parámetros dinámicos
if model_option == "KNN":
    k = st.sidebar.slider("Número de vecinos (k)", 1, 15, 5)

elif model_option == "Decision Tree":
    max_depth = st.sidebar.slider("Profundidad máxima", 1, 10, 3)

elif model_option == "Random Forest":
    n_estimators = st.sidebar.slider("Número de árboles", 10, 200, 100)

elif model_option == "SVM":
    C = st.sidebar.slider("Parámetro C", 0.1, 10.0, 1.0)

# -----------------------------
# División de datos
# -----------------------------
X = df.drop(columns=["target", "species"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# Construcción del modelo
# -----------------------------
if model_option == "KNN":
    model = KNeighborsClassifier(n_neighbors=k)

elif model_option == "Decision Tree":
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)

elif model_option == "Random Forest":
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)

elif model_option == "SVM":
    model = SVC(C=C, probability=True)

model.fit(X_train_scaled, y_train)
preds = model.predict(X_test_scaled)

# -----------------------------
# Métricas
# -----------------------------
accuracy = accuracy_score(y_test, preds)

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Accuracy")
    st.metric("Exactitud", f"{accuracy:.3f}")

with col2:
    st.subheader("📄 Reporte de Clasificación")
    report = classification_report(y_test, preds, target_names=class_names, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

# -----------------------------
# Matriz de confusión
# -----------------------------
st.subheader("🔍 Matriz de Confusión")
cm = confusion_matrix(y_test, preds)

fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names, ax=ax)
ax.set_xlabel("Predicción")
ax.set_ylabel("Real")
st.pyplot(fig)

# -----------------------------
# Visualización de datos
# -----------------------------
st.subheader("📈 Visualización de Features")
feature_x = st.selectbox("Feature X", X.columns, index=0)
feature_y = st.selectbox("Feature Y", X.columns, index=1)

fig2, ax2 = plt.subplots()
sns.scatterplot(data=df, x=feature_x, y=feature_y, hue="species", palette="Set2", ax=ax2)
st.pyplot(fig2)

# -----------------------------
# Predicción interactiva
# -----------------------------
st.subheader("🧪 Probar una nueva muestra")

input_cols = st.columns(4)
user_input = []

for i, col in enumerate(X.columns):
    val = input_cols[i].number_input(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
    user_input.append(val)

if st.button("Predecir especie"):
    sample = scaler.transform([user_input])
    pred = model.predict(sample)[0]
    proba = model.predict_proba(sample)[0]

    st.success(f"🌼 Predicción: {class_names[pred]}")

    st.subheader("Probabilidades")
    prob_df = pd.DataFrame({
        "Clase": class_names,
        "Probabilidad": proba
    })
    st.dataframe(prob_df)

