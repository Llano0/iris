import streamlit as st

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
