
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Configuración de página
st.set_page_config(page_title="Big Five Predictor & Famous Matcher", layout="centered")

st.title("🧠 Clasificador de Personalidad - Big Five Test")
st.write("Introduce tus puntajes del test Big Five (0 a 100) para conocer tu cluster, famosos similares y ver tu gráfica comparativa.")

# Cargar el dataset para buscar famosos del cluster y calcular promedios actualizados
@st.cache_data
def cargar_datos():
    url = "https://raw.githubusercontent.com/gerrymay/fcaData/refs/heads/master/oceanbig5.csv"
    df = pd.read_csv(url)
    return df

try:
    df_original = cargar_datos()
    kmeans_loaded = joblib.load('kmeans_bigFive_model.joblib')
    scaler_loaded = joblib.load('scaler_bigFive.joblib')
    
    # Asignar los clusters al dataset original usando el modelo cargado para consistencia
    features = ['op', 'co', 'ex', 'ag', 'ne']
    df_original_scaled = scaler_loaded.transform(df_original[features])
    df_original['cluster'] = kmeans_loaded.predict(df_original_scaled)
    
    st.success("✅ Modelo, escalador y base de datos cargados exitosamente.")
except Exception as e:
    st.error(f"❌ Error al cargar los componentes: {e}")
    st.stop()

# Formulario de entrada
st.subheader("Desliza para ingresar tus puntuaciones:")
op = st.slider("Apertura a la Experiencia (op):", 0.0, 100.0, 50.0, 0.1)
co = st.slider("Responsabilidad / Conciencia (co):", 0.0, 100.0, 50.0, 0.1)
ex = st.slider("Extroversión (ex):", 0.0, 100.0, 50.0, 0.1)
ag = st.slider("Amabilidad / Empatía (ag):", 0.0, 100.0, 50.0, 0.1)
eye_ne = st.slider("Neuroticismo (ne):", 0.0, 100.0, 50.0, 0.1)

if st.button("🔮 Obtener Resultados"):
    # 1. Crear DataFrame con las respuestas
    user_df = pd.DataFrame([{
        'op': op,
        'co': co,
        'ex': ex,
        'ag': ag,
        'ne': eye_ne
    }])

    # 2. Escalar y predecir
    user_scaled = scaler_loaded.transform(user_df)
    predicted_cluster = kmeans_loaded.predict(user_scaled)[0]

    st.markdown("--- ")
    st.markdown(f"### 🎉 Has sido asignado al **Cluster {predicted_cluster}**")

    # 3. Mostrar 5 famosos en ese cluster
    st.subheader("👥 Famosos en este mismo cluster:")
    famosos_cluster = df_original[df_original['cluster'] == predicted_cluster]
    st.table(famosos_cluster[['usuario', 'op', 'co', 'ex', 'ag', 'ne']].head(5))

    # 4. Calcular los promedios reales del cluster seleccionado en el dataset
    cluster_averages = famosos_cluster[features].mean().values.tolist()
    user_vals = [op, co, ex, ag, eye_ne]

    # Geometría del gráfico de radar (cerrar el círculo)
    radar_mean = cluster_averages + [cluster_averages[0]]
    radar_user = user_vals + [user_vals[0]]
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]

    # Configurar y dibujar gráfica
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # Trazar el promedio del cluster
    ax.plot(angles, radar_mean, color='#1f77b4', linewidth=2, label=f'Promedio Cluster {predicted_cluster}')
    ax.fill(angles, radar_mean, color='#1f77b4', alpha=0.25)

    # Superponer los datos ingresados por el usuario
    ax.plot(angles, radar_user, color='#e377c2', linewidth=2.5, linestyle='--', marker='o', label='Mis Resultados')
    ax.fill(angles, radar_user, color='#e377c2', alpha=0.15)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], features, fontsize=10)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title('Comparativa: Tus Datos vs Promedio de tu Cluster', size=13, y=1.1)

    st.pyplot(fig)
