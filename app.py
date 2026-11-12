import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
# CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title="Big Five Predictor & Cluster by Gerry",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ESTILO
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #2f3443;
    }
</style>
""", unsafe_allow_html=True)


# TÍTULO
st.title("🧠 Clasificador de Personalidad Big Five (OCEAN)")
st.write(
    "Ingresa tus resultados del test Big Five para descubrir tu cluster de personalidad, "
    "famosos con rasgos similares y comparar tus puntuaciones en diferentes gráficas."
)


# CARGA DE DATOS
@st.cache_data
def cargar_datos():
    url = "https://raw.githubusercontent.com/gerrymay/fcaData/refs/heads/master/oceanbig5.csv"
    df = pd.read_csv(url)

    # Mapear categorías si existe la columna
    mapping_categorias = {
        1: 'Actor/actriz',
        2: 'Cantante',
        3: 'Modelo',
        4: 'TV, series',
        5: 'Radio',
        6: 'Tecnología',
        7: 'Deportes',
        8: 'Política',
        9: 'Escritor'
    }

    if 'categoria' in df.columns:
        df['categoria_nombre'] = pd.to_numeric(
            df['categoria'], errors='coerce'
        ).map(mapping_categorias)

    return df

@st.cache_resource
def cargar_modelos():
    kmeans = joblib.load('kmeans_bigFive_model.joblib')
    scaler = joblib.load('scaler_bigFive.joblib')
    return kmeans, scaler


# CARGAR COMPONENTES
try:
    df_original = cargar_datos()
    kmeans_loaded, scaler_loaded = cargar_modelos()

    features = ['op', 'co', 'ex', 'ag', 'ne']

    # Asignar clusters al dataset completo
    df_original_scaled = scaler_loaded.transform(df_original[features])
    df_original['cluster'] = kmeans_loaded.predict(df_original_scaled)

    st.success("✅ Modelo, escalador y base de datos cargados exitosamente.")

except Exception as e:
    st.error(f"❌ Error al cargar los componentes: {e}")
    st.stop()

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.subheader("Tus Resultados Big Five")

    op = st.slider("Apertura a la experiencia (op)", 0.0, 100.0, 67.8, 0.1)
    co = st.slider("Responsabilidad / Orden (co)", 0.0, 100.0, 38.2, 0.1)
    ex = st.slider("Extroversión (ex)", 0.0, 100.0, 43.0, 0.1)
    ag = st.slider("Amabilidad / Empatía (ag)", 0.0, 100.0, 62.9, 0.1)
    ne = st.slider("Neuroticismo (ne)", 0.0, 100.0, 68.0, 0.1)

# -----------------------------
# PREDICCIÓN DEL USUARIO
# -----------------------------
user_df = pd.DataFrame([{
    'op': op,
    'co': co,
    'ex': ex,
    'ag': ag,
    'ne': ne
}])

user_scaled = scaler_loaded.transform(user_df)
predicted_cluster = int(kmeans_loaded.predict(user_scaled)[0])

famosos_cluster = df_original[df_original['cluster'] == predicted_cluster].copy()

# -----------------------------
# SECCIÓN PRINCIPAL ARRIBA
# -----------------------------
col1, col2 = st.columns([1, 1.2], gap="large")

# -------- COLUMNA IZQUIERDA
with col1:
    st.subheader(f"🎯 Cluster Asignado: Cluster {predicted_cluster}")
    st.write("Tu Cluster")
    st.markdown(f"## Cluster {predicted_cluster}")

    st.subheader(f"🌟 Famosos en tu mismo cluster ({len(famosos_cluster)} en total)")

    columnas_mostrar = ['usuario', 'op', 'co', 'ex', 'ag', 'ne']
    if 'categoria_nombre' in famosos_cluster.columns:
        columnas_mostrar.insert(1, 'categoria_nombre')

    # Usamos st.table en lugar de st.dataframe para evitar el error visual
    st.table(famosos_cluster[columnas_mostrar].head(5))

# -------- COLUMNA DERECHA
with col2:
    st.subheader("📊 Comparativa de Radar")

    cluster_averages = famosos_cluster[features].mean().values.tolist()
    user_vals = [op, co, ex, ag, ne]

    radar_mean = cluster_averages + [cluster_averages[0]]
    radar_user = user_vals + [user_vals[0]]

    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.plot(
        angles, radar_mean,
        color='#1f77b4',
        linewidth=2,
        label=f'Promedio Cluster {predicted_cluster}'
    )
    ax.fill(angles, radar_mean, color='#1f77b4', alpha=0.25)

    ax.plot(
        angles, radar_user,
        color='#e377c2',
        linewidth=2.5,
        linestyle='--',
        marker='o',
        label='Tus Resultados'
    )
    ax.fill(angles, radar_user, color='#e377c2', alpha=0.15)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), features)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    st.pyplot(fig)
    plt.close(fig)

# -----------------------------
# GRÁFICAS ADICIONALES
# -----------------------------
st.markdown("---")
st.header("📈 Otras Gráficas del Análisis")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Método del Codo",
    "Promedios por Cluster",
    "Boxplot",
    "Dispersión 2D",
    "Radar del Cluster",
    "Categorías"
])

# -----------------------------
# TAB 1: MÉTODO DEL CODO
# -----------------------------
with tab1:
    st.subheader("Método del Codo")

    inertia = []
    k_values = range(1, 11)

    for k in k_values:
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_temp.fit(df_original_scaled)
        inertia.append(kmeans_temp.inertia_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, inertia, marker='o', linestyle='-', color='b')
    ax.set_title('Método del Codo (Elbow Method)')
    ax.set_xlabel('Número de clusters (K)')
    ax.set_ylabel('Inercia')
    ax.set_xticks(list(k_values))
    ax.grid(True)

    st.pyplot(fig)
    plt.close(fig)

# -----------------------------
# TAB 2: PROMEDIOS POR CLUSTER
# -----------------------------
with tab2:
    st.subheader("Promedio de Atributos por Cluster")

    cluster_means = df_original.groupby('cluster')[features].mean()

    num_clusters = len(cluster_means.index)
    fig, axes = plt.subplots(1, num_clusters, figsize=(4 * num_clusters, 5), sharey=True)

    if num_clusters == 1:
        axes = [axes]

    for i, cluster in enumerate(cluster_means.index):
        axes[i].bar(features, cluster_means.loc[cluster], color=plt.cm.viridis(np.linspace(0.2, 0.8, len(features))))
        axes[i].set_title(f'Cluster {cluster}')
        axes[i].set_xlabel('Atributos')
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        if i == 0:
            axes[i].set_ylabel('Promedio')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# -----------------------------
# TAB 3: BOXPLOT
# -----------------------------
with tab3:
    st.subheader(f"Distribución de Rasgos de Personalidad en el Cluster {predicted_cluster}")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=famosos_cluster[features], palette='muted', ax=ax)
    ax.set_title(f'Distribución de Rasgos de Personalidad en el Cluster {predicted_cluster}')
    ax.set_xlabel('Rasgos (Big Five)')
    ax.set_ylabel('Puntuación')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    st.pyplot(fig)
    plt.close(fig)

# -----------------------------
# TAB 4: DISPERSIÓN 2D
# -----------------------------
with tab4:
    st.subheader("Gráfica de Dispersión: op vs ne")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=df_original,
        x='op',
        y='ne',
        hue='cluster',
        palette='viridis',
        s=80,
        edgecolor='k',
        alpha=0.7,
        ax=ax
    )
    ax.set_title('Gráfica de Dispersión: op vs ne coloreado por Cluster')
    ax.set_xlabel('Apertura Mental (op)')
    ax.set_ylabel('Neuroticismo (ne)')
    ax.grid(True, linestyle='--', alpha=0.5)

    st.pyplot(fig)
    plt.close(fig)

# -----------------------------
# TAB 5: RADAR SOLO DEL CLUSTER
# -----------------------------
with tab5:
    st.subheader(f"Radar del Cluster {predicted_cluster} vs Tus Resultados")

    cluster_averages = famosos_cluster[features].mean().values.tolist()
    user_vals = [op, co, ex, ag, ne]

    radar_mean = cluster_averages + [cluster_averages[0]]
    radar_user = user_vals + [user_vals[0]]
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    ax.plot(angles, radar_mean, linewidth=2, label='Promedio del Cluster')
    ax.fill(angles, radar_mean, alpha=0.25)

    ax.plot(angles, radar_user, linewidth=2, linestyle='--', marker='o', label='Tus Resultados')
    ax.fill(angles, radar_user, alpha=0.15)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), features)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    st.pyplot(fig)
    plt.close(fig)

# -----------------------------
# TAB 6: CATEGORÍAS
# -----------------------------
with tab6:
    st.subheader("Conteo de categorías en tu cluster")

    if 'categoria_nombre' in famosos_cluster.columns:
        conteo = famosos_cluster['categoria_nombre'].value_counts()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(conteo.index, conteo.values, color=plt.cm.viridis(np.linspace(0.2, 0.8, len(conteo))))
        ax.set_title(f'Conteo de categorías - Cluster {predicted_cluster}')
        ax.set_xlabel('Categoría')
        ax.set_ylabel('Cantidad')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("No existe la columna 'categoria_nombre' en el dataset.")
