import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans
from pathlib import Path

# CONFIGURACIÓN
st.set_page_config(
    page_title="Big Five Predictor & Cluster by Gerry",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

features = ['op', 'co', 'ex', 'ag', 'ne']

st.title("🧠 Clasificador de Personalidad - Big Five Test")
st.write(
    "Introduce tus puntajes del test Big Five (0 a 100) "
    "para conocer tu cluster, famosos similares y "
    "visualizar las gráficas del análisis."
)

# CARGAR DATOS
@st.cache_data
def cargar_datos():
    url = (
        "https://raw.githubusercontent.com/gerrymay/"
        "fcaData/refs/heads/master/oceanbig5.csv"
    )

    df = pd.read_csv(url)

    categorias = {
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
        codigos = pd.to_numeric(
            df['categoria'], errors='coerce'
        )
        df['categoria_nombre'] = (
            codigos.map(categorias).fillna('Otra categoría')
        )

    return df


@st.cache_resource
def cargar_modelos():
    carpeta = Path(__file__).resolve().parent

    kmeans = joblib.load(
        carpeta / 'kmeans_bigFive_model.joblib'
    )
    scaler = joblib.load(
        carpeta / 'scaler_bigFive.joblib'
    )

    return kmeans, scaler


try:
    df_original = cargar_datos()
    kmeans_loaded, scaler_loaded = cargar_modelos()

    df_original_scaled = scaler_loaded.transform(
        df_original[features]
    )

    df_original['cluster'] = kmeans_loaded.predict(
        df_original_scaled
    )

except Exception as e:
    st.error(f"Error al cargar componentes: {e}")
    st.stop()


# BARRA LATERAL
with st.sidebar:

    st.header("Tus Resultados Big Five")

    op = st.slider(
        "Apertura a la Experiencia (op)",
        0.0, 100.0, 50.0, 0.1
    )

    co = st.slider(
        "Responsabilidad / Conciencia (co)",
        0.0, 100.0, 50.0, 0.1
    )

    ex = st.slider(
        "Extroversión (ex)",
        0.0, 100.0, 50.0, 0.1
    )

    ag = st.slider(
        "Amabilidad / Empatía (ag)",
        0.0, 100.0, 50.0, 0.1
    )

    ne = st.slider(
        "Neuroticismo (ne)",
        0.0, 100.0, 50.0, 0.1
    )

    st.divider()
    st.caption("Los resultados se actualizan automáticamente.")


# PREDECIR CLUSTER
user_df = pd.DataFrame([{
    'op': op,
    'co': co,
    'ex': ex,
    'ag': ag,
    'ne': ne
}])

user_scaled = scaler_loaded.transform(user_df)

predicted_cluster = int(
    kmeans_loaded.predict(user_scaled)[0]
)

famosos_cluster = df_original[
    df_original['cluster'] == predicted_cluster
].copy()

cluster_averages = (
    famosos_cluster[features].mean().values.tolist()
)

user_vals = [op, co, ex, ag, ne]


# RESULTADOS PRINCIPALES
st.divider()

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:

    st.subheader(
        f"🎯 Cluster Asignado: Cluster {predicted_cluster}"
    )

    st.metric(
        "Tu Cluster",
        f"Cluster {predicted_cluster}"
    )

    st.subheader("🌟 Famosos en tu mismo cluster")

    st.write(
        f"**{len(famosos_cluster)} famosos en total**"
    )

    columnas = [
        'usuario', 'categoria_nombre',
        'op', 'co', 'ex', 'ag', 'ne'
    ]

    columnas = [
        c for c in columnas
        if c in famosos_cluster.columns
    ]

    st.dataframe(
        famosos_cluster[columnas].head(5),
        use_container_width=True,
        hide_index=True
    )


with col2:

    st.subheader("📊 Comparativa de Radar")

    radar_mean = cluster_averages + [cluster_averages[0]]
    radar_user = user_vals + [user_vals[0]]

    angles = np.linspace(
        0, 2 * np.pi,
        len(features),
        endpoint=False
    ).tolist()

    angles += angles[:1]

    fig, ax = plt.subplots(
        figsize=(6, 6),
        subplot_kw=dict(polar=True)
    )

    ax.plot(
        angles, radar_mean,
        color='#1f77b4',
        linewidth=2,
        label=f'Promedio Cluster {predicted_cluster}'
    )

    ax.fill(
        angles, radar_mean,
        color='#1f77b4',
        alpha=0.25
    )

    ax.plot(
        angles, radar_user,
        color='#e377c2',
        linewidth=2.5,
        linestyle='--',
        marker='o',
        label='Mis Resultados'
    )

    ax.fill(
        angles, radar_user,
        color='#e377c2',
        alpha=0.15
    )

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_thetagrids(
        np.degrees(angles[:-1]),
        features
    )

    ax.set_ylim(0, 100)

    ax.legend(
        loc='upper right',
        bbox_to_anchor=(1.3, 1.1)
    )

    ax.set_title(
        'Tus Datos vs Promedio del Cluster',
        fontsize=13,
        pad=25
    )

    st.pyplot(fig)
    plt.close(fig)


# GRÁFICAS DEL NOTEBOOK
st.divider()
st.header("📈 Análisis de Clusters Big Five")

tabs = st.tabs([
    "Método del Codo",
    "Promedios",
    "Boxplot",
    "Dispersión 2D",
    "Dispersión 3D",
    "Categorías"
])


# 1. MÉTODO DEL CODO
with tabs[0]:

    st.subheader("Método del Codo (Elbow Method)")

    @st.cache_data
    def calcular_inercias(datos):
        inertia = []
        k_values = range(1, min(10, len(datos)) + 1)

        for k in k_values:
            modelo = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10
            )
            modelo.fit(datos)
            inertia.append(modelo.inertia_)

        return list(k_values), inertia

    k_values, inertia = calcular_inercias(
        df_original_scaled
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        k_values,
        inertia,
        marker='o',
        linestyle='-',
        color='b'
    )

    ax.set_title('Método del Codo (Elbow Method)')
    ax.set_xlabel('Número de clusters (K)')
    ax.set_ylabel('Inercia (Inertia)')
    ax.set_xticks(k_values)
    ax.grid(True)

    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "Esta gráfica permite analizar cómo disminuye "
        "la inercia al aumentar el número de clusters."
    )


# 2. PROMEDIOS POR CLUSTER
with tabs[1]:

    st.subheader("Promedio de Atributos por Cluster")

    cluster_means = (
        df_original.groupby('cluster')[features].mean()
    )

    clusters = sorted(df_original['cluster'].unique())

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(clusters),
        figsize=(max(12, 4 * len(clusters)), 5),
        sharey=True,
        squeeze=False
    )

    fig.suptitle(
        'Promedio de Atributos por Cluster',
        fontsize=16,
        y=1.05
    )

    colores = plt.cm.viridis(
        np.linspace(0.1, 0.9, len(features))
    )

    for i, cluster in enumerate(clusters):

        ax = axes[0, i]

        ax.bar(
            features,
            cluster_means.loc[cluster],
            color=colores
        )

        ax.set_title(f'Cluster {cluster}')
        ax.set_xlabel('Atributos')
        ax.set_ylabel('Promedio' if i == 0 else '')
        ax.grid(
            axis='y',
            linestyle='--',
            alpha=0.7
        )

    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


# 3. BOXPLOT
with tabs[2]:

    st.subheader(
        f"Distribución de Rasgos - Cluster {predicted_cluster}"
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(
        data=famosos_cluster[features],
        palette='muted',
        ax=ax
    )

    ax.set_title(
        f'Distribución de Rasgos de Personalidad en el Cluster {predicted_cluster}',
        fontsize=14
    )

    ax.set_xlabel('Rasgos (Big Five)')
    ax.set_ylabel('Puntuación')
    ax.grid(
        axis='y',
        linestyle='--',
        alpha=0.7
    )

    st.pyplot(fig)
    plt.close(fig)


# 4. DISPERSIÓN 2D
with tabs[3]:

    st.subheader("Dispersión 2D por Cluster")

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

    ax.set_title(
        'Gráfica de Dispersión: op vs ne coloreado por Cluster'
    )

    ax.set_xlabel('Apertura Mental (op)')
    ax.set_ylabel('Neuroticismo (ne)')
    ax.legend(title='Cluster')
    ax.grid(
        True,
        linestyle='--',
        alpha=0.5
    )

    st.pyplot(fig)
    plt.close(fig)


# 5. DISPERSIÓN 3D
with tabs[4]:

    st.subheader("Dispersión 3D Interactiva")

    df_3d = df_original.copy()

    # Tratar cluster como categoría para diferenciar colores
    df_3d['cluster_color'] = (
        df_3d['cluster'].astype(str)
    )

    fig3d = px.scatter_3d(
        df_3d,
        x='op',
        y='ne',
        z='ex',
        color='cluster_color',
        color_discrete_sequence=px.colors.sequential.Viridis,
        title='Dispersión 3D Interactiva: op vs ne vs ex (por Cluster)',
        labels={
            'op': 'Apertura (op)',
            'ne': 'Neuroticismo (ne)',
            'ex': 'Extroversión (ex)',
            'cluster_color': 'Cluster'
        }
    )

    fig3d.update_traces(
        marker=dict(size=4, opacity=0.75)
    )

    st.plotly_chart(
        fig3d,
        use_container_width=True
    )


# 6. CONTEO DE CATEGORÍAS
with tabs[5]:

    st.subheader("Distribución de Categorías por Cluster")

    if 'categoria_nombre' in df_original.columns:

        cluster_seleccionado = st.selectbox(
            "Selecciona el cluster",
            sorted(df_original['cluster'].unique()),
            index=sorted(
                df_original['cluster'].unique()
            ).index(predicted_cluster)
        )

        datos_cluster = df_original[
            df_original['cluster'] == cluster_seleccionado
        ]

        conteo = (
            datos_cluster['categoria_nombre']
            .value_counts()
        )

        fig, ax = plt.subplots(figsize=(10, 5))

        colores = plt.cm.viridis(
            np.linspace(0.1, 0.9, len(conteo))
        )

        ax.bar(
            conteo.index,
            conteo.values,
            color=colores
        )

        ax.set_title(
            f'Categorías de Famosos - Cluster {cluster_seleccionado}'
        )

        ax.set_xlabel('Categoría')
        ax.set_ylabel('Cantidad')
        ax.tick_params(
            axis='x',
            rotation=35
        )

        ax.grid(
            axis='y',
            linestyle='--',
            alpha=0.5
        )

        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    else:
        st.warning(
            "No se encontró la columna categoria."
        )


st.divider()

st.caption(
    "Big Five Test | Clasificación de personalidad con K-Means"
)
