import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Segmentação de Mercado — OLX PB",
    page_icon="🏘️",
    layout="wide",
)

# ── Carregamento e limpeza (mesmo pipeline do previsao_preco.py) ───────────────

TIPOS_VALIDOS = ["Apartamento", "Casa", "Chácara", "Cobertura", "Galpão", "Kitnet", "Loft", "Loja"]

TIPO_KEYWORDS = {
    "Apartamento": ["apartamento", "apto", " ap ", "flat", "studio"],
    "Casa":        ["casa", "sobrado", "residência", "chalé", "bangalô"],
    "Chácara":     ["chácara", "sítio", "fazenda", "haras"],
    "Cobertura":   ["cobertura", "penthouse", "pent haus", "pent-haus"],
    "Galpão":      ["galpão", "armazém", "depósito", "barracão"],
    "Kitnet":      ["kitnet", "quitinete", "kit net"],
    "Loft":        ["loft"],
    "Loja":        ["loja", "ponto comercial", "sala comercial", "consultório"],
}

def inferir_tipo(texto: str) -> str:
    t = texto.lower()
    for tipo, palavras in TIPO_KEYWORDS.items():
        if any(p in t for p in palavras):
            return tipo
    return ""

BAIRROS_PB = [
    "Manaíra", "Tambaú", "Cabo Branco", "Bessa", "Jardim Oceania", "Intermares",
    "Bancários", "Altiplano", "Miramar", "Aeroclube", "Tambauzinho", "Portal do Sol",
    "Estados", "João Paulo II", "Muçumagro", "Jardim Camboinha", "Mangabeira",
    "Castelo Branco", "Funcionários", "Torre", "Cristo", "Expedicionários", "Brisamar",
    "Água Fria", "Valentina", "Roger", "Mandacaru", "Rangel", "Grotão", "Anatólia",
    "Planalto da Boa Esperança", "Centro", "Jardim 13 de Maio", "Jardim Luna",
    "Costa e Silva", "Penha", "Varjão", "Colinas do Sul", "Paratibe", "Jardim São Paulo",
    "Trincheiras", "Jaguaribe", "Ilha do Bispo", "Padre Zé", "Cuiá", "Mata do Buraquinho",
    "Catolé", "Alto Branco", "Centenário", "Liberdade", "Mirante", "Bodocongó",
    "Palmeira", "Cruzeiro", "Dinamérica", "Sandra Cavalcante",
    "Ponta de Campina", "Camboinha",
]
BAIRROS_PB.sort(key=len, reverse=True)

def extrair_bairro_titulo(titulo: str) -> str:
    t = titulo.lower()
    for bairro in BAIRROS_PB:
        if bairro.lower() in t:
            return bairro
    return ""

@st.cache_data
def load_data():
    df = pd.read_csv("dataset_olx_raw.csv", encoding="utf-8-sig")
    df.columns = [c.lstrip("﻿").strip() for c in df.columns]

    for col in ["preco", "area_m2", "quartos", "banheiros", "garagens"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["titulo", "descricao", "tipo", "bairro", "cidade"]:
        df[col] = df[col].fillna("").str.strip()

    mask_imovel = df["tipo"] == "Imóvel"
    texto = df.loc[mask_imovel, "titulo"] + " " + df.loc[mask_imovel, "descricao"]
    df.loc[mask_imovel, "tipo"] = texto.apply(inferir_tipo)
    df = df[df["tipo"].isin(TIPOS_VALIDOS)].copy()

    mask_sem_cidade = df["cidade"] == ""
    partes = df.loc[mask_sem_cidade, "bairro"].str.split(",")
    df.loc[mask_sem_cidade, "cidade"] = partes.str[0].str.split("Hoje").str[0].str.strip()
    df.loc[mask_sem_cidade, "bairro"] = (
        partes.str[1].str.split("Hoje").str[0].str.strip()
        if partes.str.len().gt(1).any() else ""
    )
    df["cidade"] = df["cidade"].str.split("Hoje").str[0].str.strip()
    df["bairro"] = df["bairro"].str.split("Hoje").str[0].str.strip()

    cidade_valida = df["cidade"].str.match(
        r"^[A-Za-záàãâéêíóôõúüçÁÀÃÂÉÊÍÓÔÕÚÜÇ][A-Za-záàãâéêíóôõúüçÁÀÃÂÉÊÍÓÔÕÚÜÇ\s]{2,}$", na=False
    )
    df.loc[~cidade_valida, "cidade"] = ""

    bairro_valido = df["bairro"].str.match(
        r"^[A-Za-záàãâéêíóôõúüçÁÀÃÂÉÊÍÓÔÕÚÜÇ][A-Za-záàãâéêíóôõúüçÁÀÃÂÉÊÍÓÔÕÚÜÇ\s\d°º]{2,}$", na=False
    )
    df.loc[~bairro_valido, "bairro"] = ""

    def normalizar_bairro(b):
        if not b:
            return ""
        b_low = b.lower()
        for bairro in BAIRROS_PB:
            if bairro.lower() in b_low:
                return bairro
        return b

    df["bairro"] = df["bairro"].apply(normalizar_bairro)

    mask_sem_bairro = df["bairro"] == ""
    df.loc[mask_sem_bairro, "bairro"] = (
        df.loc[mask_sem_bairro, "titulo"].apply(extrair_bairro_titulo)
    )

    texto_completo = df["titulo"].str.lower() + " " + df["descricao"].str.lower()
    df = df[~texto_completo.str.contains("repasse", na=False)]

    df = df[df["preco"].between(50_000, 20_000_000)]
    df = df[df["quartos"].fillna(0).between(0, 15)]
    df = df[df["banheiros"].fillna(0).between(0, 15)]
    df = df[df["garagens"].fillna(0).between(0, 10)]
    df = df[df["area_m2"].fillna(50).between(5, 2_000)]

    return df


df = load_data()


# ── Clusterização K-Means ──────────────────────────────────────────────────────

@st.cache_resource
def clusterizar(n_clusters: int = 4):
    d = df[["preco", "area_m2", "quartos", "banheiros", "garagens"]].copy()
    d = d.dropna()

    # Normaliza as features antes do K-Means — sem isso preço domina tudo
    # porque está em escala de centenas de milhares enquanto quartos vai de 0 a 10
    scaler = StandardScaler()
    X = scaler.fit_transform(d)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    d["cluster_raw"] = km.fit_predict(X)

    # Rotula os clusters por ordem crescente de mediana de preço:
    # 0 = Popular, 1 = Médio, 2 = Alto Padrão, 3 = Luxo
    ordem = (
        d.groupby("cluster_raw")["preco"]
        .median()
        .sort_values()
        .index.tolist()
    )
    rotulos = {
        4: ["Popular", "Médio", "Alto Padrão", "Luxo"],
        3: ["Popular", "Médio", "Alto Padrão"],
        5: ["Popular", "Econômico", "Médio", "Alto Padrão", "Luxo"],
    }
    nomes = rotulos.get(n_clusters, [f"Cluster {i}" for i in range(n_clusters)])
    mapa_nome = {cluster_id: nomes[i] for i, cluster_id in enumerate(ordem)}
    d["segmento"] = d["cluster_raw"].map(mapa_nome)

    return d, km, scaler, mapa_nome


# ── Interface ─────────────────────────────────────────────────────────────────

st.title("🏘️ Segmentação de Mercado por Perfil")
st.markdown(
    "Algoritmo **K-Means** agrupa automaticamente os imóveis em segmentos "
    "com base em preço, área e quantidade de cômodos — sem precisar rotular nada manualmente."
)
st.divider()

n_clusters = st.sidebar.slider("Número de segmentos", min_value=3, max_value=5, value=4)

with st.spinner("Rodando K-Means..."):
    df_seg, km_model, scaler, mapa_nome = clusterizar(n_clusters)

# Junta o segmento de volta no dataframe completo pra ter tipo e cidade disponíveis
df_plot = df.loc[df_seg.index].copy()
df_plot["segmento"] = df_seg["segmento"]

# Métricas gerais
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total de imóveis", f"{len(df_plot):,}")
for i, nome in enumerate(df_seg["segmento"].unique()):
    qtd = (df_plot["segmento"] == nome).sum()
    pct = qtd / len(df_plot) * 100
    [m2, m3, m4][i % 3].metric(nome, f"{qtd:,} ({pct:.0f}%)")

st.divider()

col_scatter, col_perfil = st.columns([2, 1])

# ── Scatter principal: Área x Preço colorido por segmento ─────────────────────
with col_scatter:
    st.subheader("Imóveis por segmento")

    CORES = {
        "Popular":     "#4fc3f7",
        "Econômico":   "#81c784",
        "Médio":       "#ffb74d",
        "Alto Padrão": "#f06292",
        "Luxo":        "#ce93d8",
        "Cluster 0":   "#4fc3f7",
        "Cluster 1":   "#ffb74d",
        "Cluster 2":   "#f06292",
        "Cluster 3":   "#ce93d8",
        "Cluster 4":   "#81c784",
    }

    # Limita eixo Y pra não deixar o gráfico achatado por um outlier
    df_vis = df_plot[df_plot["preco"] <= 3_000_000].copy()

    fig = px.scatter(
        df_vis,
        x="area_m2",
        y="preco",
        color="segmento",
        color_discrete_map=CORES,
        hover_data=["tipo", "cidade", "quartos", "banheiros"],
        labels={"area_m2": "Área (m²)", "preco": "Preço (R$)", "segmento": "Segmento"},
        opacity=0.6,
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

# ── Perfil de cada segmento ────────────────────────────────────────────────────
with col_perfil:
    st.subheader("Perfil dos segmentos")

    # Ordena os segmentos do mais barato pro mais caro pra exibir em sequência
    ordem_display = (
        df_plot.groupby("segmento")["preco"]
        .median()
        .sort_values()
        .index.tolist()
    )

    for seg in ordem_display:
        sub = df_plot[df_plot["segmento"] == seg]
        cor = CORES.get(seg, "#888")

        with st.container():
            st.markdown(f"**:{cor[1:]}[{seg}]** — {len(sub):,} imóveis")
            c1, c2 = st.columns(2)
            c1.metric("Preço mediano",  f"R$ {sub['preco'].median():,.0f}")
            c2.metric("Área mediana",   f"{sub['area_m2'].median():.0f} m²")
            c1.metric("Quartos (md)",   f"{sub['quartos'].median():.0f}")
            c2.metric("Banheiros (md)", f"{sub['banheiros'].median():.0f}")
            st.divider()

st.divider()

# ── Distribuição de tipos por segmento ────────────────────────────────────────
st.subheader("Composição por tipo de imóvel")

tipo_seg = (
    df_plot.groupby(["segmento", "tipo"])
    .size()
    .reset_index(name="qtd")
)
# Normaliza dentro de cada segmento pra mostrar % em vez de contagem absoluta
tipo_seg["pct"] = tipo_seg.groupby("segmento")["qtd"].transform(lambda x: x / x.sum() * 100)

fig2 = px.bar(
    tipo_seg,
    x="segmento",
    y="pct",
    color="tipo",
    barmode="stack",
    labels={"segmento": "Segmento", "pct": "% de imóveis", "tipo": "Tipo"},
    category_orders={"segmento": ordem_display},
    color_discrete_sequence=px.colors.qualitative.Set2,
)
fig2.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02))
st.plotly_chart(fig2, use_container_width=True)

# ── Simulador: em qual segmento cai meu imóvel? ───────────────────────────────
st.divider()
st.subheader("Em qual segmento está o seu imóvel?")

s1, s2, s3, s4, s5 = st.columns(5)
preco_sim  = s1.number_input("Preço (R$)",  50_000, 20_000_000, 400_000, step=10_000)
area_sim   = s2.number_input("Área (m²)",   5, 2_000, 80)
quartos_s  = s3.slider("Quartos",    0, 10, 2)
banhs_s    = s4.slider("Banheiros",  0, 8,  1)
garagens_s = s5.slider("Garagens",   0, 6,  1)

if st.button("Classificar →", type="primary"):
    entrada = np.array([[preco_sim, area_sim, quartos_s, banhs_s, garagens_s]])
    entrada_norm = scaler.transform(entrada)
    cluster_raw = km_model.predict(entrada_norm)[0]
    segmento_resultado = mapa_nome[cluster_raw]
    cor_res = CORES.get(segmento_resultado, "#888")

    st.success(f"**Segmento: {segmento_resultado}**")

    # Mostra onde esse imóvel ficaria em relação à mediana do segmento
    sub_res = df_plot[df_plot["segmento"] == segmento_resultado]
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Preço vs mediana do segmento",
        f"R$ {preco_sim:,.0f}",
        delta=f"R$ {preco_sim - sub_res['preco'].median():,.0f}",
    )
    c2.metric(
        "Área vs mediana do segmento",
        f"{area_sim} m²",
        delta=f"{area_sim - sub_res['area_m2'].median():.0f} m²",
    )
    c3.metric("Imóveis nesse segmento", f"{len(sub_res):,}")
