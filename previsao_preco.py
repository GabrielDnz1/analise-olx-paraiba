import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(
    page_title="Previsão de Preço — OLX PB",
    page_icon="📈",
    layout="wide",
)

# Tipos aceitos no modelo — qualquer outra coisa é descartada na limpeza
TIPOS_VALIDOS = ["Apartamento", "Casa", "Chácara", "Cobertura", "Galpão", "Kitnet", "Loft", "Loja"]

# O campo "tipo" no CSV tem muitos registros como "Imóvel" (genérico).
# Essas palavras-chave permitem reclassificar esses casos pelo título e descrição.
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
    """Tenta descobrir o tipo do imóvel a partir do texto livre do anúncio."""
    t = texto.lower()
    for tipo, palavras in TIPO_KEYWORDS.items():
        if any(p in t for p in palavras):
            return tipo
    return ""  # não identificado — será descartado depois


# Bairros conhecidos usados pra extrair localização do título quando o campo vem vazio.
# Ordenados do mais longo pro mais curto pra evitar match parcial
# (ex: "Jardim Oceania" antes de "Oceania").
BAIRROS_PB = [
    # João Pessoa
    "Manaíra", "Tambaú", "Cabo Branco", "Bessa", "Jardim Oceania", "Intermares",
    "Bancários", "Altiplano", "Miramar", "Aeroclube", "Tambauzinho", "Portal do Sol",
    "Estados", "João Paulo II", "Muçumagro", "Jardim Camboinha", "Mangabeira",
    "Castelo Branco", "Funcionários", "Torre", "Cristo", "Expedicionários", "Brisamar",
    "Água Fria", "Valentina", "Roger", "Mandacaru", "Rangel", "Grotão", "Anatólia",
    "Planalto da Boa Esperança", "Centro", "Jardim 13 de Maio", "Jardim Luna",
    "Costa e Silva", "Penha", "Varjão", "Colinas do Sul", "Paratibe", "Jardim São Paulo",
    "Trincheiras", "Jaguaribe", "Ilha do Bispo", "Padre Zé", "Cuiá", "Mata do Buraquinho",
    # Campina Grande
    "Catolé", "Alto Branco", "Centenário", "Liberdade", "Mirante", "Bodocongó",
    "Palmeira", "Cruzeiro", "Dinamérica", "Sandra Cavalcante",
    # Cabedelo / outros
    "Ponta de Campina", "Intermares", "Camboinha",
]
BAIRROS_PB.sort(key=len, reverse=True)


def extrair_bairro_titulo(titulo: str) -> str:
    """Procura um bairro conhecido no título do anúncio."""
    t = titulo.lower()
    for bairro in BAIRROS_PB:
        if bairro.lower() in t:
            return bairro
    return ""


# ── Carregamento e limpeza dos dados ──────────────────────────────────────────

@st.cache_data
def load_data():
    df = pd.read_csv("dataset_olx_raw.csv", encoding="utf-8-sig")

    # Remove BOM do nome das colunas se houver
    df.columns = [c.lstrip("﻿").strip() for c in df.columns]

    # Converte colunas numéricas — erros viram NaN
    for col in ["preco", "area_m2", "quartos", "banheiros", "garagens"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["titulo", "descricao", "tipo", "bairro", "cidade"]:
        df[col] = df[col].fillna("").str.strip()

    # Reclassifica os registros com tipo "Imóvel" usando palavras-chave
    mask_imovel = df["tipo"] == "Imóvel"
    texto = df.loc[mask_imovel, "titulo"] + " " + df.loc[mask_imovel, "descricao"]
    df.loc[mask_imovel, "tipo"] = texto.apply(inferir_tipo)

    df = df[df["tipo"].isin(TIPOS_VALIDOS)].copy()

    # O campo bairro no CSV às vezes vem no formato "Cidade, BairroHoje, HH:MM".
    # Quando cidade está vazia, extrai cidade e bairro desse campo composto.
    mask_sem_cidade = df["cidade"] == ""
    partes = df.loc[mask_sem_cidade, "bairro"].str.split(",")
    df.loc[mask_sem_cidade, "cidade"] = (
        partes.str[0].str.split("Hoje").str[0].str.strip()
    )
    df.loc[mask_sem_cidade, "bairro"] = (
        partes.str[1].str.split("Hoje").str[0].str.strip()
        if partes.str.len().gt(1).any()
        else ""
    )

    # Garante que sobras do sufixo "Hoje" sejam removidas
    df["cidade"] = df["cidade"].str.split("Hoje").str[0].str.strip()
    df["bairro"] = df["bairro"].str.split("Hoje").str[0].str.strip()

    # Remove timestamps e strings inválidas que acabaram no campo cidade (ex: "07:02")
    cidade_valida = df["cidade"].str.match(
        r"^[A-Za-záàãâéêíóôõúüçÁÀÃÂÉÊÍÓÔÕÚÜÇ][A-Za-záàãâéêíóôõúüçÁÀÃÂÉÊÍÓÔÕÚÜÇ\s]{2,}$",
        na=False,
    )
    df.loc[~cidade_valida, "cidade"] = ""

    bairro_valido = df["bairro"].str.match(
        r"^[A-Za-záàãâéêíóôõúüçÁÀÃÂÉÊÍÓÔÕÚÜÇ][A-Za-záàãâéêíóôõúüçÁÀÃÂÉÊÍÓÔÕÚÜÇ\s\d°º]{2,}$",
        na=False,
    )
    df.loc[~bairro_valido, "bairro"] = ""

    # Alguns bairros vieram com lixo concatenado (ex: "Jardim Oceania12 de mar").
    # Verifica se o campo contém um bairro conhecido e extrai só ele.
    def normalizar_bairro(b: str) -> str:
        if not b:
            return ""
        b_low = b.lower()
        for bairro in BAIRROS_PB:
            if bairro.lower() in b_low:
                return bairro
        return b

    df["bairro"] = df["bairro"].apply(normalizar_bairro)

    # Só 3% dos registros tinham bairro preenchido no CSV original.
    # Busca o bairro no título pra aumentar a cobertura (~65% após essa etapa).
    mask_sem_bairro = df["bairro"] == ""
    df.loc[mask_sem_bairro, "bairro"] = (
        df.loc[mask_sem_bairro, "titulo"].apply(extrair_bairro_titulo)
    )

    # Remove repasses — são transferências de contrato, não vendas diretas
    texto_completo = df["titulo"].str.lower() + " " + df["descricao"].str.lower()
    df = df[~texto_completo.str.contains("repasse", na=False)]

    # Filtra faixa de preço válida (venda) e remove outliers numéricos
    df = df[df["preco"] >= 50_000]
    df = df[df["preco"] <= 20_000_000]
    df = df[df["quartos"].fillna(0).between(0, 15)]
    df = df[df["banheiros"].fillna(0).between(0, 15)]
    df = df[df["garagens"].fillna(0).between(0, 10)]
    df = df[df["area_m2"].fillna(50).between(5, 2_000)]

    return df


df = load_data()


# ── Treinamento do modelo ──────────────────────────────────────────────────────

@st.cache_resource
def treinar():
    d = df[["tipo", "cidade", "bairro", "quartos", "banheiros", "garagens", "area_m2", "preco"]].copy()
    d = d.dropna(subset=["tipo", "cidade", "preco"])
    d = d[(d["tipo"] != "") & (d["cidade"] != "")]

    d["bairro"] = d["bairro"].fillna("").replace("", "Desconhecido")
    for col in ["quartos", "banheiros", "garagens", "area_m2"]:
        d[col] = d[col].fillna(-1)

    # Label encoding para tipo e cidade — o modelo só aceita números
    le_t, le_c = LabelEncoder(), LabelEncoder()
    d["te"] = le_t.fit_transform(d["tipo"])
    d["ce"] = le_c.fit_transform(d["cidade"])

    # Target encoding para bairro:
    # em vez de um número arbitrário (LabelEncoder), cada bairro recebe a mediana
    # de preço dos imóveis naquele bairro. Assim o modelo recebe um valor em R$
    # que tem significado real — bairros caros entram como números altos.
    preco_global = d["preco"].median()
    bairro_map = d.groupby("bairro")["preco"].median().to_dict()
    d["be"] = d["bairro"].map(bairro_map).fillna(preco_global)

    X = d[["te", "ce", "be", "quartos", "banheiros", "garagens", "area_m2"]].values
    y = d["preco"].values

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Restrições monotônicas: garante que mais quartos, área maior e bairro mais
    # valorizado sempre aumentem o preço — sem isso o modelo pode aprender padrões
    # contraditórios nos dados (ex: 2 quartos mais caro que 3).
    # 0 = sem restrição | 1 = deve crescer com a variável
    restricoes = [0, 0, 1, 1, 1, 1, 1]  # tipo, cidade, bairro, quartos, banheiros, garagens, area

    mdl = HistGradientBoostingRegressor(
        max_iter=300, random_state=42, monotonic_cst=restricoes
    )
    mdl.fit(X_treino, y_treino)

    y_pred = mdl.predict(X_teste)
    r2  = r2_score(y_teste, y_pred)
    mae = mean_absolute_error(y_teste, y_pred)

    return mdl, le_t, le_c, bairro_map, preco_global, r2, mae, d, X_teste, y_teste


with st.spinner("Treinando modelo..."):
    mdl, le_t, le_c, bairro_map, preco_global, r2, mae, df_model, Xte, yte = treinar()


# Importância calculada por permutação porque HistGradientBoosting não expõe
# feature_importances_ diretamente. A ideia: embaralha uma variável de cada vez
# e mede quanto o R² piora — quanto mais piora, mais importante a variável é.
@st.cache_data
def calc_importancia():
    res = permutation_importance(mdl, Xte, yte, n_repeats=8, random_state=42, n_jobs=-1)
    return res.importances_mean


# ── Interface ─────────────────────────────────────────────────────────────────

st.title("📈 Previsão de Preço de Imóvel")
st.markdown(
    "Modelo **Gradient Boosting** treinado sobre os anúncios da OLX Paraíba para "
    "estimar o valor de mercado de um imóvel com base em suas características."
)
st.divider()

m1, m2, m3 = st.columns(3)
m1.metric("Imóveis no treino", f"{len(df_model):,}")
m2.metric("R² do modelo",      f"{r2:.1%}")
m3.metric("Erro médio (MAE)",  f"R$ {mae:,.0f}")

st.divider()

col_form, col_imp, col_dist = st.columns([1, 1, 1.5])

# ── Formulário de simulação ───────────────────────────────────────────────────
with col_form:
    st.subheader("Simule um imóvel")

    tipos_v   = sorted(TIPOS_VALIDOS)
    cidades_v = sorted(c for c in df["cidade"].unique() if c)

    t_sel = st.selectbox("Tipo",   tipos_v)
    c_sel = st.selectbox("Cidade", cidades_v)

    # Lista só os bairros que existem na cidade selecionada
    bairros_cidade = sorted(
        b for b in df.loc[df["cidade"] == c_sel, "bairro"].unique() if b
    )
    b_opts = ["Não especificado"] + bairros_cidade
    b_sel_nome = st.selectbox("Bairro", b_opts)

    a_sel  = st.number_input("Área (m²)",  10, 1000, 70)
    q_sel  = st.slider("Quartos",          0, 10, 2)
    ba_sel = st.slider("Banheiros",        0, 8,  1)
    g_sel  = st.slider("Garagens",         0, 6,  1)

    if st.button("Estimar preço →", type="primary", use_container_width=True):
        te = le_t.transform([t_sel])[0] if t_sel in le_t.classes_ else 0
        ce = le_c.transform([c_sel])[0] if c_sel in le_c.classes_ else 0

        bairro_enc = b_sel_nome if b_sel_nome != "Não especificado" else "Desconhecido"
        be = bairro_map.get(bairro_enc, preco_global)

        pred = mdl.predict([[te, ce, be, q_sel, ba_sel, g_sel, a_sel]])[0]

        # Busca imóveis similares no dataset pra mostrar uma faixa de referência
        sub = df_model[
            (df_model["tipo"] == t_sel) &
            (df_model["cidade"] == c_sel)
        ]
        if b_sel_nome != "Não especificado":
            sub_b = sub[sub["bairro"] == b_sel_nome]
            if len(sub_b) >= 3:
                sub = sub_b

        st.metric("Preço estimado", f"R$ {pred:,.0f}")

        if len(sub) >= 3:
            st.caption(
                f"Referência ({c_sel}{' — ' + b_sel_nome if b_sel_nome != 'Não especificado' else ''}): "
                f"R$ {sub['preco'].quantile(0.25):,.0f} – R$ {sub['preco'].quantile(0.75):,.0f}"
            )

# ── Importância das variáveis ─────────────────────────────────────────────────
with col_imp:
    st.subheader("O que mais influencia o preço?")
    feat_names = ["Tipo", "Cidade", "Bairro", "Quartos", "Banheiros", "Garagens", "Área m²"]

    with st.spinner("Calculando importância das variáveis..."):
        importancias = calc_importancia()

    imp_df = (
        pd.DataFrame({"Feature": feat_names, "Importância": importancias})
        .sort_values("Importância")
    )
    fig = px.bar(
        imp_df, x="Importância", y="Feature", orientation="h",
        color="Importância", color_continuous_scale="Blues",
        labels={"Importância": "Importância relativa", "Feature": ""},
    )
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

# ── Distribuição de preços por tipo ──────────────────────────────────────────
with col_dist:
    st.subheader("Distribuição de preços por tipo")

    # Pega os 6 tipos com maior mediana de preço pra não poluir o gráfico
    tipos_plot = (
        df_model.groupby("tipo")["preco"]
        .median()
        .sort_values(ascending=False)
        .head(6)
        .index.tolist()
    )

    df_plot = df_model[
        df_model["tipo"].isin(tipos_plot) &
        df_model["preco"].between(20_000, 3_000_000)
    ]

    fig2 = px.box(
        df_plot, x="tipo", y="preco",
        color="tipo",
        labels={"tipo": "", "preco": "Preço (R$)"},
        category_orders={"tipo": tipos_plot},
    )
    fig2.update_layout(showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)
