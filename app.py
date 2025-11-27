"""App Streamlit simples para prever preco (BRL) usando FIPE 2022."""

import joblib
import numpy as np
import pandas as pd
import streamlit as st


def carregar_modelo():
    """Carrega o pickle com modelo e colunas."""
    try:
        dados = joblib.load("modelo_carros.pkl")
    except FileNotFoundError:
        st.error("modelo_carros.pkl nao encontrado. Rode train_model.py antes.")
        st.stop()
    return dados["modelo"], dados["colunas"]


def carregar_opcoes():
    """Ler o CSV para sugerir marcas/combustiveis/cambios/modelos conhecidos."""
    try:
        df = pd.read_csv("data/fipe_2022.csv")
        for col in ["brand", "model", "fuel", "gear"]:
            df[col] = df[col].astype(str).str.strip().str.lower()
        marcas_chave = {"gm - chevrolet", "vw - volkswagen", "fiat", "ford", "toyota"}
        df = df[df["brand"].isin(marcas_chave)]
        marcas = sorted(df["brand"].unique())
        combustiveis = sorted(df["fuel"].unique())
        cambios = sorted(df["gear"].unique())
        modelos_por_marca = {
            marca: sorted(df[df["brand"] == marca]["model"].unique()) for marca in marcas
        }
        return marcas, combustiveis, cambios, modelos_por_marca
    except FileNotFoundError:
        return [], [], [], {}


def preparar_entrada(entrada: dict, colunas_modelo: list) -> pd.DataFrame:
    """Aplica get_dummies e reindex para alinhar com o treino."""
    df = pd.DataFrame([entrada])
    for col in ["brand", "model", "fuel", "gear"]:
        df[col] = df[col].astype(str).str.strip().str.lower()
    # Mantem todas as categorias para nao perder informacao em uma unica linha
    df = pd.get_dummies(df, drop_first=False)
    df = df.reindex(columns=colunas_modelo, fill_value=0)
    return df


def main():
    st.title("Estimativa de preco (FIPE 2022)")
    st.write("Preencha os campos e veja o valor estimado em reais.")

    modelo, colunas_modelo = carregar_modelo()
    marcas, combustiveis, cambios, modelos_por_marca = carregar_opcoes()

    # Entradas principais
    brand = st.selectbox("Marca", marcas or ["acura"])
    modelos_opcoes = modelos_por_marca.get(brand, [])
    model = st.selectbox("Modelo", modelos_opcoes or ["modelo"])
    fuel = st.selectbox("Combustivel", combustiveis or ["gasoline"])
    gear = st.selectbox("Cambio", cambios or ["manual"])
    year_model = st.number_input("Ano do modelo", min_value=1950, max_value=2024, value=1995, step=1)

    if st.button("Calcular preco"):
        entrada = {
            "brand": brand,
            "model": model,
            "fuel": fuel,
            "gear": gear,
            "year_model": year_model,
        }

        dados_prontos = preparar_entrada(entrada, colunas_modelo)
        preco_log = modelo.predict(dados_prontos)[0]
        preco = np.expm1(preco_log)
        st.success(f"Preco estimado: R$ {preco:,.2f}")


if __name__ == "__main__":
    main()
