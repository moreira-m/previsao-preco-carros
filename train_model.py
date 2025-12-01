from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def carregar_dados() -> pd.DataFrame:
    caminho = Path("data/fipe_2022.csv")
    if not caminho.exists():
        raise FileNotFoundError("Arquivo dataset nao encontrado")

    df = pd.read_csv(caminho)
    colunas = [
        "brand",
        "model",
        "fuel",
        "gear",
        "year_model",
        "avg_price_brl",
    ]
    df = df[colunas].dropna()
    return df


def normalizar_texto(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["brand", "model", "fuel", "gear"]:
        df[col] = df[col].astype(str).str.strip().str.lower()
    return df


def filtrar_marcas(df: pd.DataFrame) -> pd.DataFrame:
    marcas_chave = {
        "gm - chevrolet",
        "vw - volkswagen",
        "fiat",
        "ford",
        "toyota",
    }
    return df[df["brand"].isin(marcas_chave)]


def filtrar_todas_marcas(df: pd.DataFrame) -> pd.DataFrame:
    return df


def preparar_dados(df: pd.DataFrame):
    df = normalizar_texto(df)


    # usar apenas marcas comuns
    df = filtrar_marcas(df)


    # usar todas as marcas
    # df = filtrar_todas_marcas(df)




    X = df.drop("avg_price_brl", axis=1)
    y = df["avg_price_brl"]

    X_dummies = pd.get_dummies(X, drop_first=False)
    return X_dummies, y


def treinar_modelo(X: pd.DataFrame, y: pd.Series):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    y_train_log = np.log1p(y_train)

    modelo = LinearRegression()
    modelo.fit(X_train, y_train_log)

    y_pred_log = modelo.predict(X_test)
    y_pred = np.expm1(y_pred_log)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.4f}")

    joblib.dump({"modelo": modelo, "colunas": X.columns.tolist()}, "modelo_carros.pkl")
    print("Modelo salvo em modelo_carros.pkl")


def main():
    df = carregar_dados()
    X, y = preparar_dados(df)
    treinar_modelo(X, y)


if __name__ == "__main__":
    main()
