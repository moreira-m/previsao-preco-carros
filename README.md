
## Passo a passo
1. Crie e ative o ambiente: `python3 -m venv venv && source venv/bin/activate`
2. Instale dependencias: `pip install -r requirements.txt`
3. Rode o treino: `python train_model.py` (gera `modelo_carros.pkl` e mostra MAE/RMSE/R2)
4. Rode o app: `streamlit run app.py` e abra o link mostrado
5. Preencha marca, modelo, combustivel, cambio e ano do modelo para ver o preco estimado (opcoes limitadas a marcas populares)

## Como o modelo funciona
- Usa as colunas `brand`, `model`, `fuel`, `gear`, `year_model` como entrada e o alvo `avg_price_brl`. O dataset e filtrado para marcas populares (Chevrolet, Volkswagen, Fiat, Ford, Toyota) para manter simplicidade.
- Converte categorias com `pandas.get_dummies` (sem `drop_first`) e divide treino/teste.
- Treina uma `LinearRegression` no `log1p` do preco para lidar com valores altos e reverte com `expm1` na previsao.
- O app aplica o mesmo `get_dummies`, alinha colunas e retorna o valor em R$.
