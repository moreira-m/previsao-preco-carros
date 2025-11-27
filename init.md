# Projeto: Previsão de Preço de Carros Usados (Versão Simples – KISS)

> **IMPORTANTE (para a IA/Codex):**  
> Todo o projeto deve seguir o princípio **KISS – Keep It Simple, Stupid**.  
> O foco é: **código simples, funcional e fácil de explicar em uma apresentação rápida (até 10 minutos)**.

---

## Objetivo do projeto

Criar um mini‐projeto em **Python** que:

1. Usa um **dataset pequeno** de carros usados (pode ser fictício, mas coerente).
2. Treina um modelo muito simples de **regressão** para prever o preço do carro (`selling_price`).
3. Salva o modelo treinado em arquivo.
4. Cria uma **interface web em Streamlit** onde o usuário informa dados do carro e recebe a previsão de preço.
5. Tenha código **curto, direto e bem comentado em português**, fácil de entender.

---

## Regras gerais de simplicidade

- Nada de arquitetura complexa, múltiplas camadas ou padrões avançados.
- **Não usar classes** se não for estritamente necessário; prefira funções simples e código sequencial.
- Comentários em **português simples**, explicando blocos de código de forma didática.
- Evitar arquivos desnecessários: use apenas os arquivos listados neste documento.
- O foco é um projeto **didático**, não um sistema profissional.

---

## Estrutura de pastas desejada

Crie exatamente esta estrutura:

```text
previsao-preco-carros/
  ├─ data/
  │    └─ car-details.xlsx
  ├─ train_model.py
  ├─ app.py
  ├─ requirements.txt
  └─ README.md
