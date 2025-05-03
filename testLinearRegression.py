import os
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan

# Configurações gerais
warnings.simplefilter('ignore')
sns.set(style="whitegrid")


def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')


def carregar_dados(caminho):
    return pd.read_csv(caminho)


def preencher_dados(data):
    for col in data.select_dtypes(include='object').columns:
        data[col].fillna(data[col].mode().iloc[0], inplace=True)
    for col in data.select_dtypes(include='number').columns:
        data[col].fillna(data[col].median(), inplace=True)
    return data


def analise_estatistica(data):
    desc = data.describe().T
    desc["mediana"] = data.median(numeric_only=True)
    print("\nEstatísticas Descritivas:\n", desc)
    return desc


def preparar_dummies(data, var_dependente):
    cat_vars = data.select_dtypes(include='object').columns.tolist()
    print("\nVariáveis categóricas:", cat_vars)

    data_dummies = pd.get_dummies(data, columns=cat_vars, drop_first=True)

    X = data_dummies.drop(columns=[var_dependente])
    y = data_dummies[var_dependente]

    X = sm.add_constant(X).astype(float)
    y = y.astype(float)

    return X, y, cat_vars


def ajustar_modelo_regressao(X, y):
    modelo = sm.OLS(y, X).fit()
    print("\nResumo do Modelo de Regressão:\n")
    print(modelo.summary())
    return modelo


def diagnostico_multicolinearidade(X):
    print("\nDiagnóstico de Multicolinearidade (VIF):")
    vif_df = pd.DataFrame()
    vif_df["variavel"] = X.columns
    vif_df["VIF"] = [variance_inflation_factor(
        X.values, i) for i in range(X.shape[1])]
    print(vif_df)
    return vif_df


def diagnostico_heterocedasticidade(modelo):
    fitted_vals = modelo.fittedvalues
    residuos = modelo.resid

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=fitted_vals, y=residuos)
    plt.axhline(0, linestyle='--', color='red')
    plt.xlabel("Valores Ajustados")
    plt.ylabel("Resíduos")
    plt.title("Resíduos vs Valores Ajustados")
    plt.tight_layout()
    plt.show()

    bp_test = het_breuschpagan(residuos, modelo.model.exog)
    labels = ["LM Statistic", "p-value", "F-value", "F p-value"]
    resultado = dict(zip(labels, bp_test))
    print("\nTeste de Breusch-Pagan:\n", resultado)
    return resultado


def comparar_modelos(modelo1, modelo2):
    print("\nComparação entre Modelo 1 e Modelo 2:")
    print(f"Modelo 1 - R² ajustado: {modelo1.rsquared_adj:.4f}")
    print(f"Modelo 2 - R² ajustado: {modelo2.rsquared_adj:.4f}")

    f_stat = (modelo1.ssr - modelo2.ssr) / \
        (modelo1.df_resid - modelo2.df_resid)
    f_den = modelo2.ssr / modelo2.df_resid
    f_valor = f_stat / f_den

    print(f"\nEstatística F aproximada: {f_valor:.4f}")
    print("Interprete: se a diferença de R² ajustado for pequena, o modelo mais simples pode ser preferido.")

    if modelo2.rsquared_adj > modelo1.rsquared_adj:
        print("Modelo 2 tem melhor ajuste ajustado.")
    else:
        print("Modelo 1 ainda tem melhor explicação dos dados.")


def recomendar_melhorias(modelo, X):
    coeficientes = modelo.params.sort_values()
    print("\nSugestões para melhorar o tempo de resposta com base no modelo:")
    print("-" * 50)

    for var, valor in coeficientes.items():
        if "SSD" in var or "armazenamento" in var:
            print("Discos SSD estão associados a menor tempo de resposta.")
        if "ram" in var and valor < 0:
            print("Aumento de RAM pode reduzir o tempo de resposta.")
        if "cpu" in var and valor < 0:
            print("Mais núcleos de CPU podem melhorar o desempenho.")
        if "latencia" in var and valor > 0:
            print(
                "Alta latência da rede ou disco está relacionada a respostas mais lentas.")
    print("-" * 50)


def main():
    clear_terminal()
    caminho = "data/dataset_20.csv"
    var_dependente = "tempo_resposta"

    data = carregar_dados(caminho)
    print("Valores faltantes antes do preenchimento:\n", data.isnull().sum())

    data = preencher_dados(data)
    print("\nValores faltantes após preenchimento:\n", data.isnull().sum())

    analise_estatistica(data)
    X_full, y, cat_vars = preparar_dummies(data, var_dependente)

    modelo1 = ajustar_modelo_regressao(X_full, y)
    diagnostico_multicolinearidade(X_full)
    diagnostico_heterocedasticidade(modelo1)

    # Remover variável com VIF alto ou impacto fraco (exemplo: latencia_ms)
    X_reduzido = X_full.drop(columns=["latencia_ms"])
    modelo2 = ajustar_modelo_regressao(X_reduzido, y)

    comparar_modelos(modelo1, modelo2)
    recomendar_melhorias(modelo1, X_full)


if __name__ == "__main__":
    main()
