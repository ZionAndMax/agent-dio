# 🛡️ Sentinel-AI: Agentes Autônomos para Detecção de Anomalias no Mercado Financeiro

![Status do Projeto](https://img.shields.io/badge/Status-Conclu%C3%ADdo-brightgreen)
![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue)
![Area](https://img.shields.io/badge/Setor-Fintech-gold)

## 📌 Visão Geral
Este projeto documenta a implementação de uma infraestrutura baseada em **Agentes de Inteligência Artificial** para atuar na análise e detecção de anomalias dentro do mercado financeiro. O objetivo do sistema não é apenas reagir aos dados, mas interpretar o terreno, identificando movimentações atípicas que sinalizem fraudes ou flutuações de mercado fora do padrão.

Para garantir a eficiência da campanha de análise, o foco principal recai sobre a **Engenharia de Atributos (Feature Engineering)**. Ao invés de entregar dados brutos ao algoritmo, o agente é "educado" com variáveis de contexto (como faixas de horário e alertas de volume) para refinar sua capacidade de tomada de decisão.

---

## 🏗️ Arquitetura do Sistema
A operação funciona em um fluxo contínuo e automatizado (Pipeline Unificado):
1. **A Forja de Dados:** Criação automatizada de um dataset sintético (`transacoes_financeiras.csv`) que simula o fluxo transacional e injeta alvos de teste.
2. **O Estrategista (AI Agent):** O núcleo de processamento que emprega `Isolation Forest` e recebe instruções de comportamento (Prompting) para isolar as ameaças nos dados recém-gerados.

---

## 🛠️ Stack Tecnológica
- **Python 3.x**
- **Pandas:** Estruturação e manipulação tática da base de dados.
- **Scikit-Learn:** Modelagem matemática e detecção de *outliers*.
- **Numpy:** Operações de alta performance.

---

## 📂 Implementação Completa (Pipeline End-to-End)

Todo o ecossistema — desde a geração da base em `.csv` até a educação e execução do agente — está centralizado no script abaixo. Copie, cole e execute. O terreno é forjado e a varredura é feita na mesma investida.

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os

# ==========================================
# 1. FORJA DE DADOS: O CAMPO DE BATALHA
# ==========================================
def generate_financial_base(filename="transacoes_financeiras.csv"):
    """Gera o arquivo .csv com milhares de transações e injeta anomalias estratégicas."""
    np.random.seed(42)
    n_records = 1000
    
    data = {
        'id_transacao': range(1, n_records + 1),
        'valor': np.random.uniform(10, 500, n_records),
        'hora': np.random.randint(8, 22, n_records), # Horário comercial
        'tipo_conta': np.random.choice(['Pessoal', 'Empresarial'], n_records)
    }
    
    df = pd.DataFrame(data)
    
    # Injetando anomalias estratégicas (transações de alto valor na madrugada)
    anomalias = [
        {'id_transacao': 1001, 'valor': 15000.0, 'hora': 3, 'tipo_conta': 'Pessoal'},
        {'id_transacao': 1002, 'valor': 12000.0, 'hora': 4, 'tipo_conta': 'Pessoal'},
        {'id_transacao': 1003, 'valor': 9000.0, 'hora': 2, 'tipo_conta': 'Empresarial'}
    ]
    
    df = pd.concat([df, pd.DataFrame(anomalias)], ignore_index=True)
    df.to_csv(filename, index=False)
    print(f"[LOG] Terreno mapeado. Base '{filename}' forjada com sucesso.")

# ==========================================
# 2. O ESTRATEGISTA: AGENTE IA
# ==========================================
class FinancialAgent:
    def __init__(self, prompt_config):
        self.role = "Analista Técnico de Segurança Financeira"
        self.prompt_instruction = prompt_config
        self.model = IsolationForest(contamination=0.01, random_state=42)
        self.scaler = StandardScaler()

    def educate_agent(self, df):
        """
        Feature Engineering: Traduzindo o contexto para o agente.
        """
        df['is_madrugada'] = df['hora'].apply(lambda x: 1 if 0 <= x <= 6 else 0)
        df['alerta_volume'] = df['valor'].apply(lambda x: 1 if x > 2000 else 0)
        df['valor_log'] = np.log1p(df['valor'])
        return df

    def execute_campaign(self, csv_filepath):
        print(f"\n🚀 Iniciando operação...")
        print(f"| Papel do Agente: {self.role}")
        print(f"| Diretriz: {self.prompt_instruction}\n")
        
        df = pd.read_csv(csv_filepath)
        processed_df = self.educate_agent(df)
        
        features = ['valor_log', 'hora', 'is_madrugada', 'alerta_volume']
        X = self.scaler.fit_transform(processed_df[features])
        
        processed_df['status_seguranca'] = self.model.fit_predict(X)
        return processed_df

# ==========================================
# 3. EXECUÇÃO DA MISSÃO
# ==========================================
if __name__ == "__main__":
    arquivo_base = "transacoes_financeiras.csv"
    
    generate_financial_base(arquivo_base)
    
    diretriz = "Analise o fluxo transacional. Isole movimentações que divirjam do padrão de normalidade corporativo e de varejo, priorizando variáveis de tempo e volume."
    agente = FinancialAgent(diretriz)
    
    resultado = agente.execute_campaign(arquivo_base)
    ameacas_detectadas = resultado[resultado['status_seguranca'] == -1]
    
    print(f"📊 Varredura concluída. Alvos identificados na base:")
    print("-" * 60)
    print(ameacas_detectadas[['id_transacao', 'valor', 'hora', 'is_madrugada']])
    print("-" * 60)
