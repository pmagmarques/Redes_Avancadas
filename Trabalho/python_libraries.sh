#!/bin/bash

# Atualizar e instalar dependências básicas
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv build-essential

# Criar um ambiente virtual
python3 -m venv venv
source venv/bin/activate

# Instalar bibliotecas essenciais para IA, machine learning, deep learning, etc.

# IA e machine learning
pip install numpy pandas scipy matplotlib scikit-learn seaborn

# Deep Learning e Redes Neurais
pip install tensorflow keras pytorch torchvision

# Problemas de Transporte e Logística (como Caixeiro Viajante)
pip install ortools gurobipy

# Estatísticas
pip install statsmodels scipy

# Análise de Fraudes
pip install xgboost lightgbm catboost

# Reconhecimento de Imagem
pip install opencv-python pillow scikit-image

# Visão Computacional
pip install tensorflow-hub tensorflow-keras-vis

# Redes Neurais e Aprendizado por Reforço
pip install stable-baselines3 gym

# Problema do Caixeiro Viajante (Traveling Salesman Problem)
pip install tsplib95

# Reconhecimento de Fraude (Machine Learning para Fraude)
pip install fraud-detection

# Realidade Aumentada e Virtual (AR/VR)
pip install pyglet pygame

# AML, KYC, e ATF
pip install pycryptodome cryptography

# Churn Detection (Detecção de Churn)
pip install imbalanced-learn

# Redefinir o cache do pip
pip cache purge

# API Google para carregar datasets (Stock Market Data)
# Instalar a biblioteca Google API
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib

# Autenticação com a API do Google
echo "Para configurar a API da Google para carregar os dados do mercado de ações, siga as instruções a seguir:"
echo "1. Crie um projeto no Google Cloud: https://console.cloud.google.com/"
echo "2. Ative a API do Google Finance ou o Google Cloud Storage (se necessário)."
echo "3. Crie credenciais (arquivo JSON) para autenticação."
echo "4. Baixe o arquivo JSON das credenciais e mova-o para este diretório."

# Carregar o dataset de Stock Market (exemplo usando Google Finance API)
echo "Agora, use o seguinte código Python para acessar e carregar os dados do Google Stock Market."

echo "import os
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Caminho para o arquivo de credenciais JSON
credentials_path = 'path_to_your_credentials.json'

# Autenticação com a API do Google
credentials = service_account.Credentials.from_service_account_file(credentials_path)

# Criação de um serviço de API
service = build('finance', 'v1', credentials=credentials)

# Exemplo de requisição de dados de mercado de ações
stock_data = service.stocks().list(symbol='GOOG').execute()
print(stock_data)"
