import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import json
import random
import streamlit as st
import os
# (Gensim não é mais necessário, pois estamos usando TF-IDF)

@st.cache_resource
def load_all_models_and_data():
    # --- Downloads do NLTK ---
    print("Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('punkt_tab')
    print("NLTK data downloaded.")
    
    print("Iniciando carregamento de modelos (TF-IDF)...")
    
    # --- Caminhos dos Arquivos ---
    base_path = os.path.dirname(__file__) # Pega a pasta 'src'
    model_h5_path = os.path.join(base_path, 'model.h5')
    vectorizer_pkl_path = os.path.join(base_path, 'vectorizer.pkl') # <-- USA O VETORIZADOR
    classes_pkl_path = os.path.join(base_path, 'classes.pkl')
    intents_json_path = os.path.join(base_path, '..', 'data', 'intents.json')

    # --- Carregando os Arquivos Corretos ---
    try:
        model = load_model(model_h5_path)
        vectorizer = pickle.load(open(vectorizer_pkl_path, 'rb')) # <-- CARREGA O VETORIZADOR
        classes = pickle.load(open(classes_pkl_path, 'rb'))
        data_file = open(intents_json_path, encoding='utf-8').read()
        intents = json.loads(data_file)
    except FileNotFoundError as e:
        st.error(f"ERRO: Não foi possível encontrar um arquivo essencial: {e}")
        st.error(f"Caminho procurado: {e.filename}")
        st.stop()
        
    lemmatizer = WordNetLemmatizer()
    
    print("...Modelos e dados carregados com sucesso!")
    
    # --- A CORREÇÃO ESTÁ AQUI ---
    # Retorna os 5 itens que o app.py espera:
    return model, vectorizer, classes, intents, lemmatizer

# --- Funções de processamento ATUALIZADAS para TF-IDF ---

def clean_up_sentence_and_lemmatize(sentence, lemmatizer):
    ignore_words = ['?', '!', '.', ',']
    sentence_words = nltk.word_tokenize(sentence)
    lemmatized_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words if word not in ignore_words]
    # Retorna a frase limpa como uma string única
    return " ".join(lemmatized_words)

def get_tfidf_vector(sentence_string, vectorizer):
    # Transforma a string única em um vetor TF-IDF
    vector = vectorizer.transform([sentence_string]).toarray()
    return vector

def predict_class(sentence_vector, model, classes):
    res = model.predict(sentence_vector)[0]
    
    ERROR_THRESHOLD = 0.05 
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_bot_response(ints, intents_json):
    tag = "sem_resposta" # Define 'sem_resposta' como padrão
    if ints: 
        tag = ints[0]['intent']
        
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
            
    if 'result' not in locals():
         for i in list_of_intents:
            if i['tag'] == 'sem_resposta':
                result = random.choice(i['responses'])
                break
                
    return result

# --- A FUNÇÃO PRINCIPAL QUE O SITE VAI CHAMAR (ATUALIZADA) ---

def get_response_from_message(message, model, vectorizer, classes, intents, lemmatizer):
    # 1. Limpa e lematiza a frase
    user_string_clean = clean_up_sentence_and_lemmatize(message, lemmatizer)
    # 2. Vetoriza com TF-IDF
    user_vec = get_tfidf_vector(user_string_clean, vectorizer)
    # 3. Prevê
    ints = predict_class(user_vec, model, classes)
    # 4. Obtém resposta
    res = get_bot_response(ints, intents)
    return res