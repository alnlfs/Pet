import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import json
import random
import gensim
import streamlit as st
import os

@st.cache_resource
def load_all_models_and_data():
    print("Iniciando carregamento de modelos...")
    
    # Caminhos relativos (funciona localmente e no Streamlit)
    base_path = os.path.dirname(__file__) # Pega a pasta 'src'
    model_h5_path = os.path.join(base_path, 'model.h5')
    w2v_model_path = os.path.join(base_path, 'dog_w2v.model')
    words_pkl_path = os.path.join(base_path, 'words.pkl')
    classes_pkl_path = os.path.join(base_path, 'classes.pkl')
    intents_json_path = os.path.join(base_path, '..', 'data', 'intents.json') # Sobe um nível, entra em 'data'

    try:
        model = load_model(model_h5_path)
        w2v_model = gensim.models.Word2Vec.load(w2v_model_path)
        words = pickle.load(open(words_pkl_path, 'rb'))
        classes = pickle.load(open(classes_pkl_path, 'rb'))
        data_file = open(intents_json_path, encoding='utf-8').read()
        intents = json.loads(data_file)
    except FileNotFoundError as e:
        st.error(f"ERRO: Não foi possível encontrar um arquivo essencial: {e}")
        st.error(f"Caminho procurado: {e.filename}")
        st.stop()
        
    VECTOR_SIZE = w2v_model.vector_size
    lemmatizer = WordNetLemmatizer()
    
    print("...Modelos e dados carregados com sucesso!")
    return model, w2v_model, words, classes, intents, lemmatizer, VECTOR_SIZE

# --- Funções de processamento ---
def get_sentence_vector(sentence_tokens, w2v_model, lemmatizer, VECTOR_SIZE):
    sentence_vec = np.zeros(VECTOR_SIZE)
    count = 0
    for word in sentence_tokens:
        lemmatized_word = lemmatizer.lemmatize(word.lower())
        if lemmatized_word in w2v_model.wv:
            sentence_vec += w2v_model.wv[lemmatized_word]
            count += 1
    if count > 0:
        sentence_vec /= count
    return sentence_vec

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return sentence_words 

def predict_class(sentence_vec, model, classes):
    res = model.predict(np.array([sentence_vec]))[0]
    
    ERROR_THRESHOLD = 0.05 
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_bot_response(ints, intents_json):
    tag = "sem_resposta" # Define 'sem_resposta' como padrão
    if ints: # Se o modelo deu uma previsão acima do limiar
        tag = ints[0]['intent']
        
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
            
    # Se algo der errado, ele ainda usará a resposta de "sem_resposta"
    if 'result' not in locals():
         for i in list_of_intents:
            if i['tag'] == 'sem_resposta':
                result = random.choice(i['responses'])
                break
                
    return result

def get_response_from_message(message, model, w2v_model, words, classes, intents, lemmatizer, VECTOR_SIZE):
    user_tokens = clean_up_sentence(message)
    user_vec = get_sentence_vector(user_tokens, w2v_model, lemmatizer, VECTOR_SIZE)
    ints = predict_class(user_vec, model, classes)
    res = get_bot_response(ints, intents)
    return res