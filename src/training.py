import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import random
from sklearn.feature_extraction.text import TfidfVectorizer

# Garante que temos todos os pacotes NLTK
nltk.download('wordnet')
nltk.download('punkt')

lemmatizer = WordNetLemmatizer()

classes = []
data_for_processing = [] # Nossa nova lista de pares (pattern, tag)
ignore_words = ['?', '!', '.', ',']

# Carregando o JSON
try:
    data_file = open('../data/intents.json', encoding='utf-8').read() 
except FileNotFoundError:
    print("ERRO: 'intents.json' não encontrado na pasta 'data/'. Verifique o caminho.")
    exit()

intents = json.loads(data_file)
print("Processando intents...")

# --- ETAPA 1: Criar pares (pattern, tag) e lista de classes ---
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # 1. Limpa e lematiza o pattern (frase)
        w = nltk.word_tokenize(pattern)
        lemmatized_words = [lemmatizer.lemmatize(word.lower()) for word in w if word not in ignore_words]
        processed_pattern = " ".join(lemmatized_words)
        
        # 2. Adiciona o par (frase limpa, tag) à nossa lista
        data_for_processing.append((processed_pattern, intent['tag']))

        # 3. Adiciona a tag à lista de classes (se for nova)
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

classes = sorted(list(set(classes)))
pickle.dump(classes, open('classes.pkl', 'wb'))

# --- ETAPA 2: Embaralhar os dados ANTES de vetorizar ---
random.shuffle(data_for_processing)

# Separa os dados em X (patterns) e Y (tags)
training_patterns = [data[0] for data in data_for_processing]
training_tags = [data[1] for data in data_for_processing]

print(f"Total de {len(training_patterns)} padrões de treino embaralhados.")

# --- ETAPA 3: Vetorizar X (TF-IDF) e Y (One-Hot) ---
print("Criando dados de treino com TF-IDF...")


# 1. Cria e treina o vetorizador TF-IDF
vectorizer = TfidfVectorizer()
train_x = vectorizer.fit_transform(training_patterns).toarray()
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb')) # Salva o vetorizador

# 2. Cria o train_y (One-Hot encode)
output_empty = [0] * len(classes)
train_y = []
for tag in training_tags:
    output_row = list(output_empty)
    output_row[classes.index(tag)] = 1
    train_y.append(output_row)

# Converte para numpy arrays
train_x = np.array(train_x)
train_y = np.array(train_y)

print("Dados de treino X e Y criados e alinhados com sucesso.")

# --- ETAPA 4: Construir e Treinar o Modelo Keras ---
print("Construindo o novo modelo Keras...")
model = Sequential()
model.add(Dense(128, input_shape=(train_x.shape[1],), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Usando o 'adam' e 500 epochs que já definimos
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Vamos reduzir as epochs para 200, pois 'adam' e TF-IDF aprendem rápido
# e 500 pode estar causando "overfitting" (passando do ponto).
model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

model.save('model.h5')
print("Novo modelo 'src/model.h5' (TF-IDF CORRIGIDO) salvo.")