import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import random
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('wordnet')
nltk.download('punkt')

lemmatizer = WordNetLemmatizer()

classes = []
data_for_processing = [] 
ignore_words = ['?', '!', '.', ',']

try:
    data_file = open('../data/intents.json', encoding='utf-8').read() 
except FileNotFoundError:
    print("ERRO: 'intents.json' não encontrado na pasta 'data/'. Verifique o caminho.")
    exit()

intents = json.loads(data_file)
print("Processando intents...")

for intent in intents['intents']:
    for pattern in intent['patterns']:
        
        w = nltk.word_tokenize(pattern)
        lemmatized_words = [lemmatizer.lemmatize(word.lower()) for word in w if word not in ignore_words]
        processed_pattern = " ".join(lemmatized_words)
        
        
        data_for_processing.append((processed_pattern, intent['tag']))

     
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

classes = sorted(list(set(classes)))
pickle.dump(classes, open('classes.pkl', 'wb'))

random.shuffle(data_for_processing)

training_patterns = [data[0] for data in data_for_processing]
training_tags = [data[1] for data in data_for_processing]

print(f"Total de {len(training_patterns)} padrões de treino embaralhados.")

print("Criando dados de treino com TF-IDF...")

vectorizer = TfidfVectorizer()
train_x = vectorizer.fit_transform(training_patterns).toarray()
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb')) 


output_empty = [0] * len(classes)
train_y = []
for tag in training_tags:
    output_row = list(output_empty)
    output_row[classes.index(tag)] = 1
    train_y.append(output_row)


train_x = np.array(train_x)
train_y = np.array(train_y)

print("Dados de treino X e Y criados e alinhados com sucesso.")

print("Construindo o novo modelo Keras...")
model = Sequential()
model.add(Dense(128, input_shape=(train_x.shape[1],), activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(train_y[0]), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


history = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

model.save('model.h5')
print("Novo modelo 'src/model.h5' (TF-IDF CORRIGIDO) salvo.")


import matplotlib.pyplot as plt

# G1: Evolução do aprendizado
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Acurácia (Treino)')
plt.title('Evolução da Acurácia do Modelo')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.savefig('grafico_acuracia.png')
print("Gráfico 1 salvo: grafico_acuracia.png")

# G2: Perda ao longo do tempo
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Erro (Loss)', color='red')
plt.title('Diminuição do Erro do Modelo')
plt.xlabel('Épocas')
plt.ylabel('Erro')
plt.legend()
plt.savefig('grafico_erro.png')
print("Gráfico 2 salvo: grafico_erro.png")

# G3: Frases por intenção
intent_counts = []
intent_labels = []
for intent in intents['intents']:
    intent_labels.append(intent['tag'])
    intent_counts.append(len(intent['patterns']))

plt.figure(figsize=(12, 8))
plt.barh(intent_labels, intent_counts, color='skyblue')
plt.xlabel('Número de Frases de Exemplo')
plt.title('Balanceamento dos Dados de Treino')
plt.tight_layout()
plt.savefig('grafico_distribuicao.png')
print("Gráfico 3 salvo: grafico_distribuicao.png")