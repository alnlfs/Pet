import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import random
import gensim

nltk.download('wordnet') # Baixa o dicionário para lematização

print("Carregando modelo Word2Vec 'dog_w2v.model'...")
w2v_model = gensim.models.Word2Vec.load("dog_w2v.model")
VECTOR_SIZE = w2v_model.vector_size 
lemmatizer = WordNetLemmatizer()

words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

# Carregando o JSON (agora ele procura na pasta ../data/)
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
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

def get_sentence_vector(sentence_tokens):
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

training = []
output_empty = [0] * len(classes)
print("Criando dados de treino com Word2Vec...")

for doc in documents:
    pattern_tokens = doc[0]
    sentence_vec = get_sentence_vector(pattern_tokens)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([sentence_vec, output_row])

random.shuffle(training)
train_x = np.array([i[0] for i in training])
train_y = np.array([i[1] for i in training])
print("Dados de treino criados com sucesso.")

print("Construindo o novo modelo Keras...")
model = Sequential()
model.add(Dense(128, input_shape=(VECTOR_SIZE,), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)
model.save('model.h5')
print("Novo modelo 'src/model.h5' salvo.")