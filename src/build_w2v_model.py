import wikipediaapi
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
import string

# Configurar o NLTK (necessário na primeira vez)
nltk.download('punkt')
nltk.download('punkt_tab') # Garantindo que temos o tokenizer de português

print("Iniciando coleta de dados da Wikipédia...")
wiki_api = wikipediaapi.Wikipedia(
    user_agent='PetBot-Project (seu.email@exemplo.com)',
    language='pt',
    extract_format=wikipediaapi.ExtractFormat.WIKI
)

paginas_wiki = [
    # Cães
    "Cão", "Adestramento de cães", "Raças de cães", "Comportamento canino",
    "Golden Retriever", "Labrador Retriever", "Buldogue", "Poodle", "Pastor-alemão",
    "Rottweiler", "Shih-tzu", "Lulu-da-pomerânia", "Pug", "Dachshund",
    # Gatos
    "Gato", "Gato persa", "Siamês (gato)", "Maine Coon", "Sphynx (gato)",
    "Gato-de-bengala", "Ragdoll", "Abissínio", "Scottish Fold",
    # Saúde
    "Veterinária", "Vacinação", "Castração", "Cinomose", "Parvovirose canina",
    "Leishmaniose visceral canina", "Raiva (doença)", "Otite", "Dermatite atópica",
    "Pulga", "Carrapato", "Vermífugo",
    # Cuidados
    "Ração", "Dieta canina", "Dieta felina", "Enriquecimento ambiental", "Caixa de areia"
]

corpus_completo = []
for pagina in paginas_wiki:
    try:
        texto = wiki_api.page(pagina).text
        print(f"Coletado: {pagina}")
        corpus_completo.append(texto)
    except Exception as e:
        print(f"Erro ao baixar {pagina}: {e}")

print("Coleta finalizada. Iniciando pré-processamento...")
texto_completo = " ".join(corpus_completo)

def limpar_e_tokenizar(texto):
    tokens_processados = []
    for frase in sent_tokenize(texto, language='portuguese'):
        frase_limpa = frase.lower().translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(frase_limpa, language='portuguese')
        tokens = [palavra for palavra in tokens if palavra.isalpha() and len(palavra) > 2]
        if tokens:
            tokens_processados.append(tokens)
    return tokens_processados

dados_treino_w2v = limpar_e_tokenizar(texto_completo)
print(f"Processamento finalizado. {len(dados_treino_w2v)} frases tokenizadas.")

print("Iniciando treinamento do modelo Word2Vec...")
w2v_model = gensim.models.Word2Vec(
    sentences=dados_treino_w2v,
    vector_size=100,
    window=5,
    min_count=2, # Reduzimos para 2 para capturar mais palavras
    workers=4
)

# Salva o modelo treinado DENTRO da pasta 'src'
w2v_model.save("dog_w2v.model")
print("\nTreinamento concluído!")
print("Modelo salvo como 'src/dog_w2v.model'.")