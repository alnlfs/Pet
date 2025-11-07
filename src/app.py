import streamlit as st
from chatbot_engine import load_all_models_and_data, get_response_from_message

# --- Carregar os modelos (Usa o cache do engine) ---
model, w2v_model, words, classes, intents, lemmatizer, VECTOR_SIZE = load_all_models_and_data()

# --- Configuração da Página ---
st.set_page_config(page_title="PetBot - Seu Amigo Pet", page_icon="🐾")

st.title("🐾 PetBot - Chatbot sobre Pets")
st.caption("Um assistente de IA treinado para responder suas dúvidas.")

# --- Container de Chat ---
# Cria um container com uma altura fixa para as mensagens do chat.
chat_container = st.container(height=500)

# --- Inicialização do Histórico do Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(
        {"role": "assistant", "content": "Olá! Eu sou o PetBot. Como posso ajudar você com seu pet hoje?"}
    )

# --- Exibir mensagens antigas DENTRO do container ---
for message in st.session_state.messages:
    with chat_container.chat_message(message["role"]):
        chat_container.markdown(message["content"])

# --- Receber nova entrada do usuário (Fica fixo no final) ---
if prompt := st.chat_input("Qual sua dúvida?"):
    
    # 1. Adiciona e exibe a mensagem do usuário (DENTRO do container)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_container.chat_message("user"):
        chat_container.markdown(prompt)

    # 2. Pega a resposta do bot (chamando nosso engine)
    bot_response = get_response_from_message(
        prompt, model, w2v_model, words, classes, intents, lemmatizer, VECTOR_SIZE
    )

    # 3. Adiciona e exibe a resposta do bot (DENTRO do container)
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    with chat_container.chat_message("assistant"):
        chat_container.markdown(bot_response)