import streamlit as st
from chatbot_engine import load_all_models_and_data, get_response_from_message


model, vectorizer, classes, intents, lemmatizer = load_all_models_and_data()

st.set_page_config(page_title="PetBot - Seu Amigo Pet", page_icon="🐾")

st.title("🐾 PetBot - Chatbot sobre Pets")
st.caption("Um assistente de IA treinado para responder suas dúvidas.")


chat_container = st.container(height=500)

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(
        {"role": "assistant", "content": "Olá! Eu sou o PetBot. Como posso ajudar você com seu pet hoje?"}
    )

for message in st.session_state.messages:
    with chat_container.chat_message(message["role"]):
        chat_container.markdown(message["content"])

if prompt := st.chat_input("Qual sua dúvida?"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with chat_container.chat_message("user"):
        chat_container.markdown(prompt)

    bot_response = get_response_from_message(
        prompt, model, vectorizer, classes, intents, lemmatizer
    )

    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    with chat_container.chat_message("assistant"):
        chat_container.markdown(bot_response)