🐾 PetBot: Seu Assistente Virtual para Cães e Gatos
Seja bem-vindo ao repositório do PetBot! Este projeto foi desenvolvido como parte da disciplina de Inteligência Artificial, com o objetivo de criar um assistente ára auxiliar e tirar dúvidas comuns de tutores de pets de forma rápida e acessível.

🔗 Clique aqui para testar o PetBot ao vivo no Streamlit!

💡 A Ideia do Projeto
A gente sabe que cuidar de um pet gera muitas dúvidas: "pode dar tal comida?", "quando vacinar?", "como conseguir castração gratuita?". A ideia do PetBot é centralizar essas respostas em um chat simples, que entende o que você pergunta, sem que você precise usar termos técnicos exatos.

O diferencial aqui é que ele não é apenas um sistema de regras (if/else). Ele usa uma Rede Neural para tentar "entender" a intenção da sua frase, mesmo que você escreva de um jeito diferente do previsto.

🛠️ Como ele foi construído?
Para fazer o bot funcionar de verdade, passei por algumas etapas de evolução técnica:

Processamento de Texto (NLP): Usei a biblioteca NLTK para limpar o texto do usuário (tirar pontuação, colocar em minúsculas, lematizar).

A "Tradução" (TF-IDF): No início, tentei usar Word2Vec, mas percebi que para este escopo, o TF-IDF (do Scikit-Learn) oferecia uma precisão muito maior (chegando a 100% nos testes locais) para diferenciar tópicos parecidos, como "vacina" e "doença".

O Cérebro (Deep Learning): A classificação é feita por uma rede neural densa construída com TensorFlow/Keras. Ela recebe a frase "matematizada" e decide qual é a melhor resposta no banco de dados.

Interface Web: Para tirar o bot do terminal e colocar na web, usei o Streamlit, que é rápido e eficiente para demos de Data Science.

📚 O que ele sabe responder?
Treinei o PetBot para responder sobre diversos tópicos, incluindo:

🐶 Passeios: Diferencia as necessidades de cães e gatos.

💉 Saúde: Vacinas essenciais, vermífugos e cuidados com dentes.

🏥 Utilidade Pública: Lista endereços e regras para castração gratuita em SP (Programa SP 156) e hospitais públicos.

⚠️ Alertas: Alimentos tóxicos e perigos de remédios humanos.

🐱 Comportamento: Xixi no lugar errado, arranhadores, latidos e miados excessivos.
