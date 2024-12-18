import streamlit as st
from main import ask_question
from main import exit_conversation


# Configuración de la página
st.set_page_config(page_title="Chat con Daniela", page_icon="👩‍💼")

# Título
st.title("💬 Chat con Daniela")
st.caption("Tu asesora en marca personal")

# Inicializar el historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input del usuario
if prompt := st.chat_input("¿En qué puedo ayudarte?"):
    # Mostrar mensaje del usuario
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Guardar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Obtener y mostrar respuesta
    with st.spinner('Pensando...'):
        response = ask_question(prompt, st.session_state.messages)
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # Guardar respuesta del asistente
        st.session_state.messages.append({"role": "assistant", "content": response})

# Botón para nueva conversación
with st.sidebar:
    if st.button("Nueva Conversación"):
        st.session_state.messages = []
        st.rerun()