import streamlit as st
import os
from dotenv import load_dotenv
from Config.config import authenticate_gcp  # Importa la autenticación de GCP
from Maria import iniciar_conversacion 
from langchain_core.messages import HumanMessage # Importa la función de conversación
from SubAgentes.Podcat import iniciar_conversacion_podcast2

# 🔹 Autenticación con Google Cloud antes de cualquier otra cosa
authenticate_gcp()

# 🔹 Obtener la ruta correcta al archivo .env
current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(os.path.dirname(current_dir), '.env')

# 🔹 Cargar variables de entorno
load_dotenv(env_path)

# Page configuration
st.set_page_config(
    page_title="María - Asistente Virtual",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for Claude-like styling
st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
    }
    .stTextInput {
        border-radius: 10px;
    }
    .stButton>button {
        border-radius: 20px;
        background-color: #6e07f3;
        color: white;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Page header
col1, col2 = st.columns([2,1])
with col1:
    st.title("💬 Conversación con María")
    st.caption("Tu asistente virtual para contenido digital")

# Initialize session state if not exists
if "messages" not in st.session_state:
    st.session_state.messages = []

# Main conversation
iniciar_conversacion_podcast2()
#iniciar_conversacion()