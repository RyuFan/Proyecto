import os
from dotenv import load_dotenv
import vertexai
from google.oauth2 import service_account

# Cargar variables de entorno
def load_env():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))  # Ir a la raíz del proyecto
    env_path = os.path.join(base_dir, '.env')  
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        print(f"⚠️ Archivo .env no encontrado en: {env_path}")

load_env()

# Definir variables de entorno
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", 'lina-ai-447416')
LOCATION = os.getenv("GOOGLE_CLOUD_REGION", 'us-central1')

print(f"Inicializando Vertex AI con:")
print(f"  Project ID: {PROJECT_ID}")
print(f"  Location: {LOCATION}")


# Función de autenticación
def authenticate_gcp():
    try:
        # Obtener credenciales desde variable de entorno
        credentials_json = os.getenv('GCP_SA_KEY')
        if not credentials_json:
            raise ValueError("❌ No se encontró la variable de entorno GCP_SA_KEY")

        # Crear un archivo temporal con las credenciales
        import json
        import tempfile
        
        credentials_dict = json.loads(credentials_json)
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            json.dump(credentials_dict, temp_file)
            temp_credentials_path = temp_file.name

        credentials = service_account.Credentials.from_service_account_file(
            temp_credentials_path,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )

        # Limpiar archivo temporal
        os.unlink(temp_credentials_path)
        return credentials

    except Exception as e:
        raise Exception(f"❌ Error al autenticar con GCP: {str(e)}")

# Llamar a la autenticación
authenticate_gcp()


# Llamar a la autenticación antes de usar cualquier servicio
#authenticate_gcp()

#-------------------------------------------------



