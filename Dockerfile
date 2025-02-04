# Usa una imagen base de Python
FROM python:3.11-slim

# Establece el directorio de trabajo en el contenedor
WORKDIR /App

# Copia los archivos de requisitos y los instala
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del código de la aplicación
COPY . .

# Expone el puerto en el que se ejecutará Streamlit
EXPOSE 8080

# Comando para ejecutar la aplicación Streamlit (esto se anula con el comando en docker-compose.yml)
CMD ["python", "-m", "streamlit", "run", "Agente/main.py", "--server.port", "8080", "--server.enableCORS", "false"]