import streamlit as st
import logging
import os
import ffmpeg
import re
import subprocess
from typing import TypedDict
from IPython.display import Audio, Image
from google.cloud import texttospeech
from langchain.schema.document import Document
from langchain_community.retrievers import (
    ArxivRetriever,
    PubMedRetriever,
    WikipediaRetriever,
)
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from google.cloud import storage
import uuid

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

from Config.config import PROJECT_ID
from langgraph.graph import END, StateGraph, START
from Config.config import authenticate_gcp
authenticate_gcp()



memory = MemorySaver()

class AgentState(TypedDict):
    revision_number: int
    max_revisions: int
    search_count: int
    max_searches: int
    task: str
    outline: str
    queries: list
    content: list
    draft: str
    critique: str
    tool_calls: list

PodcastModel = ChatVertexAI(model="gemini-1.5-flash", temperature=0)

@tool
def search_arxiv(query: str) -> list[Document]:
    """Search for relevant publications on arXiv"""
    retriever = ArxivRetriever(load_max_docs=2, get_full_documents=True)
    docs = retriever.invoke(query)
    return docs if docs else ["No results found on arXiv"]

@tool
def search_pubmed(query: str) -> list[Document]:
    """Search for information on PubMed"""
    retriever = PubMedRetriever()
    docs = retriever.invoke(query)
    return docs if docs else ["No results found on PubMed"]

@tool
def search_wikipedia(query: str) -> list[Document]:
    """Search for information on Wikipedia"""
    retriever = WikipediaRetriever()
    docs = retriever.invoke(query)
    return docs if docs else ["No results found on Wikipedia"]

OUTLINE_PROMPT = """You are an expert writer tasked with writing a high level outline of an engaging 2-minute podcast.
Write such an outline for the user provided topic. Give an outline of the podcast along with any
relevant notes or instructions for the sections."""

def podcast_outline_node(state: AgentState):
    messages = [
        SystemMessage(content=OUTLINE_PROMPT),
        HumanMessage(content=state["task"]),
    ]
    response = PodcastModel.invoke(messages)
    return {"outline": response.content}

RESEARCH_PLAN_PROMPT = """You are a researcher tasked with providing information that can
be used when writing the following podcast. Generate one search query consisting of a few
keywords that will be used to gather any relevant information. Do not output any information
other than the query consisting of a few words.

These were the past queries, do not repeat keywords from past queries in your newly generated query:
---
{queries}"""

def research_plan_node(state: AgentState):
    messages = [
        SystemMessage(content=RESEARCH_PLAN_PROMPT.format(queries="\n".join(state["queries"]))),
        HumanMessage(content=state["task"]),
    ]
    response = PodcastModel.invoke(messages)
    queries = state["queries"]
    if queries:
        queries.append(response.content)
    else:
        queries = [response.content]
    return {"queries": queries}

RESEARCH_TASK_PROMPT = """Use the available search tools and search queries to find information
relevant to the podcast. Try searching different sources to obtain different articles. Try using
different search tools than what was used previously so that you can obtain a broader range of
information.

These are the previous tool calls, so you can choose a different tool:
---
{tool_calls}
---
These are the previous search results, so you can aim for different sources and content:
---
{content}"""

def research_agent_node(state: AgentState):
    tool_calls = state["tool_calls"]
    content = state["content"]
    queries = state["queries"]
    
    if not queries:
        raise ValueError("No queries available for research.")
    
    query = queries[-1]
    messages = [
        SystemMessage(content=RESEARCH_TASK_PROMPT.format(tool_calls=tool_calls, content=content)),
        HumanMessage(content=query),
    ]
    tools = [search_arxiv, search_pubmed, search_wikipedia]
    model_with_tools = PodcastModel.bind_tools(tools)
    response_tool_calls = model_with_tools.invoke(messages)
    tool_calls.append(response_tool_calls) if tool_calls else [response_tool_calls]
    tool_node = ToolNode(tools)
    response = tool_node.invoke({"messages": [response_tool_calls]})
    for message in response.get("messages", []):
        if isinstance(message, ToolMessage):
            content.insert(0, message.content)
    return {"content": content, "tool_calls": tool_calls, "search_count": state["search_count"] + 1}

def should_continue_tools(state: AgentState):
    return "generate_script" if state["search_count"] > state["max_searches"] else "research_plan"

def clean_agent_result(data):
    agent_result = str(data)
    agent_result = re.sub(r"[^\x00-\x7F]+", " ", agent_result)
    agent_result = re.sub(r"\\\\n", "\n", agent_result)
    agent_result = re.sub(r"\\n", "", agent_result)
    agent_result = re.sub(r"\\'", "'", agent_result)
    return agent_result

WRITER_PROMPT = """
You are a writing assistant tasked with writing engaging 2-minute podcast scripts.

- Generate the best podcast script possible for the user's request and the initial outline.
- The script MUST strictly alternate lines between the two hosts, separating each host's line with a newline.
- Add an intro phrase and outro phrase to start and end the podcast, and use a fun, random name for the podcast show.
- Given a critique, respond with a revised version of your previous script.
- Include lively back-and-forth chatter, reflections, and expressions of amazement between the hosts.
- Cite at least THREE pieces of research throughout the script, choosing the most relevant research for each point.
- DO NOT include ANY of the following:
    - Speaker labels (e.g., "Host 1:", "Host 2:")
    - Sound effect descriptions (e.g., "[Sound of waves]")
    - Formatting instructions (e.g., "(Emphasis)", "[Music fades in]")
    - Any other non-dialogue text.
- Use this format for citations, including the month and year if available:
    "In [Month, Year], [Organization] found that..."
    "Research from [Organization] in [Month, Year] showed that..."
    "Back in [Month, Year], a study by [Organization] suggested that..."
---
Utilize all of the following search results and context as needed:
{content}
---
If this is a revision, the critique will be provided below:
{critique}"""

def generate_script_node(state: AgentState):
    messages = [
        SystemMessage(content=WRITER_PROMPT.format(content=state["content"], critique=state.get("critique", ""))),
        HumanMessage(content=f"{state['task']}\n\nHere is my outline:\n\n{state['outline']}"),
    ]
    response = PodcastModel.invoke(messages)
    cleaned_script = clean_agent_result(response.content)
    state["generate_script"] = {"draft": cleaned_script}
    state["search_count"] = 0
    state["revision_number"] = state.get("revision_number", 1) + 1
    return state

CRITIQUE_PROMPT = """You are a producer grading a podcast script.
Generate critique and recommendations for the user's submission.
Provide detailed recommendations, including requests for conciceness, depth, style, etc."""

def perform_critique_node(state: AgentState):
    messages = [
        SystemMessage(content=CRITIQUE_PROMPT),
        HumanMessage(content=state["draft"]),
    ]
    response = PodcastModel.invoke(messages)
    return {"critique": response.content}

RESEARCH_CRITIQUE_PROMPT = """You are a writing assistant tasked with providing information that can
be used when making any requested revisions (as outlined below).
Generate one search query consisting of a few keywords that will be used to gather any relevant
information. Do not output any information other than the query consisting of a few words.

---

These were the past queries, so you can vary the query that you generate:

{queries}"""

def research_critique_node(state: AgentState):
    messages = [
        SystemMessage(content=RESEARCH_CRITIQUE_PROMPT.format(queries=state["queries"])),
        HumanMessage(content=state["critique"]),
    ]
    response = PodcastModel.invoke(messages)
    queries = state.get("queries", [])
    queries.append(response.content) if queries else [response.content]
    return {"queries": queries}

def should_continue(state: AgentState):
    return END if state["revision_number"] > state["max_revisions"] else "perform_critique"

def Agente():
    return "¡Bienvenido/a! Estoy aquí para ayudarte. ¿En qué puedo asistirte hoy?"

def Solicitud_Tema():
    if "user_topic" not in st.session_state:
        st.session_state.user_topic = ""

    user_topic = st.text_input("Por favor, ingrese el tema para el podcast:", st.session_state.user_topic).strip()

    if st.button("Confirmar tema"):
        if not user_topic:
            st.warning("No se ingresó ningún tema. Usando tema predeterminado: Tecnología y su impacto en la educación")
            user_topic = "Tecnología y su impacto en la educación"
        else:
            st.write(f"Tema seleccionado: {user_topic}")
        st.session_state.user_topic = user_topic

    if not st.session_state.user_topic:
        st.stop()

    initial_state = {
        "task": st.session_state.user_topic,
        "revision_number": 1,
        "max_revisions": 2,
        "search_count": 0,
        "max_searches": 2,
        "content": [],
        "queries": [],
        "tool_calls": [],
        "outline": "",
        "draft": "",
        "critique": "",
    }
    return initial_state

def confirm_script_node(state: AgentState):
    podcast_script = state["generate_script"]["draft"]
    parsed_script = [text for text in (line.strip() for line in podcast_script.splitlines()) if text]
    st.write("Este es el borrador del guion generado:\n")
    for line in parsed_script:
        st.write(f"- {line}")
    state["confirm_script"] = {"parsed_script": parsed_script}
    return state

def should_continue_confirm(state: AgentState):
    confirmation = st.radio("¿Estás satisfecho con este guion?", ("Sí", "No"), index=None, key="confirmation_radio")
    if confirmation == "Sí":
        st.write("¡Guion confirmado! Procediendo a la generación del audio...")
        state["next_node"] = "generate_audio"
        return {"status": "confirmed"}
    elif confirmation == "No":
        st.write("Se requiere una revisión adicional. Regresando al nodo de planificación.")
        state["next_node"] = "podcast_outline"
        return {"status": "needs_revision"}
    st.stop()

def generate_audio(state: AgentState, client):
    parsed_script = state["confirm_script"]["parsed_script"]
    audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)
    audio_files = []
    for count, line in enumerate(parsed_script):
        synthesis_input = texttospeech.SynthesisInput(text=line)
        voice_name = "en-US-Journey-O" if count % 2 == 0 else "en-US-Journey-D"
        voice = texttospeech.VoiceSelectionParams(language_code="en-US", name=voice_name)

        response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        filename = f"part-{str(count)}.mp3"
        audio_files.append(filename)
        with open(filename, "wb") as out:
            out.write(response.audio_content)
            st.write(f"Audio content written to file {filename}")

    # Combine audio files using ffmpeg
    unique_id = str(uuid.uuid4())
    podcast_filename = f"gemini-podcast-{unique_id}.mp3"
    with open("filelist.txt", "w") as f:
        for file in audio_files:
            f.write(f"file '{file}'\n")

    subprocess.run(["ffmpeg", "-f", "concat", "-safe", "0", "-i", "filelist.txt", "-c", "copy", podcast_filename])
    st.write(f"Podcast content written to file {podcast_filename}")

    # Upload to Google Cloud Storage
    bucket_name = "mariaai"  # Replace with your GCS bucket name
    destination_blob_name = f"podcasts/{podcast_filename}"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(podcast_filename)

    st.write(f"Podcast uploaded to GCS: gs://{bucket_name}/{destination_blob_name}")

    # Clean up temporary files
    os.remove("filelist.txt")
    for file in audio_files:
        os.remove(file)

    state["podcast_filename"] = podcast_filename
    state["gcs_uri"] = f"gs://{bucket_name}/{destination_blob_name}"
    return state

def desplegado(state: AgentState):
    try:
        if "podcast_filename" not in state:
            raise ValueError("No se encontró el archivo de audio en el estado.")
        podcast_filename = state["podcast_filename"]
        abs_path = os.path.abspath(podcast_filename)
        st.write("\n" + "="*50)
        st.write("🎙️ ¡Escuche su podcast generado por IA!")
        st.write("Esta celda reproduce el podcast final generado por el agente de IA.")
        st.write("\n💡 El objeto Audio de IPython.display se usa para incrustar el reproductor de audio.")
        st.write("▶️ El podcast comenzará a reproducirse automáticamente.")
        st.write("\n🎵 ¡Disfruta de tu podcast creado por IA! 🎧")
        st.write(f"📁 Archivo generado: {abs_path}")
        st.write("="*50 + "\n")
        st.audio(podcast_filename)
        return {"status": "success", "message": "El podcast ha sido generado y se está reproduciendo.", "podcast_path": abs_path}
    except Exception as e:
        error_message = f"Error al desplegar el audio: {str(e)}"
        st.write("\n❌", error_message)
        return {"status": "error", "error_message": error_message}

podcast_workflow = StateGraph(AgentState)
podcast_workflow.add_node("Agente", Agente)
podcast_workflow.add_node("Input", Solicitud_Tema)
podcast_workflow.add_node("podcast_outline", podcast_outline_node)
podcast_workflow.add_node("research_plan", research_plan_node)
podcast_workflow.add_node("research_agent", research_agent_node)
podcast_workflow.add_node("generate_script", generate_script_node)
podcast_workflow.add_node("confirm_script", confirm_script_node)
podcast_workflow.add_node("perform_critique", perform_critique_node)
podcast_workflow.add_node("research_critique", research_critique_node)
podcast_workflow.add_node("generate_audio", generate_audio)
podcast_workflow.add_node("Desplegado", desplegado)
podcast_workflow.set_entry_point("podcast_outline")
podcast_workflow.add_edge(START, "Agente")
podcast_workflow.add_edge("Agente", "Input")
podcast_workflow.add_edge("Input", "podcast_outline")
podcast_workflow.add_edge("podcast_outline", "research_plan")
podcast_workflow.add_edge("research_plan", "research_agent")
podcast_workflow.add_edge("research_agent", "generate_script")
podcast_workflow.add_edge("generate_script", "confirm_script")
podcast_workflow.add_edge("confirm_script", "generate_audio")
podcast_workflow.add_edge("perform_critique", "research_critique")
podcast_workflow.add_edge("research_critique", "research_agent")
podcast_workflow.add_conditional_edges("research_agent", should_continue, {"generate_script": "generate_script", "research_plan": "research_plan"})
podcast_workflow.add_conditional_edges("generate_script", should_continue, {"perform_critique": "perform_critique"})
podcast_workflow.add_conditional_edges("confirm_script", should_continue_confirm, {"generate_audio": "generate_audio", "podcast_outline": "podcast_outline"})
podcast_workflow.add_edge("generate_audio", "Desplegado")
podcast_workflow.add_edge("Desplegado", END)
graph1 = podcast_workflow.compile(checkpointer=memory)

def iniciar_conversacion_podcast2():
    try:
        st.write("¡Hola! Soy María, tu asistente para la creación de podcasts. 😊")
        state = Solicitud_Tema()
        st.write("María: Iniciando el flujo de trabajo del podcast... Por favor, espera.")
        current_node = "podcast_outline"
        while current_node is not None:
            st.write(f"María: Ejecutando nodo '{current_node}'...")
            try:
                if current_node == "podcast_outline":
                    outline_result = podcast_outline_node(state)
                    state.update(outline_result)
                    st.write("María: Outline generado.")
                elif current_node == "research_plan":
                    if "queries" not in state:
                        state["queries"] = []
                    plan_result = research_plan_node(state)
                    state.update(plan_result)
                    st.write("María: Plan de investigación generado.")
                elif current_node == "research_agent":
                    if "tool_calls" not in state:
                        state["tool_calls"] = []
                    if "content" not in state:
                        state["content"] = []
                    agent_result = research_agent_node(state)
                    state.update(agent_result)
                    if should_continue_tools(state) == "generate_script":
                        current_node = "generate_script"
                        continue
                elif current_node == "generate_script":
                    script_result = generate_script_node(state)
                    state.update(script_result)
                    current_node = "confirm_script"
                    continue
                elif current_node == "confirm_script":
                    confirmation_result = confirm_script_node(state)
                    state.update(confirmation_result)
                    current_node = "should_continue_confirm"
                    continue
                elif current_node == "should_continue_confirm":
                    confirm_result = should_continue_confirm(state)
                    state.update(confirm_result)
                    current_node = state["next_node"]
                    continue
                elif current_node == "generate_audio":
                    client = texttospeech.TextToSpeechClient(client_options={"api_endpoint": "texttospeech.googleapis.com", "quota_project_id": PROJECT_ID})
                    audio_result = generate_audio(state, client)
                    state.update(audio_result)
                    current_node = "Desplegado"
                    continue
                elif current_node == "Desplegado":
                    if "podcast_filename" in state:
                        desplegado_result = desplegado(state)
                        state.update(desplegado_result)
                    else:
                        raise ValueError("Error: No se encontró el archivo de audio para desplegar.")
                    break
                next_nodes = [edge[1] for edge in podcast_workflow.edges if edge[0] == current_node]
                if not next_nodes:
                    break
                current_node = next_nodes[0]
                if current_node == END:
                    break
            except Exception as node_error:
                st.write(f"Error en el nodo {current_node}: {str(node_error)}")
                raise
        if state.get("audio_status") == "completed":
            st.write(f"María: ¡El podcast se ha generado exitosamente! Archivo: {state.get('podcast_filename')}")
        else:
            st.write("María: El flujo ha terminado.")
    except Exception as e:
        st.write(f"María: Ocurrió un error al ejecutar el flujo de trabajo: {e}")
        st.write(f"Detalles del error: {str(e)}")






