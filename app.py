# =====================================================================
# Importing necessary libraries
# =====================================================================
import os
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_community.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import Chroma
# from langchain.prompts import PromptTemplate
from langchain.prompts import load_prompt
from streamlit import session_state as ss
from pymongo import MongoClient
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from pymongo import MongoClient
import uuid
import json
import time
import datetime
import random





# Add a theme customization option
st.set_page_config(
    page_title="Olivier RAYMOND - Virtual Assistant",
    page_icon=":robot:",
)


# Function to check if a string is valid JSON
def is_valid_json(data):
    try:
        json.loads(data)
        return True
    except json.JSONDecodeError:
        return False

# Retrieve MongoDB password from environment variables or Streamlit secrets
if "mongodB_pass" in os.environ:
    mongodB_pass = os.getenv("mongodB_pass")
else:
    mongodB_pass = st.secrets["mongodB_pass"]

# Define the MongoDB URI with the password - Store conversations for deeper analysis
uri = "//bouabideoumayma:" + mongodB_pass + "@cluster0.wxn9x8j.mongodb.net/"
try:
    client = MongoClient(uri)
    client.admin.command("ping")  # Check if the server is reachable
    print("Connection successful!")
except Exception as e:
    print(f"Connection failed: {e}")
    
# Function to initialize a MongoDB connection (cached for efficiency)
@st.cache_resource
def init_connection():
    return MongoClient(uri, server_api=ServerApi('1'))

# Initialize the MongoDB client
client = init_connection()

# Access specific MongoDB collections for storing conversations
db = client['conversations_db']
conversations_collection = db['conversations']

# Retrieve OpenAI API key from environment variables or Streamlit secrets
if "OPENAI_API_KEY" in os.environ:
    openai_api_key = os.getenv("OPENAI_API_KEY")
else:
    openai_api_key = st.secrets["OPENAI_API_KEY"]

# Create Streamlit title and provide additional information about the bot
st.title("Olivier RAYMOND - Virtual Assistant")

with st.expander("⚠️ Informations importantes"):
    st.write(
        """
        Bienvenue dans l'assistant virtuel d'Olivier RAYMOND !  
        Cet outil interactif est conçu pour vous fournir des informations détaillées sur le parcours professionnel, les compétences et les projets d'Olivier.  
        L'assistant utilise un modèle de langage avancé pour générer des réponses, mais veuillez noter que les réponses peuvent être automatiques et ne reflètent pas toujours toutes les informations disponibles.

        **Important :**  
        - Ce chatbot est encore en développement et est destiné à fournir une assistance sur le parcours, les études et les aspirations d'Olivier RAYMOND.
        - Les réponses sont basées uniquement sur les données fournies et peuvent ne pas être exhaustives.
        - Si une question dépasse le cadre des informations disponibles, l'assistant fournira une réponse appropriée.

        Nous vous remercions pour votre patience et vos retours pour améliorer l'expérience !
        """
    )




# Define the path for files and templates
path = os.path.dirname(__file__)

# Load prompt template from a JSON file to query openai
prompt_template = path + "/templates/template.json"
prompt = load_prompt(prompt_template)


# Define the FAISS index path - Loading embedings
faiss_index = path + "/faiss_index"

# Define paths for the data sources
data_source = path + "/data/about_me.csv"
pdf_source = path + "/data/resume.pdf"


# Function to store conversation data into MongoDB
def store_conversation(conversation_id, user_message, bot_message, answered):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {
        "conversation_id": conversation_id,
        "timestamp": timestamp,
        "user_message": user_message,
        "bot_message": bot_message,
        "answered": answered
    }
    conversations_collection.insert_one(data)

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Load or create a FAISS vector store
if os.path.exists(faiss_index):
    # Load existing FAISS index
    vectors = FAISS.load_local(faiss_index, embeddings, allow_dangerous_deserialization=True)
else:
    # Create a new FAISS index from the data sources
    if data_source:
        # Load data from PDF and CSV sources
        pdf_loader = PyPDFLoader(pdf_source)
        pdf_data = pdf_loader.load_and_split()
        csv_loader = CSVLoader(file_path=data_source, encoding="utf-8")
        csv_data = csv_loader.load()
        data = pdf_data + csv_data
        vectors = FAISS.from_documents(data, embeddings)
        vectors.save_local("faiss_index")

# Create a retriever from the FAISS vector store
retriever = vectors.as_retriever(search_type="similarity",
                                 search_kwargs={"k": 10, "include_metadata": True, "score_threshold": 0.6})

# Define the conversational chain using LangChain
chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(temperature=0.3, model_name='gpt-3.5-turbo', openai_api_key=openai_api_key),
    retriever=retriever, return_source_documents=True, verbose=True, chain_type="stuff",
    max_tokens_limit=8194, combine_docs_chain_kwargs={"prompt": prompt})

# Function to select a random spinner message
def get_spinner_message():
    messages = [
        "Je traite votre question...",
        "Un instant, je m'en occupe...",
        "Je réfléchis à votre demande...",
        "Laissez-moi analyser cela pour vous...",
        "En cours de traitement, merci de patienter...",
        "Je recherche les meilleures informations pour vous...",
        "Je prépare une réponse détaillée...",
        "Traitement de votre question en cours...",
        "Veuillez patienter, je compile les données...",
        "J'examine votre question avec attention..."
    ]
    return random.choice(messages)

# Function to handle conversational interactions
def conversational_chat(query):
    with st.spinner(get_spinner_message()):
        # Be conversational and ask a follow up questions to keep the conversation going"
        result = chain({
            "system": "Tu es l'assistant personnal (chatbot) d'Olivier RAYMOND, une ressource interactive complète pour explorer le parcours, les compétences et l'expertise d'Olivier RAYMOND. Soit poli et fournit des réponses basées uniquement sur les informations disponibles dans le contexte. Ne t'appuis pas sur tes connaissances externes.",
            "question": query,
            "chat_history": st.session_state['history']
        })

    # Validate and parse the JSON response
    if (is_valid_json(result["answer"])):
        data = json.loads(result["answer"])
    else:
        # Fallback response if JSON parsing fails
        data = json.loads(
            '{"answered":"false", "response":"Hmm... Il semble qu\'il y ait un problème technique. Je rencontre des difficultés pour répondre à votre question. Veuillez réessayer ou poser une autre question concernant le parcours professionnel ou les qualifications d\'Olivier RAYMOND. Merci pour votre compréhension.", "questions":["Quel est l\'expérience professionnelle d\'Olivier ?","Sur quels projets Olivier a-t-il travaillé ?","Quels sont les objectifs de carrière d\'Olivier ?"]}')

    # Access data fields
    answered = data.get("answered")
    response = data.get("response")
    questions = data.get("questions")

    full_response = "--"

    # Update conversation history and generate response
    st.session_state['history'].append((query, response))

    if ('Je ne réponds qu\'à des questions' in response) or (response == ""):
        full_response = """
            Je ne suis pas en mesure de répondre à votre question. Mes réponses se limitent aux informations relatives au parcours et aux compétences professionnelles d'Olivier RAYMOND. Pour plus de précisions, je vous invite à consulter son profil LinkedIn : [Linkedin](https://www.linkedin.com/in/olivier-raymond/).
           
            \n - Quel est le parcours académique d'Olivier RAYMOND ?
            \n - Pouvez-vous lister l'expérience professionnelle d'Olivier RAYMOND ?
            \n - Quelles compétences techniques Olivier RAYMOND possède-t-il ?
            """
        store_conversation(st.session_state["uuid"], query, full_response, answered)

    else:
        markdown_list = ""
        for item in questions:
            markdown_list += f"- {item}\n"
        full_response = response + "\n\nVoici des exemples de questions supplémentaires que vous pouvez poser :\n\n" + markdown_list
        store_conversation(st.session_state["uuid"], query, full_response, answered)
    return (full_response)

# Initialize session states for managing user interactions
if "uuid" not in st.session_state:
    st.session_state["uuid"] = str(uuid.uuid4())
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"
if "messages" not in st.session_state:
    st.session_state.messages = []

    # Display the initial assistant message
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        welcome_message = """
            Bienvenue. Je suis **l'assistant virtuel d'Olivier RAYMOND**. 
            Je suis conçu pour fournir des réponses précises et concises sur son parcours et ses compétences.

            - Quelle est la formation d'Olivier RAYMOND ?
            - Pouvez-vous détailler son expérience professionnelle ?
            - Quelles sont ses principales compétences techniques ?

            Je suis là pour vous aider. Que souhaitez-vous savoir ?
            """
        message_placeholder.markdown(welcome_message)

if 'history' not in st.session_state:
    st.session_state['history'] = []

# Display past chat messages from the session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input and generate assistant responses
if prompt := st.chat_input("Demandez-moi des informations sur Olivier RAYMOND"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        user_input = prompt
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        full_response = conversational_chat(user_input)
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})


