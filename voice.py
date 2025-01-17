import os
import re
import json
import random
import time
import PyPDF2  # PyPDF2 for PDF processing
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.schema import Document  # Added import for Document
import speech_recognition as sr
import pyttsx3
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure NLTK stopwords are downloaded
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    import lib
    lib.download('stopwords')
    lib.download('punkt')
    stop_words = set(stopwords.words('english'))

# Load the JSON file for configuration
with open('config.json', 'r') as file:
    config = json.load(file)

# Set up the GROQ API key
os.environ["GROQ_API_KEY"] = config["GROQ_API_KEY"]

# Initialize the recognizer and TTS engine
recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()

# Create 'uploads' folder if it doesn't exist
uploads_folder = 'uploads'
if not os.path.exists(uploads_folder):
    os.makedirs(uploads_folder)
    print(f"'{uploads_folder}' folder created.")

def preprocess_text(text):
    """
    Cleans and normalizes text for better embeddings.
    - Removes special characters, punctuation, and extra spaces.
    - Converts text to lowercase.
    - Removes stopwords.
    """
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text.lower())
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered_tokens)

def preprocess_document(document_path):
    """Loads and preprocesses a PDF document using PyPDF2."""
    documents = []
    with open(document_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            page_text = page.extract_text()
            page_text = preprocess_text(page_text)  # Preprocess text here
            # Create Document object and append to documents list
            documents.append(Document(page_content=page_text))
    return documents

def set_voice_for_gender(gender="male"):
    voices = tts_engine.getProperty('voices')
    
    if gender == "male":
        # Find the male voice
        male_voice = next((voice for voice in voices if 'male' in voice.name.lower()), None)
        if male_voice:
            tts_engine.setProperty('voice', male_voice.id)
        else:
            print("Male voice not found. Using default voice.")
    elif gender == "female":
        # Find the female voice
        female_voice = next((voice for voice in voices if 'female' in voice.name.lower()), None)
        if female_voice:
            tts_engine.setProperty('voice', female_voice.id)
        else:
            print("Female voice not found. Using default voice.")
    else:
        print("Gender not specified. Using default voice.")

def text_to_speech(text, gender="male"):
    """Converts text to speech using pyttsx3."""
    set_voice_for_gender(gender)  # Set the voice based on gender
    tts_engine.say(text)
    tts_engine.runAndWait()

def listen_for_command():
    """Listens for voice commands and returns them as text."""
    try:
        with sr.Microphone() as source:
            print("Listening for your command...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
            return recognizer.recognize_google(audio).lower()
    except sr.UnknownValueError:
        print("Sorry, I didn't catch that.")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return ""

def integrate_rlhf(prompt, response):
    """Applies RLHF without affecting the response."""
    class SimulatedRewardModel:
        @staticmethod
        def evaluate(prompt, response):
            return len(response) / (len(prompt) + 1)
    
    reward_model = SimulatedRewardModel()
    reward_score = reward_model.evaluate(prompt, response)
    print(f"RLHF Reward Score: {reward_score}")
    return response

def initialize_model():
    """Initializes the QA chain and vector store using PDF files from the 'uploads' folder."""
    global qa_chain, patient_context

    # Get all PDF files from the 'uploads' folder
    pdf_files = [f for f in os.listdir(uploads_folder) if f.endswith('.pdf')]

    if not pdf_files:
        print("No PDF files found in the 'uploads' folder.")
        return

    documents = []
    for pdf_file in pdf_files:
        document_path = os.path.join(uploads_folder, pdf_file)
        documents.extend(preprocess_document(document_path))

    text_splitter = CharacterTextSplitter(
        chunk_size=30000,
        chunk_overlap=400
    )
    
    # Split the document into manageable chunks
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()
    
    persist_directory = "doc_db"
    
    # Create vector store from the documents' chunks
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    retriever = vectordb.as_retriever()

    llm = ChatGroq(
        model="gemma2-9b-it",
        temperature=0.7,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    global patient_context
    patient_context = """
    You are the patient named from document, whose details are provided in the following context:
    [Extracted medical history from the document]
    Respond as if you are the patient, using natural conversation, emotions, and human-like expressions.
    """

def process_query(query, gender="male"):
    """Processes a query using the QA chain and applies RLHF."""
    global qa_chain  # Declare qa_chain as global
    
    if not qa_chain:
        print("Model is not initialized.")
        return

    patient_prompt = f"{patient_context}\n\nDoctor: {query}\nPatient:"
    
    # Process query through QA chain and get response 
    response_data = qa_chain.invoke({"query": patient_prompt})
    result_text = response_data["result"]

    optimized_response = integrate_rlhf(patient_prompt, result_text)
    
    # Use the gender-specific voice to speak the response
    text_to_speech(optimized_response, gender)
    print(f"Patient: {optimized_response}")

# Example usage within a query loop:
if __name__ == "__main__":
    initialize_model()

    while True:
        print("Say 'begin' to start or 'exit' to end the program.")
        command = listen_for_command()
        
        if command == "exit":
            print("Exiting program.")
            break
        elif command == "begin":
            while True:
                print("Listening for your query (Say 'exit' to stop)...")
                query = listen_for_command()
                if query == "exit":
                    print("Stopping interaction. Say 'start' to begin again.")
                    break
                elif query:
                    # Based on query or external context, choose gender ('male' or 'female')
                    # For simplicity, this can be toggled or set dynamically
                    process_query(query, gender="female")  # Example to use female voice
                else:
                    print("No valid query detected.")
