import os
import re
import json
import time
import speech_recognition as sr
import pyttsx3
from flask import Flask, render_template, request, jsonify
from threading import Thread
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.schema import Document

app = Flask(__name__)

# Initialize recognizer and TTS engine
recognizer = sr.Recognizer()
tts_engine = pyttsx3.init()

# Ensure NLTK stopwords are downloaded
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    import lib
    lib.download('stopwords')
    lib.download('punkt')
    stop_words = set(stopwords.words('english'))

# Load the configuration from JSON
with open('config.json', 'r') as file:
    config = json.load(file)

# Initialize model and other global variables
qa_chain = None
patient_context = None

# Create 'uploads' folder if it doesn't exist
uploads_folder = 'uploads'
if not os.path.exists(uploads_folder):
    os.makedirs(uploads_folder)

# Function for text-to-speech
def set_voice_for_gender(gender="male"):
    voices = tts_engine.getProperty('voices')
    if gender == "male":
        male_voice = next((voice for voice in voices if 'male' in voice.name.lower()), None)
        if male_voice:
            tts_engine.setProperty('voice', male_voice.id)
    elif gender == "female":
        female_voice = next((voice for voice in voices if 'female' in voice.name.lower()), None)
        if female_voice:
            tts_engine.setProperty('voice', female_voice.id)

def text_to_speech(text, gender="male"):
    set_voice_for_gender(gender)
    tts_engine.say(text)
    tts_engine.runAndWait()

# Listen for commands
def listen_for_command():
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
            return recognizer.recognize_google(audio).lower()
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        return ""

def process_query(query, gender="male"):
    global qa_chain
    if not qa_chain:
        return "Model is not initialized."

    patient_prompt = f"{patient_context}\n\nDoctor: {query}\nPatient:"
    response_data = qa_chain.invoke({"query": patient_prompt})
    result_text = response_data["result"]
    text_to_speech(result_text, gender)
    return result_text

# Initialize the model and embeddings
def initialize_model():
    global qa_chain, patient_context
    # Your model initialization logic here...

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_interaction', methods=['POST'])
def start_interaction():
    command = listen_for_command()
    if command == "begin":
        return jsonify({"message": "Starting interaction... please ask your query!"})
    elif command == "exit":
        return jsonify({"message": "Exiting interaction."})
    return jsonify({"message": "No valid command detected."})

@app.route('/process_query', methods=['POST'])
def process_query_route():
    query = request.json.get('query')
    gender = request.json.get('gender', 'male')
    response = process_query(query, gender)
    return jsonify({"response": response})

if __name__ == "__main__":
    initialize_model()
    app.run(debug=True)
