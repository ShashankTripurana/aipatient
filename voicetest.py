import pyttsx3

# Initialize the TTS engine
tts_engine = pyttsx3.init()

def set_voice_for_gender(gender="male"):
    voices = tts_engine.getProperty('voices')
    
    if gender == "male":
        # Find the male voice
        male_voice = next((voice for voice in voices if 'male' in voice.name.lower()), None)
        if male_voice:
            tts_engine.setProperty('voice', male_voice.id)
            print("Male voice set.")
        else:
            print("Male voice not found. Using default voice.")
    elif gender == "female":
        # Find the female voice
        female_voice = next((voice for voice in voices if 'female' in voice.name.lower()), None)
        if female_voice:
            tts_engine.setProperty('voice', female_voice.id)
            print("Female voice set.")
        else:
            print("Female voice not found. Using default voice.")
    else:
        print("Gender not specified. Using default voice.")

def text_to_speech(text, gender="male"):
    """Converts text to speech using pyttsx3."""
    set_voice_for_gender(gender)  # Set the voice based on gender
    tts_engine.say(text)
    tts_engine.runAndWait()

# Test function
if __name__ == "__main__":
    text = "Hello! This is a test for the TTS system."
    text_to_speech(text, gender="female")  # Change to "male" or "female" for different voices
