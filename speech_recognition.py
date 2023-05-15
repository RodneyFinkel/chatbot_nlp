from speech_recognition import Recognizer, AudioFile

recognizer = Recognizer()

with AudioFile('chile.wav') as audio_file:
    audio = recognizer(audio_file)
    
text = recognizer.recognize_google(audio) # hits a google api to process this
print(text)