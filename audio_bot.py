from speech_recognition import Recognizer, AudioFile
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')

recognizer = Recognizer()
analyzer = SentimentIntensityAnalyzer()

with Audiofile('') as audio_file:
    audio = recognizer.record(audio_file)
    
text = recognizer.recognize_google(audio)
print(text)

scores = analyzer.polarity_scores(text)
print(scores)

if scores['compund'] > 0:
    print('positve speech')
else:
    print('negative speech')



