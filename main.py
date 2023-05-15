import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise  import cosine_similarity
import pandas
import wikipedia

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

text = wikipedia.page('Manchester United').content

lemmatizer = WordNetLemmatizer()


def lemma_me(sent):
    sentence_tokens = nltk.word_tokenize(sent.lower())
    pos_tags = nltk.pos_tag(sentence_tokens)

    sentence_lemmas = []
    for token, pos_tag in zip(sentence_tokens, pos_tags):
        if pos_tag[1][0].lower() in ['n', 'v', 'a', 'r']:
            lemma = lemmatizer.lemmatize(token, pos_tag[1][0].lower())
            sentence_lemmas.append(lemma)
    
    return sentence_lemmas
    
    
def process(text, question):   
    sentence_tokens = nltk.sent_tokenize(text)
    sentence_tokens.append(question)

    tv = TfidfVectorizer(tokenizer=lemma_me)
    tf = tv.fit_transform(sentence_tokens)
    tf2 = tf.toarray()

    # df = pandas.DataFrame(tf2, columns=tv.get_feature_names())
    values = cosine_similarity(tf[-1], tf)
    index = values.argsort()[0][-2]
    values_flat = values.flatten()
    values_flat.sort()
    coeff = values_flat[-2]
    if coeff > 0.2:
        return sentence_tokens[index]
        
    
while True:
    question = input('Hi, what do you want to know?\n')    
    output = process(text=text, question=question)
    if output:
        print(output)
    elif question == 'quit':
        break
    else:
        print('I dont know')
    
