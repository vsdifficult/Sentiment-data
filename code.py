from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
import matplotlib.pyplot as plt
import nltk, spacy

from sklearn.linear_model import LogisticRegression, LinearRegression 
from sklearn.model_selection import train_test_split

global analyzer, npl
analyzer = SentimentIntensityAnalyzer() 

def tokenizeee(text): 
    text_with_str = str(text)  
    stop_words_ru = stopwords.words('russian') 
    stop_words_eng = stopwords.words('english')
    tokens = word_tokenize(text_with_str)
    # strip_whitespace = [text_with_str.strip() for string in text_with_str]
    # remove_stops = [text_with_str.replace() for text_with_str in strip_whitespace ]
    return [word for word in tokens if word not in stop_words_ru]

def analys_( file_path: str): 
            with open(file_path, 'r', encoding='utf-8') as file: 
                wprds = file.readlines()
                sentence = str(tokenizeee(wprds))
            vs = analyzer.polarity_scores(sentence)
            print(f"Text: {sentence}")
            print(f"Positive: {vs['pos']}")
            print(f"Negative: {vs['neg']}") 
            print(f"Neutral: {vs['neu']}") 
            return {"Positive": float(vs['pos']), "Negative": float(vs['neg'])}
