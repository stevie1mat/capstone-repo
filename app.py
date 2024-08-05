import streamlit as st
import pandas as pd
import gensim
import gensim.corpora as corpora
import spacy
from gensim.models import CoherenceModel
from gensim.models.ldamodel import LdaModel
from sklearn.feature_extraction.text import TfidfVectorizer
import pyLDAvis.gensim
import pyLDAvis
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
import re
import os
from spacy.cli import download

# Ensure the spaCy model is installed
model_name = 'en_core_web_sm'
try:
    nlp = spacy.load(model_name, disable=['parser', 'ner'])
except IOError:
    download(model_name)
    nlp = spacy.load(model_name, disable=['parser', 'ner'])

# Load stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

# Function to preprocess data
def preprocess_data(texts):
    data = [re.sub('\S*@\S*\s?', '', sent) for sent in texts]
    data = [re.sub('\s+', ' ', sent) for sent in data]
    data = [re.sub("\'", " ", sent) for sent in data]
    return data

# Function to tokenize and lemmatize text
def lemmatize_text(texts):
    def sent_to_words(sentences):
        for sentence in sentences:
            yield gensim.utils.simple_preprocess(str(sentence).encode('utf-8'), deacc=True)

    def remove_stopwords(texts):
        return [[word for word in gensim.utils.simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    data_words = list(sent_to_words(texts))
    data_words_nostops = remove_stopwords(data_words)
    data_lemmatized = lemmatization(data_words_nostops, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    return data_lemmatized

# Streamlit App
st.title('Topic Modeling on BBC News Articles')

# Load and preprocess data
directory = st.text_input('Directory Path to Data', 'https://stevenmathew.dev/demo/bbc/')
if directory:
    subdirs = ['business', 'entertainment', 'politics', 'sport', 'tech']
    bbc = []

    for subdir in subdirs:
        dir_path = os.path.join(directory, subdir)
        if not os.path.exists(dir_path):
            st.write(f"Directory {dir_path} does not exist.")
            continue
        for filename in os.listdir(dir_path):
            filepath = os.path.join(dir_path, filename)
            try:
                with open(filepath, 'r') as file:
                    data = file.read()
                escape = ['\n']
                for elem in escape:
                    data = data.replace(elem, ' ')
                dict1 = {'Filename': filename.split('.')[0], 'Contents': data.lower(), 'Category': subdir}
                bbc.append(dict1)
            except Exception as e:
                st.write(f"Error processing file {filepath}: {e}")

    bbc = pd.DataFrame(bbc)
    bbc = bbc.drop_duplicates(subset=['Contents'], keep='first')
    reindexed_data = bbc["Contents"].values.tolist()
    data = preprocess_data(reindexed_data)
    data_lemmatized = lemmatize_text(data)

    # Create Dictionary and Corpus
    id2word = corpora.Dictionary(data_lemmatized)
    corpus = [id2word.doc2bow(text) for text in data_lemmatized]

    # Build LDA model
    lda_model = LdaModel(corpus=corpus,
                         id2word=id2word,
                         num_topics=5,
                         random_state=100,
                         update_every=1,
                         chunksize=100,
                         passes=10,
                         alpha='auto',
                         per_word_topics=True)

    st.write("LDA Topics:")
    topics = lda_model.print_topics()
    for topic in topics:
        st.write(topic)

    # Compute Model Perplexity and Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    st.write(f'Coherence Score: {coherence_lda}')

    # Visualize the topics-keywords
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    st.write(pyLDAvis.display(vis))
