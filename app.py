import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout
from sklearn.model_selection import train_test_split
import os

# Load your keyword data
keywords_df = pd.read_csv('keyword.csv')

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the "text" column to TF-IDF matrix
tfidf_matrix = tfidf_vectorizer.fit_transform(keywords_df['Text'].fillna(''))

# Example function for keyword matching with multiple results
def match_top_pages(user_prompt, keywords_df, tfidf_matrix, top_n=5):
    processed_prompt = user_prompt.lower()  # Convert to lowercase
    keywords = processed_prompt.split()  # Simple split by whitespace
    
    relevance_scores = {}
    
    for keyword in keywords:
        matches = keywords_df[keywords_df.apply(lambda x: keyword in x.values, axis=1)]
        
        for index, row in matches.iterrows():
            relevance_score = row[['Score1', 'Score2', 'Score3', 'Score4', 'Score5']].sum()
            page_name = row['Title']
            if page_name in relevance_scores:
                relevance_scores[page_name] += relevance_score
            else:
                relevance_scores[page_name] = relevance_score
    
    if not relevance_scores:
        user_tfidf = tfidf_vectorizer.transform([user_prompt])
        cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
        top_indices = cosine_similarities.argsort()[-top_n:][::-1]
        
        for index in top_indices:
            page_name = keywords_df.iloc[index]['Title']
            relevance_score = cosine_similarities[index]
            relevance_scores[page_name] = relevance_score
    
    sorted_pages = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)
    top_pages = sorted_pages[:top_n]
    
    return top_pages

# Load the user path data
user_path = pd.read_csv('user_paths.csv')

expanded_data = []
for _, row in user_path.iterrows():
    study_path = eval(row['page_name'])
    timestamp = eval(row['time_spent'])
    for i in range(len(study_path) - 1):
        current_page = study_path[i]
        next_page = study_path[i + 1]
        time_spent = timestamp[i]
        expanded_data.append([current_page, next_page, time_spent])

expanded_df = pd.DataFrame(expanded_data, columns=['current_page', 'next_page', 'time_spent'])
max_time_spent = expanded_df['time_spent'].max()
expanded_df['rating'] = expanded_df['time_spent'] / max_time_spent

interactions = expanded_df[['current_page', 'next_page', 'rating']].copy()

label_encoder = LabelEncoder()
all_pages = pd.concat([interactions['current_page'], interactions['next_page']]).unique()
label_encoder.fit(all_pages)

interactions['current_page_encoded'] = label_encoder.transform(interactions['current_page'])
interactions['next_page_encoded'] = label_encoder.transform(interactions['next_page'])

num_pages = len(all_pages)
embedding_dim = 50

input_current_page = Input(shape=(1,))
input_next_page = Input(shape=(1,))

embedding_layer = Embedding(num_pages, embedding_dim)

flattened_current_page = Flatten()(embedding_layer(input_current_page))
flattened_next_page = Flatten()(embedding_layer(input_next_page))

concatenated = Concatenate()([flattened_current_page, flattened_next_page])

dense_1 = Dense(128, activation='relu')(concatenated)
dropout = Dropout(0.5)(dense_1)
dense_2 = Dense(64, activation='relu')(dropout)
output_layer = Dense(1, activation='sigmoid')(dense_2)

ncf_model = Model(inputs=[input_current_page, input_next_page], outputs=output_layer)

model_path = 'https://github.com/stevie1mat/capstone-repo/blob/main/model.keras'

if os.path.exists(model_path):
    try:
        ncf_model = load_model(model_path)
        st.write("Loaded pre-trained model successfully!")
    except Exception as e:
        st.write("Error loading model:", e)
else:
    # Split data
    train_data, test_data = train_test_split(interactions[['current_page_encoded', 'next_page_encoded', 'rating']],
                                             test_size=0.2,
                                             random_state=42)

    ncf_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    ncf_model.fit(
        [train_data['current_page_encoded'], train_data['next_page_encoded']],
        train_data['rating'],
        batch_size=64,
        epochs=10,
        validation_data=(
            [test_data['current_page_encoded'], test_data['next_page_encoded']],
            test_data['rating']
        )
    )
    
    # Save the model after training
    ncf_model.save(model_path)
    st.write("Model trained and saved successfully!")

# Streamlit App
st.title('Learning Path Recommender System')

user_prompt = st.text_input("Enter your prompt:")

if st.button("Get Recommendations"):
    top_n = 3
    path_length = 3
    top_pages = match_top_pages(user_prompt, keywords_df, tfidf_matrix, top_n=top_n)

    for page_name, _ in top_pages:
        prompt_page_encoded = label_encoder.transform([page_name])[0]
        predicted_pages = ncf_model.predict([prompt_page_encoded * np.ones(num_pages), np.arange(num_pages)])
        confidence_scores = predicted_pages.flatten()
        top_predicted_indices = np.argsort(confidence_scores)[::-1][:path_length]
        predicted_paths = label_encoder.inverse_transform(top_predicted_indices)
        
        st.write(f"Top page: {page_name}")
        st.write(f"Predicted paths: {predicted_paths.tolist()}")
        st.write(f"Confidence scores: {confidence_scores[top_predicted_indices].tolist()}")
