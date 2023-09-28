import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity


def get_bert_embedding(text):
    inputs = tokenizer(
        text, return_tensors="pt", max_length=512, truncation=True, padding="max_length"
    )
    with torch.no_grad():
        outputs = model(**inputs)
    return np.array(
        outputs["last_hidden_state"][:, 0, :].cpu().tolist()
    ).squeeze()  # Ensure 1D shape (768,)


def get_combined_bert_embedding(
    row, columns=["Course Name", "Course Description", "Skills"]
):
    combined_embedding = np.zeros((768,))
    for col in columns:
        combined_embedding += get_bert_embedding(row[col])
    return combined_embedding / len(columns)


def recommend_courses(query, n=5):
    query_embedding = get_bert_embedding(query)
    cosine_similarities = cosine_similarity(
        query_embedding.reshape(1, -1), embedding_matrix
    ).flatten()

    ranked_scores = cosine_similarities * \
        df["Course Rating"].astype(float).values
    top_indices = ranked_scores.argsort()[-n:][::-1]

    return df[["Course Name", "Course URL"]].iloc[top_indices]


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

file_path = "coursera_embedings.pkl"

if os.path.exists(file_path):
    df = pd.read_pickle(file_path)
else:
    df = pd.read_csv("coursera_data.csv")

    tqdm.pandas()
    df["embedding"] = df.progress_apply(
        lambda row: get_combined_bert_embedding(row), axis=1
    )

    df.to_pickle(file_path)

df["Course Rating"] = pd.to_numeric(df["Course Rating"], errors="coerce")
df["Course Rating"].fillna(df["Course Rating"].mean(), inplace=True)

embedding_matrix = np.vstack(df["embedding"].values)

query = "automation testing"
recommended_courses = recommend_courses(query, n=5)
print(recommended_courses)
