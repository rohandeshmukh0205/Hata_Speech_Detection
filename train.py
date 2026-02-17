import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.svm import OneClassSVM
import joblib


data = pd.read_csv("data/labeled_data.csv")

# Keep only normal sentences (class == 2)
normal_data = data[data['class'] == 2]
normal_sentences = normal_data['tweet'].tolist()

print("Normal samples:", len(normal_sentences))

# Load Sentence Embedding model..
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Convert sentences to embeddings
X_train = embedder.encode(normal_sentences, show_progress_bar=True)

# Train One-Class SVM
ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
ocsvm.fit(X_train)

# Save model
joblib.dump(ocsvm, "model.pkl")
joblib.dump(embedder, "embedder.pkl")

print("Model trained and saved successfully.")
