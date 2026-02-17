import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
import joblib

# Load model
ocsvm = joblib.load("model.pkl")
embedder = joblib.load("embedder.pkl")

# Load dataset
data = pd.read_csv("data/labeled_data.csv")

# Prepare test set (mixed)
test_sentences = data['tweet'].tolist()
true_labels = data['class'].tolist()

# Convert to embeddings
X_test = embedder.encode(test_sentences, show_progress_bar=True)

# Predict
predictions = ocsvm.predict(X_test)

# Convert predictions
# One-Class SVM:
#  1 = normal
# -1 = anomaly

pred_labels = []
for p in predictions:
    if p == 1:
        pred_labels.append(2)  # normal
    else:
        pred_labels.append(0)  # anomaly (hate/offensive)

# Convert true labels to binary
binary_true = []
for t in true_labels:
    if t == 2:
        binary_true.append(2)
    else:
        binary_true.append(0)

print(classification_report(binary_true, pred_labels))
