import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import lightgbm as lgb

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
# Set random seeds
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} ⚡")

os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# 1. Load Data
data_dir = "Data" if os.path.exists("Data") else "data"
train_df = pd.read_csv(os.path.join(data_dir, "train.csv")).dropna(subset=["cleaned_text", "sentiment"])
test_df = pd.read_csv(os.path.join(data_dir, "test.csv")).dropna(subset=["cleaned_text", "sentiment"])

# Sample a subset for fast training
# Sample a subset for fast training without dropping grouping columns
train_sample = pd.concat([
    group.sample(n=min(len(group), 4000), random_state=42) 
    for _, group in train_df.groupby("sentiment")
], ignore_index=True)

test_sample = pd.concat([
    group.sample(n=min(len(group), 1000), random_state=42) 
    for _, group in test_df.groupby("sentiment")
], ignore_index=True)

X_train, y_train = train_sample["cleaned_text"].values, train_sample["sentiment"].values
X_test, y_test = test_sample["cleaned_text"].values, test_sample["sentiment"].values

results = {}

def plot_cm(cm, model_name):
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=['Neg', 'Neu', 'Pos'], yticklabels=['Neg', 'Neu', 'Pos'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f"plots/cm_{model_name.lower().replace(' ', '_')}.png")
    plt.close()

# TF-IDF Setup
tfidf = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")

# ==========================================
# Model 1: Logistic Regression (Baseline)
# ==========================================
print("\n--- Training Model 1: Logistic Regression ---")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_vec, y_train)
preds_lr = lr.predict(X_test_vec)

acc_lr = accuracy_score(y_test, preds_lr)
f1_lr = f1_score(y_test, preds_lr, average="weighted")
results["Logistic Regression"] = {"Accuracy": acc_lr, "F1-Score": f1_lr}
plot_cm(confusion_matrix(y_test, preds_lr), "Logistic Regression")
joblib.dump(lr, "models/logistic_regression.pkl")
print(f"LR Accuracy: {acc_lr:.4f} | F1: {f1_lr:.4f}")

# ==========================================
# Model 2: LightGBM
# ==========================================
print("\n--- Training Model 2: LightGBM ---")
lgbm = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
lgbm.fit(X_train_vec.astype(np.float32), y_train)
preds_lgb = lgbm.predict(X_test_vec.astype(np.float32))

acc_lgb = accuracy_score(y_test, preds_lgb)
f1_lgb = f1_score(y_test, preds_lgb, average="weighted")
results["LightGBM"] = {"Accuracy": acc_lgb, "F1-Score": f1_lgb}
plot_cm(confusion_matrix(y_test, preds_lgb), "LightGBM")
joblib.dump(lgbm, "models/lightgbm.pkl")
print(f"LightGBM Accuracy: {acc_lgb:.4f} | F1: {f1_lgb:.4f}")

# ==========================================
# Model 3: BiLSTM (PyTorch)
# ==========================================
print("\n--- Training Model 3: BiLSTM ---")
word2idx = {"<PAD>": 0, "<UNK>": 1}
for text in X_train:
    for word in text.split():
        if word not in word2idx and len(word2idx) < 10000:
            word2idx[word] = len(word2idx)

def tokenize_seq(texts, max_len=60):
    seqs = []
    for t in texts:
        tokens = [word2idx.get(w, 1) for w in t.split()][:max_len]
        tokens += [0] * (max_len - len(tokens))
        seqs.append(tokens)
    return np.array(seqs)

X_train_seq = torch.tensor(tokenize_seq(X_train), dtype=torch.long)
y_train_seq = torch.tensor(y_train, dtype=torch.long)
X_test_seq = torch.tensor(tokenize_seq(X_test), dtype=torch.long)
y_test_seq = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(list(zip(X_train_seq, y_train_seq)), batch_size=64, shuffle=True)

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_sz, emb_dim, hidden_dim, out_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab_sz, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, out_dim)

    def forward(self, x):
        _, (h, _) = self.lstm(self.emb(x))
        out = torch.cat((h[-2], h[-1]), dim=1)
        return self.fc(out)

bilstm = BiLSTMClassifier(len(word2idx) + 1, 64, 64, 3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(bilstm.parameters(), lr=0.002)

for epoch in range(4):
    bilstm.train()
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        loss = criterion(bilstm(batch_x), batch_y)
        loss.backward()
        optimizer.step()

bilstm.eval()
with torch.no_grad():
    preds_bilstm = bilstm(X_test_seq.to(device)).argmax(dim=1).cpu().numpy()

acc_lstm = accuracy_score(y_test, preds_bilstm)
f1_lstm = f1_score(y_test, preds_bilstm, average="weighted")
results["BiLSTM"] = {"Accuracy": acc_lstm, "F1-Score": f1_lstm}
plot_cm(confusion_matrix(y_test, preds_bilstm), "BiLSTM")
torch.save(bilstm.state_dict(), "models/bilstm.pth")
print(f"BiLSTM Accuracy: {acc_lstm:.4f} | F1: {f1_lstm:.4f}")

# ==========================================
# Model 4: Fine-Tuned DistilBERT
# ==========================================
print("\n--- Training Model 4: Fine-tuned DistilBERT ---")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Lightweight sub-sample for local speed
sub_train_x = list(X_train[:1500])
sub_train_y = list(y_train[:1500])
sub_test_x = list(X_test[:500])
sub_test_y = list(y_test[:500])

train_enc = tokenizer(sub_train_x, truncation=True, padding=True, max_length=64, return_tensors="pt")
test_enc = tokenizer(sub_test_x, truncation=True, padding=True, max_length=64, return_tensors="pt")

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
    def __len__(self):
        return len(self.labels)

bert_train_loader = DataLoader(TextDataset(train_enc, sub_train_y), batch_size=16, shuffle=True)
bert_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3).to(device)
bert_opt = torch.optim.AdamW(bert_model.parameters(), lr=5e-5)
bert_model.train()
for batch in bert_train_loader:
    bert_opt.zero_grad()
    inputs = {k: v.to(device) for k, v in batch.items()}
    loss = bert_model(**inputs).loss
    loss.backward()
    bert_opt.step()

bert_model.eval()
with torch.no_grad():
    bert_inputs = {k: v.to(device) for k, v in test_enc.items()}
    preds_bert = bert_model(**bert_inputs).logits.argmax(dim=1).cpu().numpy()

acc_bert = accuracy_score(sub_test_y, preds_bert)
f1_bert = f1_score(sub_test_y, preds_bert, average="weighted")
results["DistilBERT"] = {"Accuracy": acc_bert, "F1-Score": f1_bert}
plot_cm(confusion_matrix(sub_test_y, preds_bert), "DistilBERT")
bert_model.save_pretrained("models/distilbert")
tokenizer.save_pretrained("models/distilbert")
print(f"DistilBERT Accuracy: {acc_bert:.4f} | F1: {f1_bert:.4f}")

# ==========================================
# Final Analytics Summary
# ==========================================
summary_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
print("\n================ SOTA MODELS PERFORMANCE ================")
print(summary_df.to_string(index=False))
print("==========================================================")
summary_df.to_csv("plots/performance_metrics.csv", index=False)

# Bar chart comparison
plt.figure(figsize=(7, 4))
sns.barplot(data=summary_df, x="Model", y="Accuracy", palette="Blues_d")
plt.title("Model Comparison: Accuracy")
plt.ylim(0.5, 1.0)
plt.tight_layout()
plt.savefig("plots/model_comparison_accuracy.png")
plt.close()
print("All confusion matrices and performance charts saved to /plots! 🎀")