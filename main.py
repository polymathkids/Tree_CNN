import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import defaultdict
import numpy as np

# 1. Dataset definition
class BOMDataset(Dataset):
    def __init__(self, dataframe, token_freq_table, vocab):
        self.data = dataframe
        self.token_freq_table = token_freq_table
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        token_ids = [self.vocab.get(tok, 0) for tok in item['tokens']]
        token_tensor = torch.tensor(token_ids[:50], dtype=torch.long)
        freq_vec = self.token_freq_table.loc[item['tokens']].mean().fillna(0).values.astype(np.float32)
        return token_tensor, torch.tensor(freq_vec), torch.tensor(item['wbs_label'])

# 2. CNN model
class BOMCNN(nn.Module):
    def __init__(self, vocab_size, num_labels, freq_dim, embedding_dim=64):
        super(BOMCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embedding_dim, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc1 = nn.Linear(64 + freq_dim, 128)
        self.out = nn.Linear(128, num_labels)

    def forward(self, tokens, freq_vec):
        x = self.embedding(tokens).permute(0, 2, 1)  # [B, E, T]
        x = F.relu(self.conv1(x))
        x = self.pool(x).squeeze(-1)  # [B, 64]
        x = torch.cat([x, freq_vec], dim=1)
        x = F.relu(self.fc1(x))
        return self.out(x)

# 3. Tree-split logic
def split_by_parent_assembly(df):
    return dict(tuple(df.groupby('parent_assembly')))

# 4. Training function
def train_cnn_model(train_df, token_freq_table, vocab, num_labels):
    dataset = BOMDataset(train_df, token_freq_table, vocab)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = BOMCNN(vocab_size=len(vocab), num_labels=num_labels, freq_dim=token_freq_table.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        for tokens, freq_vec, labels in dataloader:
            logits = model(tokens, freq_vec)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model

# 5. Main execution logic
def train_tree_cnn(bom_df, token_freq_table, vocab, num_labels):
    tree_splits = split_by_parent_assembly(bom_df)
    trained_models = {}
    for parent, df in tree_splits.items():
        if len(df) > 10:  # Only train if enough samples
            print(f"Training model for parent assembly: {parent}")
            trained_models[parent] = train_cnn_model(df, token_freq_table, vocab, num_labels)
    return trained_models

'''
	•	bom_df includes columns: tokens, wbs_label, and parent_assembly.
	•	token_freq_table is a DataFrame where each row is a token and each column is a WBS label.
	•	vocab is a dictionary mapping tokens to indices.
	•	wbs_label is already encoded as integers; if not, you can preprocess with LabelEncoder.

'''
# read in the BOM data
bom_df = pd.read_csv('bom_data.csv')  # Adjust path as necessary
# encode the WBS labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
bom_df['wbs_label'] = le.fit_transform(bom_df['wbs_label'])

# use trained distilbert tokenenizer to tokenize the tokens
from transformers import DistilBertTokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
def tokenize_bom(df):
    df['tokens'] = df['tokens'].apply(lambda x: tokenizer.tokenize(x))  # Limit to 50 tokens
    return df
bom_df = tokenize_bom(bom_df)

# read in the token frequency table
token_freq_table = pd.read_csv('token_freq_table.csv', index_col=0)  # Adjust path as necessary
# create a vocabulary mapping
vocab = {token: idx + 1 for idx, token in enumerate(token_freq_table.index)}  # Start from 1
# number of unique WBS labels
num_labels = len(le.classes_)
# train the tree CNN model
trained_models = train_tree_cnn(bom_df, token_freq_table, vocab, num_labels)
# Save the trained models
import torch
for parent, model in trained_models.items():
    torch.save(model.state_dict(), f'model_{parent}.pth')
    print(f"Model for {parent} saved.")
# Example usage of the trained models
def predict_with_model(model, tokens, freq_vec):
    model.eval()
    with torch.no_grad():
        tokens_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # Add batch dimension
        freq_tensor = torch.tensor(freq_vec, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
        logits = model(tokens_tensor, freq_tensor)
        return logits.argmax(dim=1).item()
# Example prediction
def example_prediction(model, example_tokens, example_freq_vec):
    return predict_with_model(model, example_tokens, example_freq_vec)
# Example usage
example_tokens = [vocab.get(tok, 0) for tok in bom_df['tokens'].iloc[0][:50]]  # Example tokens
example_freq_vec = token_freq_table.loc[bom_df['tokens'].iloc[0]].mean().fillna(0).values.astype(np.float32)
example_model = trained_models[bom_df['parent_assembly'].iloc[0]]
predicted_label = example_prediction(example_model, example_tokens, example_freq_vec)
print(f"Predicted label: {predicted_label} ({le.inverse_transform([predicted_label])[0]})")
# Example output
# Predicted label: 2 (Assembly A)
# Note: Ensure that the paths to the CSV files and the tokenizer are correct.
# The above code provides a complete implementation of a CNN model for predicting WBS labels from BOM data.
# The model is trained on a tree-split dataset based on parent assemblies.
# The example prediction shows how to use the trained model to predict a label for a given set of tokens and frequency vector.
# The code includes necessary imports, dataset preparation, model definition, training logic, and example usage.
# The model is saved for each parent assembly, allowing for later use or evaluation.
