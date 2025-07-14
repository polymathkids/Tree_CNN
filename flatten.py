# recommend_flattening.py

import torch
import torch.nn.functional as F
import pandas as pd
from transformers import DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

class FlatteningRecommender:
    def __init__(self, model_name='distilbert-base-uncased', similarity_threshold=0.85, device='cpu'):
        self.device = torch.device(device)
        self.model = DistilBertModel.from_pretrained(model_name).to(self.device)
        self.similarity_threshold = similarity_threshold

    def compute_embeddings(self, token_id_lists, max_len=32):
        """
        token_id_lists: List of lists of token IDs for each part
        Returns: Tensor of embeddings [N, H]
        """
        input_tensor = torch.zeros((len(token_id_lists), max_len), dtype=torch.long)
        for i, tok_ids in enumerate(token_id_lists):
            length = min(len(tok_ids), max_len)
            input_tensor[i, :length] = torch.tensor(tok_ids[:length])
        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # [B, hidden_size]
        return embeddings.cpu()

    def recommend(self, df, token_column='token_ids', label_column='wbs_label', name_column='assembly_name'):
        """
        df: DataFrame with columns: [assembly_name, wbs_label, token_ids]
        Returns: DataFrame of similar assembly pairs with flattening recommendations
        """
        results = []
        grouped = df.groupby(label_column)

        for wbs_label, group in tqdm(grouped, desc="Analyzing WBS labels"):
            if len(group) < 2:
                continue

            token_lists = group[token_column].tolist()
            names = group[name_column].tolist()
            embeddings = self.compute_embeddings(token_lists)
            sim_matrix = cosine_similarity(embeddings)

            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    sim = sim_matrix[i, j]
                    if sim >= self.similarity_threshold:
                        parent, child = sorted([names[i], names[j]], key=len)
                        results.append({
                            "Parent": parent,
                            "Child": child,
                            "WBS_Label": wbs_label,
                            "Similarity": round(sim, 4),
                            "Recommend_Flattening": True
                        })

        return pd.DataFrame(results)