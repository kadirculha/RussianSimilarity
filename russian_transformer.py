import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
 

class SentenceTransformer:
    def __init__(self, model_name: str):
        self.model_name = model_name

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def calculate_cosine_similarity(self, embeddings):
        similarity = cosine_similarity(embeddings[0].reshape(1, -1), embeddings[1].reshape(1, -1))
        return similarity[0][0]

    def get_similarity(self, sentences:list):

        # Load model from HuggingFace Hub
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)

        # Tokenize sentences
        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

        # Compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        # İki cümle arasındaki cosine similarity'yi hesapla
        similarity_score = self.calculate_cosine_similarity(sentence_embeddings)

        return similarity_score
