from sentence_transformers import SentenceTransformer

class TextEmbedder:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed_text(self, text):
        return self.model.encode(text).tolist()

text_embedder = TextEmbedder()