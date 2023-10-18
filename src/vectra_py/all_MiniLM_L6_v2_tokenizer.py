from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


class OSSTokenizer:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Load model from HuggingFace Hub
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def decode(self, tokens):
        pass

    def encode(self, text):
        try:
            if len(text) > 1:  # if text is a list of strings
                data = [self.tokenizer.encode(item) for item in text]
                return data
            else:
                data = self.tokenizer.encode(text)
                return data
        except Exception as e:
            print('encoding error', e)
            return None
