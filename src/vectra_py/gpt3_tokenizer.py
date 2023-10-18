import tiktoken
# from tiktoken import encode, decode


class GPT3Tokenizer:
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.encoding = tiktoken.encoding_for_model(model_name)

    def decode(self, tokens):
        return self.encoding.decode(tokens)

    def encode(self, text):
        return self.encoding.encode(text)
