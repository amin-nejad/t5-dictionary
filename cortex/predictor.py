import torch
from transformers import AutoTokenizer, AutoModelWithLMHead


class PythonPredictor:
    def __init__(self, config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained("t5-large")
        self.model = AutoModelWithLMHead.from_pretrained("t5-large").to(self.device)

    def predict(self, payload):
        tokens = self.tokenizer.encode(payload["text"], return_tensors="pt").to(
            self.device
        )
        output = self.model.generate(
            input_ids=tokens,
            max_length=512,
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
        )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
