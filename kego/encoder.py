from abc import ABC, abstractmethod

from transformers import AutoModelForSequenceClassification, AutoTokenizer


class EncoderBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def encode(self, text: str):
        raise NotImplementedError(f"{self.__class__} needs to implement encode!")


class EncoderHuggingFace(EncoderBase):
    def __init__(self, model_name: str):
        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def encode(self, text: str):
        encoded_input = self.tokenizer(text, return_tensors="pt")
        output = self.model(**encoded_input)
        return output
