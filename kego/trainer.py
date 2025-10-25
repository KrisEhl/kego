from .models import ModelBase


class Trainer:
    def __init__(self, model: ModelBase):
        self.model = model

    def train(self):
        self.model.train()
