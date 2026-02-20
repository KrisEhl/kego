import torch
from skorch import NeuralNetBinaryClassifier


class AMPNeuralNetBinaryClassifier(NeuralNetBinaryClassifier):
    """NeuralNetBinaryClassifier with automatic mixed precision (fp16)."""

    def initialize(self):
        super().initialize()
        self.amp_scaler_ = torch.amp.GradScaler("cuda")
        return self

    def infer(self, x, **fit_params):
        with torch.amp.autocast("cuda"):
            return super().infer(x, **fit_params)

    def train_step_single(self, batch, **fit_params):
        try:
            from skorch.dataset import unpack_data
        except ImportError:
            from skorch.utils import unpack_data

        self._set_training(True)
        Xi, yi = unpack_data(batch)
        y_pred = self.infer(Xi, **fit_params)
        loss = self.get_loss(y_pred, yi, X=Xi, training=True)
        self.amp_scaler_.scale(loss).backward()
        return {"loss": loss, "y_pred": y_pred}

    def train_step(self, batch, **fit_params):
        self.optimizer_.zero_grad()
        step = self.train_step_single(batch, **fit_params)
        self.amp_scaler_.step(self.optimizer_)
        self.amp_scaler_.update()
        return step
