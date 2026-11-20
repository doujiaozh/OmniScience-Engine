import torch
import logging

logger = logging.getLogger(__name__)


class LightAIModel:
    def __init__(self):
        logger.info("[OSE-AI] Loading Lightweight AI Model...")
        self.weights = torch.randn(20, 1)

    def predict(self, simulated_states: torch.Tensor):
        flat_state = simulated_states.view(1, -1)
        prediction = torch.sigmoid(torch.matmul(flat_state, self.weights))
        return prediction.item()

    def train_and_predict(self, simulated_states):
        return self.predict(simulated_states)

    def predict_fixed(self, simulated_states):
        flat = simulated_states.view(-1)
        mean_val = torch.mean(flat).item()
        import math

        return 1.0 / (1.0 + math.exp(-mean_val * 0.1))
