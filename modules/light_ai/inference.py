import torch
import torch.nn as nn
import torch.optim as optim
import logging

logger = logging.getLogger(__name__)

class LightAIModel(nn.Module):
    def __init__(self):
        super().__init__()
        logger.info("[OSE-AI] Initializing LightAIModel...")
        self.net = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
    def forward(self, x):
        return self.net(x)
    def train_and_predict(self, states):
        try:
            x = states.view(1, 10).float()
            target = torch.tensor([[1.0]], dtype=torch.float32)
            for _ in range(5):
                self.optimizer.zero_grad()
                loss = self.criterion(self(x), target)
                loss.backward()
                self.optimizer.step()
            return self(x).item()
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
