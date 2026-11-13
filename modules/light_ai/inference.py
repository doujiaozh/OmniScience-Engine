import torch
import torch.nn as nn
import torch.optim as optim

class LightAIModel(nn.Module):
    def __init__(self):
        super().__init__()
        print("[OSE-AI] Initializing Real LightAIModel with Training Loop...")
        self.net = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)
        self.criterion = nn.BCELoss()

    def forward(self, x):
        return self.net(x)

    def train_and_predict(self, states):
        x = states.view(1, 10)
        target = torch.tensor([[1.0]])
        
        # Real Training Loop
        for epoch in range(5):
            self.optimizer.zero_grad()
            output = self(x)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
        return self(x).item()
