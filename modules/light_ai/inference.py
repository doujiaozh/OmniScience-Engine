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
        # 改用 MSELoss，对云端环境的精度要求更宽容
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.net(x)

    def train_and_predict(self, states):
        # 强制将 states 转换为 float32，并拉平形状
        x = states.view(1, 10).float()
        target = torch.tensor([[1.0]], dtype=torch.float32)
        
        for epoch in range(5):
            self.optimizer.zero_grad()
            output = self(x)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
        return self(x).item()