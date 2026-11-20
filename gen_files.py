import os


def w(p, c):
    d = os.path.dirname(p)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(c)


w(
    "modules/bio_processing/processor.py",
    """import torch

class BioProcessor:
    def __init__(self):
        print("[OSE-Bio] Initializing Real BioProcessor...")

    def process(self, seq):
        mapping = {"A": 0, "T": 1, "C": 2, "G": 3}
        idx = [mapping.get(b, 0) for b in seq]
        one_hot = torch.nn.functional.one_hot(torch.tensor(idx), num_classes=4).float()
        return one_hot.flatten().unsqueeze(0)
""",
)

w(
    "modules/phys_sim/accelerator.py",
    """import torch
import numpy as np
from scipy.integrate import odeint

class PhysSimAccelerator:
    def __init__(self):
        print("[OSE-Phys] Initializing Real PhysSim Accelerator...")

    def simulate(self, bio_feats, steps=10):
        x0 = bio_feats.mean().item()
        t = np.linspace(0, steps * 0.1, steps)

        def damped_oscillator(y, t, k, c):
            x, v = y
            dydt = [v, -k * x - c * v]
            return dydt

        sol = odeint(damped_oscillator, [x0, 0.0], t, args=(1.0, 0.1))
        return torch.tensor(sol[:, 0]).float().unsqueeze(0).unsqueeze(-1)
""",
)

w(
    "modules/light_ai/inference.py",
    """import torch
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
""",
)

w(
    "ose_core/scheduler.py",
    """from modules.bio_processing.processor import BioProcessor
from modules.phys_sim.accelerator import PhysSimAccelerator
from modules.light_ai.inference import LightAIModel
import time

class OSEScheduler:
    def __init__(self):
        self.bio = BioProcessor()
        self.phys = PhysSimAccelerator()
        self.ai = LightAIModel()

    def run(self, seq):
        print("=== Starting OSE Pipeline ===")
        t0 = time.time()
        
        # 1. Bioinformatics
        b_out = self.bio.process(seq)
        
        # 2. Physics Simulation
        p_out = self.phys.simulate(b_out)
        
        # 3. AI Inference & Training
        score = self.ai.train_and_predict(p_out)
        
        print(f"Done! Real AI Score: {round(score, 4)}")
        print(f"Time: {round(time.time() - t0, 2)} s")
""",
)

w(
    "main.py",
    """from ose_core.scheduler import OSEScheduler

if __name__ == "__main__":
    engine = OSEScheduler()
    engine.run("ATCGATCGATCGATCG")
""",
)

w(
    ".github/workflows/python-app.yml",
    """name: OSE CI

on: [push]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python 3.10
      uses: actions/setup-python@v4
      with:
        python-version: "3.10"
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install torch==2.2.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
        pip install scipy biopython
    - name: Run pipeline
      run: python main.py
""",
)

w(
    ".github/ISSUE_TEMPLATE/bug_report.md",
    """---
name: Bug report
about: Create a report to help us improve
---

**Describe the bug**
A clear and concise description of what the bug is.

**Environment:**
 - OS: Windows 7
 - Python: 3.11
""",
)

w(
    "LICENSE",
    """MIT License

Copyright (c) 2026 doujiaozh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
""",
)

w("requirements.txt", "torch==2.2.1\nscipy\nbiopython\n")
w(".gitignore", "__pycache__/\n*.py[cod]\n.vscode/\n*.log\n")

print("All premium files generated successfully!")
