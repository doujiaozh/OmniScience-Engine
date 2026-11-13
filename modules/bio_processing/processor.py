import torch

class BioProcessor:
    def __init__(self):
        print("[OSE-Bio] Initializing Real BioProcessor...")

    def process(self, seq):
        mapping = {"A": 0, "T": 1, "C": 2, "G": 3}
        idx = [mapping.get(b, 0) for b in seq]
        one_hot = torch.nn.functional.one_hot(torch.tensor(idx), num_classes=4).float()
        return one_hot.flatten().unsqueeze(0)
