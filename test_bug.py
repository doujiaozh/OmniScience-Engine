import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from modules.bio_processing.processor import BioProcessor

bio = BioProcessor()
feat1 = bio.process("AAAAAAAA")
feat2 = bio.process("ATCGATCG")
print("--- Seq: AAAAAAAA ---")
print("Old buggy x0:", feat1.mean().item())
print(
    "New physical x0:",
    torch.matmul(feat1.view(-1, 4), torch.tensor([1.0, 2.0, 3.0, 4.0])).mean().item(),
)
print("--- Seq: ATCGATCG ---")
print("Old buggy x0:", feat2.mean().item())
print(
    "New physical x0:",
    torch.matmul(feat2.view(-1, 4), torch.tensor([1.0, 2.0, 3.0, 4.0])).mean().item(),
)
