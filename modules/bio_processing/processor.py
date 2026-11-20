import torch
import logging

logger = logging.getLogger(__name__)


class BioProcessor:
    def __init__(self):
        logger.info("[OSE-Bio] Initializing Real DNA BioProcessor...")

    def process_genomic_data(self, raw_sequence: str):
        mapping = {"A": 0, "T": 1, "C": 2, "G": 3, "a": 0, "t": 1, "c": 2, "g": 3}
        numeric_seq = [mapping.get(base, 0) for base in raw_sequence]

        seq_tensor = torch.tensor(numeric_seq[:64], dtype=torch.float32)
        if len(seq_tensor) < 64:
            padding = torch.zeros(64 - len(seq_tensor))
            seq_tensor = torch.cat((seq_tensor, padding))

        logger.info(f"  - {len(raw_sequence)} bases into 64-dim tensor.")
        return {
            "feature_vector": seq_tensor.unsqueeze(0),
            "seq_length": len(raw_sequence),
        }

    def process(self, seq):
        return self.process_genomic_data(seq)
