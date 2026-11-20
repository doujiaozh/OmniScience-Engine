from Bio import SeqIO
from io import StringIO


def parse_fasta(fasta_text):
    print("[OSE-Parser] Parsing FASTA data...")
    records = list(SeqIO.parse(StringIO(fasta_text), "fasta"))
    return [str(r.seq) for r in records]
