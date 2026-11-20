import os
import requests

def fetch_proteins(limit=100):
    filepath = "data/proteins.fasta"
    if os.path.exists(filepath):
        print("[OSE-Data] Loading cached local protein sequences...")
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

    print(f"[OSE-Data] Downloading {limit} sequences from CNCB...")
    url = "https://ngdc.cncb.ac.cn/genbase/api/file/fasta"
    acc_list = ["C_AA001108.1", "C_AA001109.1", "C_AA001110.1"]
    try:
        fasta_records = []
        for acc in acc_list[:min(limit, len(acc_list))]:
            res = requests.get(url, params={"acc": acc}, timeout=10)
            res.raise_for_status()
            fasta_records.append(res.text.strip())
        combined_fasta = "\n".join(fasta_records)
        os.makedirs("data", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(combined_fasta)
        return combined_fasta
    except Exception as e:
        print(f"[OSE-Data] Error: {e}")
        return ""
