import os
import requests

def fetch_proteins(limit=100):
    # 1. 本地缓存：如果已下载过，直接秒读本地文件
    filepath = "data/proteins.fasta"
    if os.path.exists(filepath):
        print('[OSE-Data] Loading cached local protein sequences...')
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

    # 2. 国家生物信息中心 GenBase API
    print(f'[OSE-Data] Downloading real protein sequences from CNCB GenBase (limit={limit})...')
    url = "https://ngdc.cncb.ac.cn/genbase/api/file/fasta"
    # 使用真实的测试蛋白 Accession 编号（如人类血红蛋白）
    acc_list = ["C_AA001108.1", "C_AA001109.1", "C_AA001110.1"]
    
    try:
        fasta_records = []
        for acc in acc_list[:min(limit, len(acc_list))]:
            params = {"acc": acc}
            res = requests.get(url, params=params, timeout=10)
            res.raise_for_status()
            fasta_records.append(res.text.strip())
        
        combined_fasta = "\n".join(fasta_records)
        
        # 3. 将下载的数据存入本地
        os.makedirs("data", exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(combined_fasta)
            
        return combined_fasta
    except Exception as e:
        print(f"[OSE-Data] CNCB Network Error ({e}). Using built-in demo sequences.")
        # 4. 兜底数据：如果国内网络也异常，直接使用内置的真实蛋白质片段，确保程序永不卡死
        return """>sp|P69905|HBA_HUMAN Hemoglobin subunit alpha
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSH
>sp|P68871|HBB_HUMAN Hemoglobin subunit beta
MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLST
>sp|P0DTC2|SPIKE_SARS2 Spike glycoprotein
MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHST"""