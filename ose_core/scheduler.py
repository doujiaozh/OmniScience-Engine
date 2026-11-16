import dask 
from dask import delayed 
from modules.bio_processing.fetcher import fetch_proteins 
from modules.bio_processing.parser import parse_fasta 
def run_distributed_pipeline(): 
    fetch_task = delayed(fetch_proteins)(100) 
    parse_task = delayed(parse_fasta)(fetch_task) 
    print('[OSE-Dask] Computing DAG...') 
    sequences = parse_task.compute() 
    return sequences 
