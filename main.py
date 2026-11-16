import sys, os 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))) 
from ose_core.scheduler import run_distributed_pipeline 
if __name__ == '__main__': 
    print('=== Starting Real Protein Data Pipeline ===') 
    seqs = run_distributed_pipeline() 
    print('Success! Loaded', len(seqs), 'real protein sequences.') 
    print('Example sequence:', seqs[0][:50]) 
