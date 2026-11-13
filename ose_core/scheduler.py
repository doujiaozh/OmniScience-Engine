import ray 
import torch 
from modules.bio_processing.processor import BioProcessor 
from modules.phys_sim.accelerator import PhysSimAccelerator 
from modules.light_ai.inference import LightAIModel 
 
@ray.remote 
class BioActor(BioProcessor): pass 
 
@ray.remote 
class PhysActor(PhysSimAccelerator): pass 
 
@ray.remote 
class AIActor(LightAIModel): pass 
 
class OSEScheduler: 
    def __init__(self): 
        if not ray.is_initialized(): 
            ray.init(ignore_reinit_error=True, num_cpus=4) 
        self.bio_node = BioActor.remote() 
        self.phys_node = PhysActor.remote() 
        self.ai_node = AIActor.remote() 
 
    def run_pipeline(self, gene_sequence): 
        print("Starting pipeline:", gene_sequence[:10]) 
        bio_future = self.bio_node.process_genomic_data.remote(gene_sequence) 
        phys_future = self.phys_node.simulate_dynamics.remote(bio_future["feature_vector"], 10) 
        ai_future = self.ai_node.predict.remote(phys_future) 
        result = ray.get(ai_future) 
        print("Done! Score:", round(result, 4)) 
        return result 
