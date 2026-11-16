import time 
from modules.bio_processing.processor import BioProcessor 
from modules.phys_sim.accelerator import PhysSimAccelerator 
from modules.light_ai.inference import LightAIModel 
 
class OSEScheduler: 
    def __init__(self): 
        self.bio = BioProcessor() 
        self.phys = PhysSimAccelerator() 
        self.ai = LightAIModel() 
 
    def run(self, seq): 
        print("=== Starting OSE Pipeline ===") 
        t0 = time.time() 
        b_out = self.bio.process(seq) 
        p_out = self.phys.simulate(b_out) 
        score = self.ai.train_and_predict(p_out) 
        print("Done! Real AI Score:", round(score, 4)) 
        print("Time:", round(time.time() - t0, 2), "s") 
