import time
import logging
from modules.bio_processing.processor import BioProcessor
from modules.phys_sim.accelerator import PhysSimAccelerator
from modules.light_ai.inference import LightAIModel

logger = logging.getLogger(__name__)

class OSEScheduler:
    def __init__(self):
        self.bio = BioProcessor()
        self.phys = PhysSimAccelerator()
        self.ai = LightAIModel()
    def run(self, seq):
        logger.info("Starting OSE Pipeline...")
        t0 = time.time()
        try:
            b_out = self.bio.process(seq)
            p_out = self.phys.simulate(b_out)
            score = self.ai.train_and_predict(p_out)
            logger.info(f"Done! Real AI Score: {round(score, 4)}")
            logger.info(f"Time: {round(time.time() - t0, 2)} s")
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise
