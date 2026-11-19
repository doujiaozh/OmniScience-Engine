import sys, os 
import logging 
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s") 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))) 
from ose_core.scheduler import OSEScheduler 
engine = OSEScheduler() 
print("--- Testing GGGGGGGGGGGGGGGG ---") 
engine.run("GGGGGGGGGGGGGGGG") 
