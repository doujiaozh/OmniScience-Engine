import sys, os
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ose_core.scheduler import OSEScheduler

if __name__ == "__main__":
    try:
        engine = OSEScheduler()
        engine.run("ATCGATCGATCGATCG")
    except Exception as e:
        logging.error(f"System Exit due to error: {e}")
