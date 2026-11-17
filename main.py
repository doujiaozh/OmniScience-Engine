import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ose_core.scheduler import OSEScheduler

if __name__ == "__main__":
    engine = OSEScheduler()
    engine.run("ATCGATCGATCGATCG")
