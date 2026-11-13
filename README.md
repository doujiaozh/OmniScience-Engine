# OmniScience-Engine (OSE) 
 
A Lightweight, End-to-End AI4Science Framework for Bioinformatics, Physics Simulation, and Lightweight AI. 
 
## Introduction 
OmniScience-Engine (OSE) is an ambitious open-source project aiming to bridge the gap between heterogeneous biological data, physical dynamics simulation, and lightweight AI inference. 
 
## The Win7 Hardcore Challenge 
This MVP was successfully built and executed on a legacy Windows 7 system with Python 3.11. 
We bypassed numerous legacy DLL limitations (e.g., shm.dll, rpds, pydantic_core) to achieve a seamless end-to-end pipeline in 0.07 seconds! 
 
## Core Architecture (MVP Version) 
The current standalone MVP includes: 
1. BioProcessor: One-hot encoding of real DNA sequences. 
2. PhysSimAccelerator: Numerical integration of a Damped Harmonic Oscillator ODE. 
3. LightAIModel: A PyTorch Multi-Layer Perceptron (MLP) performing inference on the simulated states. 
 
## Quick Start 
git clone https://github.com/your-username/OmniScience-Engine.git 
cd OmniScience-Engine 
pip install -r requirements.txt 
python main.py 
 
## Roadmap 
[x] Standalone MVP pipeline (Bio + Phys + AI) 
[x] Win7 compatibility hacking 
[ ] Integration with Ray Distributed Framework (Linux environment) 
[ ] Real-world PDB protein dataset support 
 
## License 
MIT License 

## CI Status
![OSE CI](https://github.com/doujiaozh/OmniScience-Engine/actions/workflows/python-app.yml/badge.svg)

## Live Demo Output
```text
[OSE-Bio] Initializing Real BioProcessor...
[OSE-Phys] Initializing Real PhysSim Accelerator...
[OSE-AI] Initializing Real LightAIModel with Training Loop...
Done! Real AI Score: 0.5982
Time: 0.65 s
```
