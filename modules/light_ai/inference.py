import torch 
 
class LightAIModel: 
    def __init__(self): 
        print("[OSE-AI] Loading Lightweight AI Model...") 
        self.weights = torch.randn(64, 1) 
 
    def predict(self, simulated_states): 
        final_state = simulated_states[:, -1, :] 
        prediction = torch.sigmoid(torch.matmul(final_state, self.weights)) 
        return prediction.item() 
