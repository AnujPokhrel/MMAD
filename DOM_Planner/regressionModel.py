import torch
import torch.nn as nn
import pickle
from utils import map_range
import pdb

class PitchLRTorch(nn.Module):
    def __init__(self, input_dim):
        super(PitchLRTorch, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)  # Linear layer with bias

    def forward(self, x):
        return self.linear(x)

class RollLRTorch(nn.Module):
    def __init__(self, input_dim):
        super(RollLRTorch, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)  # Linear layer with bias

    def forward(self, x):
        return self.linear(x)

class RollPitchRegressionModels():
    def __init__(self, roll_model_pickle = 'roll_lr_model.pkl', pitch_model_pickle = 'pitch_lr_model.pkl', input_dim=5, device='cuda'):
        self.device = device        
        self.roll_torch_md = RollLRTorch(5).to(self.device)
        self.pitch_torch_md = PitchLRTorch(5).to(self.device)


        self.acc_max = 1940
        self.desc_max = 10550

        with open('roll_lr_model.pkl', 'rb') as f:
            
            roll_ln_model = pickle.load(f)
        
        with open('pitch_lr_model.pkl', 'rb') as f:
            pitch_ln_model = pickle.load(f)

        roll_weights = roll_ln_model.coef_
        roll_bias = roll_ln_model.intercept_
        pitch_weights = pitch_ln_model.coef_
        pitch_bias = pitch_ln_model.intercept_
        # pdb.set_trace()
        with torch.no_grad():
            self.roll_torch_md.linear.weight = nn.Parameter(torch.tensor(roll_weights, dtype=torch.float32, device=self.device).unsqueeze(0))
            self.roll_torch_md.linear.bias = nn.Parameter(torch.tensor(roll_bias, dtype=torch.float32, device=self.device))

        with torch.no_grad():
            self.pitch_torch_md.linear.weight = nn.Parameter(torch.tensor(pitch_weights, dtype=torch.float32, device=self.device).unsqueeze(0))
            self.pitch_torch_md.linear.bias = nn.Parameter(torch.tensor(pitch_bias, dtype=torch.float32, device=self.device))
        # pdb.set_trace()
        # print(f"model is in: {next(self.roll_torch_md.parameters()).device}")

    #Input to the model rpm, rpm_dot, steering, pitch_angle, roll_angle
    def model_dynamics(self, states, actions):
        #Aniket's old implementation
            # # Simplified dynamics: actions -> accelerations in roll and pitch
            # accelerations = torch.zeros_like(states[:, :2])
            # accelerations[:, 0] = actions[:, 1] * 0  # Example coefficient for roll
            # desc_mask = actions[:, 1] < 0
            # acc_mask = actions[:, 1] > 0
            # # accelerations[:, 1] = map_range(actions[:, 0], -desc_max, acc_max,  46.76, -8.6) 
            # accelerations[desc_mask, 1] = map_range(actions[desc_mask, 0], -self.desc_max,  0,  46.76, 0) * 2  
            # accelerations[acc_mask, 1]  = map_range(actions[acc_mask, 0] ,  0,  self.acc_max,   0, -8.6) ** 2 /-(8.6**2)
            
            # return accelerations
        accelerations = torch.zeros_like(states[:, :2])
        angles = states[:, :2] 
        rpm = states[:, 4]      # Extract rpm from the states
        steering_angle = states[:, 5]
        rpm_dot = actions[:, 0]

        model_inputs = torch.transpose(torch.stack([rpm, rpm_dot, steering_angle, angles[:, 1], angles[:, 0]]), 0,1)
        # pdb.set_trace()
        accelerations[:, 0] = self.roll_torch_md(model_inputs).squeeze(-1) * 0.0
        accelerations[:, 1] = self.pitch_torch_md(model_inputs).squeeze(-1)

        return accelerations

    

   