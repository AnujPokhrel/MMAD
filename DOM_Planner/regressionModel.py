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
        with torch.no_grad():
            self.roll_torch_md.linear.weight = nn.Parameter(torch.tensor(roll_weights, dtype=torch.float32, device=self.device).unsqueeze(0))
            self.roll_torch_md.linear.bias = nn.Parameter(torch.tensor(roll_bias, dtype=torch.float32, device=self.device))

        with torch.no_grad():
            self.pitch_torch_md.linear.weight = nn.Parameter(torch.tensor(pitch_weights, dtype=torch.float32, device=self.device).unsqueeze(0))
            self.pitch_torch_md.linear.bias = nn.Parameter(torch.tensor(pitch_bias, dtype=torch.float32, device=self.device))

    #Input to the model rpm, rpm_dot, steering, pitch_angle, roll_angle
    def model_dynamics(self, states, actions):
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

    

   