import torch
import torch.nn as nn
import pdb

# ## No normalization
class RollPitchAcclnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(13, 16), 
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.05), 
            nn.Dropout(0.2),
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.05),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.05),
            nn.Dropout(0.2),
            nn.Linear(16, 3) # Output: roll_accln, pitch_accln
        )

    def forward(
            self,
            rpm: torch.Tensor,
            rpm_dot: torch.Tensor,
            steering: torch.Tensor,
            steering_dot: torch.Tensor,
            angles: torch.Tensor,
            angular_vels: torch.Tensor,
    ):
        # x = torch.concat([rpm, rpm_dot, steering, steering_dot, angles, roll_pitch_accln], dim=-1) # Directly concatenate inputs
        x = torch.concat([rpm, rpm_dot, steering, steering_dot, angles, angular_vels], dim=-1) # Directly concatenate inputs
        x = self.fc1(x)
        return x

class RpmModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_all = nn.Sequential(
            nn.Linear(22, 32), 
            nn.ReLU(), 
            nn.Dropout(0.4),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(8, 2)
        )

    def forward(
            self,
            rpm: torch.Tensor,
            steering: torch.Tensor,
            cmd_vel: torch.Tensor,
    ):
        x = torch.concat([rpm, steering, cmd_vel], dim=-1) # Directly concatenate inputs        
        x = self.fc_all(x)
        return x








