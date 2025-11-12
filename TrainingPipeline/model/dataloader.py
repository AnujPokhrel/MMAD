from pathlib import Path
import copy
import pickle

# import albumentations as A
import numpy as np
from scipy.signal import periodogram
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pdb
import icecream as ic


def merge(base_dict: dict, new_dict: dict):
    """Merges two dictionary together

    base_dict (dict): The base dictionary to be updated
    new_dict (dict): The new data to be added to the base dictionary
    """
    # assert base_dict is None, "Base dictionary cannot be None"
    assert (
        base_dict.keys() == new_dict.keys()
    ), "The two dictionaries must have the same keys"
    for key in base_dict.keys():
        if key == 'patches_found':
            continue
        base_dict[key].extend(new_dict[key])

    return base_dict

def convolution_wrapper(signal, window):
    return np.convolve(signal, np.ones(window)/window, mode='same')

def derivative_wrapper(signal, dt):
    return np.diff(signal) / dt

def normalize_steering(value):
    if value < 0:  # Negative normalization
        return (value - (-1.0)) / (0 - (-1.0)) * (-1)  # Scale to [-1, 0]
    else:  # Positive normalization
        return (value - 0) / (0.68 - 0)  # Scale to [0, 1]

def normalize_st_angles(value):
    new_min = -np.pi / 6
    new_max = np.pi / 6
    old_min = -1.0
    old_max = 1.0
    return new_min + (value - old_min) * (new_max - new_min) / (old_max - old_min)

def normalize_std_dev(value, mean, std):
    return (torch.tensor(value).float() - mean) / (std + 0.000006)

def normalize_min_max(value, min, max):
    return (torch.tensor(value).float() - min) / (max - min + 0.000006)

class TurntableDataset(Dataset):
    def __init__(self, root: str, data_stats: str,  seed: int=42):
        torch.manual_seed(seed)
        files = list(Path(root).glob("*.pkl"))
        self.data = dict()
        for file in files:
            with file.open("rb") as f:
                data = pickle.load(f)
            if bool(self.data):
                self.data = merge(self.data, data)
            else:
                self.data = data

        # load stats
        self.stats = None
        with open(data_stats, 'rb') as f:
            self.stats = pickle.load(f)
        
        self.roll_accln_mean = self.stats['roll_accln_mean']
        self.roll_accln_std = self.stats['roll_accln_std']
        self.pitch_accln_mean = self.stats['pitch_accln_mean']
        self.pitch_accln_std = self.stats['pitch_accln_std']
        self.yaw_accln_mean = self.stats['yaw_accln_mean']
        self.yaw_accln_std = self.stats['yaw_accln_std']

    def __len__(self):
        return len(self.data['rpm'])

    def __getitem__(self, idx):
        # Inputs (normalized)
        rpm = normalize_min_max(self.data['rpm'][idx],
                                self.stats['rpm_min'],
                                self.stats['rpm_max'])

        rpm_dot = normalize_std_dev(self.data['rpm_dot'][idx],
                                   self.stats['rpm_dot_mean'],
                                   self.stats['rpm_dot_std'])
        
        steering = torch.tensor(self.data['steering'][idx]).float()

        steering_dot = normalize_std_dev(self.data['steering_dot'][idx],
                                    self.stats['steering_dot_mean'],
                                    self.stats['steering_dot_std'])

        s_roll = torch.sin(torch.tensor(self.data['roll'][idx]).float())
        c_roll = torch.cos(torch.tensor(self.data['roll'][idx]).float())
        s_pitch = torch.sin(torch.tensor(self.data['pitch'][idx]).float())
        c_pitch = torch.cos(torch.tensor(self.data['pitch'][idx]).float())
        s_yaw = torch.sin(torch.tensor(self.data['yaw'][idx]).float())
        c_yaw = torch.cos(torch.tensor(self.data['yaw'][idx]).float())

        roll_vel = torch.tensor(self.data['roll_vel'][idx]).float()
        pitch_vel = torch.tensor(self.data['pitch_vel'][idx]).float()
        yaw_vel = torch.tensor(self.data['yaw_vel'][idx]).float()

        # Outputs (Normalized acclnerations)
        roll_accln_norm = torch.tensor(self.data['roll_accln'][idx]).float()
        pitch_accln_norm = torch.tensor(self.data['pitch_accln'][idx]).float()
        yaw_accln_norm = torch.tensor(self.data['yaw_accln'][idx]).float()

        cur_angles = torch.stack((s_roll, c_roll, s_pitch, c_pitch, s_yaw, c_yaw))
        return (
            rpm,                #model input
            rpm_dot,            #model input
            steering,           #model input
            steering_dot,       #model input   
            cur_angles,         #model input
            torch.stack((roll_accln_norm, pitch_accln_norm, yaw_accln_norm)),  # Normalized outputs
            torch.stack((roll_vel, pitch_vel, yaw_vel)) # velocities that could be used later
        )
    
    def generate_tensor(self, data):
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).float()
        elif isinstance(data, list):
            return torch.tensor(data).float()
        elif isinstance(data, tuple):
            return torch.tensor(data).float()
        elif isinstance(data, torch.Tensor):
            return data.float()


if __name__ == "__main__":
    pass