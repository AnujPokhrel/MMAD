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

def normalize_std_dev(value, mean, std):
    return (torch.tensor(value).float() - torch.tensor(mean).float()) / (torch.tensor(std).float() + 0.000006)

def normalize_min_max(value, min, max):
    return (torch.tensor(value).float() - torch.tensor(min).float()) / (torch.tensor(max).float() - torch.tensor(min).float() + 0.000006)

class RPMDataset(Dataset):
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
        

    def __len__(self):
        return len(self.data['rpm'])

    def __getitem__(self, idx):
        # Inputs (normalized)
        rpm = normalize_min_max(self.data['rpm'][idx],
                                self.stats['rpm_min'],
                                self.stats['rpm_max'])

        steering = torch.tensor(self.data['steering'][idx]).float()

        cmd_vel = normalize_min_max(self.data['cmd_vel_history'][idx],
                                    self.stats['cmd_vel_min'],
                                    self.stats['cmd_vel_max'])
        # Outputs (Normalized acclnerations)
        rpm_dot = normalize_std_dev(self.data['rpm_dot'][idx],
                                   self.stats['rpm_dot_mean'],
                                   self.stats['rpm_dot_std'])

        steering_dot = normalize_std_dev(self.data['steering_dot'][idx],
                                    self.stats['steering_dot_mean'],
                                    self.stats['steering_dot_std'])
        
        ground_truth = torch.stack((rpm_dot, steering_dot))
        cmd_vel = cmd_vel.flatten()
        return (
            rpm,                #model input
            steering,           #model input
            cmd_vel,           #model input
            ground_truth        #model output
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