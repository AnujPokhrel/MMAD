from datetime import datetime
from pathlib import Path
import sys
import argparse
from functools import partial
import copy
import gc

try:
    sys.path.append(str(Path(".").resolve()))
except Exception as e:
    raise RuntimeError("Can't append root directory of the project to the path") from e

from rich import print
import comet_ml
from comet_ml.integration.pytorch import log_model, watch
import numpy as np
import json

# import seaborn as sns
from tqdm import tqdm
import torch
from torchvision import models
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Subset
from icecream import ic, install
import os

from model.model import RpmModel
from model.rpm_dataloader import RPMDataset
from utils.nn import (
    check_grad_norm,
    save_checkpoint,
    load_checkpoint,
    op_counter,
)
from utils.helpers import get_conf, timeit, init_logger, to_world_torch, to_robot_torch
import pdb

class Learner:
    def __init__(self, cfg_dir: str):
        # load config file and initialize the logger and the device
        self.cfg = get_conf(cfg_dir)
        self.cfg.directory.model_name = self.cfg.logger.experiment_name
        self.cfg.directory.model_name += f"--{datetime.now():%m-%d-%H-%M}"
        # self.cfg.directory.model_name += f"-{self.cfg.model.img_backbone.name}-{self.cfg.train_params.loss}-{'pretrained' if self.cfg.model.img_backbone.pretrained else 'random'}-{datetime.now():%m-%d-%H-%M}"
        self.cfg.logger.experiment_name = self.cfg.directory.model_name
        self.cfg.directory.save = str(
            Path(self.cfg.directory.save) / self.cfg.directory.model_name
        )
        if self.cfg.train_params.debug:
            install()
            ic.enable()
            ic.configureOutput(prefix=lambda: f"{datetime.now():%Y-%m-%d %H:%M:%S} |> ")
            torch.autograd.set_detect_anomaly(True)
            self.cfg.logger.disabled = True
        else:
            ic.disable()
            torch.autograd.set_detect_anomaly(True)
            # matplotlib.use("Agg")
        self.logger = init_logger(self.cfg)
        self.logger.log_code(file_name="model/model.py")
        self.logger.log_code(file_name="model/rpm_dataloader.py")
        self.device = self.init_device()

        # fix the seed for reproducibility
        torch.random.manual_seed(self.cfg.train_params.seed)
        torch.cuda.manual_seed(self.cfg.train_params.seed)
        torch.cuda.manual_seed_all(self.cfg.train_params.seed)
        torch.backends.cudnn.benchmark = True
        # ic(self.logger)
        # creating dataset interface and dataloader for trained data
        self.data, self.val_data = self.init_dataloader()
        # create model and initialize its weights and move them to the device
        self.model = self.init_model(self.cfg.model)
        num_params = [x.numel() for x in self.model.parameters()]
        trainable_params = [
            x.numel() for x in self.model.parameters() if x.requires_grad
        ]
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Number of parameters: {sum(num_params) / 1e6:.2f}M")
        print(
            f"{datetime.now():%Y-%m-%d %H:%M:%S} - Number of trainable parameters: {sum(trainable_params) / 1e6:.2f}M"
        )
        # initialize the optimizer
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            **self.cfg.adamw)

        # criterion
        self.criterion = torch.nn.MSELoss() 

        # initialize the learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.cfg.train_params.epochs
        )

        # if resuming, load the checkpoint
        self.val_loss = []
        self.if_resume()

        #Early stopping initialization
        self.early_stopping_enabled = self.cfg.train_params.early_stopping.enabled
        self.early_stopping_patience = self.cfg.train_params.early_stopping.patience
        self.early_stopping_delta = self.cfg.train_params.early_stopping.delta
        self.early_stopping_counter = 0
        self.early_stopping_best = float('inf')

    def train(self):
        """Trains the model"""

        while self.epoch <= self.cfg.train_params.epochs:
            running_loss = []

            bar = tqdm(
                self.data,
                desc=f"Epoch {self.epoch:03}/{self.cfg.train_params.epochs:03}, training: ",
            )
            for data in bar:
                self.iteration += 1
                with self.logger.train():
                    (loss, grad_norm), t_train = (
                        self.forward_batch(data)
                    )
                    t_train /= self.data.batch_size
                    running_loss.append(loss)

                    bar.set_postfix(
                        loss=loss,
                        Grad_Norm=grad_norm,
                        Time=t_train,
                    )

                    self.logger.log_metrics(
                        {
                            "batch_loss": loss,
                            "grad_norm": grad_norm,
                        },
                        epoch=self.epoch,
                        step=self.iteration,
                    )
                
                # Step the scheduler after the optimizer step

            bar.close()

            self.validate()
           
            self.e_loss.append(np.mean(running_loss))

            self.logger.log_metrics(
                {
                    "train_loss": self.e_loss[-1],
                    "val_loss": self.val_loss[-1]
                },
                epoch=self.epoch,
                step=self.iteration,
            )

            self.scheduler.step() 
            print(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} Epoch {self.epoch:03}, "
                + f"Iteration {self.iteration:05} summary: train Loss: "
                + f"[green]{self.e_loss[-1]:.4f}[/green] | Val loss: [red]{self.val_loss[-1]:.4f}[/red] | "
                + f"Current LR: {self.optimizer.param_groups[0]['lr']:.6f} | " 
                + f"Time: {t_train:.6f} seconds\n"
            )
        
            if self.epoch % self.cfg.train_params.save_every == 0 or (
                self.val_loss[-1] < self.best
                and self.epoch >= self.cfg.train_params.start_saving_best
            ):
                self.save()
            
            #Early Stopping conditions
            if self.early_stopping_enabled and self.epoch >= self.cfg.train_params.start_saving_best:
                current_val_loss = self.val_loss[-1]
                if current_val_loss < (self.early_stopping_best - self.early_stopping_delta):
                    self.early_stopping_best = current_val_loss
                    self.early_stopping_counter = 0
                else:
                    self.early_stopping_counter += 1
                    if self.early_stopping_counter >= self.early_stopping_patience:
                        print(f"Early stopping triggered after {self.early_stopping_patience} epochs without improvement.")
                        break

            gc.collect()
            torch.cuda.empty_cache()
            self.epoch += 1
        
        data = next(iter(self.data))
        rpm = data[0].to(device=self.device).view(-1, 1)
        steering = data[1].to(device=self.device).view(-1, 1)
        cmd_vel = data[2].to(device=self.device)

        macs, params = op_counter(self.model, sample=(rpm, steering, cmd_vel))
        print("macs = ", macs, " | params = ", params)
        with open(Path(self.cfg.directory.save) / "loss.json", "w") as f:
            json.dump(self.e_loss, f)

        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Training is DONE!")

    #myway
    @timeit
    def forward_batch(self, data):
        """Forward pass of a batch"""
        self.model.train()
        rpm = data[0].to(device=self.device).view(-1, 1)
        steering = data[1].to(device=self.device).view(-1, 1)
        cmd_vel_history = data[2].to(device=self.device)
        gt_roll_pitch_dot = data[3].to(device=self.device)

        roll_pitch_dot_pred_norm = self.model(rpm, steering, cmd_vel_history)
        loss = self.criterion(roll_pitch_dot_pred_norm, gt_roll_pitch_dot)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.cfg.train_params.grad_clipping > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.train_params.grad_clipping
            )
        
        grad_norm = check_grad_norm(self.model)

        return loss.detach().item(), grad_norm
    
    #myway
    @timeit
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        running_loss = []
        
        bar = tqdm(
            self.val_data,
            desc=f"Epoch {self.epoch:03}/{self.cfg.train_params.epochs:03}, validating"
        )

        for data in bar:
            rpm = data[0].to(device=self.device).view(-1, 1)
            steering = data[1].to(device=self.device).view(-1, 1)
            cmd_vel_history = data[2].to(device=self.device)
            gt_roll_pitch_dot = data[3].to(device=self.device)

            roll_pitch_dot_pred_norm = self.model(rpm, steering, cmd_vel_history)
            loss = self.criterion(roll_pitch_dot_pred_norm, gt_roll_pitch_dot)

            running_loss.append(loss.item())
            bar.set_postfix(loss=loss.item())
        
        bar.close()
        loss = np.mean(running_loss)
        self.val_loss.append(loss)

    #myway
    def init_device(self):
        """Initializes the device"""
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the device!")
        is_cuda_available = torch.cuda.is_available()
        device = self.cfg.train_params.device

        if "cpu" in device:
            print(f"Performing all the operations on CPU.")
            return torch.device(device)

        elif "cuda" in device:
            if is_cuda_available:
                device_idx = device.split(":")[1]
                if device_idx == "a":
                    print(
                        f"Performing all the operations on CUDA; {torch.cuda.device_count()} devices."
                    )
                    self.cfg.dataloader.batch_size *= torch.cuda.device_count()
                    return torch.device(device.split(":")[0])
                else:
                    print(f"Performing all the operations on CUDA device {device_idx}.")
                    return torch.device(device)
            else:
                print("CUDA device is not available, falling back to CPU!")
                return torch.device("cpu")
        else:
            raise ValueError(f"Unknown {device}!")

    #myway
    def init_dataloader(self):
        """Initializes the dataloaders"""
        print(
            f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the train and val dataloaders!"
        )
        dataset = RPMDataset(**self.cfg.dataset)
        val_dataset = RPMDataset(**self.cfg.val_dataset)
        # creating dataset interface and dataloader for val data
        if self.cfg.train_params.debug:
            dataset = Subset(dataset, list(range(self.cfg.dataloader.batch_size * 2)))
            val_dataset = Subset(
                val_dataset, list(range(self.cfg.dataloader.batch_size * 2))
            )

        data = DataLoader(dataset, **self.cfg.dataloader)
        self.cfg.dataloader.update({"shuffle": False})  # for val dataloader
        val_data = DataLoader(val_dataset, **self.cfg.dataloader)

        # log dataset status
        self.logger.log_parameters(
            {"train_len": len(dataset), "val_len": len(val_dataset)}
        )
        print(
            f"Training consists of {len(dataset)} samples, and validation consists of {len(val_dataset)} samples."
        )
        return data, val_data 

    #myway
    def if_resume(self):
        if self.cfg.logger.resume:
            # load checkpoint
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - LOADING checkpoint!!!")
            save_dir = self.cfg.directory.load
            checkpoint = load_checkpoint(save_dir, self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            # self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            self.epoch = checkpoint["epoch"] + 1
            self.e_loss = checkpoint["e_loss"]
            self.iteration = checkpoint["iteration"] + 1
            self.best = checkpoint["best"]
            print(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} "
                + f"LOADING checkpoint was successful, start from epoch {self.epoch}"
                + f" and loss {self.best}"
            )
        else:
            self.epoch = 1
            self.iteration = 0
            self.best = np.inf
            self.e_loss = []

        self.logger.set_epoch(self.epoch)

    #myway
    def save(self, name=None):
        model = self.model
        if isinstance(self.model, torch.nn.DataParallel):
            model = model.module

        checkpoint = {
            "time": str(datetime.now()),
            "epoch": self.epoch,
            "iteration": self.iteration,
            "model": model.state_dict(),
            "model_name": type(model).__name__,
            "optimizer": self.optimizer.state_dict(),
            "optimizer_name": type(self.optimizer).__name__,
            # "lr_scheduler": self.lr_scheduler.state_dict(),
            "best": self.best,
            "e_loss": self.e_loss,
        }

        if name is None:
            save_name = f"{self.cfg.directory.model_name}-E{self.epoch}"
        else:
            save_name = name

        if self.val_loss[-1] < self.best:
            self.best = self.val_loss[-1].item()
            checkpoint["best"] = self.best
            save_checkpoint(checkpoint, True, self.cfg.directory.save, save_name)
            if self.cfg.logger.upload_model:
                # upload only the current checkpoint
                log_model(self.logger, checkpoint, model_name=save_name)
        else:
            save_checkpoint(checkpoint, False, self.cfg.directory.save, save_name)
            if self.cfg.logger.upload_model:
                # upload only the current checkpoint
                log_model(self.logger, checkpoint, model_name=save_name)

    #myway
    def init_model(self, cfg):
        """Initializes the model"""
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the model!")

        #load the model
        model = RpmModel()

        return model.to(self.device) 

 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", default="conf/config_rpm", type=str)
    args = parser.parse_args()
    cfg_path = args.conf
    print(cfg_path)
    learner = Learner(cfg_path)
    learner.train()