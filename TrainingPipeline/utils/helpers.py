"""General utility functions"""
from time import time
import functools
import logging

import torch
from omegaconf import OmegaConf, DictConfig
import comet_ml
from datetime import datetime
import pdb

def get_conf(name: str):
    """Returns yaml config file in DictConfig format

    Args:
        name: (str) name of the yaml file
    """
    name = name if name.split(".")[-1] == "yaml" else name + ".yaml"
    cfg = OmegaConf.load(name)
    return cfg


def timeit(fn):
    """Calculate time taken by fn().

    A function decorator to calculate the time a function needed for completion on GPU.
    returns: the function result and the time taken
    """
    # first, check if cuda is available
    cuda = True if torch.cuda.is_available() else False
    if cuda:

        @functools.wraps(fn)
        def wrapper_fn(*args, **kwargs):
            torch.cuda.synchronize()
            t1 = time()
            result = fn(*args, **kwargs)
            torch.cuda.synchronize()
            t2 = time()
            take = t2 - t1
            return result, take

    else:

        @functools.wraps(fn)
        def wrapper_fn(*args, **kwargs):
            t1 = time()
            result = fn(*args, **kwargs)
            t2 = time()
            take = t2 - t1
            return result, take

    return wrapper_fn

def init_logger(cfg: DictConfig):
    """Initializes the cometml logger

    Args:
        cfg: (DictConfig) the configuration
    """
    print(
        f"{datetime.now():%Y-%m-%d %H:%M:%S} - INITIALIZING the logger for experiment {cfg.logger.experiment_name}!"
    )
    logger = None
    cfg_full = cfg
    cfg = cfg.logger
    # Check to see if there is a key in environment:
    EXPERIMENT_KEY = cfg.experiment_key

    # First, let's see if we continue or start fresh:
    CONTINUE_RUN = cfg.resume
    if EXPERIMENT_KEY and CONTINUE_RUN:
        # There is one, but the experiment might not exist yet:
        api = comet_ml.API()  # Assumes API key is set in config/env
        try:
            api_experiment = api.get_experiment_by_id(EXPERIMENT_KEY)
        except Exception:
            api_experiment = None
        if api_experiment is not None:
            CONTINUE_RUN = True
            # We can get the last details logged here, if logged:
            # step = int(api_experiment.get_parameters_summary("batch")["valueCurrent"])
            # epoch = int(api_experiment.get_parameters_summary("epochs")["valueCurrent"])

    if CONTINUE_RUN:
        # 1. Recreate the state of ML system before creating experiment
        # otherwise it could try to log params, graph, etc. again
        # ...
        # 2. Setup the existing experiment to carry on:
        logger = comet_ml.ExistingExperiment(
            previous_experiment=EXPERIMENT_KEY,
            log_env_details=cfg.log_env_details,  # to continue env logging
            log_env_gpu=True,  # to continue GPU logging
            log_env_cpu=True,  # to continue CPU logging
            auto_histogram_weight_logging=True,
            auto_histogram_gradient_logging=True,
            auto_histogram_activation_logging=True,
        )
        # Retrieved from above APIExperiment
        # self.logger.set_epoch(epoch)

    else:
        # 1. Create the experiment first
        #    This will use the COMET_EXPERIMENT_KEY if defined in env.
        #    Otherwise, you could manually set it here. If you don't
        #    set COMET_EXPERIMENT_KEY, the experiment will get a
        #    random key!
        if cfg.online:
            logger = comet_ml.Experiment(
                disabled=cfg.disabled,
                project_name=cfg.project,
                log_env_details=cfg.log_env_details,
                log_env_gpu=True,  # to continue GPU logging
                log_env_cpu=True,  # to continue CPU logging
                auto_histogram_weight_logging=True,
                auto_histogram_gradient_logging=True,
                auto_histogram_activation_logging=True,
            )
            logger.set_name(cfg.experiment_name)
            logger.add_tags(cfg.tags.split())
            logger.log_parameters(cfg_full)
        else:
            logger = comet_ml.OfflineExperiment(
                disabled=cfg.disabled,
                project_name=cfg.project,
                offline_directory=cfg.offline_directory,
                auto_histogram_weight_logging=True,
                log_env_details=cfg.log_env_details,
                log_env_gpu=True,  # to continue GPU logging
                log_env_cpu=True,  # to continue CPU logging
            )
            logger.set_name(cfg.experiment_name)
            logger.add_tags(cfg.tags.split())
            logger.log_parameters(cfg_full)

    return logger


def to_world_torch(Robot_frame, P_relative):
    SE3 = True

    if not isinstance(Robot_frame, torch.Tensor):
        Robot_frame = torch.tensor(Robot_frame, dtype=torch.float32)
    if not isinstance(P_relative, torch.Tensor):
        P_relative = torch.tensor(P_relative, dtype=torch.float32)

    if len(Robot_frame.shape) == 1:
        Robot_frame = Robot_frame.unsqueeze(0)

    if len(P_relative.shape) == 1:
        P_relative = P_relative.unsqueeze(0)
  
    if len(Robot_frame.shape) > 2 or len(P_relative.shape) > 2:
        raise ValueError(f"Input must be 1D for  unbatched and 2D for batched got input dimensions {Robot_frame.shape} and {P_relative.shape}")

    if Robot_frame.shape != P_relative.shape:
        raise ValueError("Input tensors must have same shape")
    
    if Robot_frame.shape[-1] != 6 and Robot_frame.shape[-1] != 3:
        raise ValueError(f"Input tensors 1 must have last dim equal to 6 for SE3 and 3 for SE2 got {Robot_frame.shape[-1]}")
    
    if P_relative.shape[-1] != 6 and P_relative.shape[-1] != 3:
        raise ValueError(f"Input tensors 2 must have last dim equal to 6 for SE3 and 3 for SE2 got {P_relative.shape[-1]}")
    
    if Robot_frame.shape[-1] == 3:
        SE3 = False
        Robot_frame_ = torch.zeros((Robot_frame.shape[0], 6), device=Robot_frame.device, dtype=Robot_frame.dtype)
        Robot_frame_[:, [0,1,5]] = Robot_frame
        Robot_frame = Robot_frame_
        P_relative_ = torch.zeros((P_relative.shape[0], 6), device=P_relative.device, dtype=P_relative.dtype)
        P_relative_[:, [0,1,5]] = P_relative
        P_relative = P_relative_
        
    """ Assemble a batch of SE3 homogeneous matrices from a batch of 6DOF poses """
    batch_size = Robot_frame.shape[0]
    ones = torch.ones_like(P_relative[:, 0])
    transform = torch.zeros_like(Robot_frame)
    T1 = torch.zeros((batch_size, 4, 4), device=Robot_frame.device, dtype=Robot_frame.dtype)
    T2 = torch.zeros((batch_size, 4, 4), device=P_relative.device, dtype=P_relative.dtype)

    R1 = euler_to_rotation_matrix(Robot_frame[:, 3:])
    R2 = euler_to_rotation_matrix(P_relative[:, 3:])
    
    T1[:, :3, :3] = R1
    T2[:, :3, :3] = R2
    T1[:, :3,  3] = Robot_frame[:, :3]
    T2[:, :3,  3] = P_relative[:, :3]
    T1[:,  3,  3] = 1
    T2[:,  3,  3] = 1 

    T_tf = torch.matmul(T2, T1)
    transform[:, :3] = torch.matmul(T1, torch.cat((P_relative[:, :3], ones.unsqueeze(-1)), dim=1).unsqueeze(2)).squeeze(dim=2)[:, :3]
    transform[:, 3:] = extract_euler_angles_from_se3_batch(T_tf)

    if not SE3:
        transform = transform[:, [0,1,5]]

    return transform

def euler_to_rotation_matrix(euler_angles):
    """ Convert Euler angles to a rotation matrix """
    # Compute sin and cos for Euler angles
    cos = torch.cos(euler_angles)
    sin = torch.sin(euler_angles)
    zero = torch.zeros_like(euler_angles[:, 0])
    one = torch.ones_like(euler_angles[:, 0])
    # Constructing rotation matrices (assuming 'xyz' convention for Euler angles)
    R_x = torch.stack([one, zero, zero, zero, cos[:, 0], -sin[:, 0], zero, sin[:, 0], cos[:, 0]], dim=1).view(-1, 3, 3)
    R_y = torch.stack([cos[:, 1], zero, sin[:, 1], zero, one, zero, -sin[:, 1], zero, cos[:, 1]], dim=1).view(-1, 3, 3)
    R_z = torch.stack([cos[:, 2], -sin[:, 2], zero, sin[:, 2], cos[:, 2], zero, zero, zero, one], dim=1).view(-1, 3, 3)

    return torch.matmul(torch.matmul(R_z, R_y), R_x)

def extract_euler_angles_from_se3_batch(tf3_matx):
    # Validate input shape
    if tf3_matx.shape[1:] != (4, 4):
        raise ValueError("Input tensor must have shape (batch, 4, 4)")

    # Extract rotation matrices
    rotation_matrices = tf3_matx[:, :3, :3]

    # Initialize tensor to hold Euler angles
    batch_size = tf3_matx.shape[0]
    euler_angles = torch.zeros((batch_size, 3), device=tf3_matx.device, dtype=tf3_matx.dtype)

    # Compute Euler angles
    euler_angles[:, 0] = torch.atan2(rotation_matrices[:, 2, 1], rotation_matrices[:, 2, 2])  # Roll
    euler_angles[:, 1] = torch.atan2(-rotation_matrices[:, 2, 0], torch.sqrt(rotation_matrices[:, 2, 1] ** 2 + rotation_matrices[:, 2, 2] ** 2))  # Pitch
    euler_angles[:, 2] = torch.atan2(rotation_matrices[:, 1, 0], rotation_matrices[:, 0, 0])  # Yaw

    return euler_angles

def to_robot_torch(Robot_frame, P_relative):
    SE3 = True

    if not isinstance(Robot_frame, torch.Tensor):
        Robot_frame = torch.tensor(Robot_frame, dtype=torch.float32)
    
    if not isinstance(P_relative, torch.Tensor):
        P_relative = torch.tensor(P_relative, dtype=torch.float32)

    if len(Robot_frame.shape) == 1:
        Robot_frame = Robot_frame.unsqueeze(0)

    if len(P_relative.shape) == 1:
        P_relative = P_relative.unsqueeze(0)
  
    if len(Robot_frame.shape) > 2 or len(P_relative.shape) > 2:
        raise ValueError(f"Input must be 1D for  unbatched and 2D for batched got input dimensions {Robot_frame.shape} and {P_relative.shape}")
    
    if Robot_frame.shape != P_relative.shape:
        raise ValueError("Input tensors must have same shape")
    
    if Robot_frame.shape[-1] != 6 and Robot_frame.shape[-1] != 3:
        raise ValueError(f"Input tensors must have last dim equal to 6 for SE3 and 3 for SE2 got {Robot_frame.shape[-1]}")
    
    if Robot_frame.shape[-1] == 3:
        SE3 = False
        Robot_frame_ = torch.zeros((Robot_frame.shape[0], 6), device=Robot_frame.device, dtype=Robot_frame.dtype)
        Robot_frame_[:, [0,1,5]] = Robot_frame
        Robot_frame = Robot_frame_
        P_relative_ = torch.zeros((P_relative.shape[0], 6), device=P_relative.device, dtype=P_relative.dtype)
        P_relative_[:, [0,1,5]] = P_relative
        P_relative = P_relative_
        
    """ Assemble a batch of SE3 homogeneous matrices from a batch of 6DOF poses """
    batch_size = Robot_frame.shape[0]
    ones = torch.ones_like(P_relative[:, 0])
    transform = torch.zeros_like(Robot_frame)
    T1 = torch.zeros((batch_size, 4, 4), device=Robot_frame.device, dtype=Robot_frame.dtype)
    T2 = torch.zeros((batch_size, 4, 4), device=P_relative.device, dtype=P_relative.dtype)

    T1[:, :3, :3] = euler_to_rotation_matrix(Robot_frame[:, 3:])
    T2[:, :3, :3] = euler_to_rotation_matrix(P_relative[:, 3:])
    T1[:, :3,  3] = Robot_frame[:, :3]
    T2[:, :3,  3] = P_relative[:, :3]
    T1[:,  3,  3] = 1
    T2[:,  3,  3] = 1 
    
    T1_inv = torch.inverse(T1)
    tf3_mat = torch.matmul(T2, T1_inv)
    
    transform[:, :3] = torch.matmul(T1_inv, torch.cat((P_relative[:,:3], ones.unsqueeze(-1)), dim=1).unsqueeze(2)).squeeze(dim=2)[:, :3]
    transform[:, 3:] = extract_euler_angles_from_se3_batch(tf3_mat)
    
    if not SE3:
        transform = transform[:, [0,1,5]]
    
    return transform