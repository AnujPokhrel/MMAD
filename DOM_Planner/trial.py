import torch
import traceback
import time
import numpy as np
from functools import lru_cache

def compute_control(self, current_pose, vehicle, last_control):
    try:
        t0 = time.time()
        self.running_cost.zero_()
        cur_pose = current_pose.repeat(self.num_samples, 1).cuda() 
        # current_state = torch.tensor([current_pos.x, current_pos.y, current_yaw, vehicle.GetVehicle().GetSpeed()], device=self.device)
        speed = vehicle.GetVehicle().GetSpeed()
        last_control_all_samples = last_control.repeat(self.num_samples, 1).cuda()

        goal_angle = np.arctan2(self.goal[1] - current_pose[1], self.goal[0] - current_pose[0])
        angle_diff = self.normalize_angle(goal_angle - current_pose[2])
        roll_pitch_vals = torch.zeros((self.num_samples, self.horizon, 2), device=self.device)
        torch.normal(0, self.sigma, out=self.noise)

        for t in range(self.horizon):
            last_control_all_samples = (self.control_sequence[t, 1] + self.noise[t, 1]).cuda()
            last_control_all_samples[:, 0].clamp_(self.min_speed, self.max_speed)
            last_control_all_samples[:, 1].clamp_(self.min_steer_angle, self.max_steer_angle)

            delta_pose = ackermann_bench(last_control_all_samples[:, 0], last_control_all_samples[:, 1], self.wheel_base, self.dt)
            cur_pose = to_world_torch(cur_pose, delta_pose[:, (0,1,5)])

            self.poses[:, t, :] = cur_pose.clone()
            if torch.isnan(cur_pose).any():
                print("NaN is in poses")

            if vehicle is not None:
                roll_pitch = vehicle.GetVehicle().GetRot().Q_to_Euler123()
                roll_pitch_vals[:, t] = torch.tensor([roll_pitch.x , roll_pitch.y], device=self.device)
            
            self.calcualte_parallel_cost(self, self.poses, roll_pitch_vals, last_control_all_samples, t, self.noise[t])
        
        self.running_cost -= torch.min(self.running_cost)
        self.running_cost /= -self.lambda_
        torch.exp(self.running_cost, out=self.running_cost)
        weights = self.running_cost / torch.sum(self.running_cost)

        weights = weights.unsqueeze(1).expand(self.horizon, self.num_samples, 2)
        weights_temp = weights.mul(self.noise)
        self.ctrl_change.copy_(weights_temp.sum(dim=1))
        self.ctrl += self.ctrl_change

        self.ctrl[:, 0].clamp_(self.min_vel, self.max_vel)
        self.ctrl[:, 1].clamp_(self.min_steer_angle, self.max_steer_angle)

        print("MPPI Loop: %4.5f ms" % ((time.time() - t0) * 1000.0))

        return self.ctrl[:, 1], self.ctrl[:, 0]

    except Exception as e:
        print(f"Error in compute Control: {str(e)}")
        traceback.print_exc()
        return 0.0, 1.0

           

def ackermann_bench(throttle, steering, wheel_base=0.324, dt=0.1):
    if not isinstance(throttle, torch.Tensor):
        throttle = torch.tensor(throttle, dtype=torch.float32)
    if not isinstance(steering, torch.Tensor):
        steering = torch.tensor(steering, dtype=torch.float32)
    if not isinstance(wheel_base, torch.Tensor):
        wheel_base = torch.tensor(wheel_base, dtype=torch.float32)
    if not isinstance(dt, torch.Tensor):
        dt = torch.tensor(dt, dtype=torch.float32)
    if throttle.shape != steering.shape:
        raise ValueError("throttle and steering must have the same shape")
    if len(throttle.shape) == 0:
        throttle = throttle.unsqueeze(0)
    
    deltaPose = torch.zeros(throttle.shape[0], 6, dtype=torch.float32)
    
    dtheta = (throttle / wheel_base) * torch.tan(steering) * dt
    dx = throttle * torch.cos(dtheta) * dt
    dy = throttle * torch.sin(dtheta) * dt

    deltaPose[:, 0], deltaPose[:, 1], deltaPose[:, 5] = dx, dy, dtheta

    return deltaPose.squeeze()
            
@lru_cache(maxsize=2048)
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

@lru_cache(maxsize=2048)
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

@lru_cache(maxsize=2048)
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