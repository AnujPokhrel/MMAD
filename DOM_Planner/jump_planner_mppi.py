import rospy
import os
import time
import numpy as np
import torch
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from callbacks import ImuProcessor, LosiManager
from utils import *
from functools import lru_cache 
from nn_model import RollPitchAcclnModel
import pickle
import matplotlib.pyplot as plt
import matplotlib
from planners import MPPI

plantime = 2.0
class JumpPlanner:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device to run the planner
        print(f"Device set to: {self.device}\n")
        rospy.init_node("jump_planner")                                             # Initialize ROS node
        self.witIMU = ImuProcessor(imu_topic="/losi/witmotion_imu/imu",             # Initialize IMU processor
                              mag_topic="/losi/witmotion_imu/magnetometer", 
                              sampling_rate=200,
                              method='madgwick', use_mag=False, tait_bryan=False)       
        self.losi = LosiManager()                                                            # Initialize LOSI manager
        self.path_pub = rospy.Publisher("/MPPI_paths", Path, queue_size=10)                # Path publisher
        self.stats_pub = rospy.Publisher("/planner_stats", Float32MultiArray, queue_size=10) # Path publisher
        
        self.action_stat = Float32MultiArray(data=np.zeros((1000), dtype=np.float32).tolist())

        '''load model and stats file for the neural net model
        model: RollPitchAcclnModel
            input:  [rpm, rpm_dot, steering, steering_dot, 
                    [sin(roll), cos(roll), sin(pitch), cos(pitch), sin(yaw), cos(yaw)]
                    roll_dot, pitch_dot, yaw_dot]
               
            output: [roll_accln, pitch_accln, yaw_accln]
        '''
        with open("DOM_Planner/stats_physics.pkl", "rb") as f:
            self.stats = pickle.load(f)

        # model_path = "DOM_Planner/precession_all_data--02-16-00-46/shifted_accln_gt--02-22-14-29-E54-best.pth"
        model_path = "DOM_Planner/precession_all_data--02-16-00-46/inc_manual_data--02-20-16-39-E52-best.pth"
        self.nnModel = RollPitchAcclnModel()
        self.nnModel.load_state_dict(torch.load(model_path)['model'])
        self.nnModel.to(self.device)
        self.nnModel.eval()

        # self.stats['rpm_dot_min'] = -5000
        # self.stats['rpm_dot_max'] = 5000


        self.time_step = 0.1                                                           # Time step for the planner
        self.planner_rate = 10                                                          # Planner rate
        self.plan_horizon = int(plantime/self.time_step)                        # Planning horizon in steps
        self.pose_count = 10                                                            # Pose count for the path publisher
        self.desc_max = -self.stats['rpm_dot_min']                                      # Maximum deceleration
        self.acc_max = self.stats['rpm_dot_max']                                        # Maximum acceleration
        self.str_dot_max = self.stats['steering_dot_max']                               # Maximum steering acceleration in rad/s^2

        self.action_graph = torch.zeros((1000, 2), dtype=torch.float32, device=self.device)  # Action graph for the planner

        '''State variables 
        [   0    1      2       3       4       5       6       7  ]
        [roll, pitch, yaw, delRoll, delPitch, delYaw, rpm, steering]
        '''
        self.state_space = torch.zeros((8,), dtype=torch.float32, device=self.device, requires_grad=False)         

        self.state_limits = torch.tensor([[0, -1], [self.stats['rpm_max'], 1]],             # State limits
                                        dtype=torch.float32, 
                                        device=self.device)
        
        self.best_action = torch.zeros((2,),dtype=torch.float32,                        # Best action from the planner
                                        device=self.device, requires_grad=False)   

        self.action_limits = torch.tensor([[self.stats['rpm_dot_min'], -6.5], [self.stats['rpm_dot_max'], 6.5]],             # State limits
                                     dtype=torch.float32, 
                                     device=self.device)

        self.mppi = MPPI(
            dynamics_func=self.get_next_state,
            cost_func=self.cost_function,
            nx = 8,
            nu = 2,
            horizon = 30, #self.plan_horizon,
            num_samples = 2000,
            lambda_= 0.3,
            u_min = self.action_limits[0],
            u_max = self.action_limits[1],
            noise_sigma=2000,
            device=self.device,
            adaptive_noise=False,
            return_trajectory=True
        )

        self.near_threshold = torch.tensor([[50, 0.15], [50, 0.15]], dtype=torch.float32, device=self.device) 
        self.acc_cache = torch.zeros((10, 2), dtype=torch.float32, device=self.device)                      
        self.action_cache = torch.zeros((10, self.mppi.num_samples, 2), dtype=torch.float32, device=self.device)                      
        self.init_cache_cnt = 0                                                         # Cache initialization flag
        
        
        # Goal states
        self.goal_angles = [[-0.5,  2.7,  -0.59, -2.35,  0.0, 0.0, 0.0,],            #Goal roll angles for the planner
                            [-0.3, -0.4,  0.75,   0.67, 0.0, 0.0, 0.0],              #Goal pitch angles for the planner
                            [0.14, -3.0, -0.46, -2.68, 0.0, 0.0, 0.0]]              #Goal yaw angles for the planner

        self.goal_state = torch.tensor([self.goal_angles[0][0], 
                                        self.goal_angles[1][0],
                                        self.goal_angles[2][0],        #Goal Angles
                                        0.0, 0.0, 0.0,                 #Goal angular velocities
                                        500.0, 0.0 ],                  #Goal RPM and steering angle
                                        dtype=torch.float32, 
                                        device=self.device, requires_grad=False)               
        
        self.goal_fail_ctr = 0
        self.goal_index = 1                                                             # Index of goal pitch
        self.plan_ctr = 0                                                               # counter for holding the position at goal for 3 sec
        self.action_idx = 0
        self.best_trajectory = torch.zeros((10, 8), dtype=torch.float32, device=self.device)

        # Planner timer at Event
        self.losi.ctrl_cmd['Thr_Cmd'] = 0.3
        self.losi.ctrl_cmd['Str_Cmd'] = 0.0
        self.losi.send_cmd()

        '''PLanner stats publisher  [[Current_roll, Current_pitch, Current_yaw, roll_vel,  pitch_vel,  yaw_vel, RPM, Steering],
                                    [Goal_roll,     Goal_pitch,    Goal_yaw,    roll_vel,  pitch_vel,  yaw_vel, RPM, Steering],
                                    [diff_roll,     diff_pitch,    diff_yaw,    droll_vel, dpitch_vel, dyaw_vel, dRPM, dSteering],
                                    [P_roll,        P_pitch,        P_yaw,     p_roll_vel, p_pitch_vel, p_yaw_vel, RPM, Steering],
                                    [Throttle_cmd, Steering_cmd,    Time,      Goal_index, 0,           0,         0,   0      ]]
        '''
        self.planner_stats = Float32MultiArray(data=[0, 0])

        input("Press Enter to start the planner")
        rospy.Timer(rospy.Duration(1.0 / self.planner_rate), self.run_planner) 
        rospy.Timer(rospy.Duration(self.time_step), self.get_cmd) 
        rospy.spin()

    @torch.no_grad()
    def get_next_state(self, state, action, dt):
        '''Get the next state from the current state and action
        state: torch.tensor [batch_size, 8]
            [roll, pitch, yaw, roll_vel, pitch_vel, yaw_vel, rpm, steering]
        action: torch.tensor [batch_size, 2]
            [rpm_dot, steering_dot]
        dt: float   
            time step
        '''
        ###############################################################
        # Extract the components from the current state and action
        ###############################################################
        next_state = torch.zeros_like(state)
        
        x_angles = state[:, :3]
        x_angular_vels = state[:, 3:6]
        x_rpm = state[:, 6]
        x_steering = state[:, 7]

        u_rpm_dot = action[:, 0]
        u_steering_dot = action[:, 1]

        x_angle_components = torch.zeros_like(state[:, :6])
        x_angle_components[:, [0,2,4]] = torch.sin(x_angles)
        x_angle_components[:, [1,3,5]] = torch.cos(x_angles)

        ################################################################################################
        # calculate the realizable action values based on the current state and action values
        ################################################################################################
        '''
        The realizable action values are calculated moving average of last acceleration vs current acceleration
        '''
        alpha_thr = 0.1
        alpha_str = 0.1
        u_rpm_dot = (1-alpha_thr) * self.action_cache[-1, :, 0] + alpha_thr * u_rpm_dot
        u_steering_dot = (1-alpha_str) * self.action_cache[-1, :,  1] + alpha_str * u_steering_dot
        self.action_cache[-1, :, 0] = u_rpm_dot
        self.action_cache[-1, :, 1] = u_steering_dot


        ##########################################################################
        # Clip the action values based on the current state limits
        ##########################################################################
        mask_speed_low = 0 - x_rpm >= (self.action_limits[0, 0]*dt)
        u_rpm_dot[mask_speed_low] = torch.maximum(u_rpm_dot[mask_speed_low], 
                                                0-x_rpm[mask_speed_low]/dt)
        
        mask_speed_high = 1980.0 - x_rpm <= (self.action_limits[1, 0]*dt)
        u_rpm_dot[mask_speed_high] = torch.minimum(u_rpm_dot[mask_speed_high], 
                                                    (1980.0 - x_rpm[mask_speed_high])/dt)


        mask_str_low = -1 - x_steering >= -self.action_limits[0, 1]*dt
        u_steering_dot[mask_str_low] = torch.maximum(u_steering_dot[mask_str_low], 
                                                    (-1 - x_steering[mask_str_low])/dt)

        mask_srt_high = 1 - x_steering <= self.action_limits[1, 1]*dt
        u_steering_dot[mask_srt_high] = torch.minimum(u_steering_dot[mask_srt_high], 
                                                     (1 - x_steering[mask_srt_high])/dt)

        ##############################################################################################
        # Update the next state RPM and steering values based on the action values
        ##############################################################################################
        '''
        This is simply the integration of the action values with the current state
        x_rpm{t+1} = x_rpm{t} + u{t}*dt
        x_steering{t+1} = x_steering{t} + u{t}*dt
        '''
        next_state[:, 6] = x_rpm + u_rpm_dot*dt
        next_state[:, 7] = x_steering + u_steering_dot*dt

        ##############################################################################################
        # Normalize the state and action values based on the stats
        ##############################################################################################
        x_rpm = (x_rpm - self.stats['rpm_min']) / (self.stats['rpm_max'] - self.stats['rpm_min'])
        u_rpm_dot = (u_rpm_dot - self.stats['rpm_dot_mean']) / self.stats['rpm_dot_std']
        u_steering_dot = (u_steering_dot - self.stats['steering_dot_mean']) / self.stats['steering_dot_std']
        # x_angular_vels[:, 0] = (x_angular_vels[:, 0] - self.stats['roll_vel_mean']) / self.stats['roll_vel_std']
        # x_angular_vels[:, 1] = (x_angular_vels[:, 1] - self.stats['pitch_vel_mean']) / self.stats['pitch_vel_std']
        # x_angular_vels[:, 2] = (x_angular_vels[:, 2] - self.stats['yaw_vel_mean']) / self.stats['yaw_vel_std']
        
        ##############################################################################################
        # Get the acceleration values from the neural network model and unnormalize
        ##############################################################################################
        rpy_acc = self.nnModel(rpm = x_rpm.unsqueeze(1),
                               rpm_dot = u_rpm_dot.unsqueeze(1),
                               steering = x_steering.unsqueeze(1),
                               steering_dot = u_steering_dot.unsqueeze(1),
                               angles = x_angle_components,
                               angular_vels = x_angular_vels)
        # rpy_acc[:, 0] = rpy_acc[:, 0] * self.stats['roll_accln_std'] + self.stats['roll_accln_mean']
        # rpy_acc[:, 1] = rpy_acc[:, 1] * self.stats['pitch_accln_std'] + self.stats['pitch_accln_mean']
        # rpy_acc[:, 2] = rpy_acc[:, 2] * self.stats['yaw_accln_std'] + self.stats['yaw_accln_mean']
        
        ##############################################################################################
        # Update the next state angular velocities based on the acceleration values
        # The next state angles based on the angular displacements using SE3 transformations
        ##############################################################################################           
        '''
        For the Angular velocities and Angular displacements we use the simple kinematic equations
        x_vel{t+1} = x_vel{t} + x_acc{t} * dt
        x_ang_displacement{t+1} = x_vel{t} * dt + 0.5 * x_acc{t} * dt^2 

        The angular displacements are in the current state robot frame, 
        we need to convert them to the world frame using 6DOF transformations
        '''
        next_state[:, 3:6] = x_angular_vels + rpy_acc * dt
        ang_displacements  = x_angular_vels * dt + 0.5 * rpy_acc * dt**2
        
        # for 6DoF transformations the cur_state and delta_state has to be in the form [x, y, z, roll, pitch, yaw,]
        cur_state = torch.zeros_like(state[:, :6])
        delta_state = torch.zeros_like(state[:, :6])
        cur_state[:, 3:] = x_angles
        delta_state[:, 3:] = ang_displacements
        
        next_state[:, :3] = to_world_torch(cur_state, delta_state)[:, 3:]

        return next_state

    @torch.no_grad()
    def cost_function(self, x_state, x_next, u, t, horizon, dt):
        '''Cost function for the MPPI
        x_state: torch.tensor   [batch_size, 8]
            [roll, pitch, yaw, roll_vel, pitch_vel, yaw_vel, rpm, steering]
        x_next: torch.tensor    [batch_size, 8]
            [roll, pitch, yaw, roll_vel, pitch_vel, yaw_vel, rpm, steering]
        u: torch.tensor         [batch_size, 2]
            [throttle, steering]
        t: int                  
            time step
        horizon: int
            planning horizon
        dt: float
            time step
        '''
        ################################################################################################
        # Extract the components from the current state, next state, goal state and action
        ################################################################################################
        # x_roll = x_state[:, 0]
        # x_pitch = x_state[:, 1]
        # x_yaw = x_state[:, 2]
        # x_roll_rate = x_state[:, 3]
        # x_pitch_rate = x_state[:, 4]
        # x_yaw_rate = x_state[:, 5]
        # x_rpm = x_state[:, 6]
        # x_steering = x_state[:, 7]

        # x_next_roll = x_next[:, 0]
        # x_next_pitch = x_next[:, 1]
        # x_next_yaw = x_next[:, 2]
        x_next_roll_rate = x_next[:, 3]
        x_next_pitch_rate = x_next[:, 4]
        x_next_yaw_rate = x_next[:, 5]
        x_next_rpm = x_next[:, 6]
        x_next_steering = x_next[:, 7]

        # x_goal_roll = self.goal_state[0]
        # x_goal_pitch = self.goal_state[1]
        # x_goal_yaw = self.goal_state[2]
        x_goal_roll_rate = self.goal_state[3]
        x_goal_pitch_rate = self.goal_state[4]
        x_goal_yaw_rate = self.goal_state[5]
        x_goal_rpm = self.goal_state[6]
        x_goal_steering = self.goal_state[7]
                
        u_rpm = u[:, 0]
        u_steering = u[:, 1]

        horizon = min(horizon, self.plan_horizon)

        is_final = (t >= horizon - 1)

        if is_final:
            r_weight = 0.0
            p_weight = 1.0
            y_weight = 1.0
            rr_weight = 0.0
            pr_weight = 1.0
            yr_weight = 0.0
            rpm_weight = 0.0
            steering_weight = 0.0
            rpm_penalty_weight = 0.0
            steering_penalty_weight = 0.0
        else:
            r_weight = 0.0
            p_weight = 1.0
            y_weight = 1.0
            rr_weight = 0.0
            pr_weight = 1.0
            yr_weight = 0.0
            rpm_weight = 0.0
            steering_weight = 0.0
            rpm_penalty_weight = 0.0
            steering_penalty_weight = 0.0

        cost_max = r_weight + p_weight + y_weight + rr_weight + pr_weight + yr_weight + rpm_weight + steering_weight + rpm_penalty_weight + steering_penalty_weight
        r_norm          = 2 * torch.pi
        p_norm          = torch.pi
        y_norm          = 2 * torch.pi
        rr_norm         = 2 * max(self.stats['roll_vel_max'], -self.stats['roll_vel_min'])
        pr_norm         = 2 * max(self.stats['pitch_vel_max'], -self.stats['pitch_vel_min'])
        yr_norm         = 2 * max(self.stats['yaw_vel_max'], -self.stats['yaw_vel_min'])       
        rpm_norm        = 2 * self.state_limits[1, 0]
        steering_norm   = 2 * self.state_limits[1, 1]

        u_rpm_norm      = self.action_limits[1, 0] - self.action_limits[0, 0]
        u_steering_norm = self.action_limits[1, 1] - self.action_limits[0, 1]



        total_cost = torch.zeros_like(x_state[:, 0])        # Initialize the total cost [batch_size]
        ################################################################################################
        # Calculate the angle diff by 6DOF transformations
        ################################################################################################
        '''
        The angle difference is calculated by converting the current state and goal state to the robot frame
        and then calculating the difference in the angles
        '''
        x_next_expanded = torch.zeros_like(x_state[:, :6])
        g_state_expanded = torch.zeros_like(x_state[:, :6])
        x_next_expanded[:, 3:] = x_state[:, :3]
        g_state_expanded[:, 3:] = self.goal_state[:3]
        ang_diffs = to_robot_torch(x_next_expanded, g_state_expanded)[:, 3:]

        ################################################################################################
        # Calculate the cost based on the angle differences
        ################################################################################################
        '''
        The cost is calculated based on the angle differences multiplied by angle weights
        '''
        total_cost      += r_weight * (ang_diffs[:, 0]/r_norm)**2
        total_cost      += p_weight * (ang_diffs[:, 1]/p_norm)**2
        total_cost      += y_weight * (ang_diffs[:, 2]/y_norm)**2
        total_cost      += rr_weight * ((x_next_roll_rate - x_goal_roll_rate)/rr_norm)**2
        total_cost      += pr_weight * ((x_next_pitch_rate - x_goal_pitch_rate)/pr_norm)**2
        total_cost      += yr_weight * ((x_next_yaw_rate - x_goal_yaw_rate)/yr_norm)**2
        total_cost      += rpm_weight * ((x_next_rpm - x_goal_rpm)/rpm_norm)**2
        total_cost      += steering_weight * ((x_next_steering - x_goal_steering)/steering_norm)**2

        ################################################################################################
        # Calculate the penalty for the states at the limits
        ################################################################################################
        '''
        The penalty is calculated based on the states at the limits for RPM and steering
        '''
        rpm_penalty = torch.zeros_like(u_rpm)
        steering_penalty = torch.zeros_like(u_steering)
        near_max_rpm = (x_next_rpm >= self.state_limits[1, 0]-self.near_threshold[1, 0])
        near_min_rpm = (x_next_rpm <= self.state_limits[0, 0]+self.near_threshold[0, 0])
        near_max_steering = (x_next_steering >= self.state_limits[1, 1]-self.near_threshold[1, 1])
        near_min_steering = (x_next_steering <= self.state_limits[0, 1]+self.near_threshold[0, 1])

        penalty_mask = near_max_rpm & (u_rpm > 0)
        rpm_penalty[penalty_mask] += (u_rpm[penalty_mask] / u_rpm_norm)**2 

        penalty_mask = near_min_rpm & (u_rpm < 0)
        rpm_penalty[penalty_mask] += (u_rpm[penalty_mask] / u_rpm_norm)**2

        penalty_mask = near_max_steering & (u_steering > 0)
        steering_penalty[penalty_mask] += (u_steering[penalty_mask] / u_steering_norm)**2
        
        penalty_mask = near_min_steering & (u_steering < 0)
        steering_penalty[penalty_mask] += (u_steering[penalty_mask] / u_steering_norm)**2


        total_cost += rpm_penalty_weight * rpm_penalty
        total_cost += steering_penalty_weight * steering_penalty

        ################################################################################################
        # Calculate the reward based on the time left
        ################################################################################################
        '''
        The reward is calculated based on the mean of the total cost as max reward multiplied with the time left   
        '''
        # if horizon > t:
        #     total_cost -= 2/(cost_max * horizon) * (horizon - t) * torch.clamp((total_cost.mean() - total_cost), 0, total_cost.mean()) 


        return total_cost

    def execute_cmd(self, action):
        if action[0] > 0:
            thr = map_range(action[0], 0.0, self.acc_max/10,  0.0, 1.0)/self.planner_rate
        else:
            thr = map_range(action[0], -self.desc_max/10, 0.0, -1.0, 0.0)/self.planner_rate

        strng = map_range(action[1], -6.5, 6.5, -1.0, 1.0)/self.planner_rate
        #why?
        if self.losi.losi_stats_data['RPM_avg'] < 20:
            thr = np.clip(thr, 0.0, 1.0)

        self.losi.ctrl_cmd['Thr_Cmd'] = np.clip(self.losi.ctrl_cmd['Thr_Cmd'] + thr,  -0.2 , 1.0)
        self.losi.ctrl_cmd['Str_Cmd'] = 0*np.clip(self.losi.ctrl_cmd['Str_Cmd'] + strng,  -1.0 , 1.0)
        self.losi.send_cmd()

    def print_table(self, r_state, f_state, g_state, action, rtime, ltime):
        # Print header with column widths
        os.system('cls' if os.name == 'nt' else 'clear')
        lines = f"|{('-' * 13):<13}|{('-' * 40):<40}|{('-' * 40):<40}|{('-' * 27):<27}|"
        print(lines)
        print(f"|{'':<12} | {'roll':<12} {'pitch':<12} {'yaw':<12} | {'roll_vel':<12} {'pitch_vel':<12} {'yaw_vel':<12} | {'rpm':<12} {'steering':<12} |")
        print(lines)

        # Print current state values
        print(f"|{'Current:':<12} | {r_state[0]:<12.2f} {r_state[1]:<12.2f} {r_state[2]:<12.2f} | "
            f"{r_state[3]:<12.2f} {r_state[4]:<12.2f} {r_state[5]:<12.2f} | {r_state[6]:<12.2f} {r_state[7]:<12.2f} |")
        
        rt_state = torch.zeros_like(r_state[:6])
        rt_state[3:] = r_state[:3]
        
        gt_state = torch.zeros_like(g_state[:6])
        gt_state[3:] = g_state[:3]

        ang_diffs = to_robot_torch(rt_state, gt_state).squeeze()[3:]

        print(f"|{'Goal:':<12} | {g_state[0]:<12.2f} {g_state[1]:<12.2f} {g_state[2]:<12.2f} | "
            f"{g_state[3]:<12.2f} {g_state[4]:<12.2f} {g_state[5]:<12.2f} | {g_state[6]:<12.2f} {g_state[7]:<12.2f} |")
        print(lines)
        print(f"|{'Ang_diff:':<12} | {ang_diffs[0]:<12.2f} {ang_diffs[1]:<12.2f} {ang_diffs[2]:<12.2f} | "
            f"{0.0:<12.2f} {0.0:<12.2f} {0.0:<12.2f} | {0.0:<12.2f} {0.0:<12.2f} |")
        print(lines)
        # Print planned state values
        print(f"|{'Planned:':<12} | {f_state[0]:<12.2f} {f_state[1]:<12.2f} {f_state[2]:<12.2f} | "
            f"{f_state[3]:<12.2f} {f_state[4]:<12.2f} {f_state[5]:<12.2f} | {f_state[6]:<12.2f} {f_state[7]:<12.2f} |")

        # Print footer
        print(lines)
        print(f"{'|Throttle_cmd |':<20} {action[0]:<34.2f}| {'Steering_cmd':<20}: {action[1]:<17.2f}| {'Time':<20} {rtime:<5.1f}|")
        print(lines)
        ltime = ltime + f" \t\t\t\tGoal Index: {self.goal_index:<2}"
        print(f"|{ltime:^101}|")
        print(lines)

        diffs = torch.zeros_like(r_state)
        diffs = r_state - g_state 
        diffs[:3] = ang_diffs

        stat_cache = np.zeros((5, 8), dtype=np.float32)

        for i in range(8):
            stat_cache[0: i] = r_state[i].cpu().numpy().item()
            stat_cache[1: i] = g_state[i].cpu().numpy().item()
            stat_cache[2: i] = diffs[i].cpu().numpy().item()
            stat_cache[3: i] = f_state[i].cpu().numpy().item() 

        stat_cache[4: 0] = action[0].cpu().numpy().item()
        stat_cache[4: 1] = action[1].cpu().numpy().item()
        stat_cache[4: 2] = float(rtime)
        stat_cache[4: 3] = float(self.goal_index)

        self.planner_stats.data = stat_cache.flatten().tolist()

        self.stats_pub.publish(self.planner_stats)

    def get_cmd(self):
        
        thr = self.mppi.mean_control[self.action_idx, 0]
        strng = self.mppi.mean_control[self.action_idx, 1]
        
        alpha_c = 0.1
        self.best_action[0] = alpha_c * thr +  (1-alpha_c) * self.best_action[0]
        self.best_action[1] = alpha_c * thr +  (1-alpha_c) * self.best_action[1]
        self.action_graph = torch.roll(self.action_graph, -1, 0)
        self.action_graph[-1, 0] = self.best_action[0]
        self.action_stat.data = thr.flatten().cpu().numpy().tolist() #self.action_graph[-1, :].flatten().cpu().numpy().tolist()

        self.execute_cmd(self.best_action.cpu().numpy())

        ltime = f"loop time: -NA- millisec"
        self.stats_pub.publish(self.action_stat)
        self.print_table(self.state_space, self.best_trajectory[], self.goal_state, self.best_action, self.plan_horizon*self.time_step, ltime)       

    def run_planner(self, event):

        if not self.witIMU.initialized or not self.losi.initialized:
            print(f"IMU Initialized:{self.witIMU.initialized}, LOSI initialized: {self.losi.initialized}")
            self.losi.send_cmd()
            return  
        
        if self.init_cache_cnt < 10: 
            self.acc_cache = torch.roll(self.acc_cache, -1, 0)
            self.acc_cache[-1, 0] = self.losi.losi_stats_data['accln']
            self.acc_cache[-1, 1] = self.losi.losi_stats_data['str_vel'] 
            self.init_cache_cnt += 1
            self.losi.send_cmd()
            return

        start_time = time.time()
        self.state_space[0:3] = torch.tensor(self.witIMU.angles, dtype=torch.float32, device=self.device)
        self.state_space[3:6] = torch.tensor(self.witIMU.velocities, dtype=torch.float32, device=self.device)
        self.state_space[6] = self.losi.losi_stats_data['RPM_avg']
        self.state_space[7] = self.losi.losi_stats_data['Str_Sens']

        self.acc_cache = torch.roll(self.acc_cache, -1, 0)
        self.acc_cache[-1, 0] = self.losi.losi_stats_data['accln']
        self.acc_cache[-1, 1] = self.losi.losi_stats_data['str_vel']
        self.action_cache = self.acc_cache.unsqueeze(1).repeat(1, self.mppi.num_samples, 1)

        planning_time = self.plan_horizon       # Planning horizon in steps = time in sec / time_step 

        if planning_time <= 2 or self.plan_ctr > 0:
            # print("Plan time zero and checking goal")
            self.goal_process()
            # return
        
        if self.mppi.return_trajectory:
            action, trajectories, best_index = self.mppi.command(self.state_space, 0.1)
            self.best_trajectory = trajectories[:, best_index, :]
        else:
            action = self.mppi.command(self.state_space, 0.1)

        self.action_idx = 0
        # if abs(action[0]) < 200:
        #     action[0] = 0

        # alpha_c = 0.1
        # self.best_action[0] = alpha_c * action[0] +  (1-alpha_c) * self.best_action[0]
        # self.best_action[1] = alpha_c * action[1] +  (1-alpha_c) * self.best_action[1]
        # self.action_graph = torch.roll(self.action_graph, -1, 0)
        # self.action_graph[-1, 0] = self.best_action[0]
        # self.action_stat.data = action[0].flatten().cpu().numpy().tolist() #self.action_graph[-1, :].flatten().cpu().numpy().tolist()

        # self.stats_pub.publish(self.action_stat)

        # publish_path(trajectory=best_trajectory.cpu().numpy(), pub=self.path_pub, p_count=self.pose_count)
        # ltime = f"loop time {((time.time() - start_time)*1e3):<4.1f} millisec"
        # best_trajectory = torch.zeros((10, 8), dtype=torch.float32, device=self.device)
        
        # self.execute_cmd(self.best_action.cpu().numpy())
        # self.losi.send_cmd()
        # self.plan_horizon -= 1
        # print(f"Best Action: {self.best_action[0].cpu().numpy()}")
        # self.print_table(self.state_space, self.best_trajectory[], self.goal_state, self.best_action, self.plan_horizon*self.time_step, ltime)       
        
    def goal_process(self):
        r_state = torch.zeros_like(self.state_space[:6])
        r_state[3:] = self.state_space[:3]
        
        g_states = torch.zeros_like(self.goal_state[:6])
        g_states[3:] = self.goal_state[:3]

        ang_diffs = torch.abs(to_robot_torch(r_state, g_states)).squeeze()[3:]

        if (ang_diffs[1:] < 0.3).all() :# and (ang_diffs[0] < 0.5):
        # if (ang_diffs[0] < 0.5):    
            self.plan_ctr += 1
            if self.plan_ctr > 1*self.planner_rate:
                self.plan_horizon = int(plantime/self.time_step)
                self.plan_ctr = 0
                self.goal_fail_ctr = 0
                self.goal_state[0] = self.goal_angles[0][self.goal_index]
                self.goal_state[1] = self.goal_angles[1][self.goal_index]
                self.goal_state[2] = self.goal_angles[2][self.goal_index]
                self.goal_index += 1
                if self.goal_index >= len(self.goal_angles[0]):
                    self.goal_index = 0
            else:
                self.plan_horizon = int(0.5/self.time_step)
        else:
            # self.goal_fail_ctr += 1
            if self.goal_fail_ctr > 10:
                print("Plan time zero and not in goal, restart?")
                input()
                self.plan_horizon = int(plantime/self.time_step)
                self.plan_ctr = 0
                self.goal_fail_ctr = 0
            self.plan_horizon = int(plantime/self.time_step)
            self.plan_ctr = 0

if __name__ == "__main__":

    os.system('cls' if os.name == 'nt' else 'clear')                             # Clear terminal
    planner = JumpPlanner()
    rospy.spin()
