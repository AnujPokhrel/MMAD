import rospy
import os
import time
import numpy as np
import torch
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from callbacks import ImuProcessor, LosiManager
from utils import *
from functools import lru_cache
import pdb
from regressionModel import RollPitchRegressionModels 
from nn_model import RollPitchAcclnModel
import pickle
from simple_pid import PID
os.system('cls' if os.name == 'nt' else 'clear')                             # Clear terminal

debug_angle = False
plantime = 1                                                                    # Planning horizon in seconds

class JumpPlanner:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device to run the planner
        print(f"Device set to: {self.device}\n")
        rospy.init_node("jump_planner")                                             # Initialize ROS node
        self.witIMU = ImuProcessor(imu_topic="/losi/witmotion_imu/imu",             # Initialize IMU processor
                              mag_topic="/losi/witmotion_imu/magnetometer", 
                              sampling_rate=200,
                              method='madgwick', use_mag=False, tait_bryan=False)       
        self.losi = LosiManager()                                                   # Initialize LOSI manager
        self.path_pub = rospy.Publisher("/planned_path", Path, queue_size=10)       # Path publisher

        self.stats_pub = rospy.Publisher("/planner_stats", Float32MultiArray, queue_size=10)       # Path publisher
        
        '''load model and stats file for the neural net model
        model: RollPitchAcclnModel
            input:  [rpm, rpm_dot, steering, steering_dot, 
                    [sin(roll), cos(roll), sin(pitch), cos(pitch), sin(yaw), cos(yaw)]
                    roll_dot, pitch_dot, yaw_dot]
               
            output: [roll_accln, pitch_accln, yaw_accln]
        '''
        
        with open("DOM_Planner/stats_physics.pkl", "rb") as f:
            self.stats = pickle.load(f)

        # model_path = "DOM_Planner/precession_all_data--02-16-00-46/.pth"
        model_path = "DOM_Planner/precession_all_data--02-16-00-46/no_norm--02-23-02-32-E53-best.pth"
        self.nnModel = RollPitchAcclnModel()
        self.nnModel.load_state_dict(torch.load(model_path)['model'])
        self.nnModel.to(self.device)
        # self.nnModel.compile()
        self.nnModel.eval()


        '''State variables 
        [   0    1      2       3       4       5       6       7  ]
        [roll, pitch, yaw, delRoll, delPitch, delYaw, rpm, steering]
        '''
        self.state_space = torch.zeros((8,), dtype=torch.float32, 
                                       device=self.device, requires_grad=False)         

        self.s_limits = torch.tensor([[0, -1], [self.stats['rpm_max'], 1]],             # State limits
                                     dtype=torch.float32, 
                                     device=self.device, requires_grad=False)
        
        self.avg_vel_max = (self.stats['roll_vel_max'] + self.stats['pitch_vel_max'] + self.stats['yaw_vel_max'])  / 3
        self.avg_vel_min = (self.stats['roll_vel_min'] + self.stats['pitch_vel_min'] + self.stats['yaw_vel_min'])  / 3
        self.abs_vel_max = max(abs(self.avg_vel_max), abs(self.avg_vel_min))

                
        # ##Flip
        self.goal_angles = [[ 0.0,  0.0, 0.0, 0.0],            #Goal roll angles for the planner
                            [ 0.0,  0.0, 0.0, 0.0],            #Goal pitch angles for the planner
                            [ 0.0,  0.0, 0.0, 0.0]]           #Goal yaw angles for the planner

        self.goal_state = torch.tensor([self.goal_angles[0][0], 
                                        self.goal_angles[1][0],
                                        self.goal_angles[2][0],        #Goal Angles
                                        0.0, 0.0, 0.0,                 #Goal angular velocities
                                        1000.0, 0.0 ],                  #Goal RPM and steering angle
                                        dtype=torch.float32, 
                                        device=self.device, requires_grad=False)               
        self.goal_fail_ctr = 0
        self.goal_index = 0                                                             # Index of goal pitch
        self.plan_ctr = 0                                                               # counter for holding the position at goal for 3 sec

        # Planning parameters
        self.time_step = 0.1                                                          # Time step for FKD model   
        self.sample_size = 4000                                                         # Number of samples to generate for action space
        self.planner_rate = 50                                                          # Rate of planner in Hz
        self.pose_count = 10                                                            # Number of poses to publish
        self.throttle_sigma = 5000                                                      # Standard deviation for throttle noise
        self.steering_sigma = 0.2                                                       # Standard deviation for steering noise  
        self.plan_horizon = int(plantime/self.time_step)                                #Planning horizon tracker
        self.acc_max = 3070 #self.stats['rpm_dot_max']                                         # Maximum wheel acceleration in rpm/s
        self.desc_max = 3900 #-self.stats['rpm_dot_min']                                      # Maximum wheel deceleration in rpm/s
        self.str_dot_max = self.stats['steering_dot_max']                               # Maximum steering acceleration in rad/s^2
        self.t_remaining = self.planner_rate * plantime                                 # Remaining time for the planner 

        # PID control variables
        self.integral = 0.0
        self.prev_error = 0.0


        self.actions = torch.empty((self.sample_size, 2),                               # Action space for the planner
                               dtype=torch.float32,
                               device=self.device, requires_grad=False)
        
        self.best_action = torch.zeros((2,),dtype=torch.float32,                        # Best action from the planner
                                        device=self.device, requires_grad=False)   

        # Planner timer at Event
        # self.losi.ctrl_cmd['Thr_Cmd'] = 0.3
        # self.losi.ctrl_cmd['Str_Cmd'] = 0.0
        # self.losi.send_cmd()

        self.max_rpm_vel = 0.0

        '''PLanner stats publisher  [[Current_roll, Current_pitch, Current_yaw, roll_vel,  pitch_vel,  yaw_vel, RPM, Steering],
                                    [Goal_roll,     Goal_pitch,    Goal_yaw,    roll_vel,  pitch_vel,  yaw_vel, RPM, Steering],
                                    [diff_roll,     diff_pitch,    diff_yaw,    droll_vel, dpitch_vel, dyaw_vel, dRPM, dSteering],
                                    [P_roll,        P_pitch,        P_yaw,     p_roll_vel, p_pitch_vel, p_yaw_vel, RPM, Steering],
                                    [Throttle_cmd, Steering_cmd,    Time,      Goal_index, 0,           0,         0,   0      ]]
        '''
        # self.planner_stats = Float32MultiArray(data=np.zeros((5,8), dtype=np.float32).flatten().tolist())

        self.action_counter = 0

        input("Press Enter to start the planner")
        rospy.Timer(rospy.Duration(1.0 / self.planner_rate), self.run_planner) 
        rospy.spin()
        
    @torch.no_grad()
    def plan_path(self, state, planning_horizon=60):    
        #set the lower and higher bound for sampling the action space
        sample_low_bound, sample_high_bound = self.best_action.clone(), self.best_action.clone()  
        sample_low_bound[0]  -= self.throttle_sigma
        sample_high_bound[0] += self.throttle_sigma
        sample_low_bound[1]  -= self.steering_sigma
        sample_high_bound[1] += self.steering_sigma

        torch.clamp_(sample_low_bound, torch.tensor([-self.desc_max, -self.str_dot_max], device=self.device),
                     torch.tensor([-100, -0.2], device=self.device))
        torch.clamp_(sample_high_bound, torch.tensor([100, 0.2], device=self.device),
                     torch.tensor([self.acc_max, self.str_dot_max], device=self.device))

        self.actions.zero_()
        self.actions[:, 0].uniform_(sample_low_bound[0], sample_high_bound[0])
        self.actions[:, 1].uniform_(sample_low_bound[1], sample_high_bound[1])

        # self.actions[:, 0] = 2000
        # self.actions[:, 0] = self.best_action[0] + torch.normal(mean=0.0, std=0.2, size=(self.sample_size,)).cuda()*(self.acc_max)
        # self.actions[:, 1] = self.best_action[1] + torch.normal(mean=0.0, std=0.05, size=(self.sample_size,)).cuda()*(self.str_dot_max)
        
        torch.clamp_(self.actions[:, 0], -self.desc_max, self.acc_max)
        torch.clamp_(self.actions[:, 1], -self.str_dot_max, self.str_dot_max)

        # Initialize trajectories with the starting state
        states = state.clone().repeat(self.sample_size, 1)
        
        total_costs     = torch.zeros(self.sample_size, dtype=torch.float32, device=self.device)
        trajectories    = torch.zeros((self.sample_size, planning_horizon, len(state)), dtype=torch.float32, device=self.device)
        prev_state      = torch.zeros_like(states[:, :6])
        next_state      = torch.zeros_like(prev_state)
        angles          = torch.zeros_like(prev_state)
        
        # states[:5, 6] = 1000
        action_cur = self.actions.clone()
        for t in range(planning_horizon):
            # Clamp the states to the limits
            torch.clamp_(states[:, 6], self.s_limits[0, 0], self.s_limits[1, 0])
            torch.clamp_(states[:, 7], self.s_limits[0, 1], self.s_limits[1, 1])

            cur_speed = states[:, 6]
            cur_str = states[:, 7]
            action_cur[:, 0] = 0.9 * action_cur[:, 0] + 0.1 * self.actions[:, 0].clone()

            mask_speed_low = cur_speed <= 400    
            action_cur[mask_speed_low, 0] = torch.maximum(action_cur[mask_speed_low, 0]*self.time_step, 
                                                        -cur_speed[mask_speed_low])/self.time_step

            mask_speed_high = cur_speed >= 1650          
            action_cur[mask_speed_high, 0] = torch.minimum(action_cur[mask_speed_high, 0]*self.time_step, 
                                                        ((1980.0 - cur_speed))[mask_speed_high])/self.time_step

            mask_str_low = cur_str <= -0.8
            action_cur[mask_str_low, 1] = torch.maximum(action_cur[mask_str_low, 1]*self.time_step, 
                                                        -1 - cur_str[mask_str_low])/self.time_step

            mask_srt_high = cur_str >= 0.8
            action_cur[mask_srt_high, 1] = torch.minimum(action_cur[mask_srt_high, 1], 
                                                        1 - cur_str[mask_srt_high])/self.time_step

            rpm = states[:, 6] + action_cur[:, 0] * self.time_step
            steering = states[:, 7] + action_cur[:, 1] * self.time_step
            
            rpmcheck = states[:5, 6]
            action_chk = action_cur[:5, 0]
            rpmdot = action_cur[:, 0]
            steering_dot = action_cur[:, 1]

            angles[:, [0,2,4]] = torch.sin(states[:, :3])
            angles[:, [1,3,5]] = torch.cos(states[:, :3])
            ang_vels = states[:, 3:6]

            # rpm_normalized = (rpm - self.stats['rpm_min']) / (self.stats['rpm_max'] - self.stats['rpm_min'])
            # rpmdot_normalized = (rpmdot - self.stats['rpm_dot_mean']) / self.stats['rpm_dot_std']
            # steering_dot_normalized = (steering_dot - self.stats['steering_dot_mean']) / self.stats['steering_dot_std']
            # ang_vels[:, 0] = (ang_vels[:, 0] - self.stats['roll_vel_mean']) / self.stats['roll_vel_std']
            # ang_vels[:, 1] = (ang_vels[:, 1] - self.stats['pitch_vel_mean']) / self.stats['pitch_vel_std']
            # ang_vels[:, 2] = (ang_vels[:, 2] - self.stats['yaw_vel_mean']) / self.stats['yaw_vel_std']

            accelerations = model_dynamics(rpm.unsqueeze(1).clone(),
                            rpmdot.unsqueeze(1).clone(),
                            steering.unsqueeze(1).clone(),
                            steering_dot.unsqueeze(1).clone(),
                            self.time_step,
                           )

            accelerations[:, [1,2]] = self.nnModel(rpm.unsqueeze(1), 
                               rpmdot.unsqueeze(1), 
                               steering.unsqueeze(1), 
                               steering_dot.unsqueeze(1), 
                               angles, 
                               ang_vels
                               )[:, [1,2]]
            
            # accelerations = self.nnModel(rpm_normalized.unsqueeze(1), 
            #                    rpmdot_normalized .unsqueeze(1), 
            #                    steering.unsqueeze(1), 
            #                    steering_dot_normalized.unsqueeze(1), 
            #                    angles, 
            #                    ang_vels
            #                    )

            # accelerations[:, 0] = accelerations[:, 0] * self.stats['roll_accln_std']    + self.stats['roll_accln_mean']
            # accelerations[:, 1] = accelerations[:, 1] * self.stats['pitch_accln_std']   + self.stats['pitch_accln_mean']
            # accelerations[:, 2] = accelerations[:, 2] * self.stats['yaw_accln_std']     + self.stats['yaw_accln_mean']
            
            # Update roll, pitch, yaw angular velocities roll_vel = roll_vel + acc * dt
            rpy_vels = states[:, 3:6] + accelerations * self.time_step
            
            # Update roll, pitch and yaw angular displacement roll_displacement = roll_vel * dt + 0.5 * acc * dt^2
            rpy_displacement = states[:, 3:6] * self.time_step + 0.5 * accelerations * (self.time_step ** 2)
            # rpy_next = states[:, 3:6] + rpy_displacement  # Update roll and pitch angles roll = roll + roll_vel * dt + 0.5 * acc * dt^2

            # roll_pitch = clamp_angle(roll_pitch)
            prev_state.zero_()
            next_state.zero_()
            prev_state[:, 3:] = states[:, :3]           # Roll, pitch and yaw angles form current state 
            next_state[:, 3:] = rpy_displacement         # roll, pitch and yaw angular displacement w.r.t. current state frame

            new_state = to_world_torch(prev_state, next_state)  # Convert roll and pitch displacement to world frame
            # new_state = prev_state + next_state  # Roll and pitch angles in world frame
            rpy_new = new_state[:, 3:]  # Roll and pitch angles in world frame
            # Update states

            states[:, :3] = rpy_new
            states[:, 3:6] = rpy_vels
            states[:, 6] = rpm
            states[:, 7] = steering

            # Save the current state in trajectories
            trajectories[:, t, :] = states

            # if t > min(10, planning_horizon - 1) :
            total_costs += self.evaluate_cost(states, t) * t
        # print(f"max_rpm: {np.max(states[:, 4]):.2f}, min_rpm: {np.min(states[:, 4]):.2f}, max_st: {np.max(states[:, 5]):.2f}, min_st: {np.min(states[:, 5]):.2f}")
        # Select the best trajectory based on cumulative cost

        # Old way to do it
        best_index = torch.argmin(total_costs)
        best_actions = self.actions[best_index]
        best_trajectory = trajectories[best_index]

        if self.state_space[6] < 10:
            torch.clamp_(best_actions[0], 0, self.acc_max)
        
        if self.state_space[6] > self.s_limits[1, 0]:
            torch.clamp_(best_actions[0], -self.desc_max, 0)

        if best_actions[0] > 0.5:
            torch.clamp_(best_actions[0], 5, self.acc_max)
        
        elif best_actions[0] < -0.5:
            torch.clamp_(best_actions[0], -self.desc_max, -5)

        else:
            best_actions[0] = 0
        
        return best_actions, best_trajectory
    
    @torch.no_grad()
    def evaluate_cost(self, c_states, time_step):
        t_states = torch.zeros_like(c_states[:, :6])
        t_states[:, 3:] = c_states[:, :3]
        
        g_states = torch.zeros_like(self.goal_state[:6])
        g_states[3:] = self.goal_state[:3]

        ang_diffs = to_robot_torch(t_states, g_states.unsqueeze(0).repeat(c_states.shape[0], 1))[:, 3:]

        # r, p ,y, rr, pr, py, rpm, str

        vel_diff   = c_states[:, 4:5] - self.goal_state[4:5].unsqueeze(0).repeat(c_states.shape[0], 1)
        vel_diff = vel_diff.mean(dim=1)
        # vel_cutoff_mask = torch.abs(vel_diff) < 0.15
        # vel_diff[vel_cutoff_mask] = 0.0 
        rpm_diff   = c_states[:, 6] - self.goal_state[6]
    
        roll_cost   = torch.abs(ang_diffs[:, 0]) / np.pi
        pitch_cost  = torch.abs(ang_diffs[:, 1]) /(0.5 * np.pi)
        yaw_cost    = torch.abs(ang_diffs[:, 2]) / np.pi
        vel_cost    = (torch.abs(vel_diff) / self.abs_vel_max).squeeze()
        rpm_cost    = torch.abs(rpm_diff) / self.s_limits[1, 0]
        # str_cost    = torch.abs(str_cost)
        velmask = torch.logical_and(torch.abs(ang_diffs[:, 1]) < 0.15, torch.abs(ang_diffs[:, 2] <0.15))
        vel_cost[velmask] = 1.9 * vel_cost[velmask]**2
        vel_cost[~velmask] = 1.5  * vel_cost[~velmask]**2
        
        


        total_cost =  (1.0 * vel_cost +   
                       6 *  pitch_cost**3 +  
                       6 *  yaw_cost**3 + 
                       0 *  roll_cost**3  +  
                       0 * rpm_cost**2)

        return total_cost*time_step/10

    def execute_cmd(self, action):
        strng = map_range(action[1], -6.5, 6.5, -0.5, 0.5)/self.planner_rate
        thr = map_range(action[0], 0.0, 1900, -0.06, 1.0)
        print(f"Throttle: {thr:.2f}")
        self.losi.ctrl_cmd['Thr_Cmd'] = thr #np.clip(self.losi.ctrl_cmd['Thr_Cmd'] + thr,  -0.2 , 1.0)
        self.losi.ctrl_cmd['Str_Cmd'] = 0 * np.clip(action[1] + strng,  -1.0 , 1.0) 
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

        if self.max_rpm_vel == 0:
            self.max_rpm_vel = r_state[4]
        else:
            self.max_rpm_vel = torch.max(torch.abs(self.max_rpm_vel), torch.abs(r_state[4]))

        print(f"Max RPM vel: {self.max_rpm_vel:.2f}")
        
        diffs = torch.zeros_like(r_state)
        diffs = r_state - g_state 
        diffs[:3] = ang_diffs

        stat_cache = np.zeros((5, 8), dtype=np.float32)

        for i in range(8):
            stat_cache[0, i] = r_state[i].cpu().numpy().item()
            stat_cache[1, i] = g_state[i].cpu().numpy().item()
            stat_cache[2, i] = diffs[i].cpu().numpy().item()
            stat_cache[3, i] = f_state[i].cpu().numpy().item() 

        stat_cache[4, 0] = action[0].cpu().numpy().item()
        stat_cache[4, 1] = action[1].cpu().numpy().item()
        stat_cache[4, 2] = float(rtime)
        stat_cache[4, 3] = float(self.goal_index)

        planner_stats = Float32MultiArray(data=stat_cache.flatten().tolist())

        self.stats_pub.publish(planner_stats)

    def run_planner(self, event):

        if not self.witIMU.initialized or not self.losi.initialized:
            print(f"IMU Initialized:{self.witIMU.initialized}, LOSI initialized: {self.losi.initialized}")
            self.losi.send_cmd()
            return  
        
        start_time = time.time()
        self.state_space[0:3] = torch.tensor(self.witIMU.angles, dtype=torch.float32, device=self.device)
        self.state_space[3:6] = torch.tensor(self.witIMU.velocities, dtype=torch.float32, device=self.device)
        self.state_space[6] = self.losi.losi_stats_data['RPM_avg']
        self.state_space[7] = self.losi.losi_stats_data['Str_Sens']
        planning_time = self.plan_horizon
        # print(f"current pitch:{self.state_space[1]:.2f},\t pitch_vel: {self.state_space[3]:.2f}, \t rpm: {self.state_space[4]:.2f}\t goal_pitch: {self.goal_state[1]:.2f}")

        self.t_remaining -= 1
        self.plan_horizon = int(np.round(self.t_remaining/self.planner_rate, 0))
        if self.plan_horizon < 2:
            self.plan_horizon = 2

        if self.losi.remote_control['Thr_Cmd'] > 0.1:
            self.losi.ctrl_cmd['Thr_Cmd'] += 0.002
            self.losi.ctrl_cmd['Thr_Cmd'] = min(self.losi.ctrl_cmd['Thr_Cmd'], 0.3)
            # print("step 1")
               #1.0 1.75sec with 1 sec break
            self.losi.ctrl_cmd['Str_Cmd'] = self.losi.remote_control['Str_Cmd']
            self.losi.send_cmd()
            self.best_action, best_trajectory = torch.tensor([4000.0, 0.0], device=self.device), torch.zeros((10, 8), dtype=torch.float32, device=self.device)
            best_cmd = self.best_action.clone()
            self.best_action, best_trajectory = self.plan_path(self.state_space, planning_horizon=planning_time)
            self.action_counter = 1
        elif self.action_counter > 0 and self.action_counter < 150 and self.losi.remote_control['Thr_Cmd'] < -0.5:
            # print("step 2")
        #     self.action_counter += 1
        #     self.losi.ctrl_cmd['Thr_Cmd'] = -1.0
        #     self.losi.ctrl_cmd['Str_Cmd'] = self.losi.remote_control['Str_Cmd']
        #     self.losi.send_cmd()
        #     self.best_action, best_trajectory = torch.tensor([4000.0, 0.0], device=self.device), torch.zeros((10, 8), dtype=torch.float32, device=self.device)
        #     self.best_action, best_trajectory = self.plan_path(self.state_space, planning_horizon=planning_time)  
        #     best_cmd = self.best_action.clone()
        # elif self.action_counter >= 50 and self.action_counter < 250:
            self.action_counter += 1
            # print("step 3")
            if planning_time <= 2 or self.plan_ctr > 0:
                # print("Plan time zero and checking goal")
                self.goal_process()
                # return
                pass

            self.best_action, best_trajectory = self.plan_path(self.state_space, planning_horizon=planning_time)
            # self.best_action = self.pid_control(self.state_space, self.time_step)
            # self.best_action, best_trajectory = self.plan_path(self.state_space, planning_horizon=10)        
            # publish_path(trajectory=best_trajectory.cpu().numpy(), pub=self.path_pub, p_count=self.pose_count)

            best_cmd = self.best_action.clone()
            self.best_action[0] = best_trajectory[-1, 6]
            self.best_action[1] = self.best_action[1]
            self.execute_cmd(self.best_action.cpu().numpy())

        elif self.action_counter >= 150:
            self.losi.ctrl_cmd['Thr_Cmd'] = 0.0
            self.losi.send_cmd()
            exit()

        else:
            print("step 0")
            self.best_action, best_trajectory = torch.tensor([4000.0, 0.0], device=self.device), torch.zeros((10, 8), dtype=torch.float32, device=self.device)
            best_cmd = self.best_action.clone()

        ltime = f"loop time {((time.time() - start_time)*1e3):<4.1f} millisec"  
        # best_trajectory = torch.zeros((10, 8), dtype=torch.float32, device=self.device)
        self.print_table(self.state_space, best_trajectory[-1], self.goal_state, best_cmd, self.t_remaining/self.planner_rate, ltime)       
        
    def goal_process(self):
        r_state = torch.zeros_like(self.state_space[:6])
        r_state[3:] = self.state_space[:3]
        
        g_states = torch.zeros_like(self.goal_state[:6])
        g_states[3:] = self.goal_state[:3]

        ang_diffs = torch.abs(to_robot_torch(r_state, g_states)).squeeze()[3:]

        if (ang_diffs[1] < 0.5) and (ang_diffs[2] < 0.5) and (ang_diffs[0] < 0.6):
        # if (ang_diffs[1] < 0.4) and (ang_diffs[2] < 0.4):
        # if (ang_diffs[0] < 0.4):    
            self.plan_ctr += 1
            if self.plan_ctr > 1 * self.planner_rate:
                self.t_remaining = self.planner_rate * plantime
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
                self.plan_horizon = 4
                self.t_remaining = 25
        else:
            # self.goal_fail_ctr += 1
            if self.goal_fail_ctr > 10:
                print("Plan time zero and not in goal, restart?")
                input()
                self.goal_fail_ctr = 0
            self.plan_horizon = int(plantime/self.time_step)
            self.t_remaining = self.planner_rate * plantime
            self.plan_ctr = 0

if __name__ == "__main__":
    planner = JumpPlanner()
    rospy.spin()
