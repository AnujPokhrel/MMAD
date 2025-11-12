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


debug_angle = False
plantime = 3                                                                    # Planning horizon in seconds

class JumpPlanner:
    def __init__(self):
        os.system('cls' if os.name == 'nt' else 'clear')                             # Clear terminal
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

        with open("DOM_Planner/stats_physics.pkl", "rb") as f:
            self.stats = pickle.load(f)

       
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
        
        # ###Goal states
        # self.goal_angles = [[0.18, -2.77,  -1.5,  -3.13,  0.0, 0.0, 0.0, ],            #Goal roll angles for the planner
        #                     [-0.56, -0.76,  1.35,  0.46,  0.0, 0.0, 0.0],            #Goal pitch angles for the planner
        #                     [ -0.11, 2.86, -1.44,  3.12, 0.0, 0.0, 0.0]]           #Goal yaw angles for the planner
        
        self.goal_angles = [[ -0.54,  -1.93, -2.56,  -3.09, 2.40,  0.84, 0.44, 0.0, 0.0],            #Goal roll angles for the planner
                    [ -0.70,  -1.01, -0.37,    0.0, -0.8,  -0.98, -0.47, 0.0, 0.0],            #Goal pitch angles for the planner
                    [  0.29,   1.75,  2.55,   2.82, -2.80, -1.09, -0.6, 0.0, 0.0]]           #Goal yaw angles for the planner
        # ###Just Roll
        # self.goal_angles = [[-0.6, 0.5, 0.7, -0.25, 0.0, 0.0, 0.0],            #Goal roll angles for the planner
        #                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],            #Goal pitch angles for the planner
        #                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]           #Goal yaw angles for the planner
        
        # ###Just Pitch
        # self.goal_angles = [[-0.0, -3.13,  0.0,  -3.14,  0.0,  0.0, 0.0],            #Goal roll angles for the planner
        #                     [-0.5, -0.7,   0.0,    0.8,  0.8,  0.0, 0.0],            #Goal pitch angles for the planner
        #                     [ 0.0,  3.10,  0.0,   3.12,  0.0,  0.0, 0.0]]           #Goal yaw angles for the planner

        self.goal_angles = [[ 0.3,  0.78, 2.57,  -2.75, 3.04,  -0.48, -0.24, 0.0, 0.0],            #Goal roll angles for the planner
                    [-0.42, -1.0, -0.96,  -0.23, 1.25,  0.96, 0.42, 0.0, 0.0],            #Goal pitch angles for the planner
                    [-0.13, -0.69, -2.68, -3.05, 3.04, -0.39, -0.07, 0.0, 0.0]]           #Goal yaw angles for the planner

        self.goal_state = torch.tensor([self.goal_angles[0][0]*0, 
                                        self.goal_angles[1][0],
                                        self.goal_angles[2][0],        #Goal Angles
                                        0.0, 0.0, 0.0,                 #Goal angular velocities
                                        500.0, 0.0 ],                  #Goal RPM and steering angle
                                        dtype=torch.float32, 
                                        device=self.device, requires_grad=False)               
        self.goal_fail_ctr = 0
        self.goal_index = 0                                                             # Index of goal pitch
        self.plan_ctr = 0                                                               # counter for holding the position at goal for 3 sec

        # Planning parameters
        self.time_step = 0.15                                                          # Time step for FKD model   
        self.sample_size = 7000                                                         # Number of samples to generate for action space
        self.planner_rate = 50                                                          # Rate of planner in Hz
        self.pose_count = 10                                                            # Number of poses to publish
        self.throttle_sigma = 5000                                                      # Standard deviation for throttle noise
        self.steering_sigma = 2.5                                                       # Standard deviation for steering noise  
        self.plan_horizon = plantime                                                    # Planning horizon tracker
        self.acc_max = self.stats['rpm_dot_max'] * 0.8                                        # Maximum wheel acceleration in rpm/s
        self.desc_max = -self.stats['rpm_dot_min']                                      # Maximum wheel deceleration in rpm/s
        self.str_dot_max = self.stats['steering_dot_max']                               # Maximum steering acceleration in rad/s^2

        self.pid_pitch_ang = PID(0.45,  0.12, 0.27)  # (P, I, D gains)
        self.pid_pitch_vel = PID(0.45, 0.12, 0.27)
        self.pid_roll_ang = PID(2.0, 0.01, 0.0)
        
        self.pid_pitch_ang.output_limits = ([-0.2, 1.0])
        self.pid_pitch_ang.sample_time = 1/100
        self.pid_pitch_ang.set_auto_mode(True, last_output=0.3)
        
        self.pid_pitch_vel.output_limits = ([-0.2, 1.0])
        self.pid_pitch_vel.sample_time = 1/self.planner_rate
        self.pid_pitch_vel.set_auto_mode(True, last_output=0.3)

        self.pid_roll_ang.output_limits = ([-1.0, 1.0])
        self.pid_roll_ang.sample_time = 1/self.planner_rate
        self.pid_roll_ang.set_auto_mode(True, last_output=0.0)


        self.pid_pitch_ang.setpoint = self.goal_state[1].cpu().numpy()
        self.pid_pitch_vel.setpoint = self.goal_state[4].cpu().numpy()
        self.pid_roll_ang.setpoint = self.goal_state[0].cpu().numpy()

        # PID control variables
        self.integral = 0.0
        self.prev_error = 0.0


        self.actions = torch.empty((self.sample_size, 2),                               # Action space for the planner
                               dtype=torch.float32,
                               device=self.device, requires_grad=False)
        
        self.best_action = torch.zeros((2,),dtype=torch.float32,                        # Best action from the planner
                                        device=self.device, requires_grad=False)   

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
        self.planner_stats = Float32MultiArray(data=np.zeros((5,8), dtype=np.float32).flatten().tolist())


        input("Press Enter to start the planner")
        rospy.Timer(rospy.Duration(1.0 / self.planner_rate), self.run_planner) 
        rospy.spin()

    def execute_cmd(self, action):
        if action[0] > 0:
            thr = map_range(action[0], 0.0, self.acc_max,  0.0, 1.0)/self.planner_rate
        else:
            thr = map_range(action[0], -self.desc_max, 0.0, -1.0, 0.0)/self.planner_rate

        strng = map_range(action[1], -6.5, 6.5, -1.0, 1.0)/self.planner_rate
        #why?
        if self.losi.losi_stats_data['RPM_avg'] < 20:
            thr = np.clip(thr, 0.0, 1.0)

        self.losi.ctrl_cmd['Thr_Cmd'] = np.clip(self.losi.ctrl_cmd['Thr_Cmd'] + thr,  -0.1 , 1.0)
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
            return  
        
        start_time = time.time()
        self.state_space[0:3] = torch.tensor(self.witIMU.angles, dtype=torch.float32, device=self.device)
        self.state_space[3:6] = torch.tensor(self.witIMU.velocities, dtype=torch.float32, device=self.device)
        self.state_space[6] = self.losi.losi_stats_data['RPM_avg']
        self.state_space[7] = self.losi.losi_stats_data['Str_Sens']
        # print(f"current pitch:{self.state_space[1]:.2f},\t pitch_vel: {self.state_space[3]:.2f}, \t rpm: {self.state_space[4]:.2f}\t goal_pitch: {self.goal_state[1]:.2f}")

        planning_time = int(self.plan_horizon/self.time_step)       # Planning horizon in steps = time in sec / time_step 
        
        if planning_time <= 2 or self.plan_ctr > 0:
            # print("Plan time zero and checking goal")
            self.goal_process()
            # return
            pass
        
        self.best_action = self.pid_control(self.state_space, self.time_step)
        ltime = f"loop time {((time.time() - start_time)*1e3):<4.1f} millisec"
        
        self.plan_horizon -= 1/self.planner_rate
        best_trajectory = torch.zeros((10, 8), dtype=torch.float32, device=self.device)

        self.print_table(self.state_space, best_trajectory[-1], self.goal_state, self.best_action, self.plan_horizon, ltime)       
        
    def goal_process(self):
        r_state = torch.zeros_like(self.state_space[:6])
        r_state[3:] = self.state_space[:3]
        
        g_states = torch.zeros_like(self.goal_state[:6])
        g_states[3:] = self.goal_state[:3]

        ang_diffs = torch.abs(to_robot_torch(r_state, g_states)).squeeze()[3:]

        if (ang_diffs[1:] < 0.5).all() and (ang_diffs[0] < 0.5):
        # if (ang_diffs[1:] < 0.3).all() :
        # if (ang_diffs[0] < 0.5):    
            self.plan_ctr += 1
            if self.plan_ctr > 10 * self.planner_rate:
                self.plan_horizon = plantime
                self.plan_ctr = 0
                self.goal_fail_ctr = 0
                self.goal_state[0] = self.goal_angles[0][self.goal_index]
                self.goal_state[1] = self.goal_angles[1][self.goal_index]
                self.goal_state[2] = self.goal_angles[2][self.goal_index]
                self.goal_index += 1
                if self.goal_index >= len(self.goal_angles[0]):
                    self.goal_index = 0
            else:
                self.plan_horizon =0.5
        else:
            # self.goal_fail_ctr += 1
            if self.goal_fail_ctr > 10:
                print("Plan time zero and not in goal, restart?")
                input()
                self.plan_horizon = plantime
                self.plan_ctr = 0
                self.goal_fail_ctr = 0
            self.plan_horizon = plantime
            self.plan_ctr = 0

    def pid_control(self, state, time_step=0.25):      
        t_states = torch.zeros_like(state[:6])
        t_states[3:] = state[:3]
        
        g_states = torch.zeros_like(self.goal_state[:6])
        g_states[3:] = self.goal_state[:3]

        ang_diffs = to_robot_torch(t_states, g_states)[:, 3:].squeeze().cpu().numpy()

        pitch_input = self.pid_pitch_ang.setpoint + ang_diffs[1]
        
        output_pitch = self.pid_pitch_ang(pitch_input)
        output_vel = self.pid_pitch_vel(self.state_space[4].cpu().numpy())
        output_roll = self.pid_roll_ang(ang_diffs[0])
        
        output = 0.8 * output_pitch + 0.2 * -output_vel
        output = np.clip(output, -self.acc_max, self.acc_max)
        cmd = self.best_action
        cmd[0] = output #float(output_roll)

        self.losi.ctrl_cmd['Thr_Cmd'] = np.clip(output,  -0.2 , 1.0)
        self.losi.ctrl_cmd['Str_Cmd'] = output_roll
        self.losi.send_cmd()
        return cmd

if __name__ == "__main__":
    planner = JumpPlanner()
    rospy.spin()
