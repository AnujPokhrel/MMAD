# DOM: "Dom, Cars Don’t Fly! — Or Do They?"  
### In-Air Vehicle Maneuver for High-Speed Off-Road Navigation  

📄 **Checkout our paper:**  
[Dom, Cars Don’t Fly! — Or Do They? (arXiv:2503.19140)](https://arxiv.org/abs/2503.19140)

🎥 **Checkout our Video (We encourage you to watch till the end for interesting footage):** 
[Video](https://www.youtube.com/watch?v=jW-c0OP8pQQ)

---

![CombinedTrajectories](./CombinedTrajectories.png)

*DOM enables accurate and timely in-air maneuvers for safe vehicle landings during high-speed off-road navigation. Top: DOM Planner prepares the vehicle for flat landing. Bottom: Without DOM, improper in-air attitude results in failure.*

---

## 🧠 Overview

When pushing the speed limit for aggressive off-road navigation on uneven terrain, vehicles inevitably become airborne.  
**DOM (Dynamics-Optimized Maneuvering)** is a motion planning framework that enables **precise in-air vehicle control** using only throttle and steering commands.  
It combines a **hybrid kinodynamic model (PHLI: PHysics and Learning-based model for In-air vehicle maneuver)** with a **fixed-horizon sampling-based motion planner (Dom Planner)** to achieve safe and timely landing poses within short airborne durations.

DOM is the first approach to demonstrate that **existing ground vehicle controls** can be repurposed to achieve accurate in-air attitude correction, ensuring stable landings and preventing mission failures caused by mid-air instabilities.

---

## 📦 Installation 

Clone the repository and install the dependencies:

```bash
git clone https://github.com/AnujPokhrel/MMAD.git
cd MMAD
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Neural Model Details
Input: [rpm, rpm_dot, steering, steering_dot, sin/cos(roll/pitch/yaw), roll_dot, pitch_dot, yaw_dot]

Output: [roll_accln, pitch_accln, yaw_accln]

Loss: Mean-squared prediction of angular accelerations.

Data: Recorded vehicle trajectories the 2-axis Gimbal.


## Training PHLI & RPMModel

```bash
python3 TrainingPipeline/train.py --config TrainingPipeline/conf/config.yaml
python3 TrainingPipeline/rpm_train.py --config TrainingPipeline/conf/config_rpm.yaml
```


## Running the indoor and outdoor DOM planner 
```bash
python3 DOM_Planner/jump_planner.py 
python3 DOM_Planner/Outdoor_jump_planner.py 
```

## Citation
If you found this work useful, please cite our manuscript:

```bibtex
@INPROCEEDINGS{pokhrel2024dom,
  author={Pokhrel, Anuj and Datar, Aniket and Xiao, Xuesu},
  booktitle={2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={Dom, cars don’t fly!—Or do they? In-Air Vehicle Maneuver for High-Speed Off-Road Navigation}, 
  year={2025},
  volume={},
  number={},
  pages={21325-21332},
  keywords={Accuracy;Navigation;Atmospheric modeling;Machine learning;Land vehicles;Automobiles;Physics;Intelligent robots},
  doi={10.1109/IROS60139.2025.11246862}
}
```
