# DOM: "Dom, Cars Don’t Fly! — Or Do They?"  
### In-Air Vehicle Maneuver for High-Speed Off-Road Navigation  

📄 **Checkout our paper:**  
[Dom, Cars Don’t Fly! — Or Do They? (arXiv:2503.19140)](https://arxiv.org/abs/2503.19140)

---

![CombinedTrajectories](./CombinedTrajectories.png)

*Above: The DOM planner predicts airborne kinodynamics for controlled in-air attitude correction (top: DOM planner, bottom: open-loop baseline).*

---

## 🧠 Overview

**DOM (Dynamics-Optimized Maneuvering)** enables off-road wheeled robots to perform stable **in-air maneuvers** by predicting 6-DoF rotational evolution during flight.  
It combines a learned neural kinodynamic model with a physics-informed planner that anticipates roll-pitch-yaw evolution and actively corrects mid-air attitude to ensure a safe landing.

---

## 📦 Requirements

Clone the repository and install the dependencies:

```bash
git clone https://github.com/<your-username>/MMAD.git
cd MMAD
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
