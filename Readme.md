# PreSimNet: A Synergistic Physics-Encoded Deep Learning Framework for Integrated Prediction and Simulation of Car-Following in Mixed-Autonomy Traffic

[![paper-status](https://img.shields.io/badge/paper-submitted-red.svg)](https://) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This is the official repository for our submitted paper: **"A Synergistic Physics-Encoded Deep Learning Framework for Integrated Prediction and Simulation of Car-Following in Mixed-Autonomy Traffic"**. All code and weights files will be made public within one week of the acceptance of the manuscript.

---

## Overview

Modeling heterogeneous car-following dynamics is a critical challenge in mixed-autonomy traffic, which comprises both autonomous (AVs) and human-driven vehicles (HVs). Existing research often treats trajectory prediction and behavior simulation as isolated tasks. This creates a trade-off between the short-term numerical accuracy of prediction models and the long-term behavioral consistency of simulation models.

To resolve this conflict, we propose **PreSimNet**, a novel, physics-encoded deep learning framework that synergistically integrates both tasks within a unified architecture. The core of PreSimNet is a shared encoder that learns type-guided features for all four interaction types (AV-AV, AV-HV, HV-AV, HV-HV). This common representation then informs two specialized decoders: an open-loop head for high-accuracy trajectory prediction, and a closed-loop Mixture-of-Experts (MoE) module that generates dynamic parameters for physics-based models to enable high-fidelity simulation.

Our experiments show that PreSimNet significantly outperforms specialized baselines in both prediction and simulation tasks. Long-term comparative experiments further validate the synergistic design, showing that the closed-loop simulation maintains behavioral realism where open-loop prediction fails due to catastrophic error accumulation.

## Framework

The core idea of PreSimNet is to bridge the gap between trajectory prediction and behavior simulation, allowing them to benefit from each other.

![Conceptual illustration of the PreSimNet framework](./assets/graphical_abstract.png)
> **Figure**: (a) Traditional trajectory prediction models focus on short-term positional accuracy. (b) Traditional behavior simulation models prioritize long-term dynamic consistency. (c) The proposed PreSimNet synergistically integrates both tasks through a shared feature encoder and specialized decoders, achieving both high-accuracy prediction and high-fidelity simulation.

The framework consists of three main components:
1.  **Trajectory Feature Learning**: Acts as a shared encoder using Mamba and Transformer blocks to extract deep spatio-temporal features from historical trajectories and explicitly predict the car-following interaction type.
2.  **Open-Loop Prediction**: A specialized decoder for direct, multi-step trajectory forecasting. It takes the shared features to generate future positions with the highest possible short-term accuracy.
3.  **Closed-Loop Simulation**: Another specialized decoder for behavior simulation. It utilizes a physics-encoded Mixture-of-Experts (MoE) to dynamically generate parameters for classic car-following models (IDM and ACC) based on the shared features. It then iteratively simulates the vehicle's future velocity and gap in a closed-loop manner, ensuring long-term behavioral realism and dynamic feasibility.

## Key Results

PreSimNet demonstrates state-of-the-art performance across multiple benchmarks.

### 1. Trajectory Prediction & Type Classification

PreSimNet achieves an average RMSE of **0.258 m** in short-term trajectory prediction and a classification accuracy of **99.23%** for car-following types, outperforming all baselines.

| Model | Position RMSE (m) | Classification Accuracy (%) | #Params | Inf. Time (ms/batch) |
| :--- | :---: | :---: | :---: | :---: |
| | **Avg.** | **Avg.** | | |
| Seq2Seq | 0.304 | 97.12% | 185,413 | 2.4 |
| Transformer | 0.295 | 98.62% | 246,085 | 2.9 |
| CS-LSTM | 0.380 | 95.68% | 247,525 | 4.9 |
| STDAN | 0.278 | 97.10% | 336,165 | 6.3 |
| BAT | 0.272 | 98.39% | 417,605 | 5.1 |
| HLTP | 0.280 | 98.96% | 527,882 | 7.4 |
| **PreSimNet (Ours)** | **0.258** | **99.23%** | **247,621** | **2.7** |

### 2. Behavior Simulation

In the closed-loop simulation task, PreSimNet achieves the lowest average RMSE for both velocity (**0.342 m/s**) and gap (**0.420 m**).

| Model | Velocity RMSE (m/s) | Gap RMSE (m) |
| :--- | :---: | :---: |
| | **Avg.** | **Avg.** |
| Personalized IDM | 0.359 | 1.352 |
| PIDL-IDM (Joint) | 1.126 | 1.085 |
| PILSTM-IDM (Joint)| 0.606 | 0.776 |
| PIT-IDM (Joint) | 0.855 | 0.677 |
| **PreSimNet (Ours)** | **0.342** | **0.420** |

### 3. Long-term Stability: Prediction vs. Simulation

Long-term (50s+) iterative rollouts highlight the synergistic advantage of our framework.

![Long-term Stability Comparison](./assets/long_term_comparison.png)
> **Figure**: In four different car-following scenarios, conventional open-loop prediction (pink line) quickly leads to unrealistic or colliding trajectories due to error accumulation. In contrast, PreSimNet's closed-loop simulation module (yellow line) remains stable and closely follows the ground truth (blue line) over extended horizons.

---

[//]: # (## Installation)

[//]: # (We recommend using `conda` to create a virtual environment for this project.)


[//]: # (```bash)

[//]: # ()
[//]: # (# 1. Clone this repository)

[//]: # ()
[//]: # (git clone [https://github.com/YourUsername/PreSimNet.git]&#40;https://github.com/YourUsername/PreSimNet.git&#41;)

[//]: # ()
[//]: # (cd PreSimNet)

[//]: # ()
[//]: # ()
[//]: # (# 2. Create and activate the Conda environment)

[//]: # ()
[//]: # (conda create -n presimnet python=3.8)

[//]: # ()
[//]: # (conda activate presimnet)

[//]: # ()
[//]: # ()
[//]: # (# 3. Install the required packages)

[//]: # ()
[//]: # (pip install -r requirements.txt)

[//]: # ()
[//]: # ()
[//]: # ([//]: # &#40;## Citation&#41;)

 


