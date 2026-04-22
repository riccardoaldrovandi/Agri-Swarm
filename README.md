# 🚀 Agri-Swarm: Autonomous Robotic Harvesting with Swarm Intelligence

**Agri-Swarm** is an intelligent simulation of a robotic swarm designed for precision agriculture. The project focuses on the autonomous identification and harvesting of fresh fruits in a dynamic 2D environment, combining Deep Learning (Perception) with Swarm Intelligence (Coordination) and Automated Planning (Navigation).

![Project Simulation Preview](https://via.placeholder.com/800x450.png?text=Agri-Swarm+Simulation+Animation+Placeholder)  
*Note: This animation shows the swarm in action: Scouts (Cyan) exploring, Employed bees (Yellow) harvesting, and Returning drones (Orange) heading to base.*

## 🧠 System Architecture

The project is built on a **Hybrid Intelligent System** (as defined in modern robotics), splitting the logic into three decoupled modules:

### 1. Perception Module (Deep Learning)
A custom-built **Convolutional Neural Network (CNN)** trained in PyTorch.
- **Agnostic Pipeline:** Automatically detects image dimensions and class counts.
- **Dynamic Flattening:** Uses a "dummy pass" strategy for architectural flexibility.
- **Inference Wrapper:** Drones use a real-time wrapper to classify 5x5 grid slices into "Fresh" or "Rotten" categories.

### 2. Coordination Module (Swarm Intelligence)
The swarm is governed by the **Artificial Bee Colony (ABC)** algorithm, optimized for spatial resource exploitation:
- **Roles:** - **Scouts:** Explore the grid using a **Tabu Search** memory to avoid redundant paths.
  - **Employed Bees:** Exploit a specific tree, performing **Chain Harvesting** to maximize payload efficiency.
  - **Onlookers:** Wait at the base station and select the most promising patches using **Roulette Wheel Selection** based on "Nectar" (fruit density).
- **Limit Mechanism:** Sources are abandoned if yields fall below a threshold, forcing a shift back to exploration.

### 3. Planning & Navigation Module (Automated Planning)
Navigation isn't just "moving to a point"; it's a deliberative process:
- **A* Pathfinding:** Drones calculate the mathematically shortest path to targets, avoiding randomly scattered obstacles.
- **Survival Logic:** Real-time battery monitoring forces drones to return to the Base Station for recharging, using a safety margin calculated via Manhattan distance.
- **Physical Constraints:** Implements a **Payload** system and a **Harvesting Timer** (5 ticks) to simulate mechanical extraction.

---

## 🛠️ Tech Stack & Implementation
- **Core:** Python 3.x, NumPy
- **Deep Learning:** PyTorch (CNN architecture)
- **Simulation/UI:** Pygame (2D Engine)
- **Planning:** A* Search Algorithm
- **Optimization:** Artificial Bee Colony (ABC)

---

## 📦 Project Structure
```text
Agri-Swarm/
├── src/
│   ├── main.py                 # Simulation Orchestrator
│   ├── perception/             # CNN, DataLoader, and Inference Wrapper
│   ├── swarm/                  # ABC Optimizer and Drone FSM logic
│   ├── environment/            # GridWorld physics and fruit generation
│   └── ui/                     # Pygame visualization engine
├── data/                       # Raw and processed fruit imagery
├── models/                     # Trained weights (.pth) and metrics
└── requirements.txt
```
