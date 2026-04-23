# Agri-Swarm: Autonomous Robotic Harvesting with Swarm Intelligence

**Agri-Swarm** is an intelligent simulation of a robotic swarm designed for precision agriculture. A fleet of 15 autonomous drones operates over a 40×40 2D orchard grid, identifying fresh fruit through a trained Convolutional Neural Network and coordinating collective harvesting via the Artificial Bee Colony (ABC) algorithm. A Genetic Algorithm runs offline to meta-optimise the ABC hyperparameters, and A\* search handles individual drone navigation.

> University of Bologna — Intelligent Systems M — A.Y. 2025/2026

![Agri-Swarm Demo](assets/demo.gif)
*Scouts (Cyan) explore the orchard, Employed bees (White) harvest trees, Returning drones (Orange) head back to base.*

---

## System Architecture

The project is built as a **Hybrid Intelligent System** with four decoupled modules:

### 1. Perception — `src/perception/`

A custom **Convolutional Neural Network** trained in PyTorch classifies 5×5 tile images into six fruit classes (apple / banana / orange × fresh / rotten).

- **Architecture:** 3 convolutional blocks (Conv2d + BatchNorm2d + ReLU + MaxPool2d) → FC(256) + Dropout(0.5) → FC(6 classes)
- **Dynamic Flattening:** A dummy-pass strategy auto-computes the flatten dimension — the model is fully resolution-agnostic.
- **Inference Wrapper:** `FruitClassifier` decouples prediction from training code; the simulation imports only this lightweight class.
- **Result:** ~94% accuracy on the held-out test set after 15 epochs.

### 2. Swarm Intelligence — `src/swarm/`

Collective coordination via the **Artificial Bee Colony (ABC)** algorithm:

| Role | Behaviour |
|---|---|
| **Scout** | Explores unmapped cells via Tabu Search random sampling |
| **Employed Bee** | Exploits a specific FoodSource; performs Chain Harvesting on multi-fruit trees |
| **Onlooker** | Waits at base; selects the best patch via Roulette Wheel Selection on nectar scores |

Fruit clusters are represented as `FoodSource` objects with a *nectar* score (fruit count raised to `nectar_exponent`). Sources are abandoned when `trials > max_trials`, freeing probability mass for other patches.

### 3. Planning & Navigation — `src/swarm/drone.py`

- **A\* Pathfinding:** Shortest obstacle-avoiding path computed once per trip (O(n log n)); subsequent steps consume the cached path in O(1).
- **Survival Logic:** Battery monitored every tick via Manhattan distance; drones return home with a 7-tick safety margin.
- **Payload & Harvest Timer:** Each harvest action costs 5 ticks and 5 battery units; a payload cap of 5 fruits forces periodic returns.
- **In-transit Scanning:** Every 5 ticks drones passively scan a 5×5 neighbourhood, turning any flight path into a scouting mission.

### 4. Genetic Algorithm — `src/genetics/evolution_main.py`

The GA runs **offline** to find the optimal hyperparameters for the ABC Hive Mind. See the dedicated section below.

---

## Genetic Algorithm — Meta-Optimisation

The Genetic Algorithm runs before the live simulation to discover the best ABC hyperparameters. The results are saved to `best_params.json` and loaded automatically by `ABCOptimizer` at runtime.

### What is being optimised?

Three ABC hyperparameters form the **chromosome** of each individual:

| Gene | Range | Best value found | Description |
|---|---|---|---|
| `max_trials` | 1 – 10 | **1** | Failed searches before a FoodSource is abandoned |
| `radius_threshold` | 1 – 8 | **8** | Manhattan radius for merging nearby detections into one FoodSource |
| `nectar_exponent` | 0.5 – 3.0 | **1.75** | Power applied to fruit count when computing nectar; controls how strongly larger clusters attract Onlookers |

### GA Configuration

| Parameter | Value |
|---|---|
| Population size | 15 individuals |
| Generations | 30 |
| Mutation rate | 0.2 (per gene, independently) |
| Max ticks per evaluation | 800 |

### Fitness Function

```
fitness = harvested_fruits × 10
        + (MAX_TICKS − ticks_taken)   ← only awarded if 100% of fruits harvested
```

Fruit collection is the primary objective (×10 weight). A speed bonus is added only when the swarm achieves complete coverage, rewarding configurations that are both thorough *and* fast.

### Genetic Operators

- **Elitism:** The best individual from each generation survives unchanged into the next, preventing regression.
- **Tournament Selection (k=3):** Three individuals are drawn at random; the fittest of the three becomes a parent. Balances selection pressure with population diversity.
- **Uniform Crossover:** Each gene is independently inherited from either parent with 50% probability.
- **Per-gene Mutation (p=0.2):** Integer genes shift by ±1; the float gene shifts by ±U(0, 0.5). All values are clamped to their valid ranges.

### Headless Evaluation

Running the real CNN inside the GA loop would take hours. Instead, each simulation uses a `PerfectHeadlessClassifier` mock that instantly returns `("fresh", 0.99)` for every image. This isolates the ABC parameters from perception noise and lets the CPU evaluate all 450 simulations (30 × 15) in minutes.

### Checkpointing

Every time a new best individual is found, `best_params.json` is updated immediately. If the process is interrupted, relaunching the script reinjects the saved champion into generation 0 so no progress is lost.

### Best Parameters Found

```json
{
    "max_trials": 1,
    "radius_threshold": 8,
    "nectar_exponent": 1.75,
    "metrics": {
        "fitness": 2370,
        "harvested_fruits": 237,
        "ticks_taken": 800
    }
}
```

### Running the GA

> The optimised `best_params.json` is already committed. Re-run only if you modify the ABC logic or the grid configuration.

```bash
python -m src.genetics.evolution_main
```

The script prints a live table of every individual's fitness score per generation and saves `best_params.json` automatically.

---

## Performance Results

| Metric | Value |
|---|---|
| Swarm efficiency | **98.6%** of fresh fruits harvested within 1 500 ticks |
| CNN test accuracy | **~94%** on the held-out test set |
| GA generations × individuals | 30 × 15 = 450 simulations evaluated |
| Best fitness score | **2 370** |
| Optimal ABC parameters | max_trials=1 · radius_threshold=8 · nectar_exponent=1.75 |

---

## Tech Stack

| Library | Role |
|---|---|
| Python 3.x | Core language |
| PyTorch | CNN architecture, training, inference |
| Pygame | Real-time 2D visualisation |
| NumPy | Grid physics, array operations |

**Algorithms:** CNN · Artificial Bee Colony · Genetic Algorithm · A\* Search · Tabu Search · Roulette Wheel Selection

---

## Project Structure

```text
Agri-Swarm/
├── src/
│   ├── main.py                  # Simulation orchestrator
│   ├── environment/
│   │   └── grid_world.py        # 2D physics engine, orchard generation
│   ├── perception/
│   │   ├── model.py             # CNN architecture (FruitDetection)
│   │   ├── data_utils.py        # Per-channel dataset statistics
│   │   ├── prepare_data.py      # Data preparation CLI
│   │   ├── data_loader.py       # PyTorch DataLoader factory
│   │   ├── training.py          # Training loop + confusion matrix
│   │   └── inference.py         # FruitClassifier inference wrapper
│   ├── swarm/
│   │   ├── abc_optimizer.py     # Hive Mind — ABC algorithm
│   │   └── drone.py             # Autonomous agent FSM + A* planner
│   ├── ui/
│   │   └── visualizer.py        # Pygame rendering engine
│   └── genetics/
│       └── evolution_main.py    # Genetic Algorithm meta-optimiser
├── data/
│   ├── raw/                     # Fruit image dataset (train / test)
│   └── processed/               # dataset_stats.json
├── models/
│   ├── fruit_classifier.pth     # Trained CNN weights
│   └── confusion_matrix.png     # Test-set confusion matrix heatmap
├── best_params.json             # GA output — optimal ABC hyperparameters
└── requirements.txt
```

---

## Getting Started

### Prerequisites

- Python 3.8+
- Fruit image dataset in `data/raw/` following the `ImageFolder` layout:
  ```
  data/raw/
  ├── train/
  │   ├── apple/
  │   ├── rotten_apple/
  │   └── ...
  └── test/
      └── ...
  ```

### Installation

```bash
git clone https://github.com/yourusername/Agri-Swarm.git
cd Agri-Swarm
pip install -r requirements.txt
```

### Step 1 — Prepare data statistics *(run once)*

```bash
python -m src.perception.prepare_data
```

Computes per-channel mean and standard deviation and caches them in `data/processed/dataset_stats.json`.

### Step 2 — Train the CNN *(optional — weights already included)*

```bash
python -m src.perception.training
```

Trains for 15 epochs and saves the best checkpoint to `models/fruit_classifier.pth`.

### Step 3 — Run the Genetic Algorithm *(optional — best_params.json already included)*

```bash
python -m src.genetics.evolution_main
```

### Step 4 — Run the simulation

```bash
python -m src.main
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*Developed for the Intelligent Systems M course at the University of Bologna — A.Y. 2025/2026.*
