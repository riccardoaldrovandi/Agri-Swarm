import os
import json
import random
import sys

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================
# Since this script lives inside the 'genetics/' folder, Python doesn't naturally 
# know where the 'src/' folder is. We append the parent directory (..) to the 
# system path so we can import our main project modules safely.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environment.grid_world import GridWorld
from src.swarm.abc_optimizer import ABCOptimizer
from src.swarm.drone import Drone

# ==============================================================================
# GENETIC ALGORITHM HYPERPARAMETERS (The "Rules of Evolution")
# ==============================================================================
POPULATION_SIZE = 15       # How many different swarms (sets of parameters) to test per generation.
GENERATIONS = 30           # How many times the population will reproduce and evolve.
MUTATION_RATE = 0.2        # 20% chance for a 'gene' to randomly change. Keeps the gene pool diverse!
CHECKPOINT_FILE = "best_params.json" # Where we save our best results so we don't lose progress.
MAX_SIMULATION_TICKS = 800 # We restrict the time. We want the swarm to be fast, not just thorough.

class PerfectHeadlessClassifier:
    """
    OPTIMIZATION TRICK (MOCKING):
    In a Genetic Algorithm, we run the simulation hundreds or thousands of times.
    If we used the real PyTorch CNN here, evaluating one generation would take hours.
    Since we only want to optimize the Swarm Intelligence (movement/logic), we 
    bypass the CNN entirely. This "dummy" classifier instantly returns perfect accuracy,
    allowing the CPU to run the simulation at blazing fast speeds.
    """
    def predict(self, image_path):
        return ["fresh"], 0.99

# ==============================================================================
# CHECKPOINTING LOGIC (State Saving & Loading)
# ==============================================================================
def save_checkpoint(params, fitness, harvested, ticks):
    """
    Saves the absolute best hyperparameters to a JSON file.
    This is crucial: if your computer crashes at Generation 29, you won't lose
    the amazing parameters it found at Generation 28.
    """
    data = {
        "max_trials": params[0],
        "radius_threshold": params[1],
        "nectar_exponent": round(params[2], 2), # Rounding makes the output cleaner
        "metrics": {
            "fitness": round(fitness, 4),
            "harvested_fruits": harvested,
            "ticks_taken": ticks
        }
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"💾 [CHECKPOINT] New best saved to {CHECKPOINT_FILE}!")

def load_checkpoint():
    """
    Checks if a JSON file exists from a previous run. If it does, we inject 
    these parameters into the new population so we resume exactly where we left off.
    """
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            data = json.load(f)
            params = [data["max_trials"], data["radius_threshold"], data["nectar_exponent"]]
            print(f"🔄 [RESUME] Found checkpoint. Resuming from: {data['metrics']}")
            return params, data["metrics"]["fitness"]
    return None, -1

# ==============================================================================
# CORE EVALUATOR (The "Fitness Test")
# ==============================================================================
def run_headless_simulation(genes):
    """
    Spins up a completely invisible (headless) instance of our GridWorld.
    It applies the current individual's 'genes' to the ABCOptimizer and tracks
    how well the swarm performs.
    """
    # 1. Unpack the genes (the hyperparameters we are testing right now)
    m_trials, r_thresh, n_exp = genes[0], genes[1], genes[2]
    
    # 2. Setup the environment without Pygame
    world = GridWorld(width=40, height=40, base_pos=(0, 0))
    classifier = PerfectHeadlessClassifier()
    
    # Inject the genes into the Hive Mind
    abc_optimizer = ABCOptimizer(
        grid_width=world.width, grid_height=world.height, 
        max_trials=m_trials, radius_threshold=r_thresh, nectar_exponent=n_exp
    )
    
    drones = [Drone(drone_id=i, start_x=0, start_y=0, max_battery=400) for i in range(15)]
    # Ensure they have high payload capacity so we are testing intelligence, not logistics
    for d in drones: d.max_payload = 5 
    
    # Count the 'Ground Truth' (total fruits available to be picked)
    total_fresh = sum(1 for fruit_list in world.fruits.values() for f in fruit_list if f['state'] == "fresh")
    
    tick_count = 0
    # 3. The high-speed simulation loop
    while tick_count < MAX_SIMULATION_TICKS:
        tick_count += 1
        for drone in drones:
            drone.step(world, classifier, abc_optimizer)
            
        # Check progress
        harvested = sum(1 for fruit_list in world.fruits.values() for f in fruit_list if f['state'] == "fresh" and f['harvested'])
        
        # Early exit: If they picked everything, stop the loop to save time!
        if harvested == total_fresh:
            break
            
    # 4. FITNESS FUNCTION (The most important math in a Genetic Algorithm)
    # The fitness score is how we grade an individual. Higher is better.
    # Base score: heavily reward picking fruits (x10 multiplier makes it the main priority)
    fitness = harvested * 10 
    
    # Speed bonus: If the swarm successfully picked 100% of the fruits, 
    # we reward them based on how many ticks they had left over.
    # E.g., finishing in 500 ticks gives a +300 bonus (800 - 500).
    if harvested == total_fresh:
        fitness += (MAX_SIMULATION_TICKS - tick_count) 
        
    return fitness, harvested, tick_count

# ==============================================================================
# GENETIC OPERATORS (The Biology of the Algorithm)
# ==============================================================================
def create_random_genes():
    """Generates a random individual (chromosome) to populate the first generation."""
    return [
        random.randint(1, 10),        # Gene 0: max_trials (Integer)
        random.randint(1, 8),         # Gene 1: radius_threshold (Integer)
        random.uniform(0.5, 3.0)      # Gene 2: nectar_exponent (Float)
    ]

def mutate(genes):
    """
    Mutation randomly tweaks a gene to prevent the algorithm from getting stuck 
    in a 'Local Minimum' (a solution that is good, but not the absolute best).
    We use max() and min() to ensure the mutation doesn't break our game physics
    (e.g., we can't have a negative radius).
    """
    if random.random() < MUTATION_RATE: # 20% chance
        genes[0] = max(1, min(10, genes[0] + random.choice([-1, 1])))
    if random.random() < MUTATION_RATE: # 20% chance
        genes[1] = max(1, min(8, genes[1] + random.choice([-1, 1])))
    if random.random() < MUTATION_RATE: # 20% chance
        genes[2] = max(0.5, min(3.0, genes[2] + random.uniform(-0.5, 0.5)))
    return genes

def crossover(p1, p2):
    """
    Recombination (Sex). Takes two parent chromosomes and builds a child by 
    randomly inheriting each gene from either Parent 1 or Parent 2 (50/50 chance).
    """
    return [p1[i] if random.random() < 0.5 else p2[i] for i in range(3)]

# ==============================================================================
# THE MAIN EVOLUTION LOOP (The "God" Function)
# ==============================================================================
def run_evolution():
    print("🧬 Starting Genetic Meta-Optimization...")
    
    # 1. Initialize the primordial soup (Generation 0)
    population = [create_random_genes() for _ in range(POPULATION_SIZE)]
    
    # Try to load previous best to give the algorithm a head start
    best_genes, best_fitness = load_checkpoint()
    if best_genes:
        population[0] = best_genes # Inject the reigning champion into slot 0
        
    for gen in range(GENERATIONS):
        print(f"\n--- GENERATION {gen + 1}/{GENERATIONS} ---")
        scored_pop = []
        
        # 2. EVALUATION PHASE
        for i, ind in enumerate(population):
            # Test every individual in the population by running a full simulation
            fitness, harvested, ticks = run_headless_simulation(ind)
            scored_pop.append((fitness, ind, harvested, ticks))
            print(f"  Ind {i+1}: Trials={ind[0]}, Rad={ind[1]}, Exp={ind[2]:.2f} -> Frutti: {harvested} in {ticks} ticks | Fit: {fitness:.1f}")
            
            # 3. CHECKPOINT PHASE
            # If this individual achieved the highest fitness score we've ever seen, save it immediately.
            if fitness > best_fitness:
                best_fitness = fitness
                best_genes = ind
                save_checkpoint(best_genes, fitness, harvested, ticks)
                
        # Sort the population so the highest fitness score is at index 0
        scored_pop.sort(key=lambda x: x[0], reverse=True)
        
        # 4. REPRODUCTION PHASE (Creating the next generation)
        new_population = []
        
        # ELITISM: The absolute best individual from this generation is guaranteed 
        # to survive into the next generation untouched. This prevents "forgetting" a good solution.
        new_population.append(scored_pop[0][1]) 
        
        # Fill the rest of the new generation (until we hit POPULATION_SIZE)
        while len(new_population) < POPULATION_SIZE:
            
            # TOURNAMENT SELECTION:
            # We pick 3 random individuals from the evaluated population.
            # We then take the 'max' (the best) of those 3 to be a parent.
            # Why do this? It gives a high chance for the strong to mate, but still 
            # allows weaker/mediocre individuals a small chance to pass on potentially useful genes.
            p1 = max(random.sample(scored_pop, 3), key=lambda x: x[0])[1]
            p2 = max(random.sample(scored_pop, 3), key=lambda x: x[0])[1]
            
            # Breed the two parents, mutate the child, and add it to the new world
            child = mutate(crossover(p1, p2))
            new_population.append(child)
            
        # The old generation dies, the new generation takes its place
        population = new_population

    print("\n🏁 Evolution Completed. The best parameters are in the JSON file.")

if __name__ == "__main__":
    run_evolution()