import numpy as np
import src.environment.grid_world
import heapq
import random

class DroneState:
    """
    Enumeration representing the physical state of the drone for animation and logic control.
    """
    EXPLORING = "exploring"   # Moving towards a target
    HARVESTING = "harvesting" # Stationary, currently extracting a fruit
    RETURNING = "returning"   # Heading to base due to low battery or full payload
    IDLE = "idle"             # Resting at base

class ABCRole:
    """
    Enumeration representing the drone's role within the Artificial Bee Colony (ABC) algorithm.
    This defines the 'mindset' of the drone when picking a new target.
    """
    SCOUT = "scout"       # Explores unknown/random areas to find new food sources
    ONLOOKER = "onlooker" # Waits at base, uses Roulette Wheel selection to pick a known good source
    EMPLOYED = "employed" # Currently working a specific, known food source patch

class Drone:
    """
    Represents a single autonomous agent in the Agri-Swarm simulation.
    It combines physical constraints (battery, payload, movement) with 
    swarm intelligence (ABC roles) and path planning (A* search).
    """
    def __init__(self, drone_id, start_x, start_y, max_battery=100):
        self.drone_id = drone_id
        self.x = start_x
        self.y = start_y

        # --- Battery Management ---
        self.max_battery = max_battery
        # Battery Desync: We initialize the drones with slightly varying battery levels (80%-100%).
        # This prevents the entire swarm from dying and returning to base at the exact same tick,
        # which would cause traffic jams and highly inefficient cyclic behavior.
        self.battery = random.randint(int(max_battery * 0.8), max_battery)

        self.state = DroneState.IDLE
        
        # Every drone begins the simulation as a Scout to map out the initial environment.
        self.abc_role = ABCRole.SCOUT 
        
        # The specific (x, y) coordinate the drone is currently trying to reach.
        self.target_pos = None  

        # --- Local Visual Memory ---
        # A dictionary acting as the drone's short-term memory.
        # Key: (x, y) coordinate, Value: Confidence score from the CNN.
        # This allows the drone to remember nearby fruits it saw during a scan.
        self.known_fresh_fruits = {}

        # --- Physical Constraints ---
        self.payload = 0           # Current number of fruits carried
        self.max_payload = 3       # Maximum carrying capacity (3 forces frequent returns to base)
        self.harvest_timer = 0     # Countdown timer for the harvesting action
        self.harvest_duration = 5  # It costs 5 simulation ticks to physically pick one fruit
    
    def scan_environment(self, grid_world, classifier):
        """
        Simulates the drone's downward-facing camera.
        It grabs a local slice of the grid, checks for unharvested fruits, 
        and uses the CNN to classify if they are fresh.
        """
        # Optimization: If our backpack is already full, we don't care about finding new fruit right now.
        # We save CPU cycles by skipping the CNN inference.
        if self.payload >= self.max_payload:
            return
        
        # Get a 5x5 grid slice (radius=2) around the drone's current position.
        _, visible_fruits = grid_world.get_local_view(self.x, self.y, radius=2)
        
        for pos, fruit_data in visible_fruits.items():
            # Only process fruits we haven't already memorized in this session.
            if pos not in self.known_fresh_fruits:
                image_path = fruit_data['image_path']
                if image_path:
                    # Pass the real image through our trained CNN.
                    predicted_class, confidence = classifier.predict(image_path)
                    
                    # If the AI is highly confident it's fresh, commit it to local memory.
                    if "fresh" in predicted_class and confidence > 0.7:
                        self.known_fresh_fruits[pos] = confidence
                        print(f"[Drone {self.drone_id}] spotted fresh fruit at {pos} (Conf: {confidence:.2f})")
    
    def move(self, dx, dy, grid_world):
        """
        Attempts to physically move the drone by (dx, dy).
        Enforces collision physics and consumes battery for the effort.
        """
        if self.battery <= 0: 
            return False # Drone is dead, cannot move.
            
        new_x, new_y = self.x + dx, self.y + dy
        
        # Ask the environment if this coordinate is inside the map and free of obstacles.
        if grid_world.is_valid_move(new_x, new_y):
            self.x, self.y = new_x, new_y
            self.battery -= 1 # Movement costs 1 unit of energy
            return True
            
        return False # Move was blocked by map boundaries or an obstacle.
    
    def start_harvesting(self):
        """
        Transitions the drone into the working state and initializes the timer.
        """
        self.state = DroneState.HARVESTING
        self.harvest_timer = self.harvest_duration
        
    def check_battery(self, base_pos):
        """
        Calculates if the drone has just enough energy to return to base.
        Uses Manhattan distance as a fast heuristic.
        """
        distance_to_base = abs(self.x - base_pos[0]) + abs(self.y - base_pos[1])
        
        # We add a safety margin of 7 to the strict distance.
        # Why 7? 5 ticks are needed if it decides to harvest right now, 
        # plus 2 ticks as a buffer for pathfinding around sudden obstacles.
        if self.battery < distance_to_base + 7:
            self.state = DroneState.RETURNING
            return True
        return False
    
    def _plan_path_astar(self, grid_world, start_pos, target_pos):
        """
        A-Star (A*) search algorithm for intelligent navigation.
        This prevents the drone from getting stuck in 'Local Minima' (e.g., U-shaped obstacles)
        by calculating the guaranteed shortest path to the target.
        """
        # The priority queue stores: (f_score, (x, y), path_history)
        queue = []
        heapq.heappush(queue, (0, start_pos, []))
        visited = set([start_pos])
        
        # Movement options: Up, Down, Left, Right
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        
        while queue:
            f_score, current, path = heapq.heappop(queue)
            
            # If we reached the target coordinate, return the sequence of moves.
            if current == target_pos: 
                return path
                
            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy
                
                if grid_world.is_valid_move(nx, ny) and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    
                    # g(n): Actual cost to reach this node (steps taken so far)
                    g_score = len(path) + 1
                    # h(n): Heuristic estimate to the target (Manhattan distance)
                    h_score = abs(nx - target_pos[0]) + abs(ny - target_pos[1])
                    
                    # f(n) = g(n) + h(n). The core of A*. We prioritize nodes with the lowest total cost.
                    heapq.heappush(queue, (g_score + h_score, (nx, ny), path + [(dx, dy)]))
                    
        return [] # Returns an empty list if the target is completely boxed in and unreachable.
    
    def _move_towards_target(self, grid_world):
        """Pathfinding execution wrapper with traffic tolerance."""
        if not self.target_pos: return
        path = self._plan_path_astar(grid_world, (self.x, self.y), self.target_pos)
        if path:
            dx, dy = path[0]
            self.move(dx, dy, grid_world)
            self.stuck_counter = 0 # Reset stuck counter if we moved
        else:
            # We are blocked (likely by another drone).
            # Don't drop the target immediately! Wait a few ticks.
            if not hasattr(self, 'stuck_counter'):
                self.stuck_counter = 0
            
            self.stuck_counter += 1
            
            # Only give up on the target if we've been stuck for 5 consecutive ticks
            if self.stuck_counter > 5 and self.target_pos != grid_world.base_pos:
                self.target_pos = None
                self.stuck_counter = 0

    def step(self, grid_world, classifier, abc_optimizer):
        """
        The main 'Brain Tick' of the agent. This is called once per frame in the main loop.
        It evaluates the FSM sequentially: Harvesting -> Survival -> Planning -> Action -> Perception.
        """
        
        # =====================================================================
        # 0. ACTIVE WORK (Harvesting Phase)
        # =====================================================================
        # If the drone is actively harvesting a fruit, it is locked in this state.
        if self.state == DroneState.HARVESTING:
            self.harvest_timer -= 1
            self.battery -= 1 # Extracting fruit requires mechanical energy.
            
            # Check if the harvesting countdown has finished.
            if self.harvest_timer <= 0:
                # Attempt to physically remove the fruit from the GridWorld.
                if grid_world.harvest_fruit(self.x, self.y):
                    self.payload += 1 # Add the fruit to our backpack.
                    print(f"[Drone {self.drone_id}] Harvested fruit! Payload: {self.payload}/{self.max_payload}")
                    
                    # --- CHAIN HARVESTING CHECK (Optimization) ---
                    # Since trees can have multiple fruits, check if there's more 
                    # fresh fruit right here on this very tree before we leave.
                    still_has_fruit = False
                    if (self.x, self.y) in grid_world.fruits:
                        # 'any' returns True if at least one item matches the condition
                        still_has_fruit = any(
                            not f['harvested'] and f['state'] == "fresh" 
                            for f in grid_world.fruits[(self.x, self.y)]
                        )

                    # Update local visual memory. If the tree is empty, we "forget" it.
                    if (self.x, self.y) in self.known_fresh_fruits and not still_has_fruit:
                        del self.known_fresh_fruits[(self.x, self.y)]
                        
                    # --- HIVE MIND COMMUNICATION ---
                    # Update the swarm's global map based on our new findings.
                    if still_has_fruit or self.known_fresh_fruits:
                        # Calculate the new 'nectar' value to keep Onlookers coming.
                        fruit_count = len(self.known_fresh_fruits) + (1 if still_has_fruit else 0)
                        abc_optimizer.register_food_source(self.x, self.y, fruit_count=fruit_count)
                    else:
                        # Area is completely clean. Tell the swarm to update the Tabu List.
                        abc_optimizer.report_search_result(self.x, self.y, found_fruit=False)
                    
                    # Harvesting action is complete. Reset the physical state.
                    self.state = DroneState.IDLE
                    
                    # --- NEXT ACTION DECISION (Stubborn Exploitation) ---
                    # The crucial logic that prevents drones from ignoring spotted fruits.
                    if still_has_fruit and self.payload < self.max_payload:
                        # CHAIN HARVEST: Tree still has fruit and we have space. Stay exactly here!
                        self.target_pos = (self.x, self.y) 
                        self.abc_role = ABCRole.EMPLOYED 
                        
                    elif self.known_fresh_fruits and self.payload < self.max_payload:
                        # STUBBORN SEARCH: This tree is empty, BUT we saw another fruit nearby!
                        # Don't wait for the planning phase. Lock onto the nearest fruit instantly.
                        next_target = min(self.known_fresh_fruits.keys(), 
                                          key=lambda p: abs(p[0] - self.x) + abs(p[1] - self.y))
                        self.target_pos = next_target
                        self.abc_role = ABCRole.EMPLOYED
                        
                    else:
                        # Backpack full, or absolutely nothing nearby. Drop the target 
                        # so the logistics phase can send us home or make us a Scout.
                        self.target_pos = None 
                        
                else:
                     # Race Condition: The timer finished, but another drone on the 
                     # same cell grabbed the last fruit just before we did.
                     print(f"[Drone {self.drone_id}] Missed target. Fruit was stolen!")
                     abc_optimizer.report_search_result(self.x, self.y, found_fruit=False)
                     self.state = DroneState.IDLE
                     self.target_pos = None 
                     
            return # We are done processing this frame.

        # =====================================================================
        # 1. SURVIVAL & LOGISTICS (Return to Base Phase)
        # =====================================================================
        # Two triggers force a return home: Critical battery OR a full backpack.
        needs_return = self.check_battery(grid_world.base_pos) or (self.payload >= self.max_payload)
        
        if needs_return:
            self.state = DroneState.RETURNING
            
            # Override whatever we were doing and set a direct course for the base.
            if self.target_pos != grid_world.base_pos:
                self.target_pos = grid_world.base_pos
                
            self._move_towards_target(grid_world)
            
            # Have we successfully landed?
            if (self.x, self.y) == grid_world.base_pos:
                # Service the drone
                self.battery = self.max_battery 
                self.payload = 0                
                self.state = DroneState.IDLE
                
                # Crucial ABC Logic: After resting, the drone naturally becomes an Onlooker.
                self.abc_role = ABCRole.ONLOOKER 
                self.target_pos = None
            return # Halt execution. Returning drones shouldn't try to pick fruit on the way.

        # =====================================================================
        # 2. SWARM PLANNING (Target Acquisition Phase)
        # =====================================================================
        # If the drone has no current objective, it asks the ABC Optimizer for one.
        if self.target_pos is None:
            self.state = DroneState.EXPLORING 
            
            if self.abc_role == ABCRole.SCOUT:
                # Ask the Tabu Search for a random, unexplored coordinate.
                self.target_pos = abc_optimizer.get_scout_target()
                
            # Both Onlookers AND Employed bees who lost their target should 
            # ask the Hive Mind where the best known food is. This prevents 
            # Employed bees from accidentally "forgetting" their job.
            elif self.abc_role in [ABCRole.ONLOOKER, ABCRole.EMPLOYED]:
                target = abc_optimizer.get_onlooker_target()
                if target:
                    self.target_pos = target
                    self.abc_role = ABCRole.EMPLOYED 
                else:
                    # Swarm memory is empty. Demote to Scout to help map the area.
                    self.abc_role = ABCRole.SCOUT
                    self.target_pos = abc_optimizer.get_scout_target()

        # =====================================================================
        # 3. ACTION (Movement Phase)
        # =====================================================================
        # Execute one step of the A* path towards the current target_pos.
        self._move_towards_target(grid_world)

        # =====================================================================
        # 4. PERCEPTION & EXPLOITATION (Arrival Phase)
        # =====================================================================
        # Have we arrived at our destination coordinate?
        if (self.x, self.y) == self.target_pos:
            
            # Look around with the CNN
            self.scan_environment(grid_world, classifier)
            
            # --- PRE-HARVEST CHECK ---
            # Don't blindly start the 5-tick harvest timer. Verify the environment 
            # ground truth to ensure a fresh, unpicked fruit actually exists here.
            if (self.x, self.y) in grid_world.fruits:
                has_available_fruit = any(
                    not f['harvested'] and f['state'] == "fresh" 
                    for f in grid_world.fruits[(self.x, self.y)]
                )
                
                if has_available_fruit:
                     # Lock in and begin the work animation/timer.
                     self.start_harvesting() 
                     return 
            
            # --- VORACIOUS LOCAL SEARCH ---
            # If we reach this line, the target was a bust (empty or stolen).
            
            # Tell the Hive Mind this specific spot is empty so it can update the Tabu list.
            abc_optimizer.report_search_result(self.x, self.y, found_fruit=False)
            
            # Instead of immediately giving up, check our local visual memory. 
            if self.known_fresh_fruits:
                # Greedy optimization: Find the absolute closest fruit we know about.
                next_target = min(self.known_fresh_fruits.keys(), 
                                  key=lambda p: abs(p[0] - self.x) + abs(p[1] - self.y))
                self.target_pos = next_target
                self.abc_role = ABCRole.EMPLOYED # Remain Employed to work this cluster
            else:
                # The area is completely dry. Demote to Scout.
                self.abc_role = ABCRole.SCOUT
                self.target_pos = None