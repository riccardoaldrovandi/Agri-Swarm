import numpy as np
import src.environment.grid_world
import heapq
import random

class DroneState:
    """Enumeration for the drone's current operational state."""
    EXPLORING = "exploring"
    HARVESTING = "harvesting" # Currently picking a fruit
    RETURNING = "returning" # Low battery, heading to base
    IDLE = "idle"

class ABCRole:
    """Enumeration for the Artificial Bee Colony roles."""
    SCOUT = "scout"       # Explores randomly
    ONLOOKER = "onlooker" # Waits at base, picks best known sources
    EMPLOYED = "employed" # Exploits a specific known source

class Drone:
    """
    Represents a single autonomous agent in the Agri-Swarm simulation.
    """
    def __init__(self, drone_id, start_x, start_y, max_battery=100):
        self.drone_id = drone_id
        self.x = start_x
        self.y = start_y

        # Battery management
        self.max_battery = max_battery
        # Battery Desync: Starts at a random value between 80% and 100%
        self.battery = random.randint(int(max_battery * 0.8), max_battery)

        self.state = DroneState.IDLE

        # Every drone starts as a Scout trying to find the first fruits
        self.abc_role = ABCRole.SCOUT 
        self.target_pos = None  # Tuple (x, y) where the drone is trying to go

        # Memory: Dictionary to store the locations of fresh fruits found
        # Key: (x, y), Value: Confidence score from CNN
        self.known_fresh_fruits = {}

        # --- Physical Constraints ---
        self.payload = 0           # How many fruits it's currently carrying
        self.max_payload = 1       # Maximum carrying capacity
        self.harvest_timer = 0     # Counter for the harvesting action
        self.harvest_duration = 5  # It takes 5 ticks to pick a fruit
    
    def scan_environment(self, grid_world, classifier):
        """
        Uses the grid's local view to find fruits and the CNN to classify them.
        """

        # If the backpack is full, we don't waste time scanning/memorizing new fruit
        if self.payload >= self.max_payload:
            return
        
        # 1. Get the local surroundings from the GridWorld
        _, visible_fruits = grid_world.get_local_view(self.x, self.y, radius=2)
        
        # 2. Process what we see
        for pos, fruit_data in visible_fruits.items():
            # If we haven't already memorized this spot
            if pos not in self.known_fresh_fruits:
                image_path = fruit_data['image_path']
                
                # 3. Use the Deep Learning model to "look" at the fruit
                if image_path:
                    predicted_class, confidence = classifier.predict(image_path)
                    
                    # 4. If the CNN thinks it's fresh, remember the location!
                    if "fresh" in predicted_class and confidence > 0.7:
                        self.known_fresh_fruits[pos] = confidence
                        print(f"[Drone {self.drone_id}] has found a fresh fruit in {pos} (Conf: {confidence:.2f})")
    
    def move(self, dx, dy, grid_world):
        """
        Attempts to move the drone by (dx, dy). Consumes battery.
        """
        if self.battery <= 0:
            return False # Dead battery
            
        new_x = self.x + dx
        new_y = self.y + dy
        
        # Check collision physics
        if grid_world.is_valid_move(new_x, new_y):
            self.x = new_x
            self.y = new_y
            self.battery -= 1 # Movement costs energy
            return True
            
        return False # Move blocked
    
    def start_harvesting(self):
        """Initiates the harvesting process."""
        self.state = DroneState.HARVESTING
        self.harvest_timer = self.harvest_duration
        
    def check_battery(self, base_pos):
        """
        Calculates if the drone has just enough battery to return home.
        Uses Manhattan distance for a grid world.
        """
        distance_to_base = abs(self.x - base_pos[0]) + abs(self.y - base_pos[1])
        
        # If battery is exactly enough to get home (plus a small safety margin of 7,considering the time to harvest))
        if self.battery < distance_to_base + 7:
            self.state = DroneState.RETURNING
            return True
        return False
    
    def _plan_path_astar(self, grid_world, start_pos, target_pos):
        """
        A* Search Algorithm.
        Finds the guaranteed shortest path avoiding obstacles using heuristics.
        """
        # The Priority Queue stores tuples: (f_score, (x, y), path_taken)
        # In heapq, the first element of the tuple (f_score) determines the priority.
        queue = []
        heapq.heappush(queue, (0, start_pos, []))
        
        visited = set([start_pos])
        
        # Directions: Up, Down, Left, Right
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        
        while queue:
            f_score, current, path = heapq.heappop(queue)
            
            # If we have reached the target, return the path
            if current == target_pos:
                return path
                
            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy
                
                # Check if the move is valid and not already visited
                if grid_world.is_valid_move(nx, ny) and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    
                    # Cost function g(n): steps taken so far
                    g_score = len(path) + 1
                    
                    # Heuristic function h(n): estimated distance to target
                    h_score = abs(nx - target_pos[0]) + abs(ny - target_pos[1])
                    
                    # f(n) = g(n) + h(n) (The core of the A* algorithm)
                    new_f_score = g_score + h_score
                    
                    # Add to priority queue
                    heapq.heappush(queue, (new_f_score, (nx, ny), path + [(dx, dy)]))
                    
        return [] # No path found
    
    def _move_towards_target(self, grid_world):
        """Smart pathfinding: calculates and executes the next best step."""
        if not self.target_pos:
            return
        
        # 1. Ask the Planning algorithm for the shortest route using A*
        path = self._plan_path_astar(grid_world, (self.x, self.y), self.target_pos)

        # 2. If a route exists, take the very first step
        if path:
            dx, dy = path[0]
            self.move(dx, dy, grid_world)
        else:
            # 3. ROBUSTNESS: If target is unreachable, drop it.
            if self.target_pos != grid_world.base_pos:
                self.target_pos = None

    def step(self, grid_world, classifier, abc_optimizer):
        """
        The main 'Brain Tick' of the drone. Called once per simulation frame.
        This handles the Finite State Machine (FSM), Swarm Intelligence roles,
        physical constraints (battery/payload), and interaction with the environment.
        """
        
        # =====================================================================
        # 0. ACTIVE WORK: Are we currently picking a fruit?
        # =====================================================================
        # If the drone is in the middle of harvesting, it cannot move or plan.
        # It must wait for the timer to finish.
        if self.state == DroneState.HARVESTING:
            self.harvest_timer -= 1
            self.battery -= 1 # Picking fruit is mechanical work, so it costs battery!
            
            # Is the picking process complete?
            if self.harvest_timer <= 0:
                # Attempt to actually remove the fruit from the grid.
                if grid_world.harvest_fruit(self.x, self.y):
                    self.payload += 1
                    print(f"[Drone {self.drone_id}] Successfully harvested 1 fruit! (Backpack Full)")
                    
                    # Remove it from our local visual memory so we don't try to pick it again
                    if (self.x, self.y) in self.known_fresh_fruits:
                        del self.known_fresh_fruits[(self.x, self.y)]
                        
                    # --- COMMUNICATION WITH HIVE MIND ---
                    # We grabbed the fruit. Now, let's tell the swarm what else is here.
                    if self.known_fresh_fruits:
                        # We see other fruits! Register this source. 
                        # The number of fruits acts as the 'nectar' value to attract Onlookers.
                        abc_optimizer.register_food_source(self.x, self.y, len(self.known_fresh_fruits))
                    else:
                        # We took the last fruit. Tell the swarm this tree is dead.
                        abc_optimizer.report_search_result(self.x, self.y, found_fruit=False)
                        
                else:
                     # Edge case: We spent 5 ticks trying to pick a fruit, but another 
                     # drone swooped in and grabbed it first.
                     print(f"[Drone {self.drone_id}] Missed target. Fruit was stolen!")
                     abc_optimizer.report_search_result(self.x, self.y, found_fruit=False)
                
                # We are done working. Reset state to IDLE and clear target 
                # so the drone figures out what to do next tick.
                self.state = DroneState.IDLE
                self.target_pos = None 
                
            return # Halt execution here for this frame.


        # =====================================================================
        # 1. SURVIVAL & LOGISTICS: Do we need to go home?
        # =====================================================================
        # A drone must return to the base if its battery is critical, 
        # OR if its backpack is full (payload reached max capacity).
        needs_return = self.check_battery(grid_world.base_pos) or (self.payload >= self.max_payload)
        
        if needs_return:
            self.state = DroneState.RETURNING
            
            # Override current target and force the drone to fly to the base
            if self.target_pos != grid_world.base_pos:
                self.target_pos = grid_world.base_pos
                
            self._move_towards_target(grid_world)
            
            # Have we successfully landed on the charging pad?
            if (self.x, self.y) == grid_world.base_pos:
                self.battery = self.max_battery # Recharge to 100%
                self.payload = 0                # Empty the backpack
                self.state = DroneState.IDLE
                
                # After resting at the base, the drone naturally becomes an Onlooker
                # waiting to be dispatched to a known good location.
                self.abc_role = ABCRole.ONLOOKER 
                self.target_pos = None
                
            return # Halt execution. We don't want returning drones to harvest.


        # =====================================================================
        # 2. SWARM PLANNING: Ask the Hive Mind for orders
        # =====================================================================
        # If we are flying aimlessly, we need to request a target coordinate.
        if self.target_pos is None:
            self.state = DroneState.EXPLORING # Gives the drone a visual active state
            
            if self.abc_role == ABCRole.SCOUT:
                # Go to a completely random unknown coordinate
                self.target_pos = abc_optimizer.get_scout_target()
                
            elif self.abc_role == ABCRole.ONLOOKER:
                # Ask the Dance Floor for the best known spot (Roulette Wheel selection)
                target = abc_optimizer.get_onlooker_target()
                if target:
                    self.target_pos = target
                    self.abc_role = ABCRole.EMPLOYED # We got a job!
                else:
                    # The Hive Mind is empty. Nobody knows where food is. 
                    # Step up and become a Scout to help the swarm.
                    self.abc_role = ABCRole.SCOUT
                    self.target_pos = abc_optimizer.get_scout_target()
                    
            elif self.abc_role == ABCRole.EMPLOYED:
                # Safe fallback: an Employed bee shouldn't be without a target.
                self.abc_role = ABCRole.SCOUT


        # =====================================================================
        # 3. ACTION: Physical Movement
        # =====================================================================
        # Use A-Star pathfinding to take one step closer to self.target_pos
        self._move_towards_target(grid_world)


        # =====================================================================
        # 4. PERCEPTION & EXPLOITATION: We arrived at the coordinates
        # =====================================================================
        if (self.x, self.y) == self.target_pos:
            
            # Fire up the CNN and scan the immediate area (Radius=2)
            self.scan_environment(grid_world, classifier)
            
            # --- PRE-HARVEST CHECK ---
            # We don't blindly harvest. We check the GridWorld's ground truth to see 
            # if a fresh fruit is actually sitting exactly under us right now.
            if (self.x, self.y) in grid_world.fruits and not grid_world.fruits[(self.x, self.y)]['harvested']:
                 if grid_world.fruits[(self.x, self.y)]['state'] == "fresh":
                     # We found it! Lock the drone into the harvesting sequence.
                     # It will stay here for the next 5 ticks.
                     self.start_harvesting() 
                     return # Halt execution
            
            # --- VORACIOUS LOCAL SEARCH ---
            # If we reach this line, the target cell was EMPTY. 
            # Either it was a bad scout coordinate, or another drone stole the fruit.
            
            # Let the swarm know this specific tile is a bust.
            abc_optimizer.report_search_result(self.x, self.y, found_fruit=False)
            
            # Wait, before we give up and fly away as a Scout... did our camera 
            # catch a glimpse of another fruit nearby during the scan?
            if self.known_fresh_fruits:
                # Yes! Don't fly across the map. Just walk over to the nearest one.
                # Calculate the Manhattan distance to all known fruits and pick the closest.
                next_target = min(self.known_fresh_fruits.keys(), 
                                  key=lambda p: abs(p[0] - self.x) + abs(p[1] - self.y))
                
                self.target_pos = next_target
                self.abc_role = ABCRole.EMPLOYED # Keep working this patch
                
            else:
                # No fruits nearby. This area is completely dry.
                # Resign our position and become a Scout to find a new patch.
                self.abc_role = ABCRole.SCOUT
                self.target_pos = None