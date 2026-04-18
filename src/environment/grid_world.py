import os
import glob
import numpy as np
import random

class Paths:
    """
    Centralized path configurations for the project.
    Storing these here makes it way easier to update directory structures later
    without having to hunt down raw strings scattered across the codebase.
    """
    MODEL_PATH = "models/fruit_classifier.pth"  
    
    # Maps our CNN class indices to the actual raw data directories.
    # This acts as the bridge between the simulated environment and our real image data.
    CLASS_LABELS = {
        0: "data/raw/apple",
        1: "data/raw/banana",
        2: "data/raw/orange",
        3: "data/raw/rottenapples",
        4: "data/raw/rottenbanana",
        5: "data/raw/rottenoranges"
    }

class CellType:
    """
    Simple enum-like class to keep track of what's occupying a grid cell.
    Using integers makes the underlying NumPy array operations much faster.
    """
    EMPTY = 0
    BASE_STATION = 1
    TREE = 2
    OBSTACLE = 3

class FruitType:
    """
    Defines the biological state of the fruit on the tree.
    These states will eventually trigger different behaviors in the drones.
    """
    FRESH = "fresh"   # Good fruit, target for harvesting
    ROTTEN = "rotten" # Bad fruit, ignore or mark as spoiled
    NONE = "none"     # Just an empty branch

class GridWorld:
    """
    The main 2D physics and state engine for the Agri-Swarm simulation.
    It manages the map layout, keeps track of where all the trees and fruits are,
    and enforces movement rules (like not flying into obstacles).
    """
    def __init__(self, width=50, height=50, base_pos=(0,0), tree_spacing=4):
        self.width = width
        self.height = height
        self.base_pos = base_pos
        self.tree_spacing = tree_spacing

        # We use a 2D NumPy array for the terrain. It's memory efficient and
        # makes spatial slicing (like checking a drone's field of view) super fast.
        self.grid = np.full((self.width, self.height), CellType.EMPTY, dtype=int)
        
        # We track fruits separately in a dictionary rather than adding a third dimension 
        # to the grid array. It's much sparser and easier to update when fruits get harvested.
        # Format -> Key: (x, y), Value: {state, type, class_idx, image_path, harvested}
        self.fruits = {} 
        
        # Kick off the world generation sequence
        self._initialize_world()

    def _initialize_world(self):
        """Builds the environment step-by-step."""
        # 1. Drop the base station at the starting coordinates
        self.grid[self.base_pos] = CellType.BASE_STATION

        # 2. Plant the trees in neat, agricultural rows
        self._generate_orchade()

        # 3. Grow fruits on those trees and link them to our real image dataset
        self._populate_fruits()

        # 4. Scatter some random obstacles (like sheds, rocks, or equipment).
        # We're setting the obstacle density to roughly 2% of the total map area.
        self._add_random_obstacles(num_obstacles=int((self.width * self.height) * 0.02))

    def _generate_orchade(self):
        """
        Plants trees in a structured grid pattern to simulate a real orchard.
        We leave a 2-cell buffer around the edges of the map.
        """
        for x in range(2, self.width - 2, self.tree_spacing):
            # We space trees closer together on the y-axis to create distinct rows
            for y in range(2, self.height - 2, 2):
                
                # Make sure we don't plant a tree right on top of the base station.
                # We want a clear takeoff/landing zone.
                if abs(x - self.base_pos[0]) + abs(y - self.base_pos[1]) > 5:
                    self.grid[x, y] = CellType.TREE
    
    def _populate_fruits(self):
        """
        Walks through every tree and decides if it gets fruit.
        If it does, we randomly pick a fruit type and state, and most importantly,
        we grab a real image path from our dataset so the CNN has something to look at later.
        """
        # Find every (x, y) coordinate that currently has a tree
        tree_indices = np.argwhere(self.grid == CellType.TREE)

        # Helper mapping to keep human-readable names tied to our CNN's integer classes
        fruit_name_mapping = {
            0: 'apple', 1: 'banana', 2: 'orange',
            3: 'rottenapples', 4: 'rottenbanana', 5: 'rottenoranges'
        }
        
        for tree_pos in tree_indices:
            x, y = tuple(tree_pos)
            
            # There's an 80% chance this tree actually grew something
            if random.random() < 0.8:
                # Of the trees with fruit, roughly 70% of the yield is good
                fresh_fruit = random.random() < 0.7  

                if fresh_fruit:
                    state = FruitType.FRESH
                    class_idx = random.choice([0, 1, 2])
                else:
                    state = FruitType.ROTTEN
                    class_idx = random.choice([3, 4, 5])
                
                # Figure out where the raw images for this specific fruit live
                folder_path = Paths.CLASS_LABELS[class_idx]
                fruit_type = fruit_name_mapping[class_idx]

                # Now, try to find a real image file we can pass to the perception module
                image_path = None
                if os.path.exists(folder_path):
                    # Grab all files in the directory (using *.* catches jpg, png, etc.)
                    images = glob.glob(os.path.join(folder_path, "*.*"))
                    if images:
                        image_path = random.choice(images)
                else:
                    # Good for catching missing data folders during setup
                    print(f"⚠️ Warning: Folder not found: -> {folder_path}")
                
                # Lock this fruit into the world's ground truth database
                self.fruits[(x, y)] = {
                    'state': state,
                    'type': fruit_type,
                    'class_idx': class_idx,       # Needed for the CNN
                    'image_path': image_path,     # The actual image the drone will "see"
                    'harvested': False            # Keeps track of what's been picked
                }

    def _add_random_obstacles(self, num_obstacles):
        """Drops physical barriers onto the grid that the drones will have to avoid."""
        placed = 0
        while placed < num_obstacles:
            rx = random.randint(0, self.width - 1)
            ry = random.randint(0, self.height - 1)
            
            # Only drop an obstacle if the spot is totally clear
            if self.grid[rx, ry] == CellType.EMPTY:
                self.grid[rx, ry] = CellType.OBSTACLE
                placed += 1
    
    def is_valid_move(self, x, y):
        """
        The collision detection system.
        Drones call this to check if their next planned step is legal.
        """
        # First check: Did they fly off the map?
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False 
        
        # Second check: Are they about to crash into a physical barrier?
        # Notice we don't block trees here, assuming drones fly over the canopy.
        if self.grid[x, y] == CellType.OBSTACLE:
            return False 
            
        return True
    
    def get_local_view(self, center_x, center_y, radius=2):
        """
        The drone's sensor suite.
        Instead of giving the drone the whole map, we only return a small window
        around its current position. This forces the swarm to actually explore.
        """
        # Figure out the bounding box of the drone's vision, clamped to the grid edges
        min_x = max(0, center_x - radius)
        max_x = min(self.width, center_x + radius + 1)
        min_y = max(0, center_y - radius)
        max_y = min(self.height, center_y + radius + 1)
        
        # Slice the numpy array to get the terrain data
        view_grid = self.grid[min_x:max_x, min_y:max_y]
        
        # Filter the fruits dictionary to only include what's currently in sight
        # and hasn't already been picked.
        visible_fruits = {}
        for vx in range(min_x, max_x):
            for vy in range(min_y, max_y):
                if (vx, vy) in self.fruits and not self.fruits[(vx, vy)]['harvested']:
                    visible_fruits[(vx, vy)] = self.fruits[(vx, vy)]
                    
        return view_grid, visible_fruits
    
    def harvest_fruit(self, x, y):
        """
        Attempts to pick a fruit at the given coordinates.
        Returns True if successful, False if the fruit is rotten, missing, or already gone.
        """
        if (x, y) in self.fruits and not self.fruits[(x, y)]['harvested']:
            # We only allow harvesting of fresh fruit. 
            if self.fruits[(x, y)]['state'] == FruitType.FRESH:
                self.fruits[(x, y)]['harvested'] = True
                return True
        return False