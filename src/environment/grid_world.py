import os
import glob
import numpy as np
import random

class Paths:
    """
    Centralized path configurations for the project.
    Storing these here makes it way easier to update directory structures later.
    """
    MODEL_PATH = "models/fruit_classifier.pth"  
    
    CLASS_LABELS = {
        0: "data/raw/test/apple",
        1: "data/raw/test/banana",
        2: "data/raw/test/orange",
        3: "data/raw/test/rottenapples",
        4: "data/raw/test/rottenbanana",
        5: "data/raw/test/rottenoranges"
    }

class CellType:
    """Integer mappings for grid contents to keep NumPy arrays blazing fast."""
    EMPTY = 0
    BASE_STATION = 1
    TREE = 2
    OBSTACLE = 3

class FruitType:
    """Biological state of the fruit."""
    FRESH = "fresh"   
    ROTTEN = "rotten" 
    NONE = "none"     

class GridWorld:
    """
    The main 2D physics and state engine for the Agri-Swarm simulation.
    Now upgraded to support multiple fruits per tree!
    """
    def __init__(self, width=50, height=50, base_pos=(0,0), tree_spacing=4):
        self.width = width
        self.height = height
        self.base_pos = base_pos
        self.tree_spacing = tree_spacing

        # 2D NumPy array for terrain and obstacles
        self.grid = np.full((self.width, self.height), CellType.EMPTY, dtype=int)
        
        # Fruits dictionary. Key: (x, y), Value: LIST of fruit dictionaries.
        # This allows a single tree to hold multiple fruits simultaneously!
        self.fruits = {} 
        
        self._initialize_world()

    def _initialize_world(self):
        """Builds the environment step-by-step."""
        self.grid[self.base_pos] = CellType.BASE_STATION
        self._generate_orchade()
        self._populate_fruits()
        self._add_random_obstacles(num_obstacles=int((self.width * self.height) * 0.02))

    def _generate_orchade(self):
        """Plants trees in a structured agricultural grid pattern."""
        for x in range(2, self.width - 2, self.tree_spacing):
            for y in range(2, self.height - 2, 2):
                # Leave a clear zone around the base station
                if abs(x - self.base_pos[0]) + abs(y - self.base_pos[1]) > 5:
                    self.grid[x, y] = CellType.TREE
    
    def _populate_fruits(self):
        """
        Walks through every tree and generates a random cluster of fruits for it.
        Links real images from the dataset to each generated fruit.
        """
        tree_indices = np.argwhere(self.grid == CellType.TREE)

        fruit_name_mapping = {
            0: 'apple', 1: 'banana', 2: 'orange',
            3: 'rottenapples', 4: 'rottenbanana', 5: 'rottenoranges'
        }
        
        for tree_pos in tree_indices:
            x, y = tuple(tree_pos)
            
            # Initialize an empty list for this tree's canopy
            self.fruits[(x, y)] = []
            
            # 80% chance the tree yields fruit. If it does, it generates 1 to 5 fruits.
            if random.random() < 0.8:
                num_fruits = random.randint(1, 5)
                
                for _ in range(num_fruits):
                    # 70% chance this specific fruit is fresh and good to pick
                    fresh_fruit = random.random() < 0.7  

                    if fresh_fruit:
                        state = FruitType.FRESH
                        class_idx = random.choice([0, 1, 2])
                    else:
                        state = FruitType.ROTTEN
                        class_idx = random.choice([3, 4, 5])
                    
                    folder_path = Paths.CLASS_LABELS[class_idx]
                    fruit_type = fruit_name_mapping[class_idx]

                    # Grab a real image for the CNN to look at
                    image_path = None
                    if os.path.exists(folder_path):
                        images = glob.glob(os.path.join(folder_path, "*.*"))
                        if images:
                            image_path = random.choice(images)
                    
                    # Append the newly grown fruit to the tree's branch (list)
                    self.fruits[(x, y)].append({
                        'state': state,
                        'type': fruit_type,
                        'class_idx': class_idx,
                        'image_path': image_path,
                        'harvested': False 
                    })

    def _add_random_obstacles(self, num_obstacles):
        """Drops physical barriers onto the grid."""
        placed = 0
        while placed < num_obstacles:
            rx = random.randint(0, self.width - 1)
            ry = random.randint(0, self.height - 1)
            if self.grid[rx, ry] == CellType.EMPTY:
                self.grid[rx, ry] = CellType.OBSTACLE
                placed += 1
    
    def is_valid_move(self, x, y):
        """Collision detection. Prevents flying off-map or into obstacles."""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False 
        if self.grid[x, y] == CellType.OBSTACLE:
            return False 
        return True
    
    def get_local_view(self, center_x, center_y, radius=2):
        """
        The drone's sensor suite.
        Returns the visible grid and the FIRST unharvested fruit on each visible tree 
        so the drone's CNN can classify the overall state of that tree.
        """
        min_x = max(0, center_x - radius)
        max_x = min(self.width, center_x + radius + 1)
        min_y = max(0, center_y - radius)
        max_y = min(self.height, center_y + radius + 1)
        
        view_grid = self.grid[min_x:max_x, min_y:max_y]
        
        visible_fruits = {}
        for vx in range(min_x, max_x):
            for vy in range(min_y, max_y):
                # If there is a tree here with a list of fruits...
                if (vx, vy) in self.fruits:
                    # Find the first fruit that hasn't been picked yet
                    for f in self.fruits[(vx, vy)]:
                        if not f['harvested']:
                            # Present this single fruit to the drone's camera
                            visible_fruits[(vx, vy)] = f
                            break # We only need to show one representative fruit per tree to the CNN
                            
        return view_grid, visible_fruits
    
    def harvest_fruit(self, x, y):
        """
        Attempts to pick EXACTLY ONE fresh fruit from the tree at (x, y).
        Because a tree can have multiple fruits, multiple drones can harvest 
        from the same tree over time!
        """
        if (x, y) in self.fruits:
            # Iterate through the canopy to find a good apple
            for fruit in self.fruits[(x, y)]:
                if not fruit['harvested'] and fruit['state'] == FruitType.FRESH:
                    # Pluck it!
                    fruit['harvested'] = True
                    return True # Success
                    
        return False # Tree is empty or only has rotten fruit left