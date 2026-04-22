import pygame
import sys
from src.swarm.drone import DroneState,ABCRole

# Define a color palette for the simulation
COLORS = {
    'BACKGROUND': (30, 30, 30),
    'GRID': (50, 50, 50),
    'BASE': (0, 150, 255),        # Bright Blue
    'TREE_EMPTY': (139, 69, 19),  # Brown (Picked clean)
    'TREE_FRESH': (34, 139, 34),  # Forest Green (Has fresh fruit!)
    'TREE_ROTTEN': (128, 128, 0), # Olive Green (Only rotten fruit left)
    'OBSTACLE': (100, 100, 100),  # Grey
    'DRONE': (255, 255, 255),     # White
    'DRONE_SCOUT': (0, 255, 255), # Cyan
    'DRONE_HARVESTING': (0, 255, 0), # Green (Currently picking)
    'DRONE_RETURNING': (255, 165, 0)   # Orange (Low battery / Full)
}

class Visualizer:
    """
    Handles the Pygame window and renders the GridWorld and Drones.
    """
    def __init__(self, width, height, cell_size=15):
        pygame.init()
        self.cell_size = cell_size
        self.screen_width = width * cell_size
        self.screen_height = height * cell_size
        
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Agri-Swarm: ABC Swarm Intelligence Simulation")
        self.clock = pygame.time.Clock()

    def draw_world(self, grid_world):
        """Renders the static and dynamic elements of the grid."""
        self.screen.fill(COLORS['BACKGROUND'])
        
        for x in range(grid_world.width):
            for y in range(grid_world.height):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                
                # Draw grid lines
                pygame.draw.rect(self.screen, COLORS['GRID'], rect, 1)
                
                cell_type = grid_world.grid[x, y]
                
                if cell_type == 1: # BASE_STATION
                    pygame.draw.rect(self.screen, COLORS['BASE'], rect)
                    
                elif cell_type == 2: # TREE
                    # Default to an empty, brown tree trunk color
                    tree_color = COLORS['TREE_EMPTY']
                    
                    if (x, y) in grid_world.fruits:
                        fruits_list = grid_world.fruits[(x, y)]
                        
                        # Tally up what is physically left on the tree branches
                        fresh_count = sum(1 for f in fruits_list if not f['harvested'] and f['state'] == "fresh")
                        rotten_count = sum(1 for f in fruits_list if not f['harvested'] and f['state'] == "rotten")
                        
                        # Change the tree canopy color based on its contents
                        if fresh_count > 0:
                            tree_color = COLORS['TREE_FRESH']
                        elif rotten_count > 0:
                            tree_color = COLORS['TREE_ROTTEN']
                            
                    # Draw the base tree
                    pygame.draw.circle(self.screen, tree_color, rect.center, self.cell_size // 2 - 2)
                    
                    # If it's a good tree, add a tiny red dot so the user can easily spot it
                    if tree_color == COLORS['TREE_FRESH']:
                        pygame.draw.circle(self.screen, (255, 50, 50), rect.center, 3)
                        
                elif cell_type == 3: # OBSTACLE
                    pygame.draw.rect(self.screen, COLORS['OBSTACLE'], rect)

    def draw_drones(self, drones):
        """Renders all drones in the swarm."""
        for drone in drones:
            rect = pygame.Rect(drone.x * self.cell_size, drone.y * self.cell_size, self.cell_size, self.cell_size)
            
            # Change color based on state/role hierarchy
            color = COLORS['DRONE']
            if drone.state == DroneState.HARVESTING:
                color = COLORS['DRONE_HARVESTING']
            elif drone.state == DroneState.RETURNING:
                color = COLORS['DRONE_RETURNING']
            elif drone.abc_role == ABCRole.SCOUT:
                color = COLORS['DRONE_SCOUT']
                
            pygame.draw.rect(self.screen, color, rect)
            
            # Small battery bar indicator (Green line above the drone)
            battery_w = (drone.battery / drone.max_battery) * self.cell_size
            pygame.draw.rect(self.screen, (0, 255, 0), (rect.x, rect.y - 3, battery_w, 2))

    def update(self):
        """
        Refreshes the display and caps the frame rate.

        Event handling has been intentionally removed from here and centralised in
        main.py. pygame.event.get() empties the event queue — having two callers
        meant the QUIT event was consumed silently by the visualizer (triggering a
        raw sys.exit()) and never reached main.py's clean-shutdown logic.
        """
        pygame.display.flip()
        self.clock.tick(10)  # 10 FPS cap