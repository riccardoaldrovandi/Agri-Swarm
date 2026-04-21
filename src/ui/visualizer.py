import pygame
import sys

# Define a color palette for the simulation
COLORS = {
    'BACKGROUND': (30, 30, 30),
    'GRID': (50, 50, 50),
    'BASE': (0, 150, 255),    # Bright Blue
    'TREE': (34, 139, 34),    # Forest Green
    'FRUIT_FRESH': (255, 50, 50), # Red
    'FRUIT_ROTTEN': (139, 69, 19), # Brown
    'OBSTACLE': (100, 100, 100), # Grey
    'DRONE': (255, 255, 255),    # White
    'DRONE_SCOUT': (0, 255, 255), # Cyan
    'DRONE_HARVESTING': (0, 255, 0), # Green
    'DRONE_RETURNING': (255, 165, 0) # Orange (Low battery)
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
                
                # Render specific cell types
                if cell_type == 1: # BASE_STATION
                    pygame.draw.rect(self.screen, COLORS['BASE'], rect)
                elif cell_type == 2: # TREE
                    pygame.draw.circle(self.screen, COLORS['TREE'], rect.center, self.cell_size // 2 - 2)
                    # Draw fruit if present on tree
                    if (x, y) in grid_world.fruits:
                        fruit = grid_world.fruits[(x, y)]
                        if not fruit['harvested']:
                            f_color = COLORS['FRUIT_FRESH'] if fruit['state'] == "fresh" else COLORS['FRUIT_ROTTEN']
                            pygame.draw.circle(self.screen, f_color, rect.center, self.cell_size // 4)
                elif cell_type == 3: # OBSTACLE
                    pygame.draw.rect(self.screen, COLORS['OBSTACLE'], rect)

    def draw_drones(self, drones):
        """Renders all drones in the swarm."""
        for drone in drones:
            rect = pygame.Rect(drone.x * self.cell_size, drone.y * self.cell_size, self.cell_size, self.cell_size)
            
            # Change color based on state/role
            color = COLORS['DRONE']
            if drone.state == "returning":
                color = COLORS['DRONE_RETURNING']
            elif drone.abc_role == "scout":
                color = COLORS['DRONE_SCOUT']
            elif drone.state == "harvesting":
                color = COLORS['DRONE_HARVESTING']
                
            pygame.draw.rect(self.screen, color, rect)
            # Small battery bar indicator
            battery_w = (drone.battery / drone.max_battery) * self.cell_size
            pygame.draw.rect(self.screen, (0, 255, 0), (rect.x, rect.y - 3, battery_w, 2))

    def update(self):
        """Refreshes the display and handles basic events."""
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        self.clock.tick(10) # 10 FPS for visibility