import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
from scipy.spatial.distance import cityblock as manhattan_dist
from math import exp

# Assume ship_layout.py is correctly set up with generate_ship_layout function
from ship_layout import generate_ship_layout

# Constants for grid cell states
EMPTY = 1
BOT = 2
ALIEN = 3
CREW_MEMBER = 4
BLOCKED = 0

# Parameters
D = 35  # Dimension of the grid
k = 3  # Detection range
alpha = 0.5  # Sensitivity of the beep detection
no_of_aliens = 1  # Number of aliens

# Generate ship layout
ship_layout = generate_ship_layout(D)

# Function definitions
def manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

# Function to generate a random position on the ship layout
def random_position(D, grid):
    possible_indices = np.argwhere(grid == 1)
    return tuple(possible_indices[random.choice(range(len(possible_indices)))])

# Determine if an alien is within detection range
def is_within_detection_range(bot_position, alien_position, k):
    return manhattan_dist(bot_position, alien_position) <= k

# Place aliens considering the detection range
def place_aliens_outside_detection(D, grid, no_of_aliens, bot_position, k):
    aliens = []
    while len(aliens) < no_of_aliens:
        potential_position = random_position(D, grid)
        if potential_position not in aliens and not is_within_detection_range(bot_position, potential_position, k):
            aliens.append(potential_position)
    return aliens

def get_valid_moves(position, grid):
    """
    Returns a list of valid moves (up, down, left, right) for a given position.
    """
    x, y = position
    moves = []
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Right, down, left, up
        if 0 <= x + dx < grid.shape[0] and 0 <= y + dy < grid.shape[1] and grid[x + dx, y + dy] == 1:
            moves.append((x + dx, y + dy))
    return moves

def move_aliens(aliens, grid):
    """
    Moves each alien to a random adjacent open cell.
    """
    new_positions = []
    for position in aliens:
        valid_moves = get_valid_moves(position, grid)
        if valid_moves:  # If there are valid moves, pick one randomly
            new_positions.append(random.choice(valid_moves))
        else:  # If no valid moves, the alien stays in its current position
            new_positions.append(position)
    return new_positions

def visualize_layout_with_details(layout, bot_position, crew_positions, alien_positions, k):
    D = layout.shape[0]  # Grid dimension

    # Copy the layout to overlay entities without altering the original layout
    visual_layout = np.copy(layout)
    
    # Setting up colors: 0 for blocked, 1 for open, 2 for detection area, 3 for bot, 4 for crew, 5 for aliens
    for x in range(max(0, bot_position[0] - k), min(D, bot_position[0] + k + 1)):
        for y in range(max(0, bot_position[1] - k), min(D, bot_position[1] + k + 1)):
            if visual_layout[x, y] == 1:  # Mark detection area on open cells
                visual_layout[x, y] = 2

    # Mark the bot, crew, and aliens on the visual layout
    visual_layout[bot_position] = 3
    for pos in crew_positions:
        visual_layout[pos] = 4
    for pos in alien_positions:
        visual_layout[pos] = 5

    # Create a custom colormap: black for blocked, darkgrey for open, blue for detection area,
    # dodgerblue for bot, lime for crew, red for aliens
    cmap = ListedColormap(['black', 'darkgrey', 'dodgerblue', 'blue', 'lime', 'red'])
    
    plt.figure(figsize=(10, 10))
    plt.imshow(visual_layout, cmap=cmap, interpolation='nearest')
    plt.xticks([]), plt.yticks([])  # Hide axis ticks
    plt.show()


# Determine if a beep is detected from crew members
def crewBeep(curr_pos, crew_positions, dist, alpha, open_cells):
    for i, crew_pos in enumerate(crew_positions):
        d = dist[i][np.where((open_cells == curr_pos).all(axis=1))[0][0]]
        p = exp(-alpha * (d-1))
        if random.random() < p:
            return True
    return False

def initialize_belief_matrix(D, bot_position, crew_position):
    belief_matrix = np.zeros((D, D))
    # Initially, mark the bot's and crew's positions with special values or keep separate
    belief_matrix[bot_position] = -1  # For example, -1 for bot
    belief_matrix[crew_position] = -2  # For example, -2 for crew
    return belief_matrix

def update_belief_for_alien(belief_matrix, bot_position, detection_zone, detected):
    # Update belief based on alien detection or absence
    if detected:
        # Increase belief within detection zone
        pass  # Implementation needed
    else:
        # Decrease belief within detection zone or adjust outside
        pass  # Implementation needed

def move_bot_towards_crew(belief_matrix, bot_position, crew_position):
    # Determine the next move based on current position and target
    pass  # Implementation needed

def account_for_alien_movement(belief_matrix):
    # Spread the probability of alien's presence to adjacent cells
    pass  # Implementation needed

# Place entities on the layout
bot_position = random_position(D, ship_layout)
crew_positions = [random_position(D, ship_layout) for _ in range(1)]  # Single crew member for this example
open_cells = np.argwhere(ship_layout == 1)
dist = np.array([[manhattan_dist(cell, crew) for cell in open_cells] for crew in crew_positions])
alien_positions = place_aliens_outside_detection(D, ship_layout, no_of_aliens, bot_position, k)
belief_matrix = initialize_belief_matrix(D, bot_position, crew_position)

# Visualization and sensor checks
#visualize_layout_with_entities(ship_layout, bot_position, crew_positions, alien_positions, k)
visualize_layout_with_details(ship_layout, bot_position, crew_positions, alien_positions, k)

# Example sensor checks
alien_detected = any(is_within_detection_range(bot_position, alien_pos, k) for alien_pos in alien_positions)
crew_beep_detected = crewBeep(bot_position, crew_positions, dist, alpha, open_cells)
print("Alien detected within range:" if alien_detected else "No alien detected within range.")
print("Beep detected from crew." if crew_beep_detected else "No beep detected from crew.")
