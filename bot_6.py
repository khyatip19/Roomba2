# bot6.py - Two Alien, Two Crew Member
# Bot 6 . Two Aliens and Two Crew Member case
# In this case, we have two aliens in play, and two crew members that needs 
# rescuing. Note that a crew-detection-beep is received if the bot receives 
# a beep from either crew member (beep = ((crew 1 detected) or (crew 2
# detected))). Note that similarly, the alien-detection square does not
#  distinguish between one or two aliens detected.
# Bot 6: Bot 6 is just Bot 1, applied in this new setting, but when the first
#  crew member is found, they are teleported away, and the updates continue 
# until the second crew member is found. Note that positively identifying a 
# square as containing an alien does not rule out other squares containing 
# an alien in this case.

import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns
import matplotlib.patches as patches

from heapq import heappush, heappop
from queue import PriorityQueue, Queue
from ship_layout import generate_ship_layout

# Constants for grid cell states
EMPTY = 1
BOT = 2
ALIEN = 3
CREW_MEMBER = 4
BLOCKED = 0

# Simulation parameters
D = 15  # Dimension of the grid
k = 3  # Sensor range for alien detection
alpha = 0.5  # Beep probability factor

def initialize_grid(D):
    """Initialize the simulation grid with blocked and unblocked cells."""
    grid_layout = generate_ship_layout(D)  # Get layout from ship_layout.py
    grid = np.where(grid_layout == 1, EMPTY, BLOCKED)  # Mark unblocked cells as EMPTY, blocked cells as BLOCKED
    return grid, grid_layout

def place_entities(grid, D, k, grid_layout):
    """
    Place the bot, two crew members, and an alien on the grid, ensuring they're not too close to each other.
    - Bot is placed randomly.
    - Crew members are placed at least k+1 cells away from the bot and each other.
    - Alien is placed at least 2*k+1 cells away from the bot to respect detection range.
    """
    bot_pos, crew_pos1, crew_pos2, alien_pos1, alien_pos2 = None, None, None, None, None

    while bot_pos is None or grid[bot_pos] != EMPTY:
        potential_pos = (np.random.randint(0, D), np.random.randint(0, D))
        if grid[potential_pos] == EMPTY and grid_layout[potential_pos]:
            bot_pos = potential_pos

    while crew_pos1 is None:
        potential_pos = (np.random.randint(0, D), np.random.randint(0, D))
        if grid[potential_pos] == EMPTY and grid_layout[potential_pos] and manhattan_distance(potential_pos[0], potential_pos[1], bot_pos[0], bot_pos[1]) > k:
            crew_pos1 = potential_pos

    while crew_pos2 is None:
        potential_pos = (np.random.randint(0, D), np.random.randint(0, D))
        if grid[potential_pos] == EMPTY and grid_layout[potential_pos] and manhattan_distance(potential_pos[0], potential_pos[1], bot_pos[0], bot_pos[1]) > k:
            crew_pos2 = potential_pos

    while alien_pos1 is None:
        potential_pos = (np.random.randint(0, D), np.random.randint(0, D))
        if grid[potential_pos] == EMPTY and grid_layout[potential_pos] and manhattan_distance(potential_pos[0], potential_pos[1], bot_pos[0], bot_pos[1]) > 2*k + 1:
            alien_pos1 = potential_pos
    
    while alien_pos2 is None:
        potential_pos = (np.random.randint(0, D), np.random.randint(0, D))
        if grid[potential_pos] == EMPTY and grid_layout[potential_pos] and manhattan_distance(potential_pos[0], potential_pos[1], bot_pos[0], bot_pos[1]) > 2*k + 1:
            alien_pos2 = potential_pos

    grid[bot_pos] = BOT
    grid[crew_pos1] = CREW_MEMBER
    grid[crew_pos2] = CREW_MEMBER
    grid[alien_pos1] = ALIEN
    grid[alien_pos2] = ALIEN

    return bot_pos, crew_pos1, crew_pos2, alien_pos1, alien_pos2

def manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def detect_alien_individual(bot_pos, alien_pos):
    """Determines if an alien is within detection range of the bot using Manhattan distance."""
    return (manhattan_distance(bot_pos[0], bot_pos[1], alien_pos[0], alien_pos[1]) <= 2*k + 1)

# def detect_alien(bot_pos, alien_pos1, alien_pos2):
#     """Determines if an alien is within detection range of the bot using Manhattan distance."""
#     return (manhattan_distance(bot_pos[0], bot_pos[1], alien_pos1[0], alien_pos1[1]) <= 2*k + 1) or (manhattan_distance(bot_pos[0], bot_pos[1], alien_pos2[0], alien_pos2[1]) <= 2*k + 1)

def detect_beep_individual(crew_pos, bot_pos):
    """Simulate beep detection from both crew members."""
    distance = manhattan_distance(crew_pos[0], crew_pos[1], bot_pos[0], bot_pos[1])
    beep_prob = np.exp(-alpha * (distance - 1))
    return np.random.uniform(0, 1.0) <= beep_prob

def get_adjacent_open_cells(x, y, grid_layout):
    """ Get open cells adjacent to (x, y) """
    adjacent_cells = []
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Right, Down, Left, Up
        if 0 <= x + dx < D and 0 <= y + dy < D:
            if grid_layout[x + dx, y + dy] == EMPTY:
                adjacent_cells.append((x + dx, y + dy))
    return adjacent_cells

def update_beliefs(bot_pos, alien_pos1, alien_pos2, crew_pos1, crew_pos2, D, k, alpha, prob_alien, prob_crew, grid_layout, crew_rescued):   
    # Detect alien and beep probabilities
    alien_detected1 = detect_alien_individual(bot_pos, alien_pos1)
    alien_detected2 = detect_alien_individual(bot_pos, alien_pos2)
    alien_detected = alien_detected1 or alien_detected2 
    beep_detected1 = detect_beep_individual(crew_pos1, bot_pos) if not crew_rescued[0] else False
    beep_detected2 = detect_beep_individual(crew_pos2, bot_pos) if not crew_rescued[1] else False
    beep_detected = beep_detected1 or beep_detected2
    
    for x in range(D):
        for y in range(D):
            if not grid_layout[(x, y)]:  # Skip updating beliefs for blocked cells
                continue

            distance = manhattan_distance(x, y, bot_pos[0], bot_pos[1])

            # Check if the cell is within the detection square of the bot
            in_detection_square = (abs(bot_pos[0] - x) <= k and abs(bot_pos[1] - y) <= k)

            # Update alien belief based on detection
            if in_detection_square:
                if alien_detected:
                    # If alien is detected inside the square, keep probabilities as they are
                    pass
                else:
                    # If alien is not detected inside the square, set probabilities to 0
                    prob_alien[x, y] = 0
            else:
                 if alien_detected:
                     # If alien is detected outside the square, set probabilities to 0
                     prob_alien[x, y] = 0

            # Update for beep detection from either crew member
            if beep_detected:
                # Increase probability for closer cells based on beep detection
                prob_crew[x, y] = prob_crew[x, y]*np.exp(-alpha * (distance - 1))
            else:
                # Decrease probability for cells outside beep range
                prob_crew[x, y] = prob_crew[x, y]*(1 - np.exp(-alpha * (distance - 1)))

    # Normalize only non-blocked cells
    total_prob_alien = np.sum(prob_alien[grid_layout == EMPTY])
    total_prob_crew = np.sum(prob_crew[grid_layout == EMPTY])
    if total_prob_alien > 0:
        prob_alien[grid_layout == EMPTY] /= total_prob_alien
    if total_prob_crew > 0:
        prob_crew[grid_layout == EMPTY] /= total_prob_crew

    # Diffuse the probabilities for the alien - If the alien is detected, set 
    # everything outside to 0 and diffuse whatever is inside and vice versa
    new_prob_alien = np.zeros((D, D))
    for x in range(D):
        for y in range(D):
            if prob_alien[x, y] > 0:
                adjacent_open_cells = get_adjacent_open_cells(x, y, grid_layout)
                distributed_prob = prob_alien[x, y] / len(adjacent_open_cells) if adjacent_open_cells else prob_alien[x, y]
                for adj_x, adj_y in adjacent_open_cells:
                    new_prob_alien[adj_x, adj_y] += distributed_prob

    # Update the alien belief matrix if there are any probabilities to diffuse
    if np.sum(new_prob_alien) > 0:
        prob_alien[:] = new_prob_alien / np.sum(new_prob_alien)

def bfs_search_next_step(start, goal, D, grid_layout):
    queue = Queue()
    queue.put(start)
    visited = {start}
    visited = []
    came_from = {start: None}

    while not queue.empty():
        current = queue.get()

        if current == goal:
            break

        # visited.append(current)
        #print(f"the current cell is : {current}")
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (current[0] + dx, current[1] + dy)

            if neighbor not in visited and 0 <= neighbor[0] < D and 0 <= neighbor[1] < D and grid_layout[neighbor] != BLOCKED:
                queue.put(neighbor)
                visited.append(neighbor)
                came_from[neighbor] = current

    # Reconstruct the path back to the start
    path = []
    if goal in came_from:
        current = goal
        while current != start:
            path.append(current)
            current = came_from[current]
        path.reverse()

    if path:
        return path[0], path  # Return the next step and the full path
    else:
        return start, None  # Indicate that no path was found

def choose_next_move(bot_pos, prob_alien, prob_crew, D, grid_layout):
    # Modify the goal selection to only consider unblocked cells
    unblocked_prob_crew = np.where(grid_layout == 1, prob_crew, 0)  # Zero out probabilities for blocked cells
    goal_pos = np.unravel_index(np.argmax(unblocked_prob_crew), unblocked_prob_crew.shape)  # Choose the highest unblocked probability
    print(f"The goal state is: {goal_pos}")
    next_step, path = bfs_search_next_step(bot_pos, goal_pos, D, grid_layout)  # Get the full path
    if path:
        return path[0], path  # Return the next step and the full path
    else:
        return bot_pos, []  # If no path, return the current position and an empty path

# def move_alien(alien_pos, D):
#     # All possible movements including staying in place
#     moves = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
#     move = random.choice(moves)  # Randomly choose a move
#     # Calculate new position and ensure it's within grid bounds
#     new_pos = (max(0, min(D - 1, alien_pos[0] + move[0])), max(0, min(D - 1, alien_pos[1] + move[1])))
#     return new_pos

def visualize_grid(grid, grid_layout, prob_alien, prob_crew, bot_pos, alien_positions, crew_positions, k, path, crew_rescued):
    plt.ion()  # Turn on interactive mode
    plt.figure(figsize=(12, 6))

    # Grid overview
    ax1 = plt.subplot(1, 3, 1)
    ax1.set_title("Grid Overview")
    blocked_cells = np.where(grid_layout == 0, True, False)  # Find blocked cells
    open_cells = np.logical_not(blocked_cells)  # Find open cells
    ax1.imshow(open_cells, cmap='BuPu', alpha=0.5)  # Color open cells
    ax1.imshow(blocked_cells, cmap='RdPu', alpha=0.5)  # Color blocked cells with a different color

     # Calculate the bounds for the detection area
    detection_left = max(bot_pos[1] - k, 0)
    detection_right = min(bot_pos[1] + k, D - 1)
    detection_top = max(bot_pos[0] - k, 0)
    detection_bottom = min(bot_pos[0] + k, D - 1)

     # Draw detection zone around bot
    detection_width = detection_right - detection_left + 1
    detection_height = detection_bottom - detection_top + 1
    detection_area = patches.Rectangle(
        (detection_left - 0.5, detection_top - 0.5),
        detection_width,
        detection_height,
        linewidth=1,
        edgecolor='purple',
        facecolor=(0.8, 0.6, 0.8, 0.2)
    )
    ax1.add_patch(detection_area)

    # Draw path
    for position in path:
        rect = patches.Rectangle((position[1]-0.5, position[0]-0.5), 1, 1, linewidth=2, edgecolor='orange', facecolor='none', zorder=10)
        ax1.add_patch(rect)

    # Plot positions of bot, alien, and crew member
    bot = ax1.scatter(bot_pos[1], bot_pos[0], c='green', label='Bot')

    # alien = ax1.scatter(alien_pos[1], alien_pos[0], c='red', label='Alien')

    # Plot each alien only
    for i, alien_pos in enumerate(alien_positions):
        # if not crew_rescued[i] and crew_pos is not None:
        ax1.scatter(alien_pos[1], alien_pos[0], c='red' if i == 0 else 'brown', label=f'Alien {i+1}')

    # Plot each crew member only if they have not been rescued
    for i, crew_pos in enumerate(crew_positions):
        if not crew_rescued[i] and crew_pos is not None:
            ax1.scatter(crew_pos[1], crew_pos[0], c='blue' if i == 0 else 'purple', label=f'Crew Member {i+1}')

    # Initialize a list to hold legend handles
    legend_handles = [bot]

   # Append aliens in legend handles
    for i, alien_pos in enumerate(alien_positions):
            # Create a temporary scatter plot for legend purposes only
        handle = ax1.scatter([], [], c='red' if i == 0 else 'brown', label=f'Alien {i+1}')
        legend_handles.append(handle) 

    # Append crew member legend handles only if they haven't been rescued
    for i, rescued in enumerate(crew_rescued):
        if not rescued:
            # Create a temporary scatter plot for legend purposes only
            handle = ax1.scatter([], [], c='blue' if i == 0 else 'purple', label=f'Crew Member {i+1}')
            legend_handles.append(handle)

    # Display the legend with dynamic handles
    ax1.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.2, -0.07))
    
    # Alien Probability
    ax2 = plt.subplot(1, 3, 2)
    ax2.set_title("Alien Probability")
    ax2.imshow(prob_alien, cmap='Reds')
    plt.colorbar(mappable=ax2.imshow(prob_alien, cmap='Reds'), ax=ax2)

    # Crew Member Probability
    ax3 = plt.subplot(1, 3, 3)
    ax3.set_title("Crew Member Probability")
    ax3.imshow(prob_crew, cmap='Blues')
    plt.colorbar(mappable=ax3.imshow(prob_crew, cmap='Blues'), ax=ax3)

    plt.draw()
    plt.pause(0.5)  # Pause for a brief moment to update the plot
    plt.close()

def debug_prints(bot_pos, alien_pos1, alien_pos2, crew_pos1, crew_pos2):
    print(f"Bot Position: {bot_pos}")
    print(f"Alien 1 Position: {alien_pos1}")
    print(f"Alien 2 Position: {alien_pos2}")
    print(f"Crew Member 1 Position: {crew_pos1}")
    print(f"Crew Member 2 Position: {crew_pos2}")

def simulate(D, k, alpha):
    grid, grid_layout = initialize_grid(D)
    bot_pos, crew_pos1, crew_pos2, alien_pos1, alien_pos2 = place_entities(grid, D, k, grid_layout)  # Adjusted for two crew members
    prob_alien = np.full((D, D), 1/(D*D - 3))  # Initial belief about aliens location
    prob_crew = np.full((D, D), 1/(D*D - 3))  # Initial belief about crew member location

    bot_alive = True
    crew_rescued = [False, False]  
    steps = 0
    path = []
 
    while bot_alive and not all(crew_rescued): 
        # Check for game over conditions
        
        update_beliefs(bot_pos, alien_pos1, alien_pos2, crew_pos1, crew_pos2, D, k, alpha, prob_alien, prob_crew, grid_layout, crew_rescued)
        
        # If the bot is adjacent to the crew member, rescue immediately
        next_step, path = choose_next_move(bot_pos, prob_alien, prob_crew, D, grid_layout)
        if prob_alien1[next_step[0],next_step[1]] >= 0.7 or prob_alien2[next_step[0],next_step[1]] >= 0.7:
            potential_bot_moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            valid_bot_moves1 = [move for move in potential_bot_moves if
                             0 <= bot_pos[0] + move[0] < D and
                             0 <= bot_pos[1] + move[1] < D and
                             grid_layout[bot_pos[0] + move[0], bot_pos[1] + move[1]] != BLOCKED]

            if valid_bot_moves1:  # Ensure there are valid moves
                max_d = 0
                actual_move = None
                for move in valid_alien_moves1:
                    dist = manhattan_distance(next_step[0],next_step[1],move[0],move[1])
                    if dist>max_d:
                        max_d=dist
                        actual_move = move
                next_step = actual_move

        bot_pos = next_step
        if (bot_pos!=crew_pos1 or bot_pos!=crew_pos2 ):
            prob_crew[bot_pos[0], bot_pos[1]] = 0

        if bot_pos == alien_pos1 or bot_pos == alien_pos2:
            print(f"Bot destroyed by alien at step {steps}.")
            break

        # Update the crew member's probability if the bot has moved into their cell
        if bot_pos == crew_pos1:
            print(f"Crew member found at {crew_pos1} in {steps} steps.")
            crew_rescued[0] = True
            prob_crew[crew_pos1[0], crew_pos1[1]] = 0
            crew_pos1 = None  # Indicate that Crew Member 1 has been rescued and is no longer on the grid
            # Normalize the probabilities for the remaining crew member
            prob_crew /= prob_crew.sum()

        # Update the crew member's probability if the bot has moved into their cell
        if bot_pos == crew_pos2:
            print(f"Crew member found at {crew_pos2} in {steps} steps.")
            crew_rescued[1] = True
            prob_crew[crew_pos2[0], crew_pos2[1]] = 0
            crew_pos2 = None  # Indicate that Crew Member 2 has been rescued and is no longer on the grid
            # Normalize the probabilities for the remaining crew member
            prob_crew /= prob_crew.sum()
        
        print(f"Bot moving to {bot_pos}")
        # Debug prints after the first move
        if steps == 0:
            debug_prints(bot_pos, alien_pos1, alien_pos2, crew_pos1, crew_pos2)

        steps += 1

        # Potential alien moves (up, down, left, right)
        potential_alien_moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        valid_alien_moves1 = [move for move in potential_alien_moves if
                             0 <= alien_pos1[0] + move[0] < D and
                             0 <= alien_pos1[1] + move[1] < D and
                             grid_layout[alien_pos1[0] + move[0], alien_pos1[1] + move[1]] != BLOCKED]

        # Choose a random valid move for the alien
        if valid_alien_moves1:  # Ensure there are valid moves
            alien_move = random.choice(valid_alien_moves1)
            alien_pos1 = (alien_pos1[0] + alien_move[0], alien_pos1[1] + alien_move[1])

        valid_alien_moves2 = [move for move in potential_alien_moves if
                             0 <= alien_pos2[0] + move[0] < D and
                             0 <= alien_pos2[1] + move[1] < D and
                             grid_layout[alien_pos2[0] + move[0], alien_pos2[1] + move[1]] != BLOCKED]

        if valid_alien_moves1:  # Ensure there are valid moves
            alien_move = random.choice(valid_alien_moves2)
            alien_pos2 = (alien_pos2[0] + alien_move[0], alien_pos2[1] + alien_move[1])

        #print(f"Alien moves to {alien_pos}")

        # Check for game over conditions
        if bot_pos == alien_pos1:
            print(f"Bot destroyed by alien at step {steps} and position {alien_pos1}.")
            break

        if bot_pos == alien_pos2:
            print(f"Bot destroyed by alien at step {steps} and position {alien_pos2}.")
            break

        # Adjust probabilities for remaining crew member(s)
        if all(crew_rescued):
            print("All crew members rescued.")
            break

        if steps >= 1000:
            print("Simulation ended without rescuing the crew member.")
            break

        # Update grid for visualization
        grid = np.full((D, D), EMPTY)
        grid[bot_pos[0], bot_pos[1]] = BOT
        grid[alien_pos1[0], alien_pos1[1]] = ALIEN
        grid[alien_pos2[0], alien_pos2[1]] = ALIEN
        if crew_pos1 is not None:
            grid[crew_pos1[0], crew_pos1[1]] = CREW_MEMBER
        if crew_pos2 is not None:
            grid[crew_pos2[0], crew_pos2[1]] = CREW_MEMBER

        # Visualizing the grid, alien, bot, and crew members
        visualize_grid(grid, grid_layout, prob_alien, prob_crew, bot_pos, [alien_pos1, alien_pos2], [crew_pos1, crew_pos2], k, path, crew_rescued)

    plt.ioff()  # Turn off interactive mode
    plt.show()  # Show the plot at the end
        
simulate(D=15, k=2, alpha=0.5)