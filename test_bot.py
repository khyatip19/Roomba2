""" bot1.py - One Alien, One Crew Member
In this case, we have a single alien in play, and a single crew member that needs rescuing.
At the start, the bot knows that the alien is equally likely to be in any open cell outside
the bot's detection square, and the crew member is equally likely to be in any cell other 
than the bot's initial cell. At every point in time, update what is known about the crew 
member and the alien based on the data received (How?). (Note: the bot necessarily has 
perfect knowledge of the cell that it is currently in.) Note, when the alien has the 
opportunity to move, the bot's knowledge of the alien should be updated accordingly (How?).
The bot should proceed by moving toward the cell most likely to contain the crew member 
(breaking ties at random), sticking (when possible) to cells that definitely do not contain
the alien. If necefssary, the bot should flee towards cells where the alien is known not to be.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns

from heapq import heappush, heappop
from queue import PriorityQueue
from ship_layout import generate_ship_layout

# Constants for grid cell states
EMPTY = 1
BOT = 2
ALIEN = 3
CREW_MEMBER = 4
BLOCKED = 0

# Simulation parameters
D = 5  # Dimension of the grid
k = 2  # Sensor range for alien detection
alpha = 0.5  # Beep probability factor

def initialize_grid(D):
    """Initialize the simulation grid with blocked and unblocked cells."""
    grid_layout = generate_ship_layout(D)  # Get layout from ship_layout.py
    grid = np.where(grid_layout == 1, EMPTY, BLOCKED)  # Mark unblocked cells as EMPTY, blocked cells as BLOCKED
    return grid, grid_layout


def place_entities(grid, D, k, grid_layout):
    """Place the bot, a crew member, and an alien on the grid, ensuring they're not too close to each other."""
    # Initialize positions as None
    bot_pos, crew_pos, alien_pos = None, None, None

    while bot_pos is None or grid[bot_pos] != EMPTY:
        potential_pos = (np.random.randint(0, D), np.random.randint(0, D))
        if grid[potential_pos] == EMPTY and grid_layout[potential_pos]:
            bot_pos = potential_pos

    while crew_pos is None:
        potential_pos = (np.random.randint(0, D), np.random.randint(0, D))
        if grid[potential_pos] == EMPTY and grid_layout[potential_pos] and manhattan_distance(potential_pos[0], potential_pos[1], bot_pos[0], bot_pos[1]) > k:
            crew_pos = potential_pos

    while alien_pos is None:
        potential_pos = (np.random.randint(0, D), np.random.randint(0, D))
        if grid[potential_pos] == EMPTY and grid_layout[potential_pos] and manhattan_distance(potential_pos[0], potential_pos[1], bot_pos[0], bot_pos[1]) > 2*k + 1:
            alien_pos = potential_pos

    grid[bot_pos] = BOT
    grid[crew_pos] = CREW_MEMBER
    grid[alien_pos] = ALIEN

    return bot_pos, crew_pos, alien_pos


def manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def detect_alien(bot_pos, alien_pos):
    """Determines if an alien is within detection range of the bot using Manhattan distance."""
    return manhattan_distance(bot_pos[0], bot_pos[1], alien_pos[0], alien_pos[1]) <= 2*k + 1

def detect_beep(crew_pos, bot_pos):
    """Simulates beep detection with a probability based on distance using the Manhattan distance."""
    distance = manhattan_distance(crew_pos[0], crew_pos[1], bot_pos[0], bot_pos[1])
    beep_prob = np.exp(-alpha * (distance - 1))
    return np.random.rand() < beep_prob

def update_beliefs(bot_pos, alien_pos, crew_pos, D, k, alpha, prob_alien, prob_crew, grid_layout):
    alien_detected = detect_alien(bot_pos, alien_pos)
    beep_detected = detect_beep(crew_pos, bot_pos)

    for x in range(D):
        for y in range(D):
            if not grid_layout[(x, y)]:  # Skip updating beliefs for blocked cells
                continue
            distance = manhattan_distance(x, y, bot_pos[0], bot_pos[1])
            if distance <= 2*k + 1:
                if alien_detected:
                    prob_alien[x, y] *= 2
                else:
                    prob_alien[x, y] /= 2
            if beep_detected:
                prob_crew[x, y] *= np.exp(-alpha * (distance - 1))
            else:
                prob_crew[x, y] *= 1 - np.exp(-alpha * (distance - 1))

    prob_alien /= np.sum(prob_alien)
    prob_crew /= np.sum(prob_crew)


from queue import Queue

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
        print(f"the current cell is : {current}")
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (current[0] + dx, current[1] + dy)

            if neighbor not in visited and 0 <= neighbor[0] < D and 0 <= neighbor[1] < D and grid_layout[neighbor] != BLOCKED:
                queue.put(neighbor)
                visited.append(neighbor)
                came_from[neighbor] = current

    # Reconstruct the path back to the start
    if goal in came_from:
        path_to_start = []
        current = goal
        while current != start:
            path_to_start.append(current)
            current = came_from[current]
        path_to_start.reverse()
        return path_to_start[0] if path_to_start else start
    else:
        print("NO GOAL!")
        return start  # Return the start position if there is no path to goal

# def choose_next_move(bot_pos, prob_alien, prob_crew, D, grid_layout):
#     goal_pos = np.unravel_index(np.argmax(prob_crew), prob_crew.shape)  # Target is the most likely crew member location
#     print(f"the goal state is : {goal_pos}")
#     next_step = bfs_search_next_step(bot_pos, goal_pos, D, grid_layout)
#     return next_step
    
def choose_next_move(bot_pos, prob_alien, prob_crew, D, grid_layout):
    # Modify the goal selection to only consider unblocked cells
    unblocked_prob_crew = np.where(grid_layout == 1, prob_crew, 0)  # Zero out probabilities for blocked cells
    goal_pos = np.unravel_index(np.argmax(unblocked_prob_crew), unblocked_prob_crew.shape)  # Choose the highest unblocked probability
    print(f"The goal state is: {goal_pos}")
    next_step = bfs_search_next_step(bot_pos, goal_pos, D, grid_layout)
    return next_step

def visualize_grid(grid, grid_layout, prob_alien, prob_crew, bot_pos, alien_pos, crew_pos):
    plt.figure(figsize=(12, 6))

    # Grid overview
    ax1 = plt.subplot(1, 3, 1)
    ax1.set_title("Grid Overview")
    blocked_cells = np.where(grid_layout == 0, True, False)  # Find blocked cells
    open_cells = np.logical_not(blocked_cells)  # Find open cells
    ax1.imshow(open_cells, cmap='Pastel1')  # Color open cells
    ax1.imshow(blocked_cells, cmap='gray')  # Color blocked cells with a different color
    ax1.scatter(bot_pos[1], bot_pos[0], c='green', label='Bot')
    ax1.scatter(alien_pos[1], alien_pos[0], c='red', label='Alien')
    ax1.scatter(crew_pos[1], crew_pos[0], c='blue', label='Crew Member')
    ax1.legend()

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

    plt.pause(0.3)
    plt.clf()

def debug_prints(bot_pos, alien_pos, crew_pos):
    print(f"Bot Position: {bot_pos}")
    print(f"Alien Position: {alien_pos}")
    print(f"Crew Member Position: {crew_pos}")

def simulate(D, k, alpha):
    grid, grid_layout = initialize_grid(D)
    prob_alien = np.full((D, D), 1/(D*D - 1))  # Initial belief about alien location
    prob_crew = np.full((D, D), 1/(D*D - 2))  # Initial belief about crew member location
    bot_pos, crew_pos, alien_pos = place_entities(grid, D, k, grid_layout)
    steps = 0

    while True:
        update_beliefs(bot_pos, alien_pos, crew_pos, D, k, alpha, prob_alien, prob_crew, grid_layout)
        bot_pos = choose_next_move(bot_pos, prob_alien, prob_crew, D, grid_layout)
        print(bot_pos)

        # Debug prints after the first move
        if steps == 0:
            debug_prints(bot_pos, alien_pos, crew_pos)

        steps += 1

        # Simulate alien movement
        alien_moves = [(0, 1), (1, 0), (0, -1), (-1, 0), (0, 0)]
        alien_move = alien_moves[np.random.choice(len(alien_moves))]
        alien_pos = (max(0, min(D - 1, alien_pos[0] + alien_move[0])), max(0, min(D - 1, alien_pos[1] + alien_move[1])))

        # Check for game over conditions
        if bot_pos == alien_pos:
            print(f"Bot destroyed by alien at step {steps}.")
            break
        if bot_pos == crew_pos:
            print(f"Crew member rescued in {steps} steps.")
            break

        if steps >= 1000:
            print("Simulation ended without rescuing the crew member.")
            break

        # Update grid for visualization
        grid = np.full((D, D), EMPTY)
        grid[bot_pos[0], bot_pos[1]] = BOT
        grid[alien_pos[0], alien_pos[1]] = ALIEN
        grid[crew_pos[0], crew_pos[1]] = CREW_MEMBER

        # Call visualization function
        # visualize_grid(grid, grid_layout, prob_alien, prob_crew, bot_pos, alien_pos, crew_pos)
        
    # plt.show() might be needed if you are not using interactive mode for matplotlib

        
simulate(D=5, k=2, alpha=0.5)