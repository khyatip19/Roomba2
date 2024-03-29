# bot1.py
# Implements Bot 1: Plans the shortest path to the Captain, ignoring alien movements.

import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from queue import PriorityQueue
from ship_layout import generate_ship_layout

def heuristic(a, b):
    """Calculate the Manhattan distance between two points"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def random_position(D, grid):
    """Generate a random position within the grid bounds that is not blocked."""
    possible_indices = []
    for y in range(D):
        for x in range(D):
            if grid[x, y] == 1:
                possible_indices.append((x,y))
    if not possible_indices:
        print("No indices available. Lower K")
    return random.choice(possible_indices)
    #return random.randint(0, D-1), random.randint(0, D-1)

def get_neighbors(position, grid):
    """Returns the valid neighbors for a position in the grid."""
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    neighbors = []
    for direction in directions:
        neighbor = (position[0] + direction[0], position[1] + direction[1])
        if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
            if grid[neighbor] == 1:  # Ensure the neighbor is an open cell
                neighbors.append(neighbor)
    return neighbors

# Reconstructs the path of the bot till the captain cell
def reconstruct_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path

# Find the shortest path from start to goal using A* pathfinding."""
def find_shortest_path(start, goal, grid):
    open_set = PriorityQueue()
    # Storing the nodes to be explored
    open_set.put((0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while not open_set.empty():
        current = open_set.get()[1]
        
        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path[0:]  
        
        for neighbor in get_neighbors(current, grid):
            tentative_g_score = g_score[current] + 1
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                # This path to neighbor is better than any previous one. Record it!
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                if not any(neighbor == q_item[1] for q_item in open_set.queue):
                    open_set.put((f_score[neighbor], neighbor))
                    
    return []  # Return an empty path if there is no path to the goal
    # pass

def place_aliens(D, grid, count, exclude_positions):
    aliens = []
    while len(aliens) < count:
        position = random_position(D, grid)
        if position not in exclude_positions:
            aliens.append(position)
            #grid[position] = 2  # Assuming '2' marks an alien, adjust as needed
    return aliens

def remove_aliens(D, grid):
    for x in range(D):
        for y in range(D):
            grid[x, y] = 1 if grid[x,y] != 0 else 0

def move_aliens(alien_positions, grid):
    new_alien_positions = []
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Possible directions: right, down, left, up
    for pos in alien_positions:
        valid_moves = [((pos[0] + d[0], pos[1] + d[1])) for d in directions if 0 <= pos[0] + d[0] < grid.shape[0] and 0 <= pos[1] + d[1] < grid.shape[1] and grid[pos[0] + d[0], pos[1] + d[1]] == 1]
        if valid_moves:
            new_pos = random.choice(valid_moves)  # Choose a random valid move
            new_alien_positions.append(new_pos)
        else:
            new_alien_positions.append(pos)  # Stay in place if no valid move
    return new_alien_positions

if __name__ == "__main__":
    D = 30 # (hardcoded for now)
    #D = random.randint(1, 30)
    ship_layout = generate_ship_layout(D)
    print(ship_layout.shape)

    # alien_count = random.randint(1, D//2)
    K_range = range(0, 101, 2)
    num_trials = 250



   