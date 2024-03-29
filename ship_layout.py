# Grid Design

import numpy as np
import random
import matplotlib.pyplot as plt

def initialize_ship_layout(D):
    layout = np.zeros((D, D), dtype=int)
    return layout

def open_initial_cell(layout):
    D = layout.shape[0]
    x, y = random.randint(1, D-2), random.randint(1, D-2)
    layout[x, y] = 1
    print(f"First Open Cell: {x, y}")

def get_neighbors(x, y, D):
    neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
    valid_neighbors = [(nx, ny) for nx, ny in neighbors if 0 <= nx < D and 0 <= ny < D]
    return valid_neighbors

def iteratively_open_cells(layout):
    D = layout.shape[0]
    changed = True
    while changed:
        changed = False
        candidates = []
        for x in range(D):
            for y in range(D):
                if layout[x, y] == 0: 
                    neighbors = get_neighbors(x, y, D)
                    open_neighbors = sum(layout[nx, ny] for nx, ny in neighbors)
                    if open_neighbors == 1:  
                        candidates.append((x, y)) 
        if candidates:
            # Randomly select one and open it
            to_open = random.choice(candidates)
            layout[to_open[0], to_open[1]] = 1
            changed = True

# Opening half of the dead ends ka blocked neighbours cells
def open_dead_end_neighbors(layout, fraction=0.5):
    D = layout.shape[0]
    dead_ends = []
    for x in range(D):
        for y in range(D):
            if layout[x, y] == 1:  # If the cell is open
                neighbors = get_neighbors(x, y, D)
                open_neighbors = sum(layout[nx, ny] for nx, ny in neighbors)
                if open_neighbors == 1:  # Exactly one open neighbor
                    dead_ends.append((x, y))
    # For about half of the dead ends, open a random blocked neighbor
    for dead_end in random.sample(dead_ends, int(len(dead_ends) * fraction)):
        neighbors = get_neighbors(dead_end[0], dead_end[1], D)
        blocked_neighbors = [(nx, ny) for nx, ny in neighbors if layout[nx, ny] == 0]
        if blocked_neighbors:
            to_open = random.choice(blocked_neighbors)
            layout[to_open[0], to_open[1]] = 1

def visualize_layout(layout):
    plt.imshow(layout, cmap='binary', interpolation='nearest')
    plt.xticks([]), plt.yticks([])  # Hide axis ticks
    plt.show()

def generate_ship_layout(D, dead_end_opening_fraction=0.5):
    layout = initialize_ship_layout(D)
    open_initial_cell(layout)
    iteratively_open_cells(layout)
    open_dead_end_neighbors(layout, fraction=dead_end_opening_fraction)
    return layout

D = 35  # Dimension of the grid
dead_end_opening_fraction = 0.5
ship_layout = generate_ship_layout(D, dead_end_opening_fraction)

print(ship_layout)
#visualize_layout(ship_layout)