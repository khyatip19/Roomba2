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
the alien. If necessary, the bot should flee towards cells where the alien is known not to be.
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import seaborn as sns
import matplotlib.patches as patches

from heapq import heappush, heappop
from queue import PriorityQueue
from ship_layout import generate_ship_layout

# Constants for grid cell states
EMPTY = 1
BOT = 2
ALIEN = 3
CREW_MEMBER = 4
BLOCKED = 0

# # Simulation parameters
# D = 15  # Dimension of the grid
# k = 3  # Sensor range for alien detection
# alpha = 0.5  # Beep probability factor

# Example parameters and simulation run
D = 35  # Dimension of the grid
k = range(3, 8)  # From 3 to 7 inclusive
alpha_range = np.linspace(0.01, 0.2, 30)

def initialize_grid(D):
    """Initialize the simulation grid with blocked and unblocked cells."""
    grid_layout = generate_ship_layout(D)  # Get layout from ship_layout.py
    grid = np.where(grid_layout == 1, EMPTY, BLOCKED)  # Mark unblocked cells as EMPTY, blocked cells as BLOCKED
    return grid, grid_layout

def reset_for_new_iteration(grid, ship_layout):
    # Reset grid and probabilities for a new iteration without changing the ship layout
    grid[:] = np.where(ship_layout == 1, EMPTY, BLOCKED)
    prob_alien = np.full((D, D), 1/(D*D - 1))
    prob_crew = np.full((D, D), 1/(D*D - 2))
    return grid, prob_alien, prob_crew

def place_entities(grid, D, k_value, grid_layout):
    grid = np.where(grid_layout == 1, EMPTY, BLOCKED)
    """Place the bot, a crew member, and an alien on the grid, ensuring they're not too close to each other."""
    # Initialize positions as None
    bot_pos, crew_pos, alien_pos = None, None, None

    while bot_pos is None or grid[bot_pos] != EMPTY:
        potential_pos = (np.random.randint(0, D), np.random.randint(0, D))
        if grid[potential_pos] == EMPTY and grid_layout[potential_pos]:
            bot_pos = potential_pos

    while crew_pos is None:
        potential_pos = (np.random.randint(0, D), np.random.randint(0, D))
        if grid[potential_pos] == EMPTY and grid_layout[potential_pos] and manhattan_distance(potential_pos[0], potential_pos[1], bot_pos[0], bot_pos[1]) > k_value:
            crew_pos = potential_pos

    while alien_pos is None:
        potential_pos = (np.random.randint(0, D), np.random.randint(0, D))
        if grid[potential_pos] == EMPTY and grid_layout[potential_pos] and manhattan_distance(potential_pos[0], potential_pos[1], bot_pos[0], bot_pos[1]) > 2*k_value + 1:
            alien_pos = potential_pos

    grid[bot_pos] = BOT
    grid[crew_pos] = CREW_MEMBER
    grid[alien_pos] = ALIEN

    return bot_pos, crew_pos, alien_pos


def manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def detect_alien(bot_pos, alien_pos, k_value):
    """Determines if an alien is within detection range of the bot using Manhattan distance."""
    return manhattan_distance(bot_pos[0], bot_pos[1], alien_pos[0], alien_pos[1]) <= 2*(k_value) + 1

def detect_beep(crew_pos, bot_pos, alpha_value):
    """Simulates beep detection with a probability based on distance using the Manhattan distance."""
    distance = manhattan_distance(crew_pos[0], crew_pos[1], bot_pos[0], bot_pos[1])
    beep_prob = np.exp(-alpha_value * (distance - 1))
    return np.random.uniform(0, 1.0) <= beep_prob

def get_adjacent_open_cells(x, y, grid_layout):
    """ Get open cells adjacent to (x, y) """
    adjacent_cells = []
    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Right, Down, Left, Up
        if 0 <= x + dx < D and 0 <= y + dy < D:
            if grid_layout[x + dx, y + dy] == EMPTY:
                adjacent_cells.append((x + dx, y + dy))
    return adjacent_cells

def update_beliefs(bot_pos, alien_pos, crew_pos, D, k_value, alpha_value, prob_alien, prob_crew, grid_layout):
    alien_detected = detect_alien(bot_pos, alien_pos, k_value)
    beep_detected = detect_beep(crew_pos, bot_pos, alpha_value)

    for x in range(D):
        for y in range(D):
            if not grid_layout[(x, y)]:  # Skip updating beliefs for blocked cells
                continue
            distance = manhattan_distance(x, y, bot_pos[0], bot_pos[1])

            # Check if the cell is within the detection square of the bot
            in_detection_square = (abs(bot_pos[0] - x) <= k_value and abs(bot_pos[1] - y) <= k_value)

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

            if beep_detected:
                prob_crew[x, y] *= np.exp(-alpha_value * (distance - 1))
            else:
                prob_crew[x, y] *= 1 - np.exp(-alpha_value * (distance - 1))

    # After updating beliefs, ensure that blocked cells have zero probability
    for x in range(D):
        for y in range(D):
            if grid_layout[x, y] == BLOCKED:
                prob_alien[x, y] = 0
                prob_crew[x, y] = 0

# After the loop, normalize the probabilities
    prob_alien = np.where(prob_alien > 0, prob_alien, 0)  # Ensure no negative probabilities
    # prob_alien /= np.sum(prob_alien)  # Normalize
    # prob_crew /= np.sum(prob_crew)

    # Normalize only non-blocked cells
    total_prob_alien = np.sum(prob_alien[grid_layout == EMPTY])
    total_prob_crew = np.sum(prob_crew[grid_layout == EMPTY])
    if total_prob_alien > 0:
        prob_alien[grid_layout == EMPTY] /= total_prob_alien
    if total_prob_crew > 0:
        prob_crew[grid_layout == EMPTY] /= total_prob_crew

    # Diffuse the probabilities for the alien
    # If the alien is detected, set everything outside to 0 and diffuse whatever is inside
    # and vice versa
    new_prob_alien = np.zeros((D, D))
    for x in range(D):
        for y in range(D):
            if prob_alien[x, y] > 0:
                adjacent_open_cells = get_adjacent_open_cells(x, y, grid_layout)
                # If the cell has adjacent open cells, diffuse the probability
                if adjacent_open_cells:
                    distributed_prob = prob_alien[x, y] / len(adjacent_open_cells)
                    for adj_x, adj_y in adjacent_open_cells:
                        new_prob_alien[adj_x, adj_y] += distributed_prob
                else:
                    # If no adjacent cells, keep the probability in the cell
                    new_prob_alien[x, y] = prob_alien[x, y]

    # Update the alien probability matrix with the new values
    prob_alien = new_prob_alien
    if np.sum(prob_alien) > 0:
      prob_alien /= np.sum(prob_alien)

    #prob_alien /= np.sum(prob_alien)  # Ensure the matrix is normalized again after diffusion


from queue import Queue

def find_path_with_risk_assessment(start, goal, ship_layout, risk_scores, risk_multiplier=10):
    D = ship_layout.shape[0]
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while not open_set.empty():
        current = open_set.get()[1]

        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from.get(current)  # Removed , None) -------------
            path.reverse()  # Reverse to get path from start to goal
            return path[1:]
            # break

        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            next = (current[0] + dx, current[1] + dy)
            if 0 <= next[0] < D and 0 <= next[1] < D and ship_layout[next[0], next[1]] == 1:  # Assume 0 as blocked changed !=0 to ==1 -------
                risk_scores = np.array(risk_scores)  # Ensure risk_scores is a NumPy array
                new_cost = cost_so_far[current] + 1 + (risk_scores[next[0], next[1]] * risk_multiplier)
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + manhattan_distance(goal[0],goal[1], next[0],next[1])
                    open_set.put((priority, next))
                    came_from[next] = current
    return []

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

    return path[0] if path else start, path

# def choose_next_move(bot_pos, prob_alien, prob_crew, D, grid_layout):
#     goal_pos = np.unravel_index(np.argmax(prob_crew), prob_crew.shape)  # Target is the most likely crew member location
#     print(f"the goal state is : {goal_pos}")
#     next_step = bfs_search_next_step(bot_pos, goal_pos, D, grid_layout)
#     return next_step

def choose_next_move(bot_pos, prob_alien, prob_crew, D, grid_layout):
    # Modify the goal selection to only consider unblocked cells
    unblocked_prob_crew = np.where(grid_layout == 1, prob_crew, 0)  # Zero out probabilities for blocked cells
    goal_pos = np.unravel_index(np.argmax(unblocked_prob_crew), unblocked_prob_crew.shape)  # Choose the highest unblocked probability
    #print(f"The goal state is: {goal_pos}")
    #next_step, path = bfs_search_next_step(bot_pos, goal_pos, D, grid_layout)  # Get the full path
    path = find_path_with_risk_assessment(bot_pos,goal_pos,ship_layout=grid_layout,risk_scores=(prob_alien))
    if path:
        return path[0], path  # Return the next step and the full path
    else:
        return bot_pos, []  # If no path, return the current position and an empty path

def visualize_grid(grid, grid_layout, prob_alien, prob_crew, bot_pos, alien_pos, crew_pos, k,path):
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

    # Draw detection grid around bot, ensuring it stays within the grid bounds
    # detection_area = patches.Rectangle(
    #     (detection_left - 0.5, detection_top - 0.5),  # Correct offset
    #     (detection_right - detection_left + 1),  # Correct width
    #     (detection_bottom - detection_top + 1),  # Correct height
    #     linewidth=1, edgecolor='purple', facecolor=(0.8, 0.6, 0.8, 0.2), linestyle='--')
    # ax1.add_patch(detection_area)
    # Draw path
    for position in path:
        rect = patches.Rectangle((position[1]-0.5, position[0]-0.5), 1, 1, linewidth=2, edgecolor='orange', facecolor='none', zorder=10)
        ax1.add_patch(rect)

    # ax1.scatter(bot_pos[1], bot_pos[0], c='green', label='Bot')
    # ax1.scatter(alien_pos[1], alien_pos[0], c='red', label='Alien')
    # ax1.scatter(crew_pos[1], crew_pos[0], c='blue', label='Crew Member')
    # ax1.legend()

    # Plot positions of bot, alien, and crew member
    bot = ax1.scatter(bot_pos[1], bot_pos[0], c='green', label='Bot')
    alien = ax1.scatter(alien_pos[1], alien_pos[0], c='red', label='Alien')
    crew_member = ax1.scatter(crew_pos[1], crew_pos[0], c='blue', label='Crew Member')

    # Display the legend outside the plot area in the bottom left corner
    ax1.legend(handles=[bot, alien, crew_member], loc='upper center', bbox_to_anchor=(0.2, -0.07))

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

def simulate(D, k_value, alpha_value, grid_layout):
    # grid, grid_layout = initialize_grid(D)
    # prob_alien = np.full((D, D), 1/(D*D - 1))  # Initial belief about alien location
    # prob_crew = np.full((D, D), 1/(D*D - 2))  # Initial belief about crew member location

    # Initialize grid based on the fixed layout
    grid = np.where(grid_layout == 1, EMPTY, BLOCKED)

    #grid, prob_alien, prob_crew = reset_for_new_iteration(np.empty((D, D)), grid_layout)
    bot_pos, crew_pos, alien_pos = place_entities(grid, D, k_value, grid_layout)

    
    # Initialize beliefs about alien and crew locations
    prob_alien = np.full((D, D), 1/(D*D - 1))  # Assuming one alien
    prob_crew = np.full((D, D), 1/(D*D - 3))  # Adjusted for two crew members, subtracting one more for the bot's cell

    bot_alive = True
    crew_rescued = False
    steps = 0
    path = []
    plt.figure(figsize=(12, 6))  # Initialize the figure outside the loop

    while bot_alive and not crew_rescued and steps < 1000:
        if bot_pos == alien_pos:
            #print(f"Bot destroyed by alien at step {steps}.")
            bot_alive = False
            break

        update_beliefs(bot_pos, alien_pos, crew_pos, D, k_value, alpha_value, prob_alien, prob_crew, grid_layout)

        # If the bot is adjacent to the crew member, rescue immediately
        if manhattan_distance(bot_pos[0], bot_pos[1], crew_pos[0], crew_pos[1]) == 1:
            #print(f"Crew member rescued by bot at position {bot_pos} in {steps} steps.")
            crew_rescued = True
            break

        next_step, path = choose_next_move(bot_pos, prob_alien, prob_crew, D, grid_layout)
        bot_pos = next_step
        if bot_pos == alien_pos:
            #print(f"Bot destroyed by alien at step {steps}.")
            bot_alive = False
            break

        # Update the crew member's probability if the bot has moved into their cell
        if bot_pos == crew_pos:
            #print(f"Crew member found at {crew_pos} in {steps} steps.")
            crew_rescued = True
            prob_crew[bot_pos] = 0
            break
        else:
            prob_crew[bot_pos] = 0
        #print(f"Bot moving to {bot_pos}")


        # Debug prints after the first move
        if steps == 0:
            debug_prints(bot_pos, alien_pos, crew_pos)

        steps += 1

         # Potential alien moves (up, down, left, right)
        potential_alien_moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        valid_alien_moves = [move for move in potential_alien_moves if
                             0 <= alien_pos[0] + move[0] < D and
                             0 <= alien_pos[1] + move[1] < D and
                             grid_layout[alien_pos[0] + move[0], alien_pos[1] + move[1]] != BLOCKED]

        # Choose a random valid move for the alien
        if valid_alien_moves:  # Ensure there are valid moves
            alien_move = random.choice(valid_alien_moves)
            alien_pos = (alien_pos[0] + alien_move[0], alien_pos[1] + alien_move[1])

        #print(f"Alien moves to {alien_pos}")

        # Check for game over conditions
        if bot_pos == alien_pos:
            #print(f"Bot destroyed by alien at step {steps}.")
            break
        if bot_pos == crew_pos:
            #print(f"Crew member rescued in {steps} steps.")
            break

        if steps >= 1000:
            #print("Simulation ended without rescuing the crew member.")
            break

        # Update grid for visualization
        grid = np.full((D, D), EMPTY)
        grid[bot_pos[0], bot_pos[1]] = BOT
        grid[alien_pos[0], alien_pos[1]] = ALIEN
        grid[crew_pos[0], crew_pos[1]] = CREW_MEMBER

        # Call visualization function
        #visualize_grid(grid, grid_layout, prob_alien, prob_crew, bot_pos, alien_pos, crew_pos,k, path)
    return steps, bot_alive, crew_rescued
    # plt.show() might be needed if you are not using interactive mode for matplotlib


#simulate(D=15, k=3, alpha=0.5)

def run_simulations_with_parameters(k_range, alpha_range, num_simulations=1):
    grid, grid_layout = initialize_grid(D)
    results = []

    for k_value in k_range:  # Iterate over each value in the k_range
        for alpha_value in alpha_range:
            total_steps = 0
            success_count = 0
            for simulation_index in range(num_simulations):
                #grid, grid_layout = initialize_grid(D)
                #prob_alien = np.full((D, D), 1/(D*D - 1))
                #prob_crew = np.full((D, D), 1/(D*D - 2))
                #bot_pos, crew_pos, alien_pos = place_entities(grid, D, k_value, grid_layout)  # Use k_value here
                steps, bot_alive, crew_rescued = simulate(D, k_value, alpha_value, grid_layout)  # And here

                total_steps += steps
                if crew_rescued:
                    success_count += 1
                print(f"Completed: k={k_value}, alpha={alpha_value}, simulation={simulation_index+1}/{num_simulations}")
            avg_steps = total_steps / num_simulations
            success_rate = success_count / num_simulations
            results.append({'k': k_value, 'alpha': alpha_value, 'avg_steps': avg_steps, 'success_rate': success_rate})  # Note: Changed k to k_value

    return pd.DataFrame(results)


# Run the simulations
#print(type(k))
results_df = run_simulations_with_parameters(k, alpha_range, num_simulations=10)

# Display the results
print(results_df)

csv_file_path = "D:/USA Docs/Rutgers/Intro to AI/Project 2/Roomba2/bot2.csv"

# Save the DataFrame to a CSV file
results_df.to_csv(csv_file_path, index=False)

print(f"Simulation results have been saved to {csv_file_path}")