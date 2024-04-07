# bot4.py - One Alien, Two Crew Member
# Bot 4: Bot 4 is Bot 1, except that the probabilities of where the crew members are
# account for the fact that there are two of them (How?), and are updated accordingly.

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
# D = 15  # Dimension of the grid
# k = 3  # Sensor range for alien detection
# alpha = 0.5  # Beep probability factor

# Example parameters and simulation run
D = 35  # Dimension of the grid
k = range(3, 8)  # From 3 to 7 inclusive
alpha_range = np.linspace(0.01, 0.2, 30) # From 0 to 1 in increments of 0.1

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

def place_entities(grid, D, k, grid_layout):
    grid = np.where(grid_layout == 1, EMPTY, BLOCKED)
    """
    Place the bot, two crew members, and an alien on the grid, ensuring they're not too close to each other.
    - Bot is placed randomly.
    - Crew members are placed at least k+1 cells away from the bot and each other.
    - Alien is placed at least 2*k+1 cells away from the bot to respect detection range.
    """
    bot_pos, crew_pos1, crew_pos2, alien_pos = None, None, None, None

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

    while alien_pos is None:
        potential_pos = (np.random.randint(0, D), np.random.randint(0, D))
        if grid[potential_pos] == EMPTY and grid_layout[potential_pos] and manhattan_distance(potential_pos[0], potential_pos[1], bot_pos[0], bot_pos[1]) > 2*k + 1:
            alien_pos = potential_pos

    grid[bot_pos] = BOT
    grid[crew_pos1] = CREW_MEMBER
    grid[crew_pos2] = CREW_MEMBER
    grid[alien_pos] = ALIEN

    return bot_pos, crew_pos1, crew_pos2, alien_pos

def manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)

def detect_alien(bot_pos, alien_pos, k_value):
    """Determines if an alien is within detection range of the bot using Manhattan distance."""
    return manhattan_distance(bot_pos[0], bot_pos[1], alien_pos[0], alien_pos[1]) <= 2*(k_value) + 1

def detect_beep_individual(crew_pos, bot_pos, alpha_value):
    """Simulate beep detection from both crew members."""
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

def beep_prob_adjustment(beep_detected, alpha_value, distance):
    """
    Adjust probability based on beep detection.
    """
    if beep_detected:
        # Increase probability for closer cells based on beep detection
        return np.exp(-alpha_value * (distance - 1))
    else:
        # Decrease probability for cells outside beep range
        return 1 - np.exp(-alpha_value * (distance - 1))

def update_beliefs(bot_pos, alien_pos, crew_pos1, crew_pos2, D, k_value, alpha_value, prob_alien, prob_crew, grid_layout, crew_rescued):   
    # Initialize or reset the crew probability map if necessary
    # prob_crew.fill(0)  # Resetting might be necessary depending on your implementation
    
    # Detect alien and beep probabilities
    alien_detected = detect_alien(bot_pos, alien_pos,k_value)
    beep_detected1 = detect_beep_individual(crew_pos1, bot_pos,alpha_value) if not crew_rescued[0] else False
    beep_detected2 = detect_beep_individual(crew_pos1, bot_pos,alpha_value) if not crew_rescued[1] else False 
    
    for x in range(D):
        for y in range(D):
            if not grid_layout[(x, y)]:  # Skip updating beliefs for blocked cells
                continue

            distance = manhattan_distance(x, y, bot_pos[0], bot_pos[1])

            # Calculate beep probabilities for each crew member
            

            beep_prob1 = np.exp(-alpha_value * distance) if not crew_rescued[0] else 0
            beep_prob2 = np.exp(-alpha_value * distance) if not crew_rescued[1] else 0

            # Combine the beep probabilities
            # combined_beep_prob = beep_detected1 + beep_detected2 - (beep_detected1 * beep_detected2)
            combined_beep_prob = beep_prob1 + beep_prob2 - beep_prob1*beep_prob2

            # sApply the combined beep probability to update the cell's probability
            #prob_crew[x, y] *= combined_beep_prob
            
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
    
            # Update for beep detection from either crew member
            if beep_detected1 or beep_detected2:
                # Increase probability for closer cells based on beep detection
                prob_crew[x, y] *= combined_beep_prob
            else:
                # Decrease probability for cells outside beep range
                prob_crew[x, y] *= (1-combined_beep_prob)

    # # Normalize prob_crew if not summing to 1
    # total_prob = np.sum(prob_crew[grid_layout != BLOCKED])
    # if total_prob > 0:
    #     prob_crew[grid_layout != BLOCKED] /= total_prob

    # Normalize only non-blocked cells
    total_prob_alien = np.sum(prob_alien[grid_layout == EMPTY])
    total_prob_crew = np.sum(prob_crew[grid_layout == EMPTY])
    if total_prob_alien > 0:
        prob_alien[grid_layout == EMPTY] /= total_prob_alien
    if total_prob_crew > 0:
        prob_crew[grid_layout == EMPTY] /= total_prob_crew

    # # Diffuse the probabilities for the alien
    # # If the alien is detected, set everything outside to 0 and diffuse whatever
    # #  is inside and vice versa
    new_prob_alien = np.zeros((D, D))
    for x in range(D):
        for y in range(D):
            if prob_alien[x, y] > 0:
                adjacent_open_cells = get_adjacent_open_cells(x, y, grid_layout)
                distributed_prob = prob_alien[x, y] / len(adjacent_open_cells) if adjacent_open_cells else prob_alien[x, y]
                for adj_x, adj_y in adjacent_open_cells:
                    new_prob_alien[adj_x, adj_y] += distributed_prob

    # # Update the alien belief matrix if there are any probabilities to diffuse
    if np.sum(new_prob_alien) > 0:
        prob_alien[:] = new_prob_alien / np.sum(new_prob_alien)

    # #prob_alien /= np.sum(prob_alien)  # Ensure the matrix is normalized again after diffusion

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

    if path:
        return path[0], path  # Return the next step and the full path
    else:
        return start, None  # Indicate that no path was found

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
        #print(f"no path from {bot_pos} to {goal_pos}")
        return bot_pos, []  # If no path, return the current position and an empty path

# def move_alien(alien_pos, D):
#     # All possible movements including staying in place
#     moves = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]
#     move = random.choice(moves)  # Randomly choose a move
#     # Calculate new position and ensure it's within grid bounds
#     new_pos = (max(0, min(D - 1, alien_pos[0] + move[0])), max(0, min(D - 1, alien_pos[1] + move[1])))
#     return new_pos

def visualize_grid(grid, grid_layout, prob_alien, prob_crew, bot_pos, alien_pos, crew_positions, k, path, crew_rescued):
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
   
    # crew_member1 = ax1.scatter(crew_positions[0][1], crew_positions[0][0], c='blue', label='Crew Member1')
    # crew_member2 = ax1.scatter(crew_positions[1][1], crew_positions[1][0], c='purple', label='Crew Member2')
    # crew_member = []

    # Plot each crew member only if they have not been rescued
    for i, crew_pos in enumerate(crew_positions):
        if not crew_rescued[i] and crew_pos is not None:
            ax1.scatter(crew_pos[1], crew_pos[0], c='blue' if i == 0 else 'purple', label=f'Crew Member {i+1}')

     # Initialize a list to hold legend handles
    legend_handles = [bot, alien]

    # Append crew member legend handles only if they haven't been rescued
    for i, rescued in enumerate(crew_rescued):
        if not rescued:
            # Create a temporary scatter plot for legend purposes only
            handle = ax1.scatter([], [], c='blue' if i == 0 else 'purple', label=f'Crew Member {i+1}')
            legend_handles.append(handle)

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

def debug_prints(bot_pos, alien_pos, crew_pos1, crew_pos2):
    print(f"Bot Position: {bot_pos}")
    print(f"Alien Position: {alien_pos}")
    print(f"Crew Member 1 Position: {crew_pos1}")
    print(f"Crew Member 2 Position: {crew_pos2}")

def simulate(D, k_value, alpha_value, grid_layout):
    #grid, grid_layout = initialize_grid(D)
    # bot_pos, crew_positions, alien_pos = place_entities(grid, D, k)
    # grid, prob_alien, prob_crew = reset_for_new_iteration(np.empty((D, D)), grid_layout)
    # bot_pos, crew_pos1, crew_pos2, alien_pos = place_entities(grid, D, k_value, grid_layout)  # Adjusted for two crew members
    # # prob_alien, prob_crew = initialize_probabilities(grid, D, bot_pos, k)
    # prob_alien = np.full((D, D), 1/(D*D - 1))  # Initial belief about alien location
    # prob_crew = np.full((D, D), 1/(D*D - 3))  # Initial belief about crew member location
    
    # Initialize grid based on the fixed layout
    grid = np.where(grid_layout == 1, EMPTY, BLOCKED)
    
    # Place the bot, two crew members, and an alien on the grid
    bot_pos, crew_pos1, crew_pos2, alien_pos = place_entities(grid, D, k_value, grid_layout)

    # Initialize beliefs about alien and crew locations
    prob_alien = np.full((D, D), 1/(D*D - 1))  # Assuming one alien
    prob_crew = np.full((D, D), 1/(D*D - 3))  # Adjusted for two crew members, subtracting one more for the bot's cell

    bot_alive = True
    crew_rescued = [False, False]  
    crew1_rescued = False
    crew2_rescued = False
    crew_saved_count = 0
    steps = 0
    path = []
    #plt.figure(figsize=(12, 6))  # Initialize the figure outside the loop
 
    while bot_alive and not all(crew_rescued): 
        # Check for game over conditions

        # beep_detected = False
        
        # Check for beep detection from each crew member individually
        # if not crew_rescued[0]:
        #     beep_detected |= detect_beep_individual(crew_pos1, bot_pos)
        # if not crew_rescued[1]:
        #     beep_detected |= detect_beep_individual(crew_pos2, bot_pos) 
        
        update_beliefs(bot_pos, alien_pos, crew_pos1, crew_pos2, D, k_value, alpha_value, prob_alien, prob_crew, grid_layout, crew_rescued)
        
        # If the bot is adjacent to the crew member, rescue immediately

        next_step, path = choose_next_move(bot_pos, prob_alien, prob_crew, D, grid_layout)
        if prob_alien[next_step[0],next_step[1]] >= 0.7:
            
            potential_bot_moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            valid_bot_moves = [move for move in potential_bot_moves if
                             0 <= bot_pos[0] + move[0] < D and
                             0 <= bot_pos[1] + move[1] < D and
                             grid_layout[bot_pos[0] + move[0], bot_pos[1] + move[1]] != BLOCKED]

            if valid_bot_moves:  # Ensure there are valid moves
                max_d = 0
                actual_move = None
                for move in valid_alien_moves:
                    dist = manhattan_distance(next_step[0],next_step[1],move[0],move[1])
                    if dist>max_d:
                        max_d=dist
                        actual_move = move
                next_step = actual_move

        bot_pos = next_step
        if (bot_pos!=crew_pos1 or bot_pos!=crew_pos2 ):
            prob_crew[bot_pos[0], bot_pos[1]] = 0

        if bot_pos == alien_pos:
            #print(f"Bot destroyed by alien at step {steps}.")
            break
        else:
            prob_alien[bot_pos[0], bot_pos[1]] = 0


        # Update the crew member's probability if the bot has moved into their cell
        if bot_pos == crew_pos1 and not crew_rescued[0]:
            #print(f"Crew member found at {crew_pos1} in {steps} steps.")
            # crew_rescued1 = True
            crew_rescued[0] = True
            crew_saved_count += 1
            # prob_crew[bot_pos] = 0
            prob_crew[crew_pos1[0], crew_pos1[1]] = 0
            # Normalize the probabilities for the remaining crew member
            prob_crew /= prob_crew.sum()


        # Update the crew member's probability if the bot has moved into their cell
        if bot_pos == crew_pos2 and not crew_rescued[1]:
            #print(f"Crew member found at {crew_pos2} in {steps} steps.")
            # crew_rescued2 = True
            crew_rescued[1] = True
            crew_saved_count += 1
            # prob_crew[bot_pos] = 0
            prob_crew[crew_pos2[0], crew_pos2[1]] = 0
            # Normalize the probabilities for the remaining crew member
            prob_crew /= prob_crew.sum()

        
        #print(f"Bot moving to {bot_pos}")
        # Debug prints after the first move
        if steps == 0:
            debug_prints(bot_pos, alien_pos, crew_pos1, crew_pos2)

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
            #print(f"Bot destroyed by alien at step {steps} and position {alien_pos}.")
            break

        # if bot_pos == crew_pos1 and not crew_rescued[0]:
            # Adjust probabilities for remaining crew member(s)

        if all(crew_rescued):
            #print("All crew members rescued.")
            break

        if steps >= 1000:
            #print("Simulation ended without rescuing the crew member.")
            break

        # Update grid for visualization
        grid = np.full((D, D), EMPTY)
        grid[bot_pos[0], bot_pos[1]] = BOT
        grid[alien_pos[0], alien_pos[1]] = ALIEN
        if crew_pos1 is not None:
            grid[crew_pos1[0], crew_pos1[1]] = CREW_MEMBER
        if crew_pos2 is not None:
            grid[crew_pos2[0], crew_pos2[1]] = CREW_MEMBER

        # Visualizing the grid, alien, bot, and crew members
        #visualize_grid(grid, grid_layout, prob_alien, prob_crew, bot_pos, alien_pos, [crew_pos1, crew_pos2], k, path, crew_rescued)
    return steps, bot_alive, crew_rescued, crew_saved_count
    #plt.ioff()  # Turn off interactive mode
    #plt.show()  # Show the plot at the end
        
#simulate(D=15, k=2, alpha=0.5)
def run_simulations_with_parameters(k_range, alpha_range, num_simulations=1):
    # Generate ship layout once
    grid, grid_layout = initialize_grid(D)
    results = []

    for k_value in k_range:  # Iterate over each value in the k_range
        for alpha_value in alpha_range:
            total_steps = 0
            success_count = 0
            total_crew_saved = 0            
            for simulation_index in range(num_simulations):
                # grid, grid_layout = initialize_grid(D)
                # prob_alien = np.full((D, D), 1/(D*D - 1))
                # prob_crew = np.full((D, D), 1/(D*D - 2))
                #bot_pos, crew_pos, alien_pos = place_entities(grid, D, k_value, grid_layout)  # Use k_value here
                steps, bot_alive, crew_rescued = simulate(D, k_value, alpha_value, grid_layout)  # And here

                total_steps += steps
                if crew_rescued:
                    success_count += 1
                total_crew_saved += crew_saved
                print(f"Completed: k={k_value}, alpha={alpha_value}, simulation={simulation_index+1}/{num_simulations}")

            avg_steps = total_steps / num_simulations
            success_rate = success_count / num_simulations
            avg_crew_saved = total_crew_saved / num_simulations  # Calculate average crew members saved
            results.append({'k': k_value, 'alpha': alpha_value, 'avg_steps': avg_steps, 'success_rate': success_rate,'avg_crew_saved': avg_crew_saved}) 

    return pd.DataFrame(results)



# Run the simulations
#print(type(k))
results_df = run_simulations_with_parameters(k, alpha_range, num_simulations=10)

# Display the results
print(results_df)


csv_file_path = "D:/USA Docs/Rutgers/Intro to AI/Project 2/Roomba2/bot5.csv"

# Save the DataFrame to a CSV file
results_df.to_csv(csv_file_path, index=False)

print(f"Simulation results have been saved to {csv_file_path}")