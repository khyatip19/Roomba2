import numpy as np
import matplotlib.pyplot as plt

# Constants
EMPTY, BOT, ALIEN, CREW_MEMBER = 0, 1, 2, 3
D, k, alpha = 35, 2, 0.5  # Grid size, sensor range, beep sensitivity

def initialize_grid():
    """Initializes a DxD grid filled with EMPTY."""
    return np.full((D, D), EMPTY)

def place_entities():
    """Places bot, crew member, and alien on the grid with initial constraints."""
    positions = np.random.choice(D*D, 3, replace=False)
    return np.array(np.unravel_index(positions, (D, D))).T  # Convert to (x, y) format

def detect_alien(bot_pos, alien_pos):
    """Returns True if an alien is within the detection range."""
    return np.linalg.norm(bot_pos - alien_pos, ord=1) <= 2*k + 1

def detect_beep(crew_pos, bot_pos):
    """Simulates beep detection with a probability based on distance."""
    distance = np.linalg.norm(crew_pos - bot_pos, ord=1)
    return np.random.rand() < np.exp(-alpha * (distance - 1))

def update_beliefs(bot_pos, prob_alien, prob_crew, alien_detected, beep_detected):
    """Updates beliefs about alien and crew member locations based on sensor readings."""
    # Update based on alien detection
    for x in range(D):
        for y in range(D):
            dist = np.linalg.norm(np.array([x, y]) - bot_pos, ord=1)
            if alien_detected and dist <= 2*k + 1:
                prob_alien[x, y] += 1
            elif not alien_detected and dist > 2*k + 1:
                prob_alien[x, y] += 1
            else:
                prob_alien[x, y] = max(prob_alien[x, y] - 1, 0)  # Avoid negative probabilities

    # Update based on beep detection
    for x in range(D):
        for y in range(D):
            dist = np.linalg.norm(np.array([x, y]) - bot_pos, ord=1)
            beep_prob = np.exp(-alpha * (dist - 1))
            if beep_detected:
                prob_crew[x, y] *= beep_prob
            else:
                prob_crew[x, y] *= (1 - beep_prob)

    # Normalize
    prob_alien /= np.sum(prob_alien)
    prob_crew /= np.sum(prob_crew)

def choose_next_move(bot_pos, prob_crew):
    """Chooses the next move towards the most probable crew member location."""
    target_pos = np.unravel_index(np.argmax(prob_crew), prob_crew.shape)
    direction = np.sign(np.array(target_pos) - bot_pos)
    return bot_pos + direction

def simulate():
    grid = initialize_grid()
    bot_pos, crew_pos, alien_pos = place_entities()
    prob_alien = np.full((D, D), 1/(D*D - 1))  # Belief about alien location
    prob_crew = np.full((D, D), 1/(D*D - 2))  # Belief about crew member location

    for step in range(1000):
        alien_detected = detect_alien(bot_pos, alien_pos)
        beep_detected = detect_beep(crew_pos, bot_pos)
        update_beliefs(bot_pos, prob_alien, prob_crew, alien_detected, beep_detected)
        
        bot_pos = choose_next_move(bot_pos, prob_crew).clip(0, D-1)  # Ensure bot stays within grid
        
        # Simulate alien movement
        alien_moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        alien_move = alien_moves[np.random.choice(len(alien_moves))]
        alien_pos = (alien_pos + alien_move).clip(0, D-1)  # Ensure alien stays within grid
        
        if np.array_equal(bot_pos, crew_pos):
            print(f"Crew member rescued in {step+1} moves.")
            break
        if np.array_equal(bot_pos, alien_pos):
            print("Bot destroyed by alien.")
            break

        # Visualization after each move
        plt.figure(figsize=(12, 6))

        # Grid overview
        plt.subplot(1, 3, 1)
        plt.title("Grid Overview")
        plt.imshow(grid, cmap='Pastel1')
        plt.scatter(bot_pos[1], bot_pos[0], color='green', label='Bot')
        plt.scatter(alien_pos[1], alien_pos[0], color='red', label='Alien')
        plt.scatter(crew_pos[1], crew_pos[0], color='blue', label='Crew Member')
        plt.legend()

        # Alien Probability Distribution
        plt.subplot(1, 3, 2)
        plt.title("Alien Probability Distribution")
        plt.imshow(prob_alien, cmap='Reds')
        plt.colorbar()

        # Crew Member Probability Distribution
        plt.subplot(1, 3, 3)
        plt.title("Crew Member Probability Distribution")
        plt.imshow(prob_crew, cmap='Blues')
        plt.colorbar()

        plt.pause(0.1)  # Pause for a brief moment to update visuals
        plt.clf()  # Clear the current figure for the next timestep

    plt.show()

# Corrections and Enhancements:
# - Ensure bot and alien movements are valid and check for collisions after each move.
# - Visualize the final state after the simulation loop.
# - Correctly apply Bayesian inference for updating beliefs based on sensor data.
# - Implement a more sophisticated decision-making process for the bot's movement, potentially considering the risk of encountering an alien.
# - The initialization of `prob_alien` and `prob_crew` should reflect the bot's initial knowledge more accurately, considering the constraints on their initial positions.
# - Incorporate a method to simulate the alien's random movement more effectively, ensuring it doesn't move into the bot or crew member's cell.
# - The update_beliefs function needs to properly account for the sensor's accuracy and the probabilities associated with detecting beeps and aliens. Bayesian updates should factor in the likelihood of receiving each piece of sensor data given the actual distances.

simulate()

# Changes made 
