import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the simulation results
bot1_df = pd.read_csv('D:/USA Docs/Rutgers/Intro to AI/Project 2/Roomba2/simulation_results_bot1.csv')
bot2_df = pd.read_csv('/D:/USA Docs/Rutgers/Intro to AI/Project 2/Roomba2/simulation_results_bot2.csv')

# Assuming the CSV has 'alpha', 'avg_steps', and 'success_rate' columns
# You might need to adjust this based on your actual CSV structure

# Calculate average metrics for comparison
avg_metrics_bot1 = bot1_df.groupby('alpha').mean().reset_index()
avg_metrics_bot2 = bot2_df.groupby('alpha').mean().reset_index()

# Plotting
sns.set_style("whitegrid")
plt.figure(figsize=(18, 6))

# Average Number of Moves
plt.subplot(1, 3, 1)
plt.plot(avg_metrics_bot1['alpha'], avg_metrics_bot1['avg_steps'], label='Bot 1', marker='o')
plt.plot(avg_metrics_bot2['alpha'], avg_metrics_bot2['avg_steps'], label='Bot 2', marker='s')
plt.xlabel('Alpha')
plt.ylabel('Average Number of Moves')
plt.title('Average Moves to Rescue')
plt.legend()

# Success Rate
plt.subplot(1, 3, 2)
plt.plot(avg_metrics_bot1['alpha'], avg_metrics_bot1['success_rate'], label='Bot 1', marker='o')
plt.plot(avg_metrics_bot2['alpha'], avg_metrics_bot2['success_rate'], label='Bot 2', marker='s')
plt.xlabel('Alpha')
plt.ylabel('Success Rate')
plt.title('Success Rate in Avoiding Alien and Rescuing Crew')
plt.legend()

# For average number of crew members saved, you would calculate similarly
# Assuming the metric exists in your dataframe. 
# If you only have a binary outcome (success/failure), this would be the same as the success rate.

plt.tight_layout()
plt.show()
