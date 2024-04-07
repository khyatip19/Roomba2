import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the simulation results
bot3_df = pd.read_csv('D:/USA Docs/Rutgers/Intro to AI/Project 2/Roomba2/simulation_results_bot3.csv')
bot4_df = pd.read_csv('D:/USA Docs/Rutgers/Intro to AI/Project 2/Roomba2/simulation_results_bot4.csv')
bot5_df = pd.read_csv('D:/USA Docs/Rutgers/Intro to AI/Project 2/Roomba2/simulation_results_bot5.csv')

# Assuming the CSV has 'alpha', 'avg_steps', and 'success_rate' columns
# You might need to adjust this based on your actual CSV structure

# Calculate average metrics for comparison
avg_metrics_bot3 = bot3_df.groupby('alpha').mean().reset_index()
avg_metrics_bot4 = bot4_df.groupby('alpha').mean().reset_index()
avg_metrics_bot5 = bot5_df.groupby('alpha').mean().reset_index()

# Plotting
sns.set_style("whitegrid")
plt.figure(figsize=(6, 4))

# Average Number of Moves
# plt.subplot(1, 3, 1)
plt.plot(avg_metrics_bot3['alpha'], avg_metrics_bot3['avg_steps'], label='Bot 3', marker='o')
plt.plot(avg_metrics_bot4['alpha'], avg_metrics_bot4['avg_steps'], label='Bot 4', marker='s')
plt.plot(avg_metrics_bot5['alpha'], avg_metrics_bot5['avg_steps'], label='Bot 5', marker='x')
plt.xlabel('Alpha')
plt.ylabel('Average Number of Moves')
plt.title('Average Moves to Rescue')
plt.legend()
plt.show()

# Success Rate
plt.figure(figsize=(6, 4))
# plt.subplot(1, 3, 2)
plt.plot(avg_metrics_bot3['alpha'], avg_metrics_bot3['success_rate'], label='Bot 3', marker='o')
plt.plot(avg_metrics_bot4['alpha'], avg_metrics_bot4['success_rate'], label='Bot 4', marker='s')
plt.plot(avg_metrics_bot5['alpha'], avg_metrics_bot5['success_rate'], label='Bot 5', marker='x')
plt.xlabel('Alpha')
plt.ylabel('Success Rate')
plt.title('Success Rate in Avoiding Alien and Rescuing Crew')
plt.legend()
plt.show()

# For average number of crew members saved, you would calculate similarly
# Assuming the metric exists in your dataframe. 
# If you only have a binary outcome (success/failure), this would be the same as the success rate.

plt.tight_layout()
plt.show()
