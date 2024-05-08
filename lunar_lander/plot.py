import pandas as pd
import matplotlib.pyplot as plt

# Specify the path to the CSV file
csv_file_path = 'lunar_lander/lunar_lander_training.csv'

# Read the CSV file into a pandas DataFrame
data = pd.read_csv(csv_file_path)

# Check if the second column exists by confirming the column index (1 because it's zero-based)
if len(data.columns) < 2:
    raise ValueError("The provided CSV file does not contain at least two columns.")

# Extract the second column (columns are zero-indexed)
second_column = data.iloc[:, 1]

# Plot the values of the second column
plt.figure(figsize=(10, 6))
plt.plot(second_column, linestyle='-', color='b')
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Plot of Second Column Values')
plt.grid(True)
plt.show()
