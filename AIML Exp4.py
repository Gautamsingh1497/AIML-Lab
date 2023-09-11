# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset (adjust the file path)
dataset_path = "Moonlanding.csv"
data = pd.read_csv(dataset_path, encoding='latin1')

# Display the first few rows to inspect the data
print(data.head())
print(data.describe())

# Convert the 'Launch Date' column to datetime format
data['Launch Date'] = pd.to_datetime(data['Launch Date'])

# Count the occurrences of each unique value in the 'Outcome' column
outcome_counts = data['Outcome'].value_counts()

# Create a bar plot of mission outcomes with custom labels and title
plt.figure(figsize=(10, 6))
bars=plt.bar(outcome_counts.index, outcome_counts.values)
plt.xlabel('Outcome')  # Label for the x-axis
plt.ylabel('Count')    # Label for the y-axis
plt.title('Mission Outcomes')  # Title for the plot

# Annotate the bars with their counts
for bar, count in zip(bars, outcome_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(count), ha='center', va='bottom')

plt.show()  # Display the plot
