import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Task 1: Load and Explore the Dataset ---
# We will use the Iris dataset, a classic for data analysis.
# We'll load it directly from the seaborn library.
print("--- Task 1: Loading and Exploring the Iris Dataset ---")

try:
    # Load the dataset
    df = sns.load_dataset('iris')
    print("Dataset loaded successfully.")

    # Display the first 5 rows to inspect the data
    print("\nFirst 5 rows of the dataset:")
    print(df.head())

    # Explore the structure of the dataset
    print("\nDataset information:")
    df.info()

    # The Iris dataset is clean and has no missing values, but this is where
    # you would handle them. For example:
    # - Filling missing values: df.fillna(df.mean(), inplace=True)
    # - Dropping missing values: df.dropna(inplace=True)
    print("\nNo missing values found in this dataset, so no cleaning is required.")

except FileNotFoundError:
    print("Error: The dataset file was not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# --- Task 2: Basic Data Analysis ---
print("\n--- Task 2: Basic Data Analysis ---")

# Compute basic statistics of the numerical columns
print("\nSummary statistics of numerical columns:")
print(df.describe())

# Group the data by 'species' and compute the mean of numerical columns
species_mean = df.groupby('species').mean()
print("\nMean of numerical columns for each species:")
print(species_mean)

# Insights from the analysis:
print("\nKey findings from the analysis:")
print("- The 'setosa' species has the smallest sepal and petal dimensions.")
print("- The 'virginica' species generally has the largest sepal and petal dimensions.")
print("- 'versicolor' falls somewhere in between.")

# --- Task 3: Data Visualization ---
print("\n--- Task 3: Data Visualization ---")
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Iris Dataset Visualizations', fontsize=16)

# Plot 1: Line chart (simulating a time series)
# We can use the index as a proxy for time to show trends.
df['index'] = range(len(df))
sns.lineplot(ax=axes[0, 0], x='index', y='sepal_length', data=df, hue='species')
axes[0, 0].set_title('Sepal Length Trend Over Index')
axes[0, 0].set_xlabel('Index')
axes[0, 0].set_ylabel('Sepal Length (cm)')
axes[0, 0].legend(title='Species')

# Plot 2: Bar chart comparing average petal length across species
# Corrected the deprecation warning by assigning 'x' to 'hue' and setting legend to False.
sns.barplot(ax=axes[0, 1], x=species_mean.index, y='petal_length', data=species_mean, hue=species_mean.index, palette='viridis', legend=False)
axes[0, 1].set_title('Average Petal Length by Species')
axes[0, 1].set_xlabel('Species')
axes[0, 1].set_ylabel('Average Petal Length (cm)')

# Plot 3: Histogram of petal width
sns.histplot(ax=axes[1, 0], x='petal_width', data=df, bins=10, kde=True, color='skyblue')
axes[1, 0].set_title('Distribution of Petal Width')
axes[1, 0].set_xlabel('Petal Width (cm)')
axes[1, 0].set_ylabel('Frequency')

# Plot 4: Scatter plot for relationship between two numerical columns
sns.scatterplot(ax=axes[1, 1], x='sepal_length', y='petal_length', data=df, hue='species', style='species', s=100)
axes[1, 1].set_title('Sepal Length vs. Petal Length')
axes[1, 1].set_xlabel('Sepal Length (cm)')
axes[1, 1].set_ylabel('Petal Length (cm)')
axes[1, 1].legend(title='Species')

# Adjust layout and display the plots
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

