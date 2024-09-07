import numpy as np
import pandas as pd
import os

def generate_thermal_dataset(grid_size=100, temp_range=(20.0, 30.0), time_steps=5, influence_radius=5):
    # Initialize the dataset with random temperatures
    base_temp = np.random.uniform(*temp_range)
    dataset = np.random.uniform(base_temp - 1, base_temp + 1, (grid_size, grid_size))

    # Choose a random point for the abnormal varying temperature
    varying_point = (np.random.randint(0, grid_size), np.random.randint(0, grid_size))
    varying_temps = np.random.uniform(temp_range[0], temp_range[1], time_steps)

    # Create a Gaussian influence around the varying point
    x = np.arange(grid_size)
    y = np.arange(grid_size)
    X, Y = np.meshgrid(x, y)

    # Initialize the dataset for the time dimension
    temporal_dataset = np.zeros((grid_size, grid_size, time_steps))

    for t in range(time_steps):
        # Compute Gaussian influence
        distance = np.sqrt((X - varying_point[0]) ** 2 + (Y - varying_point[1]) ** 2)
        influence = np.exp(-distance ** 2 / (2 * influence_radius ** 2))

        # Update the dataset with varying temperatures influenced by the Gaussian function
        temporal_dataset[:, :, t] = dataset + (varying_temps[t] - base_temp) * influence

    return temporal_dataset, varying_point

# Create a directory to save the outputs
output_dir = "thermal_dataset_output"
os.makedirs(output_dir, exist_ok=True)

num_series = 10000  # Number of time series
time_steps = 5  # Number of time steps per series

for series_idx in range(num_series):
    # Generate the dataset for this time series with updated parameters
    grid_size = 100  # Grid size
    temp_range = (20.0, 30.0)  # Random temperature range
    influence_radius = 5  # Influence radius for the abnormal point
    dataset, varying_point = generate_thermal_dataset(grid_size, temp_range, time_steps, influence_radius)

    # Create a subdirectory for each time series
    series_dir = os.path.join(output_dir, f"series_{series_idx:04d}")
    os.makedirs(series_dir, exist_ok=True)

    # Save each time step as a separate DataFrame
    for t in range(time_steps):
        df = pd.DataFrame(dataset[:, :, t])
        df.to_csv(os.path.join(series_dir, f"thermal_data_step_{t:03d}.csv"), index=False)

    # Create the additional metadata array
    additional_metadata = np.array([
        grid_size,  # size of one dimension of the grid
        grid_size,  # size of the other dimension of the grid
        time_steps,  # number of time steps
        varying_point[0],  # x-coordinate of varying point
        varying_point[1],  # y-coordinate of varying point
        influence_radius  # influence radius of the abnormal point
    ])

    # Save metadata
    metadata = pd.DataFrame({
        'grid_size': [grid_size],
        'temp_range': [temp_range],
        'time_steps': [time_steps],
        'varying_point_x': [varying_point[0]],
        'varying_point_y': [varying_point[1]],
        'influence_radius': [influence_radius],
        'dataframe_shape': [f"({grid_size}, {grid_size})"],
        'additional_metadata_array': [additional_metadata.tolist()]
    })
    metadata.to_csv(os.path.join(series_dir, "metadata.csv"), index=False)

    # Save the additional metadata array separately as a numpy file
    np.save(os.path.join(series_dir, "additional_metadata.npy"), additional_metadata)

    # Plot and save the temperature variation of the varying point over time
    temperature_variation = pd.DataFrame({
        'time_step': range(time_steps),
        'temperature': dataset[varying_point[0], varying_point[1], :]
    })
    temperature_variation.to_csv(os.path.join(series_dir, "temperature_variation.csv"), index=False)

    print(f"Data and metadata for series {series_idx:04d} have been saved in the '{series_dir}' directory.")

print("All time series data and metadata have been saved.")
