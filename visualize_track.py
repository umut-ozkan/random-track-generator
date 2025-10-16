import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def plot_track_from_csv(file_path):
    """
    Reads track data from a CSV file and plots the centerline, boundaries,
    and start/finish line.

    Args:
        file_path (str): The full path to the track's CSV file.
    """
    print(f"Visualizing track from: {os.path.basename(file_path)}")

    try:
        # Read the track data using pandas
        track_df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: File is empty {file_path}")
        return

    # Extract the centerline coordinates
    x_center = track_df['x_m']
    y_center = track_df['y_m']
    heading = track_df['heading_rad']
    
    # Get the track width (assuming it's constant for visualization)
    width_left = track_df['w_tr_left_m'].iloc[0]
    width_right = track_df['w_tr_right_m'].iloc[0]

    # --- Calculate Track Boundaries ---
    # The heading vector is (cos(h), sin(h))
    # The normal vector (perpendicular) is (-sin(h), cos(h))
    # We use the normal vector to offset the centerline and find the boundaries.
    x_left = x_center - width_left * np.sin(heading)
    y_left = y_center + width_left * np.cos(heading)

    x_right = x_center + width_right * np.sin(heading)
    y_right = y_center - width_right * np.cos(heading)
    
    # --- Create the Plot ---
    plt.figure(figsize=(12, 12))
    
    # Plot track boundaries
    plt.plot(x_left, y_left, color='black', linewidth=2, label='Track Boundary')
    plt.plot(x_right, y_right, color='black', linewidth=2)
    
    # Plot centerline
    plt.plot(x_center, y_center, '--', color='gray', linewidth=1.5, label='Centerline')
    
    # Plot Start/Finish line (track starts at (0,0) heading up)
    start_x = [-width_right, width_left]
    start_y = [0, 0]
    plt.plot(start_x, start_y, color='green', linewidth=4, label='Start/Finish Line')
    
    # --- Formatting ---
    plt.axis('equal')  # Ensures the track aspect ratio is correct
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title(f"Track Visualization: {os.path.basename(file_path)}")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.legend()


def main():
    """
    Finds all track CSV files in the specified output directory and
    generates a plot for each one.
    """
    # Directory where the track CSV files are stored
    # This should match the 'output_location' from your generator script
    input_directory = "output"

    # Find all CSV files in the directory
    search_pattern = os.path.join(input_directory, "track_*.csv")
    csv_files = glob.glob(search_pattern)

    if not csv_files:
        print(f"No track CSV files found in '{input_directory}/'.")
        print("Please make sure you have generated the tracks first.")
        return

    # Generate a plot for each file
    for file_path in csv_files:
        plot_track_from_csv(file_path)

    # Show all the plots
    plt.show()


if __name__ == "__main__":
    main()
