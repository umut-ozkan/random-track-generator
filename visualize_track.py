import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def plot_track_from_csv(file_path, ax=None):
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
    
    # Get per-point track widths
    width_left = track_df['w_tr_left_m'].to_numpy()
    width_right = track_df['w_tr_right_m'].to_numpy()

    # --- Calculate Track Boundaries ---
    # The heading vector is (cos(h), sin(h))
    # The normal vector (perpendicular) is (-sin(h), cos(h))
    # We use the normal vector to offset the centerline and find the boundaries.
    x_left = x_center - width_left * np.sin(heading)
    y_left = y_center + width_left * np.cos(heading)

    x_right = x_center + width_right * np.sin(heading)
    y_right = y_center - width_right * np.cos(heading)
    
    # --- Create/Reuse Axes ---
    created_ax = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 12))
        created_ax = True

    # Plot track boundaries
    ax.plot(x_left, y_left, color='black', linewidth=2, label='Track Boundary')
    ax.plot(x_right, y_right, color='black', linewidth=2)

    # Plot centerline
    ax.plot(x_center, y_center, '--', color='gray', linewidth=1.5, label='Centerline')

    # Plot Start/Finish line at the first centerline point using heading and widths
    x0 = float(x_center.iloc[0])
    y0 = float(y_center.iloc[0])
    h0 = float(heading.iloc[0])
    wl0 = float(width_left[0])
    wr0 = float(width_right[0])
    # Normal vector to heading
    nx = -np.sin(h0)
    ny = np.cos(h0)
    start_left_x = x0 + nx * wl0
    start_left_y = y0 + ny * wl0
    start_right_x = x0 - nx * wr0
    start_right_y = y0 - ny * wr0
    ax.plot([start_right_x, start_left_x], [start_right_y, start_left_y], color='green', linewidth=4, label='Start/Finish Line')

    # --- Formatting ---
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_title(f"Track Visualization: {os.path.basename(file_path)}")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.legend()

    if created_ax:
        plt.show()


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

    # Show tracks sequentially in a single window
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 12))
    # Ensure window is shown non-blocking once
    plt.show(block=False)
    try:
        for idx, file_path in enumerate(csv_files):
            ax.clear()
            plot_track_from_csv(file_path, ax=ax)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            # Allow GUI event loop to process so the figure refreshes
            plt.pause(0.05)
            user_input = input("Press Enter for next track, or 'q' then Enter to quit: ")
            if user_input.strip().lower().startswith('q'):
                break
    finally:
        plt.ioff()
        plt.close(fig)


if __name__ == "__main__":
    main()
