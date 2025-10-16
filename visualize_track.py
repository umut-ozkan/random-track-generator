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
        ax: Optional matplotlib axes object to plot on
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
    x_center = track_df['x_m'].to_numpy()
    y_center = track_df['y_m'].to_numpy()
    heading = track_df['heading_rad'].to_numpy()
    
    # Get per-point track widths
    width_left = track_df['w_tr_left_m'].to_numpy()
    width_right = track_df['w_tr_right_m'].to_numpy()

    # --- Calculate Track Boundaries (MATCHING GENERATOR METHOD) ---
    n_points = len(x_center)
    x_left = np.zeros(n_points)
    y_left = np.zeros(n_points)
    x_right = np.zeros(n_points)
    y_right = np.zeros(n_points)
    
    for i in range(n_points):
        # Calculate tangent vector using same method as generator
        if i == 0:
            # Forward difference
            dx = x_center[min(i+2, n_points-1)] - x_center[i]
            dy = y_center[min(i+2, n_points-1)] - y_center[i]
        elif i == n_points - 1:
            # Backward difference
            dx = x_center[i] - x_center[max(i-2, 0)]
            dy = y_center[i] - y_center[max(i-2, 0)]
        else:
            # Central difference with wider stencil
            i_prev = max(i-2, 0)
            i_next = min(i+2, n_points-1)
            dx = x_center[i_next] - x_center[i_prev]
            dy = y_center[i_next] - y_center[i_prev]
        
        # Normalize tangent
        length = np.sqrt(dx*dx + dy*dy)
        if length > 1e-10:
            dx_norm = dx / length
            dy_norm = dy / length
        else:
            dx_norm = 0
            dy_norm = 0
        
        # Perpendicular direction (rotate tangent by 90 degrees)
        # Left side
        perp_dx_left = dy_norm
        perp_dy_left = -dx_norm
        x_left[i] = x_center[i] + width_left[i] * perp_dx_left
        y_left[i] = y_center[i] + width_left[i] * perp_dy_left
        
        # Right side
        perp_dx_right = -dy_norm
        perp_dy_right = dx_norm
        x_right[i] = x_center[i] + width_right[i] * perp_dx_right
        y_right[i] = y_center[i] + width_right[i] * perp_dy_right
    
    # Apply the same smoothing as in generator (optional, for consistency)
    from scipy.ndimage import gaussian_filter1d
    x_left = gaussian_filter1d(x_left, sigma=1.5, mode='wrap')
    y_left = gaussian_filter1d(y_left, sigma=1.5, mode='wrap')
    x_right = gaussian_filter1d(x_right, sigma=1.5, mode='wrap')
    y_right = gaussian_filter1d(y_right, sigma=1.5, mode='wrap')
    
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

    # Plot Start/Finish line at the first centerline point
    ax.plot([x_right[0], x_left[0]], [y_right[0], y_left[0]], 
            color='green', linewidth=4, label='Start/Finish Line')

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
