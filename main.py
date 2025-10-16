# main.py
import os
from track_generator import TrackGenerator
from utils import Mode, SimType

def generate_multiple_tracks(n_samples: int):
    output_dir = "/output/"

    for i in range(n_samples):
        print(f"--- Generating Track {i} ---")
        try:
            track_gen = TrackGenerator(
                n_points=25,
                n_regions=10,
                min_bound=0,
                max_bound=250,
                mode=Mode.EXTEND,
                sim_type=SimType.CENTERLINE_CSV,
                plot_track=True,
                visualise_voronoi=True,
                create_output_file=True,
                output_location=output_dir,
                track_id=i,
                track_width_min=10.0,
                track_width_max=15.0
            )
            track_gen.create_track()
        except Exception as e:
            print(f"Could not generate track {i}. Error: {e}")
            continue

    print(f"\nFinished generating {n_samples} tracks.")

if __name__ == "__main__":
    num_tracks_to_generate = 5
    generate_multiple_tracks(num_tracks_to_generate)
