import os, yaml, csv, math
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, spatial, interpolate
from shapely.geometry.polygon import Point, Polygon
from shapely.geometry import LineString
from utils import *
import math
import gpxpy
import gpxpy.gpx

class TrackGenerator:
    """
    Generates a random track based on a bounded Voronoi diagram.
    Ensures that the tracks curvature is within limits and that the car starts at a straight section.
    """

    def __init__(self,
                 n_points: int,
                 n_regions: int,
                 min_bound: float,
                 max_bound: float,
                 mode: Mode,
                 plot_track: bool,
                 visualise_voronoi: bool,
                 create_output_file: bool,
                 output_location: str,
                 track_id: int = 0,
                 z_offset: float = 0,
                 lat_offset: float = 0,
                 lon_offset: float = 0,
                 sim_type: SimType = SimType.FSSIM,
                 track_width_min: float = 10.0,
                 track_width_max: float = 15.0):
                 
        # Input parameters
        self._n_points = n_points                                               # [-]
        self._n_regions = n_regions                                             # [-]
        self._min_bound = min_bound                                             # [m]
        self._max_bound = max_bound                                             # [m]
        self._bounding_box = np.array([self._min_bound, self._max_bound] * 2)   # [x_min, x_max, y_min, y_max]
        self._mode = mode
        self._sim_type = sim_type

        # Track parameters
        self._track_width_min = track_width_min                                 # [m]
        self._track_width_max = track_width_max                                 # [m]
        self._boundary_point_spacing = 5.0                                     # [m]
        self._length_start_area = 6.0                                          # [m]
        self._curvature_threshold = 1. / 3.75                                   # [m^-1]
        self._straight_threshold = 1. / 100.                                    # [m^-1]

        # Output options
        self._plot_track = plot_track
        self._visualise_voronoi = visualise_voronoi
        self._create_output_file = create_output_file
        self._output_location = output_location
        self._track_id = track_id
        self._z_offset = z_offset
        self._lat_offset = lat_offset
        self._lon_offset = lon_offset

    def calculate_variable_track_width(self, curvature, num_points=1000):
        """
        Calculate variable track width based on curvature.
        Straights get wider tracks, tight corners get narrower tracks.
        Ensures width doesn't exceed safe limits based on radius of curvature.

        Args:
            curvature (numpy.ndarray): Curvature values along the track
            num_points (int): Number of points to interpolate to

        Returns:
            numpy.ndarray: Variable track widths along the track
        """
        # Normalize curvature to 0-1 range (higher curvature = narrower track)
        max_curvature = np.max(np.abs(curvature))
        if max_curvature == 0:
            # If no curvature variation, use constant width
            return np.full(num_points, (self._track_width_min + self._track_width_max) / 2)

        # Create normalized curvature (0 = straight, 1 = tightest corner)
        normalized_curvature = np.abs(curvature) / max_curvature

        # Apply smooth transition using sigmoid-like function
        # Straights (low curvature) -> wide track
        # Tight corners (high curvature) -> narrow track
        width_range = self._track_width_max - self._track_width_min
        track_widths = self._track_width_max - width_range * (normalized_curvature ** 1.5)

        # CRITICAL: Ensure width doesn't exceed safe limits based on curvature
        # Tight limit: total width <= 0.6 * radius of curvature
        for i in range(len(track_widths)):
            if abs(curvature[i]) > 1e-6:  # avoid division by zero
                radius = 1.0 / abs(curvature[i])
                max_safe_width = radius * 0.6
                track_widths[i] = min(track_widths[i], max_safe_width)

        # Ensure minimum width is respected
        track_widths = np.maximum(track_widths, self._track_width_min)

        return track_widths

    def _create_variable_width_boundary(self, x, y, half_widths, side='left'):
        """
        Create a boundary line parallel to the centerline with variable width.
        Uses a more robust approach that prevents boundaries from touching.

        Args:
            x, y (numpy.ndarray): Centerline coordinates
            half_widths (numpy.ndarray): Half-width values for each point
            side (str): 'left' or 'right' side of the centerline

        Returns:
            shapely.geometry.LineString: The boundary line
        """
        # Create centerline
        centerline = LineString(list(zip(x, y)))
        
        # For constant or near-constant width, use simple parallel offset
        width_variation = np.std(half_widths) / np.mean(half_widths)
        if width_variation < 0.05:  # Less than 5% variation
            offset_distance = float(np.mean(half_widths))
            if side == 'right':
                offset_distance = -offset_distance
            try:
                boundary = centerline.parallel_offset(
                    offset_distance, 
                    'left' if offset_distance >= 0 else 'right',
                    join_style=2,  # mitered join
                    mitre_limit=2.0
                )
                if isinstance(boundary, LineString) and boundary.is_simple and boundary.is_ring:
                    return boundary
            except:
                pass
        
        # For variable widths, use point-by-point offset with smoothing
        boundary_points = []
        
        # Use more points for smoother boundaries
        n_points = len(x)
        
        for i in range(n_points):
            # Calculate tangent vector using neighboring points
            if i == 0:
                # Forward difference
                dx = x[min(i+2, n_points-1)] - x[i]
                dy = y[min(i+2, n_points-1)] - y[i]
            elif i == n_points - 1:
                # Backward difference
                dx = x[i] - x[max(i-2, 0)]
                dy = y[i] - y[max(i-2, 0)]
            else:
                # Central difference with wider stencil for smoother tangent
                i_prev = max(i-2, 0)
                i_next = min(i+2, n_points-1)
                dx = x[i_next] - x[i_prev]
                dy = y[i_next] - y[i_prev]
        
            # Normalize tangent
            length = np.sqrt(dx*dx + dy*dy)
            if length > 1e-10:
                dx_norm = dx / length
                dy_norm = dy / length
            else:
                # Skip degenerate points
                if i > 0:
                    boundary_points.append(boundary_points[-1])
                continue
        
            # Perpendicular direction (rotate tangent by 90 degrees)
            if side == 'left':
                perp_dx = dy_norm
                perp_dy = -dx_norm
            else:
                perp_dx = -dy_norm
                perp_dy = dx_norm
        
            # Calculate offset point
            offset_x = x[i] + half_widths[i] * perp_dx
            offset_y = y[i] + half_widths[i] * perp_dy
        
            boundary_points.append([offset_x, offset_y])
        
        if len(boundary_points) < 3:
            # Fallback to simple parallel offset
            offset_distance = float(np.mean(half_widths))
            if side == 'right':
                offset_distance = -offset_distance
            return centerline.parallel_offset(
                offset_distance,
                'left' if offset_distance >= 0 else 'right',
                join_style=2
            )
        
        # Convert to array for smoothing
        boundary_array = np.array(boundary_points)
        
        # Apply smoothing in multiple passes with decreasing sigma
        from scipy.ndimage import gaussian_filter1d
        
        # First pass: light smoothing
        smoothed_x = gaussian_filter1d(boundary_array[:, 0], sigma=1.5, mode='wrap')
        smoothed_y = gaussian_filter1d(boundary_array[:, 1], sigma=1.5, mode='wrap')
        
        boundary_array[:, 0] = smoothed_x
        boundary_array[:, 1] = smoothed_y
        
        # Close the loop
        boundary_array = np.vstack([boundary_array, boundary_array[0]])
        
        # Create LineString
        line = LineString(boundary_array)
        
        # If the line self-intersects after smoothing, try without smoothing
        if not line.is_simple:
            boundary_array = np.array(boundary_points + [boundary_points[0]])
            line = LineString(boundary_array)
            
            # If still not simple, reduce points
            if not line.is_simple:
                # Decimate points
                step = max(1, len(boundary_points) // 200)
                boundary_array = np.array(boundary_points[::step] + [boundary_points[0]])
                line = LineString(boundary_array)
        
        return line

    def _validate_track_geometry(self, track_left, track_right, track_widths, x, y):
        """
        Comprehensive validation of track geometry to catch corrupted tracks.
        
        Returns:
            bool: True if track is valid, False if corrupted
        """
        try:
            # Check 1: Boundaries must be LineStrings
            if not isinstance(track_left, LineString) or not isinstance(track_right, LineString):
                print(f"  Validation failed: Not LineStrings")
                return False
            
            # Check 2: No self-intersections
            if not track_left.is_simple or not track_right.is_simple:
                print(f"  Validation failed: Self-intersections detected")
                return False
            
            # Check 3: Boundaries must not cross or touch
            if track_left.crosses(track_right) or track_left.intersects(track_right):
                print(f"  Validation failed: Boundaries cross or touch")
                return False
            
            # Check 4: Minimum separation with touching threshold
            min_width = np.min(track_widths)
            actual_min_distance = track_left.distance(track_right)
            if actual_min_distance < 0.01:  # Less than 1cm means they're touching
                print(f"  Validation failed: Boundaries touching (dist={actual_min_distance:.4f})")
                return False
            if actual_min_distance < min_width * 0.3:  # At least 30% of minimum width
                print(f"  Validation failed: Too close (dist={actual_min_distance:.2f}, min={min_width*0.3:.2f})")
                return False
            
            # Check 5: Verify boundaries are closed loops
            if not track_left.is_ring or not track_right.is_ring:
                print(f"  Validation failed: Not closed rings")
                return False

            # Check 6: Ensure centerline is between boundaries using signed side tests
            num_checks = min(len(x), 100)
            for i in np.linspace(0, len(x)-1, num_checks, dtype=int):
                # Local tangent using central difference
                i_prev = (i - 1) % len(x)
                i_next = (i + 1) % len(x)
                tx = x[i_next] - x[i_prev]
                ty = y[i_next] - y[i_prev]
                t_len = np.hypot(tx, ty) + 1e-12
                nx = -ty / t_len  # right-hand normal
                ny = tx / t_len

                cx, cy = x[i], y[i]
                left_point = track_left.interpolate(track_left.project(Point(cx, cy)))
                right_point = track_right.interpolate(track_right.project(Point(cx, cy)))

                # Vectors from center to boundary points
                v_left = np.array([left_point.x - cx, left_point.y - cy])
                v_right = np.array([right_point.x - cx, right_point.y - cy])

                # Signed distances using chosen normal orientation
                d_left_signed = v_left[0] * nx + v_left[1] * ny
                d_right_signed = v_right[0] * (-nx) + v_right[1] * (-ny)

                # Both should be positive if left is on +n and right is on -n
                if d_left_signed <= 0 or d_right_signed <= 0:
                    print("  Validation failed: Centerline not between boundaries at some points")
                    return False

                # Also verify magnitudes roughly match half widths (with tolerance)
                expected_half = track_widths[i] / 2.0
                if d_left_signed < 0.2 * expected_half or d_right_signed < 0.2 * expected_half:
                    print("  Validation failed: Boundary too close to centerline at some points")
                    return False
            
            return True
            
        except Exception as e:
            # If any check fails with an exception, consider track invalid
            print(f"  Validation error: {e}")
            return False

    def bounded_voronoi(self, input_points, bounding_box):
        """
        Creates a Voronoi diagram bounded by the bounding box.
        Mirror input points at edges of the bounding box.
        Then create Voronoi diagram using all five sets of points.
        This prevents having a Voronoi diagram with edges going off to infinity.
        
        Args:
            input_points (numpy.ndarray): Coordinates of input points for Voronoi diagram.
            bounding_box (numpy.ndarray): Specifies the boundaries of the Voronoi diagram, [x_min, x_max, y_min, y_max].
        
        Returns:
            scipy.spatial.qhull.Voronoi: Voronoi diagram object.
        """
        
        def _mirror(boundary, axis):
            mirrored = np.copy(points_center)
            mirrored[:, axis] = 2 * boundary - mirrored[:, axis]
            return mirrored
        
        x_min, x_max, y_min, y_max = bounding_box
        
        # Mirror points around each boundary
        points_center = input_points
        points_left = _mirror(x_min, axis=0) 
        points_right = _mirror(x_max, axis=0) 
        points_down = _mirror(y_min, axis=1)
        points_up = _mirror(y_max, axis=1)
        points = np.concatenate([points_center, points_left, points_right, points_down, points_up])
        
        # Compute Voronoi
        vor = spatial.Voronoi(points)
        
        # We only need the section of the Voronoi diagram that is inside the bounding box
        vor.filtered_points = points_center
        vor.filtered_regions = np.array(vor.regions, dtype=object)[vor.point_region[:vor.npoints//5]]
        return vor

    def create_track(self):
        """
        Creates a track from the vertices of a Voronoi diagram.
        1.  Create bounded Voronoi diagram.
        2.  Select regions of Voronoi diagram based on selection mode.
        3.  Get the vertices belonging to the regions and sort them clockwise.
        4.  Interpolate between vertices.
        5.  Calculate curvature of track to check wether the curvature threshold is exceeded.
        6.  If curvature threshold is exceeded, remove vertice where the curvature is the highest from its set.
            Repeat steps 4-6 until curvature is within limimts.
        7.  Check if track does not cross itself. If so, go to step 2 and reiterate.
        8.  Find long enough straight section to place start line and start position.
        9.  Translate and rotate track to origin.
        10. Create track yaml file.
        """
        # Create bounded Voronoi diagram
        input_points = np.random.uniform(self._min_bound, self._max_bound, (self._n_points, 2))
        vor = self.bounded_voronoi(input_points, self._bounding_box)

        max_attempts = 100  # Prevent infinite loop
        attempt = 0

        while attempt < max_attempts:
            attempt += 1
            if attempt % 10 == 0:
                print(f"Track {self._track_id}: Attempt {attempt}/{max_attempts}...")
            
            if self._mode.value == 1:
                # Pick a random point and find its n closest neighbours
                random_index = np.random.randint(0, self._n_points)
                random_point_indices = [random_index]
                random_point = input_points[random_index]
                
                for i in range(self._n_regions - 1):
                    closest_point_index = closest_node(random_point, input_points, k=i+1)
                    random_point_indices.append(closest_point_index)
                    
            elif self._mode.value == 2:
                # Pick a random point, create a line extending from this point and find other points close to this line
                random_index = np.random.randint(0, self._n_points)
                random_heading = np.random.uniform(0, np.pi/2)
                random_point = input_points[random_index]
                
                start = (random_point[0] - 1./2. * self._max_bound * np.cos(random_heading), random_point[1] - 1./2. * self._max_bound * np.sin(random_heading))
                end = (random_point[0] + 1./2. * self._max_bound * np.cos(random_heading), random_point[1] + 1./2. * self._max_bound * np.sin(random_heading))
                line = LineString([start, end])
                distances = [Point(p).distance(line) for p in input_points]
                random_point_indices = np.argpartition(distances, self._n_regions)[:self._n_regions]
                
            elif self._mode.value == 3:
                # Select regions randomly
                random_point_indices = np.random.randint(0, self._n_points, self._n_regions)
            
            # From the Voronoi regions, get the regions belonging to the randomly selected points
            regions = np.array([np.array(region) for region in vor.regions], dtype=object)
            random_region_indices = vor.point_region[random_point_indices]
            random_regions = np.concatenate(regions[random_region_indices])
            
            # Get the vertices belonging to the random regions
            random_vertices = np.unique(vor.vertices[random_regions], axis=0)
            
            # Sort vertices
            sorted_vertices = clockwise_sort(random_vertices)
            sorted_vertices = np.vstack([sorted_vertices, sorted_vertices[0]])
            
            while True:
        
                # Interpolate
                tck, _ = interpolate.splprep([sorted_vertices[:,0], sorted_vertices[:,1]], s=0, per=True)
                t = np.linspace(0, 1, 1000)
                x, y = interpolate.splev(t, tck, der=0)
                dx_dt, dy_dt = interpolate.splev(t, tck, der=1)
                d2x_dt2, d2y_dt2 = interpolate.splev(t, tck, der=2)
                
                # Calculate curvature
                k = curvature(dx_dt, d2x_dt2, dy_dt, d2y_dt2)
                abs_curvature = np.abs(k)
                
                # Check if curvature exceeds threshold
                peaks, _ = signal.find_peaks(abs_curvature)
                exceeded_peaks = abs_curvature[peaks] > self._curvature_threshold
                max_peak_index = abs_curvature[peaks].argmax()
                is_curvature_exceeded = exceeded_peaks[max_peak_index]
                
                if is_curvature_exceeded:
                    # Find vertice where curvature is exceeded and delete vertice from sorted vertices. Reiterate
                    max_peak = peaks[max_peak_index]
                    peak_coordinate = (x[max_peak], y[max_peak])
                    vertice = closest_node(peak_coordinate, sorted_vertices, k=0)
                    sorted_vertices = np.delete(sorted_vertices, vertice, axis=0)
                    
                    # Make sure that first and last coordinate are the same for periodic interpolation
                    if not np.array_equal(sorted_vertices[0], sorted_vertices[-1]):
                        sorted_vertices = np.vstack([sorted_vertices, sorted_vertices[0]])
                else:
                    break
            
            # Calculate variable track widths based on curvature
            track_widths = self.calculate_variable_track_width(k)

            # Create centerline and boundaries using variable widths
            centerline = LineString(list(zip(x, y)))

            # Ensure centerline is a closed loop and simple (no self-intersections)
            is_closed = centerline.is_ring
            if not is_closed:
                centerline = LineString(list(zip(np.r_[x, x[0]], np.r_[y, y[0]])))

            track_left = self._create_variable_width_boundary(x, y, track_widths / 2, side='left')
            track_right = self._create_variable_width_boundary(x, y, track_widths / 2, side='right')

            # Smoothing handled within _create_variable_width_boundary

            # COMPREHENSIVE VALIDATION
            print(f"  Checking: left_simple={track_left.is_simple}, right_simple={track_right.is_simple}")
            print(f"  left_ring={track_left.is_ring}, right_ring={track_right.is_ring}")
            print(f"  crosses={track_left.crosses(track_right)}")
            print(f"  min_distance={track_left.distance(track_right):.2f}, min_width={np.min(track_widths):.2f}")
            if self._validate_track_geometry(track_left, track_right, track_widths, x, y):
                print(f"Track {self._track_id}: Valid track generated!")
                break
            else:
                print(f"Track {self._track_id}: Invalid geometry detected, retrying with new Voronoi regions...")
                # Continue to outer while loop to generate new track
                continue

        if attempt >= max_attempts:
            raise Exception(f"Track {self._track_id}: Failed to generate valid track after {max_attempts} attempts. Try different parameters.")

        # Calculate boundary point spacing using average track width
        avg_track_width = (self._track_width_min + self._track_width_max) / 2
        boundary_point_spacing_left = np.linspace(0, track_left.length, np.ceil(track_left.length / avg_track_width).astype(int) + 1)[:-1]
        boundary_point_spacing_right= np.linspace(0, track_right.length, np.ceil(track_right.length / avg_track_width).astype(int) + 1)[:-1]

        # Determine coordinates of boundary points
        boundary_points_left = np.asarray([[track_left.interpolate(sp).x, track_left.interpolate(sp).y] for sp in boundary_point_spacing_left])
        boundary_points_right = np.asarray([[track_right.interpolate(sp).x, track_right.interpolate(sp).y] for sp in boundary_point_spacing_right])

        # Find straight section in track that is at least the length of the start area
        # If such a section cannot be found, adjust the straight_threshold and length_start_area variables
        # There is only a chance of this happening if n_regions == 1 
        straight_threshold = self._straight_threshold if abs_curvature.min() < self._straight_threshold else abs_curvature.min() + 0.1
        straight_sections = abs_curvature[:-1] <= straight_threshold
        distances = arc_length(x, y, 1 / abs_curvature)
        length_straights = distances * straight_sections

        # Find cumulative length of straight sections
        for i in range(1, len(length_straights)):
            if length_straights[i]:
                length_straights[i] += length_straights[i-1]
                
        # Find start line and start pose
        length_start_area = self._length_start_area if length_straights.max() > self._length_start_area else length_straights.max()
        try:
            start_line_index = np.where(length_straights > length_start_area)[0][0]
        except IndexError:
            raise Exception("Unable to find suitable starting position. Try to decrease the length of the starting area or different input parameters.")
        start_line = np.array([x[start_line_index], y[start_line_index]])
        s_target = float(np.sum(distances[:start_line_index]) - length_start_area)
        # Wrap around if negative
        if s_target < 0:
            s_target = centerline.length + s_target
        p_start = centerline.interpolate(s_target)
        start_position = np.array([p_start.x, p_start.y])
        start_heading = float(np.arctan2(*(start_line - start_position)))

        # Translate and rotate track to origin
        # 1. Calculate heading and cumulative distance before transformation
        heading_rad = np.arctan2(dy_dt, dx_dt)
        segment_lengths = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        s_m = np.concatenate(([0], np.cumsum(segment_lengths)))

        # 2. Define transformation matrix
        rotation_angle = start_heading - np.pi / 2
        M = transformation_matrix(-start_position, rotation_angle)

        # 3. Transform centerline coordinates
        centerline_xy = np.c_[x, y, np.ones(len(x))]
        transformed_centerline = M.dot(centerline_xy.T)[:-1].T
        x_m = transformed_centerline[:, 0]
        y_m = transformed_centerline[:, 1]

        # 4. Transform heading
        unwrapped_heading = np.unwrap(heading_rad)
        transformed_unwrapped_heading = unwrapped_heading - rotation_angle
        heading_rad_transformed = (transformed_unwrapped_heading + np.pi) % (2 * np.pi) - np.pi

        # 5. Transform boundary points (for other output types)
        boundary_points_left = M.dot(np.c_[boundary_points_left, np.ones(len(boundary_points_left))].T)[:-1].T
        boundary_points_right = M.dot(np.c_[boundary_points_right, np.ones(len(boundary_points_right))].T)[:-1].T

        # Find the minimum bounds of the entire track including boundary points
        all_x = np.concatenate([x_m, boundary_points_left[:, 0], boundary_points_right[:, 0]])
        all_y = np.concatenate([y_m, boundary_points_left[:, 1], boundary_points_right[:, 1]])
        x_min, y_min = all_x.min(), all_y.min()

        # Determine the shift required to make all coordinates positive, with a 5m buffer
        x_shift = -x_min + 5 if x_min < 0 else 0
        y_shift = -y_min + 5 if y_min < 0 else 0

        # Apply the shift to all track data
        x_m += x_shift
        y_m += y_shift
        boundary_points_left += [x_shift, y_shift]
        boundary_points_right += [x_shift, y_shift]

        # Create track file
        if self._visualise_voronoi: self.visualise_voronoi(vor, sorted_vertices, random_point_indices, input_points, x, y)
        if self._plot_track: self.plot_track(boundary_points_left, boundary_points_right)
        if self._create_output_file:
            # Calculate variable track widths for output (interpolated to match centerline points)
            centerline_track_widths = self.calculate_variable_track_width(k, len(x_m))

            track_data = {
                'boundary_points_left': boundary_points_left,
                'boundary_points_right': boundary_points_right,
                'x_m': x_m,
                'y_m': y_m,
                'w_tr_right_m': centerline_track_widths / 2.0,
                'w_tr_left_m': centerline_track_widths / 2.0,
                'track_widths': centerline_track_widths,
                'curvature': k,
                'heading_rad': heading_rad_transformed,
                's_m': s_m
            }
            self.create_output_file(track_data) # Call the new generalized function

    def visualise_voronoi(self, vor, sorted_vertices, random_point_indices, input_points, x, y):
        """
        Visualises the voronoi diagram and the resulting track. 

        Args:
            vor (scipy.spatial.qhull.Voronoi): Voronoi diagram object.
            sorted_vertices (numpy.ndarray): Selected vertices sorted clockwise.
            random_point_indices (numpy.ndarray): Selected points.
            input_points (numpy.ndarray): All Voronoi points.
        """
        # Plot initial points
        plt.figure()
        plt.plot(vor.filtered_points[:, 0], vor.filtered_points[:, 1], 'b.')

        # Plot vertices points
        for region in vor.filtered_regions:
            vertices = vor.vertices[region, :]
            plt.plot(vertices[:, 0], vertices[:, 1], 'go')
            
        # Plot edges
        for region in vor.filtered_regions:
            vertices = vor.vertices[region + [region[0]], :]
            plt.plot(vertices[:, 0], vertices[:, 1], 'k-')

        # Plot selected vertices
        plt.scatter(sorted_vertices[:,0], sorted_vertices[:,1], color='y', s=200, label='Selected vertices')

        # Plot selected points
        plt.scatter(*input_points[random_point_indices].T, s=100, marker='x', color='b', label='Selected points')

        # Plot track
        plt.scatter(x, y)
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.axis('equal')
        plt.legend()
        plt.show()

    def plot_track(self, boundary_left, boundary_right):
        """
        Plots the resulting track. The car will start at the origin.

        Args:
            boundary_left (numpy.ndarray): Nx2 numpy array of left boundary point coordinates.
            boundary_right (numpy.ndarray): Nx2 numpy array of right boundary point coordinates.
        """
        plt.figure()
        plt.plot(boundary_left[:, 0], boundary_left[:, 1], 'k-', label='Track Limits')
        plt.plot(boundary_right[:, 0], boundary_right[:, 1], 'k-')

        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.axis('equal')
        plt.grid()
        plt.legend()
        plt.show()
        
    def create_output_file(self, track_data):
        """Writes the track data to a file based on the specified simulator type."""
        abs_path_dir = os.path.realpath(os.path.dirname(__file__))
        track_file_dir = abs_path_dir + self._output_location
        if not os.path.exists(track_file_dir):
            os.makedirs(track_file_dir)

        # --- FSSIM Output (Updated for unique filenames) ---
        if self._sim_type == SimType.FSSIM:
            track_file_name = os.path.join(track_file_dir, f'track_{self._track_id}.yaml')
            print(f"Saving FSSIM track to {track_file_name}")
            with open(track_file_name, 'w') as outfile:
                data = {
                    'cones_left': track_data['boundary_points_left'].tolist(),
                    'cones_right': track_data['boundary_points_right'].tolist(),
                    'cones_orange': [],
                    'cones_orange_big': [[4.7, 2.5], [4.7, -2.5], [7.3, 2.5], [7.3, -2.5]],
                    'starting_pose_cg': [0., 0., 0.],
                    'tk_device': [[6., 3.], [6., -3.]]
                }
                yaml.dump(data, outfile)

        # --- FSDS Output (Updated for unique filenames) ---
        elif self._sim_type == SimType.FSDS:
            track_file_name = os.path.join(track_file_dir, f'track_{self._track_id}.csv')
            print(f"Saving FSDS track to {track_file_name}")
            with open(track_file_name, 'w') as outfile:
                for boundary_point in track_data['boundary_points_left']:
                    outfile.write("blue," + str(boundary_point[0]) + ',' + str(boundary_point[1]) + ',0,0.01,0.01,0\n')

                for boundary_point in track_data['boundary_points_right']:
                    outfile.write("yellow," + str(boundary_point[0]) + ',' + str(boundary_point[1]) + ',0,0.01,0.01,0\n')

                outfile.write("big_orange,4.7,2.2,0,0.01,0.01,0\n")
                outfile.write("big_orange,4.7,-2.2,0,0.01,0.01,0\n")
                outfile.write("big_orange,7.3,2.2,0,0.01,0.01,0\n")
                outfile.write("big_orange,7.3,-2.2,0,0.01,0.01,0\n")

        # --- GPX Output (Updated for unique filenames) ---
        elif self._sim_type == SimType.GPX:
            track_file_name = os.path.join(track_file_dir, f'track_{self._track_id}.gpx')
            print(f"Saving GPX track to {track_file_name}")
            gpx = gpxpy.gpx.GPX()

            # Create first track in our GPX:
            gpx_track = gpxpy.gpx.GPXTrack()
            gpx.tracks.append(gpx_track)

            # Create points:
            for boundary_point in track_data['boundary_points_left']:
                lat  = self._lat_offset  + (boundary_point[1] / 6378100) * (180 / math.pi)
                lon = self._lon_offset + (boundary_point[0] / 6378100) * (180 / math.pi) / math.cos(self._lat_offset * math.pi/180)
                gpx.waypoints.append(gpxpy.gpx.GPXWaypoint(latitude=lat, longitude=lon, elevation=0 + self._z_offset))

            for boundary_point in track_data['boundary_points_right']:
                lat  = self._lat_offset  + (boundary_point[1] / 6378100) * (180 / math.pi)
                lon = self._lon_offset + (boundary_point[0] / 6378100) * (180 / math.pi) / math.cos(self._lat_offset * math.pi/180)
                gpx.waypoints.append(gpxpy.gpx.GPXWaypoint(latitude=lat, longitude=lon, elevation=0 + self._z_offset))

            with open(track_file_name, 'w') as outfile:
                outfile.writelines(gpx.to_xml())

        # --- NEW Centerline CSV Output ---
        elif self._sim_type == SimType.CENTERLINE_CSV:
            track_file_name = os.path.join(track_file_dir, f'track_{self._track_id}.csv')
            print(f"Saving Centerline track to {track_file_name}")
            with open(track_file_name, 'w', newline='') as outfile:
                writer = csv.writer(outfile)
                writer.writerow(['track_id', 'x_m', 'y_m', 'w_tr_right_m', 'w_tr_left_m', 'track_width_m', 'curvature', 'heading_rad', 's_m'])

                for i in range(len(track_data['x_m'])):
                    writer.writerow([
                        self._track_id,
                        track_data['x_m'][i],
                        track_data['y_m'][i],
                        track_data['w_tr_right_m'][i],
                        track_data['w_tr_left_m'][i],
                        track_data['track_widths'][i],
                        track_data['curvature'][i],
                        track_data['heading_rad'][i],
                        track_data['s_m'][i]
                    ])
