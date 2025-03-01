import pandas as pd
import numpy as np
import similaritymeasures
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from multiprocessing import Pool
import os

# Define input files dynamically
reference_file = "reference_track.csv"
correction_files = {
    "with_correction": "cropped_waypoint_with_correction.csv",
    "without_correction": "cropped_waypoint_without_correction.csv"
}

# Load the reference trajectory
df_reference = pd.read_csv(reference_file)

# Extract unique routes
unique_routes = df_reference[['Route_Type', 'Start_Index', 'End_Index']].drop_duplicates()

# Function to downsample trajectories adaptively
def grid_based_downsampling(route, num_points, grid_size=50):
    """Fast adaptive downsampling using grid-based density estimation."""
    if len(route) <= num_points:
        return route  # No downsampling needed

    x_min, x_max = np.min(route[:, 0]), np.max(route[:, 0])
    y_min, y_max = np.min(route[:, 1]), np.max(route[:, 1])

    # Define grid resolution
    x_bins = np.linspace(x_min, x_max, grid_size)
    y_bins = np.linspace(y_min, y_max, grid_size)

    # Compute grid indices for each point
    x_indices = np.digitize(route[:, 0], x_bins)
    y_indices = np.digitize(route[:, 1], y_bins)

    # Step 2: Count number of points per grid cell
    grid_counts = {}
    for i in range(len(route)):
        cell = (x_indices[i], y_indices[i])
        grid_counts[cell] = grid_counts.get(cell, 0) + 1

    # Compute probability of keeping points based on density
    probabilities = np.array([1 / grid_counts[(x_indices[i], y_indices[i])] for i in range(len(route))])
    probabilities /= probabilities.sum()  # Normalize

    # Randomly sample points based on computed probabilities
    selected_indices = np.random.choice(len(route), size=num_points, replace=False, p=probabilities)

    return route[np.sort(selected_indices)]

# Function to compute similarity measures for each route
def compute_similarity(args):
    route_type, start_idx, end_idx, run_id, autopilot_route, df_log = args

    # Extract model's trajectory
    model_route = df_log[
        (df_log['Run_ID'] == run_id) & 
        (df_log['Route_Type'] == route_type) & 
        (df_log['Start_Index'] == start_idx) & 
        (df_log['End_Index'] == end_idx)
    ][['Vehicle_X', 'Vehicle_Y']].values

    if len(model_route) < 2:
        return None  # Skip empty routes

    # Compute the number of points based on the shortest trajectory
    num_points = min(len(model_route), len(autopilot_route))

    # Downsample both routes
    model_route = grid_based_downsampling(model_route, num_points)
    autopilot_route = grid_based_downsampling(autopilot_route, num_points)

    # Compute similarity measures
    frechet_dist = similaritymeasures.frechet_dist(autopilot_route, model_route)
    dtw_dist, _ = fastdtw(autopilot_route, model_route, dist=euclidean)
    area_diff = similaritymeasures.area_between_two_curves(autopilot_route, model_route)

    try:
        curve_length_diff = similaritymeasures.curve_length_measure(autopilot_route, model_route)
        if np.isinf(curve_length_diff) or np.isnan(curve_length_diff):
            curve_length_diff = np.nan
    except:
        curve_length_diff = np.nan

    return {
        'Route_Type': route_type,
        'Run_ID': run_id,
        'Frechet_Distance': frechet_dist,
        'DTW_Distance': dtw_dist,
        'Area_Between_Curves': area_diff,
        'Curve_Length_Difference': curve_length_diff,
    }

# Process both corrected and uncorrected logs
for correction_type, file_path in correction_files.items():
    if not os.path.exists(file_path):
        print(f"Skipping: {file_path} (File not found)")
        continue
    
    print(f"Processing: {file_path}")
    df_log = pd.read_csv(file_path)

    # Store results per run
    results_per_run = []

    # Iterate through each route
    for _, route_info in unique_routes.iterrows():
        route_type = route_info['Route_Type']
        start_idx = route_info['Start_Index']
        end_idx = route_info['End_Index']

        # Extract reference trajectory
        autopilot_route = df_reference[
            (df_reference['Route_Type'] == route_type) & 
            (df_reference['Start_Index'] == start_idx) & 
            (df_reference['End_Index'] == end_idx)
        ][['Vehicle_X', 'Vehicle_Y']].values

        if len(autopilot_route) < 2:
            continue  # Skip missing autopilot routes

        # Parallel processing
        args = [(route_type, start_idx, end_idx, run_id, autopilot_route, df_log) for run_id in df_log['Run_ID'].unique()]
        
        with Pool(processes=4) as pool:  # Use 4 CPU cores
            results_per_run.extend(pool.map(compute_similarity, args))

    # Convert results to DataFrame
    results_per_run_df = pd.DataFrame([r for r in results_per_run if r is not None])

    # Save per-run results
    per_run_filename = f"route_similarity_per_run_{correction_type}.csv"
    results_per_run_df.to_csv(per_run_filename, index=False)

    # Compute per-route averages
    results_averaged_per_route = results_per_run_df.groupby("Route_Type").agg(
        Frechet_Mean=("Frechet_Distance", "mean"),
        Frechet_Std=("Frechet_Distance", "std"),
        DTW_Mean=("DTW_Distance", "mean"),
        DTW_Std=("DTW_Distance", "std"),
        Area_Mean=("Area_Between_Curves", "mean"),
        Area_Std=("Area_Between_Curves", "std"),
        Curve_Length_Mean=("Curve_Length_Difference", "mean"),
        Curve_Length_Std=("Curve_Length_Difference", "std")
    ).reset_index()

    # Save per-route averages
    avg_route_filename = f"route_similarity_averaged_per_route_{correction_type}.csv"
    results_averaged_per_route.to_csv(avg_route_filename, index=False)

    # Compute overall averages (all routes & runs combined)
    results_overall = pd.DataFrame({
        "Metric": ["Frechet", "DTW", "Area Between Curves", "Curve Length"],
        "Mean": [
            results_per_run_df["Frechet_Distance"].mean(),
            results_per_run_df["DTW_Distance"].mean(),
            results_per_run_df["Area_Between_Curves"].mean(),
            results_per_run_df["Curve_Length_Difference"].mean(),
        ],
        "Std": [
            results_per_run_df["Frechet_Distance"].std(),
            results_per_run_df["DTW_Distance"].std(),
            results_per_run_df["Area_Between_Curves"].std(),
            results_per_run_df["Curve_Length_Difference"].std(),
        ]
    })

    # Save overall averages
    overall_filename = f"route_similarity_overall_{correction_type}.csv"
    results_overall.to_csv(overall_filename, index=False)

    print(f"Saved results for {correction_type}")

print("Similarity analysis complete.")
