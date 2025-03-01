import pandas as pd

# Define input files
file_reference = "reference_track.csv"  # Autopilot (ground truth reference)
file_corrected = "waypoint_location_with_correction.csv"  # Model with correction
file_uncorrected = "waypoint_location_without_correction.csv"  # Model without correction

# Define output files
output_corrected = "cropped_waypoint_with_correction.csv"
output_uncorrected = "cropped_waypoint_without_correction.csv"

# Load reference and model data
df_reference = pd.read_csv(file_reference)
df_corrected = pd.read_csv(file_corrected)
df_uncorrected = pd.read_csv(file_uncorrected)

# Process function for cropping overshot tracks
def crop_overshot_tracks(df_model, df_reference):
    cropped_data_list = []
    route_types = df_reference["Route_Type"].unique()

    for route in route_types:
        # Filter data per route
        reference_route = df_reference[df_reference["Route_Type"] == route]
        model_route = df_model[df_model["Route_Type"] == route]

        print(f'len(reference_route["Vehicle_X"])_{route}:', len(reference_route["Vehicle_X"]))

        # Apply different cropping rules based on the route type
        if "Two-Turn" in route:
            y_threshold = reference_route["Vehicle_Y"].min()  # Crop below min Y
            model_route_cropped = model_route[model_route["Vehicle_Y"] >= y_threshold]
        elif "One-Turn" in route:
            y_threshold = reference_route["Vehicle_Y"].max()  # Crop above max Y
            model_route_cropped = model_route[model_route["Vehicle_Y"] <= y_threshold]
        else:
            model_route_cropped = model_route  # No cropping for straight routes

        cropped_data_list.append(model_route_cropped)
        print(f'len(cropped_data["Vehicle_X"])_{route}:', len(model_route_cropped["Vehicle_X"]))

    return pd.concat(cropped_data_list, ignore_index=True)

# Apply cropping for both corrected and uncorrected datasets
cropped_corrected = crop_overshot_tracks(df_corrected, df_reference)
cropped_uncorrected = crop_overshot_tracks(df_uncorrected, df_reference)

# Save cleaned datasets
cropped_corrected.to_csv(output_corrected, index=False)
cropped_uncorrected.to_csv(output_uncorrected, index=False)

print(f"Saved cropped corrected data to {output_corrected}")
print(f"Saved cropped uncorrected data to {output_uncorrected}")
