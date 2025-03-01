import carla
import numpy as np
import time
import csv
import cv2
import tensorflow as tf
import threading



import argparse

# Argument parser
parser = argparse.ArgumentParser(description="Run self-driving car simulation with or without correction.")
parser.add_argument("--correction", action="store_true", help="Enable correction for steering decisions")
args = parser.parse_args()

termination_reason = "Completed"

def normal_sample(mean, variance, num_samples=10):
    """Performs weighted sampling around the regression output using classification probabilities."""
    samples = np.random.normal(loc=mean, scale=np.sqrt(variance), size=num_samples)
    average = np.mean(samples)

    return average



def process(continuous_output, discrete_output, bin_edges, confidence_threshold=0.9):
    """
    Processes a batch of predictions using confidence-based correction with optional blending.
    """
    continuous_output = np.squeeze(continuous_output, axis=0)
    max_confidence = np.max(discrete_output, axis=1)
    bin_index_max = np.argmax(discrete_output, axis=1)
    bin_index_regression = np.digitize(continuous_output, bin_edges) - 1

    
    entropy = -np.sum(discrete_output * np.log(discrete_output + 1e-8), axis=1)
    final_output = np.full_like(continuous_output, np.nan)


    if (max_confidence >= confidence_threshold) and (bin_index_regression == bin_index_max or bin_index_regression == (bin_index_max+1) or bin_index_regression == (bin_index_max-1)):
        final_output = continuous_output

    elif (max_confidence >= confidence_threshold) and (bin_index_regression != bin_index_max and bin_index_regression != (bin_index_max+1) and bin_index_regression != (bin_index_max-1) ):
        bin_lower_bound_confident = bin_edges[bin_index_max]
        bin_upper_bound_confident = bin_edges[bin_index_max + 1]
        num_samples = 10
        random_samples = np.random.uniform(bin_lower_bound_confident, bin_upper_bound_confident, size=(num_samples, len(bin_index_max)))
        sampled_class_value = np.mean(random_samples, axis=0)
        final_output = sampled_class_value

    elif (max_confidence < 0.5) and (entropy > 1.5):
        final_output = continuous_output

    else:
        estimated_variance = np.var(discrete_output, axis=1)
        # Perform sampling using the correct bin indices
        final_output = normal_sample(mean=continuous_output, variance=estimated_variance)

    return final_output

# Load the trained model
saved_model_path = 'final_trained_model.h5' 
loaded_model = tf.keras.models.load_model(saved_model_path)

# Connect to CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# Load world and map for Town04
world = client.load_world('Town04')
map = world.get_map()
# Set the weather to ClearNoon ☀️
weather = carla.WeatherParameters.ClearNoon
world.set_weather(weather)
# Get the spectator object
spectator = world.get_spectator()

# Define spawn points and route types
spawn_points_indices = {
    "Two-Turn": [(224, 190)],
    "One-Turn": [(209, 18)],
    "Straight": [(81, 270)]
}


# Define custom bin edges based on provided logic
num_classes = 11
bin_width = 2 / num_classes  
half_width = bin_width / 2  

# Generate bin edges, ensuring the middle class contains 0
custom_intervals = np.linspace(-1, 1, num_classes + 1)
middle_index = num_classes // 2  
custom_intervals[middle_index] = -half_width  
custom_intervals[middle_index + 1] = half_width  


# Define CSV filename based on correction argument
filename = "waypoint_location_with_correction.csv" if args.correction else "waypoint_location_without_correction.csv"

# Create CSV file for location and waypoint logging

with open(filename, "w", newline="") as location_file:
    location_writer = csv.writer(location_file)
    location_writer.writerow(["Run_ID", "Route_Type", "Start_Index", "End_Index", "Timestep", 
                              "Vehicle_X", "Vehicle_Y", "Vehicle_Z", 
                              "Waypoint_X", "Waypoint_Y", "Waypoint_Z"])
    
def preprocess_image(image):
    """Resize and crop the image to match training format (448x448)."""
    image = image[:, :, :3]  # Ensure RGB
    image = cv2.resize(image, (448, 448))  # Match training resolution

    # Crop if needed (Adjust these values based on how training data was processed)
    left, top, right, bottom = 0, 218, 448, 448  # Match training cropping
    image = image[top:bottom, left:right]

    image = cv2.resize(image, (160, 160))  # Resize to model input size
    image = np.array(image).astype("float32") / 255.0
    return np.expand_dims(image, axis=0)  # Add batch dimension

def test_scenario(run_id, route_type, start_idx, end_idx):
    """Runs a single test scenario."""
    global results

    # Spawn vehicle
    start_point = map.get_spawn_points()[start_idx]
    end_point = map.get_spawn_points()[end_idx]

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('model3')[0]
    vehicle = world.spawn_actor(vehicle_bp, start_point)


    # Setup camera with 448x448 resolution (same as data collection)
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '448')
    camera_bp.set_attribute('image_size_y', '448')
    camera_bp.set_attribute('fov', '110')
    camera = world.spawn_actor(camera_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=vehicle)

    # Setup sensors
    collision_bp = blueprint_library.find('sensor.other.collision')
    lane_bp = blueprint_library.find('sensor.other.lane_invasion')

    collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
    lane_sensor = world.spawn_actor(lane_bp, carla.Transform(), attach_to=vehicle)

    # Tracking variables
    lane_invasions = 0
    collisions = 0
    deviation_list = []
    last_movement_time = time.time()
    max_idle_time = 120
    max_distance_threshold = 20
    rgb_image = None
    timestep = 0

    # Sensor callbacks
    def on_collision(event):
        nonlocal collisions
        collisions += 1
        print(f"[WARNING] Collision detected! Total collisions: {collisions}")

    def on_lane_invasion(event):
        nonlocal lane_invasions
        lane_invasions += 1

    def on_camera_image(image):
        """Convert CARLA image to numpy array and display it."""
        nonlocal rgb_image
        image_data = np.frombuffer(image.raw_data, dtype=np.uint8)
        image_data = image_data.reshape((image.height, image.width, 4))  # BGRA format
        rgb_image = cv2.cvtColor(image_data, cv2.COLOR_BGRA2RGB)  # Convert to RGB

        # ✅ Debug: Show a frame to check if it's black
        if np.mean(rgb_image) < 5:  # If all pixels are too dark
            print("[WARNING] The captured image appears to be black! Check camera settings.")



    # Attach sensors
    collision_sensor.listen(on_collision)
    lane_sensor.listen(on_lane_invasion)
    camera.listen(on_camera_image)

    # Start driving loop
    reached_goal = False
    start_time = time.time()

    # Before entering the loop, store the initial distance to the goal
    initial_distance = vehicle.get_location().distance(end_point.location)


    while not reached_goal:
        vehicle_location = vehicle.get_location()
        waypoint = map.get_waypoint(vehicle_location)
        deviation = vehicle_location.distance(waypoint.transform.location)
        deviation_list.append(deviation)


        # Check if reached goal
        distance_to_goal = vehicle_location.distance(end_point.location)

        velocity = vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

        if rgb_image is not None:

            processed_image = preprocess_image(rgb_image)
            output = loaded_model.predict(processed_image)


            # Correct the steering decision
            corrected_steering = process(output[0], output[1], custom_intervals)
            corrected_steering_apply = float(corrected_steering)/4

            # Apply correction only if the argument is enabled
            if args.correction:
                corrected_steering = process(output[0], output[1], custom_intervals)
                corrected_steering_apply = float(corrected_steering)/4
            else:
                corrected_steering = output[0]  # Use raw model output if correction is disabled
                corrected_steering_apply = float(corrected_steering)/4



            # Apply control
            if speed<15:
                throttle =0.5
            else:
                throttle = 0

            vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=corrected_steering_apply))

            # Log vehicle location and nearest waypoint
            
            with open(filename, "a", newline="") as location_file:
                location_writer = csv.writer(location_file)
                location_writer.writerow([run_id, route_type, start_idx, end_idx, timestep, 
                                  vehicle_location.x, vehicle_location.y, vehicle_location.z,
                                  waypoint.transform.location.x, waypoint.transform.location.y, waypoint.transform.location.z])


            timestep += 1

        # Termination conditions

        if collisions > 10:
            print(f"Run {run_id}: Too many collisions ({collisions}). Ending episode.")
            break  # End episode immediately


        if speed > 1:
            last_movement_time = time.time()


        if distance_to_goal < 4.0:
            print(f"Run {run_id}: Goal reached successfully.")
            reached_goal = True
            break  # Exit the loop immediately

        if time.time() - last_movement_time > max_idle_time:
            print(f"Run {run_id}: Vehicle paused too long. Terminating.")
            break

        if distance_to_goal > initial_distance + max_distance_threshold:
            print(f"Run {run_id}: Vehicle overshot target. Terminating.")
            break

        if time.time() - start_time > 600:
            print(f"Run {run_id}: Timed out.")
            break


    # Cleanup
    vehicle.destroy()
    collision_sensor.destroy()
    lane_sensor.destroy()
    camera.destroy()

print("Testing complete. Results saved.")

# Run Multiple Tests
num_trials = 10
for i in range(num_trials):
    for route_type, spawn_list in spawn_points_indices.items():
        for start_idx, end_idx in spawn_list:
            test_scenario(i, route_type, start_idx, end_idx)
