import carla
import time
import os
import csv
import random
import argparse
from datetime import datetime

# Define available CARLA towns
AVAILABLE_TOWNS = ["Town01", "Town02", "Town03", "Town04", "Town05", "Town06", "Town07", "Town10HD"]

# Main data collection script
def collect_data(town, output_dir):
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    # Load the specified town
    if town not in AVAILABLE_TOWNS:
        print(f"Warning: {town} is not a recognized CARLA town. Defaulting to Town04.")
        town = "Town04"
    
    world = client.load_world(town)
    
    # Set weather to ClearNoon
    world.set_weather(carla.WeatherParameters.ClearNoon)
    
    settings = world.get_settings()
    settings.synchronous_mode = True  # Enable synchronous mode
    settings.fixed_delta_seconds = 0.1
    world.apply_settings(settings)
    
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)
    
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.*model3*')[0]
    spawn_points = world.get_map().get_spawn_points()
    
    # Create output directories
    camera_folder = os.path.join(output_dir, "front_rgb")
    os.makedirs(camera_folder, exist_ok=True)
    log_file_path = os.path.join(output_dir, "control_commands.csv")
    
    # Open CSV log file
    log_file = open(log_file_path, 'w', newline='')
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(['Image_Fname', 'Steering', 'Throttle', 'Brake', 'Speed', 'Waypoint_X', 'Waypoint_Y', 'Waypoint_Z'])
    
    def log_control_commands(vehicle, image_filename):
        control = vehicle.get_control()
        velocity = vehicle.get_velocity()
        speed = (3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5)
        
        location = vehicle.get_location()
        waypoint = world.get_map().get_waypoint(location)
        waypoint_location = waypoint.transform.location
        
        csv_writer.writerow([image_filename, control.steer, control.throttle, control.brake, speed,
                             waypoint_location.x, waypoint_location.y, waypoint_location.z])
    
    def save_image_and_log(image, folder_path, vehicle):
        frame_number = image.frame
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        filename = f'image_{timestamp}_{frame_number:08d}.png'
        image_path = os.path.join(folder_path, filename)
        
        image.save_to_disk(image_path)
        log_control_commands(vehicle, filename)
    
    def add_rgb_camera(vehicle, transform, image_size=(448, 448), fov=110):
        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(image_size[0]))
        camera_bp.set_attribute('image_size_y', str(image_size[1]))
        camera_bp.set_attribute('fov', str(fov))
        
        rgb_camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
        rgb_camera.listen(lambda image: save_image_and_log(image, camera_folder, vehicle))
        
        return rgb_camera
    
    vehicle = world.spawn_actor(vehicle_bp, random.choice(spawn_points))
    front_transform = carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=0))
    front_rgb = add_rgb_camera(vehicle, front_transform)
    
    vehicle.set_autopilot(True, traffic_manager.get_port())
    traffic_manager.ignore_lights_percentage(vehicle, 100)
    
    try:
        image_count = 0
        total_images = 300000
        spawn_point_index = 0

        while image_count < total_images:
            world.tick()
            time.sleep(0.1)
            image_count += 1
            
            if image_count % 1000 == 0:
                print(f"Switching spawn point after {image_count} images.")
                front_rgb.stop()
                vehicle.destroy()
                vehicle = world.spawn_actor(vehicle_bp, spawn_points[spawn_point_index])
                spawn_point_index = (spawn_point_index + 1) % len(spawn_points)
                front_rgb = add_rgb_camera(vehicle, front_transform)
                vehicle.set_autopilot(True, traffic_manager.get_port())
                traffic_manager.ignore_lights_percentage(vehicle, 100)
    
    finally:
        front_rgb.stop()
        vehicle.destroy()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(False)
        log_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect CARLA driving data in a specified town and save to a specific directory.")
    parser.add_argument("--town", type=str, choices=AVAILABLE_TOWNS, default="Town04", help=f"Specify the CARLA town from: {AVAILABLE_TOWNS}")
    parser.add_argument("--output_dir", type=str, default="output_data", help="Specify the directory to save collected data.")
    args = parser.parse_args()
    
    collect_data(town=args.town, output_dir=args.output_dir)
