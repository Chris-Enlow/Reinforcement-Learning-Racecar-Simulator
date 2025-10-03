import pygame
import math
import random
import torch
import numpy as np

# Physics constants
MAX_SPEED = 8.0

# Respawn car at a specific checkpoint with proper orientation
def respawn_car_at_checkpoint(car, track, checkpoint_idx):
    if checkpoint_idx >= len(track.checkpoints):
        return False
    
    start_pos = track.checkpoints[checkpoint_idx][:2]
    noise_x = random.uniform(-20, 20)
    noise_y = random.uniform(-20, 20)
    car.pos = pygame.math.Vector2(start_pos[0] + noise_x, start_pos[1] + noise_y)
    
    car.next_checkpoint_index = (checkpoint_idx + 1) % len(track.checkpoints)
    car.checkpoints_hit = set(range(checkpoint_idx + 1))
    
    # Calculate track direction
    closest_track_idx = 0
    min_distance = float('inf')
    checkpoint_pos = pygame.math.Vector2(start_pos)
    
    for i, track_point in enumerate(track.track_points):
        distance = checkpoint_pos.distance_to(track_point)
        if distance < min_distance:
            min_distance = distance
            closest_track_idx = i
    
    look_ahead = 5
    current_point = pygame.math.Vector2(track.track_points[closest_track_idx])
    next_point = pygame.math.Vector2(track.track_points[(closest_track_idx + look_ahead) % len(track.track_points)])
    
    track_direction = (next_point - current_point).normalize()
    car.angle = math.degrees(math.atan2(track_direction.y, track_direction.x))
    car.speed = 0
    car.vel = pygame.math.Vector2(0, 0)
    
    return True

# Generate the state vector for the AI agent
def get_state_vector(car, track, track_boundaries):
    try:
        ray_distances = car.cast_rays(track_boundaries)
        normalized_rays = [min(d / 1000.0, 1.0) for d in ray_distances]
        normalized_speed = [car.speed / MAX_SPEED]
        
        # Angle to next checkpoint
        if (hasattr(track, 'checkpoints') and 
            hasattr(car, 'next_checkpoint_index') and 
            car.next_checkpoint_index < len(track.checkpoints)):
            
            next_cp = pygame.math.Vector2(track.checkpoints[car.next_checkpoint_index][:2])
            car_dir = pygame.math.Vector2(math.cos(math.radians(car.angle)),
                                        math.sin(math.radians(car.angle)))
            to_checkpoint = next_cp - car.pos

            if to_checkpoint.length_squared() > 1e-6:
                to_checkpoint = to_checkpoint.normalize()
                angle_diff = math.atan2(car_dir.cross(to_checkpoint), car_dir.dot(to_checkpoint))
                angle_diff_norm = angle_diff / math.pi
            else:
                angle_diff_norm = 0.0
        else:
            angle_diff_norm = 0.0

        # Center distance
        center_distance_norm = 0.0
        try:
            if (hasattr(track, 'checkpoints') and 
                hasattr(car, 'next_checkpoint_index') and 
                car.next_checkpoint_index < len(track.checkpoints) and
                len(track.checkpoints[car.next_checkpoint_index]) >= 4):
                
                cp_left, cp_right = track.checkpoints[car.next_checkpoint_index][2:4]
                track_midpoint = (pygame.math.Vector2(cp_left) + pygame.math.Vector2(cp_right)) / 2
                center_distance = car.pos.distance_to(track_midpoint)
                half_width = max(track_midpoint.distance_to(cp_left), 1.0)
                center_distance_norm = min(center_distance / half_width, 1.0)
        except (IndexError, AttributeError, TypeError):
            center_distance_norm = 0.0

        state_values = normalized_rays + normalized_speed + [angle_diff_norm, center_distance_norm]
        return torch.tensor(state_values, dtype=torch.float32).unsqueeze(0)
        
    except Exception as e:
        default_state = [0.5] * 9 + [0.0] * 3
        return torch.tensor(default_state, dtype=torch.float32).unsqueeze(0)

# Calculate reward for the current state and action
def calculate_reward(car, track, prev_distance, stuck_timer, last_checkpoint_count, steps, off_track_timer=0, action=None):
    reward = 0.0

    # Progress reward
    distance_improvement = prev_distance - car.distance_to_next_checkpoint
    reward += distance_improvement * 0.1

    # Track adherence and speed rewards
    if track.is_on_track(car.pos):
        reward += 0.1
        speed_factor = min(car.speed / MAX_SPEED, 1.0)
        
        base_speed_reward = speed_factor * 0.2 
        speed_bonus = speed_factor * speed_factor * 0.5
        reward += base_speed_reward + speed_bonus
        
        if car.speed > MAX_SPEED * 0.8:
            reward += 0.1
        if car.speed > MAX_SPEED * 0.9:
            reward += 0.1
            
        if car.speed < MAX_SPEED * 0.3:
            reward -= 0.1
    else:
        reward -= 3
        reward -= min(off_track_timer / 60.0, 5.0) * 2

    # Action-based rewards
    if action is not None:
        if action == 4:  # Coast
            reward -= 1  
        elif action in [0, 1, 2]:  # Accelerating
            reward += 0.5 
            if car.speed > MAX_SPEED * 0.5:
                reward += 0.02
        elif action == 3 and car.speed > MAX_SPEED * 0.7:  # Excessive braking
            reward -= 0.03

    # Checkpoint rewards with time bonus
    if len(car.checkpoints_hit) > last_checkpoint_count:
        checkpoint_bonus = 25
        
        if car.checkpoints_hit:
            time_per_checkpoint = steps / len(car.checkpoints_hit)
            if time_per_checkpoint < 120:
                if time_per_checkpoint > 0:
                    speed_multiplier = max(1.5, 240 / time_per_checkpoint)
                    checkpoint_bonus = int(checkpoint_bonus * speed_multiplier)
            
        reward += checkpoint_bonus

    # Stuck penalties
    if stuck_timer > 120:
        reward -= 1
    if stuck_timer > 240:
        reward -= 5.0

    reward -= 0.001 * steps

    # Center alignment bonus
    try:
        if (hasattr(track, 'checkpoints') and 
            car.next_checkpoint_index < len(track.checkpoints) and 
            len(track.checkpoints[car.next_checkpoint_index]) >= 4):
            cp_left, cp_right = track.checkpoints[car.next_checkpoint_index][2:4]
            track_midpoint = (pygame.math.Vector2(cp_left) + pygame.math.Vector2(cp_right)) / 2
            center_distance = car.pos.distance_to(track_midpoint)
            half_width = max(track_midpoint.distance_to(cp_left), 1.0)
            alignment = 1.0 - min(center_distance / half_width, 1.0)
            reward += alignment * 0.1
    except:
        pass

    return reward

# Apply the selected action to the car controls
def apply_action(car, action):
    controls = [False, False, False, False]  # [accel, brake, left, right]
    if action == 0:    # Accelerate
        controls[0] = True
    elif action == 1:  # Accelerate + Left
        controls[0] = True
        controls[2] = True
    elif action == 2:  # Accelerate + Right
        controls[0] = True
        controls[3] = True
    elif action == 3:  # Brake
        controls[1] = True
    # action == 4: Coast (no action)
    
    car.set_controls(*controls)

# Calculate smooth camera following
def calculate_camera_offset(car, camera_offset, screen_width=1200, screen_height=800):
    target_x = car.pos.x - screen_width / 2
    target_y = car.pos.y - screen_height / 2
    camera_offset.x += (target_x - camera_offset.x) * 0.1
    camera_offset.y += (target_y - camera_offset.y) * 0.1
    return camera_offset

# Prompt user for epsilon value or use default
def prompt_epsilon(training_state, user_input):
    while True:
        try:
            if user_input == "":
                saved_epsilon = training_state.get_current_epsilon()
                epsilon = saved_epsilon if saved_epsilon is not None else 0.25
                print(f"Using default epsilon: {epsilon:.3f}")
                return epsilon
            
            new_epsilon = float(user_input)
            if 0.0 <= new_epsilon <= 1.0:
                print(f"Epsilon manually set to: {new_epsilon:.3f}")
                return new_epsilon
            else:
                print("Invalid range. Epsilon must be between 0.0 and 1.0.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def save_checkpoint(agent, training_state, episode):
    model_path = 'models/car_dqn_model.pth'
    torch.save(agent.policy_net.state_dict(), model_path)
    training_state.save_state(agent)


def save_and_exit(training_state, agent):
    model_path = 'models/car_dqn_model.pth'
    training_state.save_state(agent)
    torch.save(agent.policy_net.state_dict(), model_path)
    print(f"\n=== Training saved ===")
    training_state.print_session_summary()