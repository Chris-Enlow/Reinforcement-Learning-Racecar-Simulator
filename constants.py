import pygame
import math
import random
import numpy as np
import torch
import os
import json
import pickle
from datetime import datetime
from shapely.geometry import Point, LineString

# Import your existing modules
from ai_agent import DQNAgent
from game_classes import Track, Car, TrainingState

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)
GRASS_GREEN = (34, 139, 34)
ASPHALT = (50, 50, 50)

# Physics constants
MAX_SPEED = 8.0
ACCELERATION = 0.3
BRAKE_DECELERATION = 0.6
FRICTION = 0.95
TURN_SPEED = 4.0
OFF_ROAD_FRICTION = 0.85
OFF_ROAD_DRAG = 0.9


def respawn_car_at_checkpoint(car, track, checkpoint_idx):
    if checkpoint_idx >= len(track.checkpoints):
        return False
    
    # Get the respawn position with some noise
    start_pos = track.checkpoints[checkpoint_idx][:2]
    noise_x = random.uniform(-20, 20)
    noise_y = random.uniform(-20, 20)
    car.pos = pygame.math.Vector2(start_pos[0] + noise_x, start_pos[1] + noise_y)
    
    # Update checkpoint tracking
    car.next_checkpoint_index = (checkpoint_idx + 1) % len(track.checkpoints)
    car.checkpoints_hit = set(range(checkpoint_idx + 1))
    
    # Calculate track direction by looking at nearby track points
    # Find the closest track point to this checkpoint
    closest_track_idx = 0
    min_distance = float('inf')
    checkpoint_pos = pygame.math.Vector2(start_pos)
    
    for i, track_point in enumerate(track.track_points):
        distance = checkpoint_pos.distance_to(track_point)
        if distance < min_distance:
            min_distance = distance
            closest_track_idx = i
    
    # Get the track direction by looking ahead a few points
    look_ahead = 5  # Look ahead 5 track points
    current_point = pygame.math.Vector2(track.track_points[closest_track_idx])
    next_point = pygame.math.Vector2(track.track_points[(closest_track_idx + look_ahead) % len(track.track_points)])
    
    # Calculate the track direction vector
    track_direction = (next_point - current_point).normalize()
    
    # Set car angle to face along the track direction
    car.angle = math.degrees(math.atan2(track_direction.y, track_direction.x))
    
    # Reset other car properties
    car.speed = 0
    car.vel = pygame.math.Vector2(0, 0)
    
    return True


def get_state_vector(car, track, track_boundaries):
    try:
        # Get ray distances
        ray_distances = car.cast_rays(track_boundaries)
        normalized_rays = [min(d / 1000.0, 1.0) for d in ray_distances]
        
        # Get speed
        normalized_speed = [car.speed / MAX_SPEED]
        
        # Get angle to next checkpoint - SAFE ACCESS
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

        # Get centerline distance - SAFE ACCESS
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

        # Combine into state vector
        state_values = normalized_rays + normalized_speed + [angle_diff_norm, center_distance_norm]
        return torch.tensor(state_values, dtype=torch.float32).unsqueeze(0)
        
    except Exception as e:
        # Return a safe default state vector
        default_state = [0.5] * 9 + [0.0] * 3  # 9 rays + speed + angle + center_distance
        return torch.tensor(default_state, dtype=torch.float32).unsqueeze(0)


def calculate_reward(car, track, prev_distance, stuck_timer, last_checkpoint_count, steps, off_track_timer=0, action=None):
    reward = 0.0

    # Progress reward (distance to checkpoint decreasing)
    distance_improvement = prev_distance - car.distance_to_next_checkpoint
    reward += distance_improvement * 0.1

    # Track adherence and movement encouragement
    if track.is_on_track(car.pos):
        reward += 0.1
        speed_factor = min(car.speed / MAX_SPEED, 1.0)
        
        # MASSIVELY increased speed rewards
        base_speed_reward = speed_factor * 0.2 
        speed_bonus = speed_factor * speed_factor * 0.5  # quadratic bonus
        reward += base_speed_reward + speed_bonus
        
        # Extra bonus for high speed
        if car.speed > MAX_SPEED * 0.8:
            reward += 0.1
        if car.speed > MAX_SPEED * 0.9:
            reward += 0.1
            
        # Penalty for being too slow on track
        if car.speed < MAX_SPEED * 0.3:  # less than 30% max speed
            reward -= 0.1
    else:
        reward -= 3
        reward -= min(off_track_timer / 60.0, 5.0) * 2  # harsher if off longer

    # Action-based rewards/penalties with speed emphasis
    if action is not None:
        if action == 4:  # Coast action
            reward -= 1  
        elif action == [0, 1, 2]:  # Accelerating action
            reward += 0.5 
            if car.speed > MAX_SPEED * 0.5:
                reward += 0.02
        # elif action in [1, 2]:  # Accelerating + Turning actions
        #     reward += 0.2 
        #     if car.speed > MAX_SPEED * 0.5:
        #         reward += 0.02

        # Small penalty for excessive braking at high speeds
        elif action == 3 and car.speed > MAX_SPEED * 0.7:
            reward -= 0.03

    # Checkpoints with TIME-BASED BONUSES
    if len(car.checkpoints_hit) > last_checkpoint_count:
        # Base checkpoint bonus
        checkpoint_bonus = 25  # increased from 20
        
        # TIME BONUS - reward faster checkpoint reaching
        if(car.checkpoints_hit != 0):
            time_per_checkpoint = steps / len(car.checkpoints_hit)
            if time_per_checkpoint < 120:  # less than 2 seconds per checkpoint = fast!
                if time_per_checkpoint > 0:
                    speed_multiplier = max(1.5, 240 / time_per_checkpoint)
                    checkpoint_bonus = int(checkpoint_bonus * speed_multiplier)
            
        reward += checkpoint_bonus

    # Stuck penalties
    if stuck_timer > 120:  # 2 seconds
        reward -= 1  # increased penalty
    if stuck_timer > 240:  # 4 seconds
        reward -= 5.0  # much higher penalty

    reward -= 0.001 * steps  # reduced from 0.002

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


def main():
    print("=== AI Racecar Training Started ===")

    training_state = TrainingState()
    state_loaded = training_state.load_state()
    
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("AI Racecar Training")
    font = pygame.font.SysFont(None, 24)
    large_font = pygame.font.SysFont(None, 36)

    # Create game objects
    track = Track()
    start_x, start_y = track.track_points[0]
    car = Car(start_x, start_y)
    track_boundaries = track.get_track_boundaries()

    # Initialize AI - Expanded action space and observations
    n_actions = 5  # Accel, Accel+Left, Accel+Right, Brake, Coast
    n_observations = 12  # 9 rays + speed + angle_to_checkpoint + center_distance
    agent = DQNAgent(n_observations, n_actions)
    
    # Load existing model and set epsilon
    model_path = 'models/car_dqn_model.pth'
    # Modified logic for manual control
    if os.path.exists(model_path):
        try:
            agent.policy_net.load_state_dict(torch.load(model_path))
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            print(f"✓ Model loaded from {model_path}")
            
            # --- Start of new manual epsilon logic ---
            while True:
                try:
                    user_input = input("➡️ Enter new epsilon (0.0 to 1.0), or press Enter for default: ")
                    if user_input == "":
                        # User pressed Enter, so use a default value
                        saved_epsilon = training_state.get_current_epsilon()
                        agent.epsilon = saved_epsilon if saved_epsilon is not None else 0.25
                        print(f"✓ Using default epsilon: {agent.epsilon:.3f}")
                        break
                    
                    new_epsilon = float(user_input)
                    if 0.0 <= new_epsilon <= 1.0:
                        agent.epsilon = new_epsilon
                        print(f"✓ Epsilon manually set to: {agent.epsilon:.3f}")
                        break
                    else:
                        print("✗ Invalid range. Epsilon must be between 0.0 and 1.0.")
                except ValueError:
                    print("✗ Invalid input. Please enter a number.")
        
        except Exception as e:
            print(f"✗ Could not load model: {e}")
            print("Starting fresh training...")


    camera_offset = pygame.math.Vector2(0, 0)
    
    # Use training state values
    num_episodes = training_state.total_episodes
    max_checkpoint_reached = training_state.max_checkpoint_reached
    best_lap_time = training_state.best_lap_time
    episode_rewards = training_state.episode_rewards
    successful_laps = training_state.successful_laps
    start_episode = training_state.current_episode

    paused = False
    
    for i_episode in range(start_episode, num_episodes):
        # Update training state
        training_state.current_episode = i_episode
    
        car.reset(start_x, start_y)
        
        # Curriculum learning: Advanced training starts at later checkpoints
        if max_checkpoint_reached > 3 and random.random() < 0.25:
            start_checkpoint_idx = random.randint(0, min(max_checkpoint_reached, len(track.checkpoints) - 1))
            respawn_car_at_checkpoint(car, track, start_checkpoint_idx)

        # Episode variables
        last_checkpoint_count = len(car.checkpoints_hit)
        stuck_timer = 0
        off_track_timer = 0
        episode_reward = 0
        steps = 0
        last_pos = car.pos.copy()
        movement_check_timer = 0
        episode_max_checkpoint = len(car.checkpoints_hit)
        
        # Calculate initial distance before the loop
        try:
            car.update_checkpoints(track)
            last_distance_to_checkpoint = getattr(car, 'distance_to_next_checkpoint', float('inf'))
        except Exception as e:
            print(f"Warning: Error initializing checkpoint distance: {e}")
            last_distance_to_checkpoint = float('inf')
        
        while True:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    # Save everything before exiting
                    training_state.max_checkpoint_reached = max_checkpoint_reached
                    training_state.best_lap_time = best_lap_time
                    training_state.episode_rewards = episode_rewards
                    training_state.successful_laps = successful_laps
                    training_state.epsilon_history.append(agent.epsilon)
                    training_state.checkpoint_progress.append(max_checkpoint_reached)
                    training_state.save_state(agent)
                    
                    torch.save(agent.policy_net.state_dict(), model_path)
                    print(f"\n=== Training saved and stopped at episode {i_episode} ===")
                    training_state.print_session_summary()
                    pygame.quit()
                    return
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    paused = not paused
                    print("Training paused" if paused else "Training resumed")

            # Check if pygame display is still active
            if not pygame.get_init() or pygame.display.get_surface() is None:
                print("Display closed, saving and exiting...")
                # Save everything before exiting
                training_state.max_checkpoint_reached = max_checkpoint_reached
                training_state.best_lap_time = best_lap_time
                training_state.episode_rewards = episode_rewards
                training_state.successful_laps = successful_laps
                training_state.epsilon_history.append(agent.epsilon)
                training_state.checkpoint_progress.append(max_checkpoint_reached)
                training_state.save_state(agent)
                torch.save(agent.policy_net.state_dict(), model_path)
                return

            if paused:
                pygame.time.wait(100)
                continue

            # Get current state using the helper function
            state = get_state_vector(car, track, track_boundaries)

            # AI selects action
            action_tensor = agent.select_action(state)
            action = action_tensor.item()
            
            # Map expanded actions to car controls
            controls = [False, False, False, False]  # [accel, brake, left, right]
            if action == 0:    # Accelerate only
                controls[0] = True
            elif action == 1:  # Accelerate + Turn Left
                controls[0] = True
                controls[2] = True
            elif action == 2:  # Accelerate + Turn Right
                controls[0] = True
                controls[3] = True
            elif action == 3:  # Brake
                controls[1] = True
            # elif action == 4: Coast (no action)
            
            car.set_controls(*controls)
            
            # Store previous state for reward calculation
            prev_distance = last_distance_to_checkpoint
            
            # Update game state with error handling and debugging
            try:
                # Add checkpoint debugging
                pre_checkpoints = len(getattr(car, 'checkpoints_hit', set()))
                pre_next_index = getattr(car, 'next_checkpoint_index', 0)
                
                car.update(track)
                
                # Prevent infinite loops in checkpoint detection
                checkpoint_update_attempts = 0
                max_checkpoint_attempts = 3
                
                while checkpoint_update_attempts < max_checkpoint_attempts:
                    try:
                        old_checkpoints = len(getattr(car, 'checkpoints_hit', set()))
                        old_next_index = getattr(car, 'next_checkpoint_index', 0)
                        
                        car.update_checkpoints(track)
                        
                        new_checkpoints = len(getattr(car, 'checkpoints_hit', set()))
                        new_next_index = getattr(car, 'next_checkpoint_index', 0)
                        
                        # Break if no changes (stable state)
                        if old_checkpoints == new_checkpoints and old_next_index == new_next_index:
                            break
                            
                        checkpoint_update_attempts += 1
                        
                        # Force break if we hit all checkpoints to prevent infinite loop
                        if hasattr(track, 'checkpoints') and new_checkpoints >= len(track.checkpoints):
                            print(f"DEBUG: All checkpoints hit ({new_checkpoints}), breaking update loop")
                            break
                            
                    except Exception as inner_e:
                        print(f"Warning: Error in checkpoint update attempt {checkpoint_update_attempts}: {inner_e}")
                        break
                
                if checkpoint_update_attempts >= max_checkpoint_attempts:
                    print(f"WARNING: Max checkpoint update attempts reached, forcing episode end")
                    done = True
                    termination_reason = "Checkpoint update loop detected"
                
                last_distance_to_checkpoint = getattr(car, 'distance_to_next_checkpoint', float('inf'))
                
            except Exception as e:
                print(f"Warning: Error updating car state: {e}")
                # Reset distance to prevent crashes
                last_distance_to_checkpoint = float('inf')
                # Force episode end to prevent further issues
                done = True
                termination_reason = "Update error occurred"
            
            # Track max checkpoints reached this episode - SAFE ACCESS
            try:
                current_checkpoints = len(getattr(car, 'checkpoints_hit', set()))
                episode_max_checkpoint = max(episode_max_checkpoint, current_checkpoints)
            except Exception as e:
                print(f"Warning: Error tracking episode checkpoints: {e}")
                # Continue without updating episode max
            
            # Update off-track timer
            if not track.is_on_track(car.pos):
                off_track_timer += 1
            else:
                off_track_timer = max(0, off_track_timer - 2)  # Reduce when back on track
            
            # Check if car is stuck (not moving much) - more sensitive detection
            if car.pos.distance_to(last_pos) < 2.0:  # increased from 1.0 to catch more cases
                movement_check_timer += 1
            else:
                movement_check_timer = 0
                last_pos = car.pos.copy()
            
            if movement_check_timer > 30:  # reduced from 60 - detect stuck faster (0.5 seconds)
                stuck_timer += 1
            else:
                stuck_timer = max(0, stuck_timer - 1)  # Gradually reduce stuck timer
            
            # Calculate reward with action context - add timeout protection
            try:
                reward = calculate_reward(car, track, prev_distance, stuck_timer, last_checkpoint_count, steps, off_track_timer, action)
                episode_reward += reward
            except Exception as e:
                print(f"Warning: Error calculating reward: {e}")
                reward = -1.0  # Default penalty for errors
                episode_reward += reward
            
            # Update progress tracking - SAFE ACCESS
            try:
                current_checkpoint_count = len(getattr(car, 'checkpoints_hit', set()))
                if current_checkpoint_count > last_checkpoint_count:
                    max_checkpoint_reached = max(max_checkpoint_reached, current_checkpoint_count)
                    last_checkpoint_count = current_checkpoint_count
                    stuck_timer = 0
                    off_track_timer = 0  # Reset off-track timer on progress
            except Exception as e:
                print(f"Warning: Error updating progress tracking: {e}")
                # Continue without updating progress tracking
            
            # Check termination conditions - SAFER CHECKS with timeout
            done = False
            termination_reason = ""
            
            try:
                # Add emergency timeout for long episodes
                if steps > 3000:  # Emergency timeout at 50 seconds
                    done = True
                    termination_reason = "Emergency timeout"
                
                # Check lap completion first with safe attribute access
                elif hasattr(car, 'race_finished') and car.race_finished:
                    if hasattr(car, 'lap_time') and car.lap_time < best_lap_time:
                        best_lap_time = car.lap_time
                    successful_laps += 1
                    done = True
                    termination_reason = "Lap completed"

                elif off_track_timer > 180:  # 3 seconds off track = reset
                    done = True
                    termination_reason = "Off track too long"
                elif stuck_timer > 180:  # reduced from 300 to 180 (3 seconds stuck)
                    done = True
                    termination_reason = "Stuck too long"
                elif steps > 2400:  # 40 seconds max per episode
                    done = True
                    termination_reason = "Time limit reached"
                    
                # Emergency checkpoint-based termination
                elif hasattr(car, 'checkpoints_hit') and len(car.checkpoints_hit) >= 20:
                    # If we have 20+ checkpoints but no race_finished flag, something's wrong
                    if not getattr(car, 'race_finished', False):
                        print(f"WARNING: {len(car.checkpoints_hit)} checkpoints but no race_finished flag")
                        done = True
                        termination_reason = "Checkpoint logic error"
                        
            except Exception as e:
                print(f"Warning: Error in termination check: {e}")
                # Force episode end if there's an error to prevent infinite loops
                done = True
                termination_reason = "Termination error occurred"

            # Get next state using the same helper function
            next_state = get_state_vector(car, track, track_boundaries)

            # Store experience and train - add timeout protection
            try:
                reward_tensor = torch.tensor([reward], dtype=torch.float32)
                done_tensor = torch.tensor([done], dtype=torch.bool)
                agent.memory.push(state, action_tensor, reward_tensor, next_state, done_tensor)
                agent.learn()
            except Exception as e:
                print(f"Warning: Error in agent training: {e}")
                # Continue without storing this experience
            
            # Update camera for smooth following
            target_x = car.pos.x - SCREEN_WIDTH / 2
            target_y = car.pos.y - SCREEN_HEIGHT / 2
            camera_offset.x += (target_x - camera_offset.x) * 0.1
            camera_offset.y += (target_y - camera_offset.y) * 0.1
            
            # Render everything with timeout protection
            try:
                # Check if display is still valid before rendering
                if not pygame.get_init() or pygame.display.get_surface() is None:
                    print("Display surface lost during rendering")
                    done = True
                    termination_reason = "Display closed"
                    break
                
                # Add a render timeout check
                if steps % 60 == 0:  # Every second, check if we should continue rendering
                    current_time = pygame.time.get_ticks()
                    if not hasattr(main, 'last_render_time'):
                        main.last_render_time = current_time
                    elif current_time - main.last_render_time > 5000:  # 5 second render timeout
                        print("WARNING: Render timeout detected")
                        done = True
                        termination_reason = "Render timeout"
                    main.last_render_time = current_time
                
                track.draw(screen, camera_offset)
            except pygame.error as e:
                if "display Surface quit" in str(e):
                    print("Display surface quit during rendering")
                    done = True
                    termination_reason = "Display closed"
                    break
                else:
                    print(f"Pygame error during rendering: {e}")
                    # Try to continue without this frame
            except Exception as e:
                print(f"General error during rendering: {e}")
                # Try to continue
            
            # Draw checkpoints with error handling
            try:
                if hasattr(track, 'checkpoints') and track.checkpoints:
                    for i, checkpoint in enumerate(track.checkpoints):
                        if len(checkpoint) < 2:  # Skip invalid checkpoints
                            continue
                            
                        checkpoint_screen_pos = (
                            checkpoint[0] - camera_offset.x,
                            checkpoint[1] - camera_offset.y
                        )
                        
                        # Different colors based on status - SAFE ACCESS
                        car_checkpoints_hit = getattr(car, 'checkpoints_hit', set())
                        car_next_checkpoint_index = getattr(car, 'next_checkpoint_index', 0)
                        
                        if i in car_checkpoints_hit:
                            # Completed checkpoint - green
                            color = GREEN
                            pygame.draw.circle(screen, color, (int(checkpoint_screen_pos[0]), int(checkpoint_screen_pos[1])), 15, 3)
                            pygame.draw.circle(screen, WHITE, (int(checkpoint_screen_pos[0]), int(checkpoint_screen_pos[1])), 8)
                        elif i == car_next_checkpoint_index:
                            # Next checkpoint - yellow (pulsing)
                            pulse = int(128 + 127 * math.sin(pygame.time.get_ticks() * 0.01))
                            color = (255, 255, pulse)
                            pygame.draw.circle(screen, color, (int(checkpoint_screen_pos[0]), int(checkpoint_screen_pos[1])), 18, 4)
                            # Add arrow pointing to it
                            font_small = pygame.font.SysFont(None, 24)
                            text = font_small.render("NEXT", True, color)
                            screen.blit(text, (checkpoint_screen_pos[0] - 20, checkpoint_screen_pos[1] - 35))
                        else:
                            # Future checkpoint - gray
                            color = GRAY
                            pygame.draw.circle(screen, color, (int(checkpoint_screen_pos[0]), int(checkpoint_screen_pos[1])), 12, 2)
                        
                        # Checkpoint number
                        small_font = pygame.font.SysFont(None, 20)
                        text = small_font.render(str(i + 1), True, WHITE)
                        text_rect = text.get_rect(center=(int(checkpoint_screen_pos[0]), int(checkpoint_screen_pos[1])))
                        screen.blit(text, text_rect)
            except Exception as e:
                print(f"Warning: Error drawing checkpoints: {e}")
                # Continue without drawing checkpoints
            
            # Draw finish line with error handling
            try:
                if hasattr(track, 'finish_line_center') and track.finish_line_center:
                    finish_screen_pos = (
                        track.finish_line_center[0] - camera_offset.x,
                        track.finish_line_center[1] - camera_offset.y
                    )
                    
                    # Draw finish line base
                    pygame.draw.circle(screen, WHITE, (int(finish_screen_pos[0]), int(finish_screen_pos[1])), 30, 4)
                    pygame.draw.circle(screen, BLACK, (int(finish_screen_pos[0]), int(finish_screen_pos[1])), 25, 2)
                    
                    # "FINISH" text
                    finish_font = pygame.font.SysFont(None, 24)
                    finish_text = finish_font.render("FINISH", True, WHITE)
                    text_rect = finish_text.get_rect(center=(finish_screen_pos[0], finish_screen_pos[1] - 45))
                    # Text background for visibility
                    bg_rect = text_rect.inflate(10, 5)
                    pygame.draw.rect(screen, BLACK, bg_rect)
                    pygame.draw.rect(screen, WHITE, bg_rect, 1)
                    screen.blit(finish_text, text_rect)
            except Exception as e:
                # Silently skip finish line drawing if there's an error
                pass
            
            # Draw car with error handling
            try:
                car_screen_pos = car.pos - camera_offset
                
                # Validate car screen position
                if (hasattr(car_screen_pos, 'x') and hasattr(car_screen_pos, 'y') and
                    not math.isnan(car_screen_pos.x) and not math.isnan(car_screen_pos.y) and
                    abs(car_screen_pos.x) < 10000 and abs(car_screen_pos.y) < 10000):
                    
                    # Safe car drawing
                    if hasattr(car, 'image') and car.image is not None:
                        car_rect = car.image.get_rect()
                        car_rect.center = (int(car_screen_pos.x), int(car_screen_pos.y))
                        screen.blit(car.image, car_rect)
                    else:
                        # Fallback: draw a simple colored rectangle if image is missing
                        pygame.draw.circle(screen, RED, (int(car_screen_pos.x), int(car_screen_pos.y)), 10)
                else:
                    print(f"WARNING: Invalid car position: {car_screen_pos}")
                    # Draw car at screen center as fallback
                    pygame.draw.circle(screen, RED, (SCREEN_WIDTH//2, SCREEN_HEIGHT//2), 10)
                    
            except Exception as e:
                print(f"Warning: Error drawing car: {e}")
                # Emergency fallback: draw a red circle at screen center
                try:
                    pygame.draw.circle(screen, RED, (SCREEN_WIDTH//2, SCREEN_HEIGHT//2), 10)
                except:
                    pass  # If even this fails, skip car drawing
            
            # Enhanced training information display - SAFE ACCESS
            try:
                total_time = datetime.now() - training_state.training_start_time
                session_time = datetime.now() - training_state.last_save_time
                
                car_checkpoints_hit = getattr(car, 'checkpoints_hit', set())
                car_speed = getattr(car, 'speed', 0.0)
                track_checkpoint_count = len(getattr(track, 'checkpoints', []))
                
                info_lines = [
                    f"Episode: {i_episode+1}/{num_episodes}",
                    f"Step: {steps}",
                    f"Reward: {episode_reward:.1f}",
                    f"Epsilon: {agent.epsilon:.3f}",
                    f"Checkpoints: {len(car_checkpoints_hit)}/{track_checkpoint_count}",
                    f"Speed: {car_speed:.1f}/{MAX_SPEED}",
                    f"Off-track: {off_track_timer/60:.1f}s",
                    f"Memory: {len(agent.memory)}/10000",
                ]
            except Exception as e:
                print(f"Warning: Error creating info display: {e}")
                info_lines = [
                    f"Episode: {i_episode+1}/{num_episodes}",
                    f"Step: {steps}",
                    f"Error in display"
                ]
            
            # Draw info with background
            try:
                # Check display validity before drawing UI
                if not pygame.get_init() or pygame.display.get_surface() is None:
                    done = True
                    termination_reason = "Display closed"
                    break
                    
                info_bg = pygame.Rect(5, 5, 280, len(info_lines) * 25 + 10)
                s = pygame.Surface((info_bg.width, info_bg.height), pygame.SRCALPHA)
                s.fill((0,0,0,180))
                screen.blit(s, (info_bg.x, info_bg.y))
                pygame.draw.rect(screen, WHITE, info_bg, 1)
                
                for i, line in enumerate(info_lines):
                    color = GREEN if "Best" in line and best_lap_time != float('inf') else WHITE
                    if "Reward" in line and episode_reward > 100:
                        color = GREEN
                    elif "Reward" in line and episode_reward < 0:
                        color = RED
                    elif "Session" in line or "Total Time" in line:
                        color = YELLOW
                    text_surface = font.render(line, True, color)
                    screen.blit(text_surface, (10, 10 + i * 25))
            except pygame.error as e:
                if "display Surface quit" in str(e):
                    print("Display surface quit during UI rendering")
                    done = True
                    termination_reason = "Display closed"
                    break
                else:
                    print(f"Pygame error during UI rendering: {e}")
            except Exception as e:
                print(f"General error during UI rendering: {e}")
            
            # Progress bars and final display
            try:
                # Check display validity one more time
                if not pygame.get_init() or pygame.display.get_surface() is None:
                    done = True
                    termination_reason = "Display closed"
                    break
                
                # Progress bar for current episode
                progress_width = 200
                progress_height = 10
                progress_x = SCREEN_WIDTH - progress_width - 10
                progress_y = 10
                
                progress = min(steps / 2400, 1.0)  # Based on max steps
                pygame.draw.rect(screen, GRAY, (progress_x, progress_y, progress_width, progress_height))
                pygame.draw.rect(screen, GREEN, (progress_x, progress_y, progress_width * progress, progress_height))
                pygame.draw.rect(screen, WHITE, (progress_x, progress_y, progress_width, progress_height), 1)
                
                # Overall training progress bar
                overall_progress = (i_episode + 1) / num_episodes
                overall_progress_y = progress_y + 20
                pygame.draw.rect(screen, GRAY, (progress_x, overall_progress_y, progress_width, progress_height))
                pygame.draw.rect(screen, BLUE, (progress_x, overall_progress_y, progress_width * overall_progress, progress_height))
                pygame.draw.rect(screen, WHITE, (progress_x, overall_progress_y, progress_width, progress_height), 1)
                
                # Progress labels
                episode_text = font.render("Episode", True, WHITE)
                overall_text = font.render("Overall", True, WHITE)
                screen.blit(episode_text, (progress_x - 60, progress_y - 5))
                screen.blit(overall_text, (progress_x - 60, overall_progress_y - 5))
                
                if paused:
                    pause_text = large_font.render("PAUSED", True, YELLOW)
                    screen.blit(pause_text, (SCREEN_WIDTH//2 - 60, SCREEN_HEIGHT//2))
                
                pygame.display.flip()
                
            except pygame.error as e:
                if "display Surface quit" in str(e):
                    print("Display surface quit during final display")
                    done = True
                    termination_reason = "Display closed"
                    break
                else:
                    print(f"Pygame error during final display: {e}")
                    # Try to continue
            except Exception as e:
                print(f"General error during final display: {e}")
                # Try to continue
            
            steps += 1
            if done:
                print(f"Episode {i_episode+1} finished: {termination_reason}, Steps: {steps}, Reward: {episode_reward:.1f}")
                break
        
        # Episode completed - update tracking variables
        episode_rewards.append(episode_reward)
        training_state.epsilon_history.append(agent.epsilon)
        training_state.checkpoint_progress.append(episode_max_checkpoint)
        
        # Update global max checkpoint reached
        max_checkpoint_reached = max(max_checkpoint_reached, episode_max_checkpoint)
        
        # Periodic updates and saves
        if (i_episode + 1) % 10 == 0:
            agent.update_target_net()
            torch.save(agent.policy_net.state_dict(), model_path)
            
            # Update and save training state
            training_state.max_checkpoint_reached = max_checkpoint_reached
            training_state.best_lap_time = best_lap_time
            training_state.episode_rewards = episode_rewards
            training_state.successful_laps = successful_laps
            training_state.save_state(agent)

        # Major milestone saves
        if (i_episode + 1) % 100 == 0:
            backup_path = f'models/car_dqn_model_episode_{i_episode+1}.pth'
            torch.save(agent.policy_net.state_dict(), backup_path)
            print(f"Backup model saved: {backup_path}")
            
            # Save a detailed backup of training state
            backup_state_path = f'configs/training_state_episode_{i_episode+1}.json'
            training_state.save_file = backup_state_path
            training_state.max_checkpoint_reached = max_checkpoint_reached
            training_state.best_lap_time = best_lap_time
            training_state.episode_rewards = episode_rewards
            training_state.successful_laps = successful_laps
            training_state.save_state(agent)
            training_state.save_file = 'configs/training_state.json'  # Reset to default
            print(f"Backup training state saved: {backup_state_path}")

    # Training complete - save final state
    training_state.current_episode = num_episodes
    training_state.max_checkpoint_reached = max_checkpoint_reached
    training_state.best_lap_time = best_lap_time
    training_state.episode_rewards = episode_rewards
    training_state.successful_laps = successful_laps
    training_state.save_state(agent)
    
    print("\n=== Training Complete! ===")
    torch.save(agent.policy_net.state_dict(), model_path)
    print(f"Final model saved as '{model_path}'")
    
    # Print comprehensive training summary
    training_state.print_session_summary()
    
    # Save final backup
    final_backup_path = f'models/car_dqn_model_FINAL.pth'
    torch.save(agent.policy_net.state_dict(), final_backup_path)
    print(f"Final backup saved as '{final_backup_path}'")
    
    pygame.quit()

if __name__ == "__main__":
    main()