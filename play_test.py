# play_test.py
"""
Loads a trained model and runs it in performance mode.
Displays the car's learned racing line with a clean UI and lap timer.
"""

import pygame
import math
import random
import numpy as np
import torch
import os
from shapely.geometry import Point, LineString

# Import your existing modules
from ai_agent import DQNAgent
from game_classes import Track, Car

# --- Constants (must be identical to training script) ---
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
MAX_SPEED = 8.0

def get_state_vector(car, track, track_boundaries):
    """
    Helper function to get the state for the AI.
    Must be identical to the one in the training script.
    """
    try:
        ray_distances = car.cast_rays(track_boundaries)
        normalized_rays = [min(d / 1000.0, 1.0) for d in ray_distances]
        normalized_speed = [car.speed / MAX_SPEED]
        
        # Safe checkpoint access
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
        
        # Simplified centerline distance
        center_distance_norm = 0.0
        
        state_values = normalized_rays + normalized_speed + [angle_diff_norm, center_distance_norm]
        return torch.tensor(state_values, dtype=torch.float32).unsqueeze(0)
        
    except Exception as e:
        print(f"Error in get_state_vector: {e}")
        # Return safe default
        default_state = [0.5] * 9 + [0.0] * 3
        return torch.tensor(default_state, dtype=torch.float32).unsqueeze(0)


def main():
    print("=== AI Performance Test Mode ===")
    
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("AI Performance Test - Press R to Reset")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 32)
    small_font = pygame.font.SysFont(None, 24)

    # Create track (no seed needed - it's deterministic)
    track = Track()
    start_x, start_y = track.track_points[0]
    car = Car(start_x, start_y)
    track_boundaries = track.get_track_boundaries()

    # --- AI Setup ---
    n_actions = 5
    n_observations = 12
    agent = DQNAgent(n_observations, n_actions)
    
    # Load the trained model
    model_path = 'car_dqn_model.pth'
    if os.path.exists(model_path):
        try:
            agent.policy_net.load_state_dict(torch.load(model_path))
            agent.policy_net.eval()  # Set to evaluation mode
            print(f"✓ Model loaded from {model_path}")
        except Exception as e:
            print(f"✗ ERROR loading model: {e}")
            return
    else:
        print(f"✗ ERROR: No trained model found at '{model_path}'. Please train a model first.")
        return

    # --- IMPORTANT: Set epsilon to 0 for pure performance (no random actions) ---
    agent.epsilon = 0.0

    camera_offset = pygame.math.Vector2(car.pos.x - SCREEN_WIDTH / 2, car.pos.y - SCREEN_HEIGHT / 2)
    
    # Lap timing variables
    lap_start_time = pygame.time.get_ticks()
    last_lap_time = 0
    best_lap_time = float('inf')
    lap_count = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                car.reset(start_x, start_y)
                lap_start_time = pygame.time.get_ticks()
                print("Car reset!")

        # --- AI Decision Making ---
        state = get_state_vector(car, track, track_boundaries)
        with torch.no_grad():
            action = agent.policy_net(state).max(1)[1].item()

        controls = [False] * 4  # [accel, brake, left, right]
        if action == 0:
            controls[0] = True
        elif action == 1:
            controls[0] = True
            controls[2] = True
        elif action == 2:
            controls[0] = True
            controls[3] = True
        elif action == 3:
            controls[1] = True
        # action 4 is coast (do nothing)
        
        car.set_controls(*controls)
        
        # --- Update & Lap Timing ---
        car.update(track)
        car.update_checkpoints(track)

        # Check for lap completion
        if car.race_finished:
            current_lap_time = car.lap_time
            last_lap_time = current_lap_time
            lap_count += 1
            
            if last_lap_time < best_lap_time:
                best_lap_time = last_lap_time
                print(f"🏆 NEW BEST LAP: {best_lap_time:.2f}s (Lap {lap_count})")
            else:
                print(f"Lap {lap_count} completed: {last_lap_time:.2f}s")
            
            # Reset for next lap
            car.reset(start_x, start_y)
            car.race_finished = False
        
        # --- Camera following ---
        target_x = car.pos.x - SCREEN_WIDTH / 2
        target_y = car.pos.y - SCREEN_HEIGHT / 2
        camera_offset.x += (target_x - camera_offset.x) * 0.1
        camera_offset.y += (target_y - camera_offset.y) * 0.1
        
        # --- Drawing ---
        track.draw(screen, camera_offset)
        
        # Draw checkpoints
        for i, checkpoint in enumerate(track.checkpoints):
            checkpoint_screen_pos = (
                checkpoint[0] - camera_offset.x,
                checkpoint[1] - camera_offset.y
            )
            
            if i in car.checkpoints_hit:
                color = GREEN
                pygame.draw.circle(screen, color, (int(checkpoint_screen_pos[0]), 
                                                   int(checkpoint_screen_pos[1])), 10, 2)
            elif i == car.next_checkpoint_index:
                color = YELLOW
                pygame.draw.circle(screen, color, (int(checkpoint_screen_pos[0]), 
                                                   int(checkpoint_screen_pos[1])), 12, 3)
            else:
                color = (100, 100, 100)
                pygame.draw.circle(screen, color, (int(checkpoint_screen_pos[0]), 
                                                   int(checkpoint_screen_pos[1])), 8, 2)
        
        # Draw car
        car_screen_pos = car.pos - camera_offset
        if hasattr(car, 'image') and car.image is not None:
            car_rect = car.image.get_rect(center=(int(car_screen_pos.x), int(car_screen_pos.y)))
            screen.blit(car.image, car_rect)
        
        # --- Clean UI for Lap Times ---
        current_time = car.lap_time
        
        # Create semi-transparent background for UI
        ui_bg = pygame.Surface((300, 140), pygame.SRCALPHA)
        ui_bg.fill((0, 0, 0, 180))
        screen.blit(ui_bg, (SCREEN_WIDTH - 310, 10))
        
        # Lap information
        info_lines = [
            f"Lap: {lap_count}",
            f"Current: {current_time:.2f}s",
            f"Last: {last_lap_time:.2f}s" if last_lap_time > 0 else "Last: --.-s",
            f"Best: {best_lap_time:.2f}s" if best_lap_time != float('inf') else "Best: --.-s",
        ]
        
        for i, text in enumerate(info_lines):
            color = WHITE
            if "Best" in text and best_lap_time != float('inf'):
                color = GREEN
            surf = font.render(text, True, color)
            screen.blit(surf, (SCREEN_WIDTH - 295, 20 + i * 35))
        
        # Additional info at bottom
        speed_text = small_font.render(f"Speed: {car.speed:.1f}/{MAX_SPEED}", True, WHITE)
        checkpoints_text = small_font.render(
            f"Checkpoints: {len(car.checkpoints_hit)}/{len(track.checkpoints)}", 
            True, WHITE
        )
        
        screen.blit(speed_text, (10, SCREEN_HEIGHT - 60))
        screen.blit(checkpoints_text, (10, SCREEN_HEIGHT - 35))
        
        # Instructions
        instructions = small_font.render("Press R to Reset | ESC to Exit", True, WHITE)
        screen.blit(instructions, (10, 10))
            
        pygame.display.flip()
        clock.tick(FPS)
        
    # Print final statistics
    print("\n=== Session Summary ===")
    print(f"Total laps completed: {lap_count}")
    print(f"Best lap time: {best_lap_time:.2f}s" if best_lap_time != float('inf') else "No completed laps")
    
    pygame.quit()

if __name__ == "__main__":
    main()