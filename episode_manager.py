import random
import pygame
from utils import respawn_car_at_checkpoint

class EpisodeManager:
    
    def __init__(self, track, training_state):
        self.track = track
        self.training_state = training_state
        self.reset_episode_state()
    
    # Reset all episode-specific state variables
    def reset_episode_state(self):
        self.stuck_timer = 0
        self.off_track_timer = 0
        self.movement_check_timer = 0
        self.episode_reward = 0
        self.steps = 0
        self.last_pos = pygame.math.Vector2(0, 0)
        self.last_checkpoint_count = 0
        self.last_distance_to_checkpoint = float('inf')
        self.episode_max_checkpoint = 0
    
    # Initialize a new episode with optional curriculum learning
    def initialize_episode(self, car, max_checkpoint_reached, curriculum_learning=True):
        start_x, start_y = self.track.track_points[0]
        car.reset(start_x, start_y)
        
        # Curriculum learning: spawn at later checkpoints sometimes
        if curriculum_learning and max_checkpoint_reached > 3 and random.random() < 0.25:
            start_checkpoint_idx = random.randint(0, min(max_checkpoint_reached, len(self.track.checkpoints) - 1))
            respawn_car_at_checkpoint(car, self.track, start_checkpoint_idx)
        
        self.reset_episode_state()
        self.last_checkpoint_count = len(car.checkpoints_hit)
        self.episode_max_checkpoint = len(car.checkpoints_hit)
        self.last_pos = car.pos.copy()
        
        # Initialize checkpoint distance
        try:
            car.update_checkpoints(self.track)
            self.last_distance_to_checkpoint = getattr(car, 'distance_to_next_checkpoint', float('inf'))
        except Exception as e:
            print(f"Warning: Error initializing checkpoint distance: {e}")
            self.last_distance_to_checkpoint = float('inf')
    
    # Update car state and handle checkpoint detection with safety checks
    def update_car_state(self, car):
        try:
            car.update(self.track)
            
            # Prevent infinite loops in checkpoint detection
            checkpoint_update_attempts = 0
            max_checkpoint_attempts = 3
            
            while checkpoint_update_attempts < max_checkpoint_attempts:
                try:
                    old_checkpoints = len(getattr(car, 'checkpoints_hit', set()))
                    old_next_index = getattr(car, 'next_checkpoint_index', 0)
                    
                    car.update_checkpoints(self.track)
                    
                    new_checkpoints = len(getattr(car, 'checkpoints_hit', set()))
                    new_next_index = getattr(car, 'next_checkpoint_index', 0)
                    
                    # Break if no changes (stable state)
                    if old_checkpoints == new_checkpoints and old_next_index == new_next_index:
                        break
                        
                    checkpoint_update_attempts += 1
                    
                    # Force break if all checkpoints hit
                    if hasattr(self.track, 'checkpoints') and new_checkpoints >= len(self.track.checkpoints):
                        break
                        
                except Exception as inner_e:
                    print(f"Warning: Error in checkpoint update attempt {checkpoint_update_attempts}: {inner_e}")
                    break
            
            if checkpoint_update_attempts >= max_checkpoint_attempts:
                print(f"WARNING: Max checkpoint update attempts reached")
                return False, "Checkpoint update loop detected"
            
            self.last_distance_to_checkpoint = getattr(car, 'distance_to_next_checkpoint', float('inf'))
            return True, ""
            
        except Exception as e:
            print(f"Warning: Error updating car state: {e}")
            self.last_distance_to_checkpoint = float('inf')
            return False, "Update error occurred"
    
    # Update stuck and off-track timers
    def update_timers(self, car):
        # Off-track timer
        if not self.track.is_on_track(car.pos):
            self.off_track_timer += 1
        else:
            self.off_track_timer = max(0, self.off_track_timer - 2)
        
        # Stuck detection
        if car.pos.distance_to(self.last_pos) < 2.0:
            self.movement_check_timer += 1
        else:
            self.movement_check_timer = 0
            self.last_pos = car.pos.copy()
        
        if self.movement_check_timer > 30:
            self.stuck_timer += 1
        else:
            self.stuck_timer = max(0, self.stuck_timer - 1)
    
    # Update checkpoint progress tracking
    def update_progress(self, car):
        try:
            current_checkpoints = len(getattr(car, 'checkpoints_hit', set()))
            self.episode_max_checkpoint = max(self.episode_max_checkpoint, current_checkpoints)
            
            if current_checkpoints > self.last_checkpoint_count:
                self.last_checkpoint_count = current_checkpoints
                self.stuck_timer = 0
                self.off_track_timer = 0
                return True  # New checkpoint reached
            return False
        except Exception as e:
            print(f"Warning: Error updating progress tracking: {e}")
            return False
    
    def check_termination(self, car):
        """Check if episode should terminate. Returns (done, reason)."""
        try:
            # Emergency timeout
            if self.steps > 3000:
                return True, "Emergency timeout"
            
            # Lap completed
            if hasattr(car, 'race_finished') and car.race_finished:
                return True, "Lap completed"
            
            # Off track too long
            if self.off_track_timer > 180:
                return True, "Off track too long"
            
            # Stuck too long
            if self.stuck_timer > 180:
                return True, "Stuck too long"
            
            # Time limit
            if self.steps > 2400:
                return True, "Time limit reached"
            
            # Emergency checkpoint-based termination
            if hasattr(car, 'checkpoints_hit') and len(car.checkpoints_hit) >= 20:
                if not getattr(car, 'race_finished', False):
                    print(f"WARNING: {len(car.checkpoints_hit)} checkpoints but no race_finished flag")
                    return True, "Checkpoint logic error"
            
            return False, ""
            
        except Exception as e:
            print(f"Warning: Error in termination check: {e}")
            return True, "Termination error occurred"
    
    def get_episode_stats(self):
        """Return dictionary of episode statistics."""
        return {
            'steps': self.steps,
            'reward': self.episode_reward,
            'max_checkpoint': self.episode_max_checkpoint,
            'stuck_timer': self.stuck_timer,
            'off_track_timer': self.off_track_timer
        }