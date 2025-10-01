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

# --- Constants ---
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
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

class Track:
    def __init__(self):
        self.track_width = 80
        self.spline_resolution = 30
        self.checkpoints = []
        self.track_points = self.generate_track()
        self.finish_line_center = None
        self.create_finish_line()

    # Calculates points on a Catmull-Rom spline for a segment.
    def _catmull_rom_spline(self, P0, P1, P2, P3, num_points=30):  # Change default from 20 to 30
        points = []
        for t in np.linspace(0, 1, num_points):
            t2, t3 = t * t, t * t * t
            x = 0.5 * ((2 * P1[0]) + (-P0[0] + P2[0]) * t + \
                (2 * P0[0] - 5 * P1[0] + 4 * P2[0] - P3[0]) * t2 + \
                (-P0[0] + 3 * P1[0] - 3 * P2[0] + P3[0]) * t3)
            y = 0.5 * ((2 * P1[1]) + (-P0[1] + P2[1]) * t + \
                (2 * P0[1] - 5 * P1[1] + 4 * P2[1] - P3[1]) * t2 + \
                (-P0[1] + 3 * P1[1] - 3 * P2[1] + P3[1]) * t3)
            points.append((x, y))
        return points

    # Generates a simple F1-style track with a sharp hairpin and chicane
    def generate_track(self):
        waypoints = [
            (SCREEN_WIDTH * 0.5, SCREEN_HEIGHT * 0.85),  # Start/Finish
            (SCREEN_WIDTH * 0.9, SCREEN_HEIGHT * 0.85),
            (SCREEN_WIDTH * 0.9, SCREEN_HEIGHT * 0.2),
            (SCREEN_WIDTH * 0.65, SCREEN_HEIGHT * 0.15),
            (SCREEN_WIDTH * 0.6, SCREEN_HEIGHT * 0.35),
            (SCREEN_WIDTH * 0.4, SCREEN_HEIGHT * 0.4),
            (SCREEN_WIDTH * 0.2, SCREEN_HEIGHT * 0.6),
            (SCREEN_WIDTH * 0.2, SCREEN_HEIGHT * 0.85)
        ]
        
        points = []
        for i in range(len(waypoints)):
            p0 = waypoints[(i - 1) % len(waypoints)]
            p1 = waypoints[i]
            p2 = waypoints[(i + 1) % len(waypoints)]
            p3 = waypoints[(i + 2) % len(waypoints)]
            points.extend(self._catmull_rom_spline(p0, p1, p2, p3, num_points=self.spline_resolution))

        num_generated_points = len(points)
        num_checkpoints = 20

        for i in range(1, num_checkpoints + 1):
            index = min((i * num_generated_points // num_checkpoints), num_generated_points - 1)
            self.checkpoints.append((points[index][0], points[index][1], i))

        self.track_points = points
        return points

    
    def create_finish_line(self):
        if self.track_points:
            self.finish_line_center = self.track_points[0]
    
    def is_near_finish_line(self, pos):
        if not self.finish_line_center:
            return False
        return math.hypot(pos[0] - self.finish_line_center[0], pos[1] - self.finish_line_center[1]) < 40

    def is_on_track(self, pos):
        min_distance = float('inf')
        for i in range(len(self.track_points)):
            p1 = self.track_points[i]
            p2 = self.track_points[(i + 1) % len(self.track_points)]
            min_distance = min(min_distance, self._point_to_line_distance(pos, p1, p2))
        return min_distance <= self.track_width / 2

    def _point_to_line_distance(self, point, line_start, line_end):
        x, y = point; x1, y1 = line_start; x2, y2 = line_end
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return math.hypot(x - x1, y - y1)
        t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / (dx*dx + dy*dy)))
        return math.hypot(x - (x1 + t * dx), y - (y1 + t * dy))
        
    # Calculate smooth track boundaries without pinching
    def get_track_boundaries(self):
        inner_points = []
        outer_points = []
        
        for i in range(len(self.track_points)):
            # Get previous, current, and next points
            prev_point = self.track_points[(i - 1) % len(self.track_points)]
            curr = self.track_points[i]
            next_point = self.track_points[(i + 1) % len(self.track_points)]
            
            # Calculate two direction vectors
            dx1 = curr[0] - prev_point[0]
            dy1 = curr[1] - prev_point[1]
            length1 = math.sqrt(dx1*dx1 + dy1*dy1)
            
            dx2 = next_point[0] - curr[0]
            dy2 = next_point[1] - curr[1]
            length2 = math.sqrt(dx2*dx2 + dy2*dy2)
            
            # Average the two perpendicular directions for smoother boundaries
            if length1 > 0 and length2 > 0:
                # Normalize both direction vectors
                norm_dx1, norm_dy1 = dx1 / length1, dy1 / length1
                norm_dx2, norm_dy2 = dx2 / length2, dy2 / length2
                
                # Average the normalized directions
                avg_dx = (norm_dx1 + norm_dx2) / 2
                avg_dy = (norm_dy1 + norm_dy2) / 2
                
                # Normalize the average
                avg_length = math.sqrt(avg_dx*avg_dx + avg_dy*avg_dy)
                if avg_length > 0:
                    avg_dx /= avg_length
                    avg_dy /= avg_length
                
                # Get perpendicular direction (rotate 90 degrees)
                perp_x = -avg_dy
                perp_y = avg_dx
                
            elif length2 > 0:
                # Use only next direction if previous is invalid
                perp_x = -dy2 / length2
                perp_y = dx2 / length2
            elif length1 > 0:
                # Use only previous direction if next is invalid
                perp_x = -dy1 / length1
                perp_y = dx1 / length1
            else:
                # Fallback: no valid directions
                perp_x, perp_y = 0, 1
            
            # Calculate boundary points with consistent width
            half_width = self.track_width / 2
            outer_x = curr[0] + perp_x * half_width
            outer_y = curr[1] + perp_y * half_width
            inner_x = curr[0] - perp_x * half_width
            inner_y = curr[1] - perp_y * half_width
            
            outer_points.append((outer_x, outer_y))
            inner_points.append((inner_x, inner_y))
        
        return LineString(inner_points), LineString(outer_points)

    def draw(self, screen, camera_offset):
        screen.fill(GRASS_GREEN)
        if len(self.track_points) < 3:
            return
            
        # Use the improved boundary calculation
        inner_boundary, outer_boundary = self.get_track_boundaries()
        
        # Convert LineString coordinates to screen coordinates
        track_surface_points = []
        
        # Add outer boundary points
        for coord in outer_boundary.coords:
            screen_x = coord[0] - camera_offset.x
            screen_y = coord[1] - camera_offset.y
            track_surface_points.append((screen_x, screen_y))
        
        # Add inner boundary points in reverse order to complete the polygon
        for coord in reversed(list(inner_boundary.coords)):
            screen_x = coord[0] - camera_offset.x
            screen_y = coord[1] - camera_offset.y
            track_surface_points.append((screen_x, screen_y))
        
        if len(track_surface_points) >= 3:
            pygame.draw.polygon(screen, ASPHALT, track_surface_points)

class Car:
    def __init__(self, x, y):
        self.original_image = pygame.Surface((30, 15), pygame.SRCALPHA)
        pygame.draw.rect(self.original_image, RED, (0, 0, 30, 15), border_radius=3)
        pygame.draw.rect(self.original_image, (60,60,60), (5, 3, 20, 9), border_radius=2)
        self.reset(x, y)


    def reset(self, x, y):
        self.pos = pygame.math.Vector2(x, y)
        self.vel = pygame.math.Vector2(0, 0)
        self.angle = 0.0
        self.speed = 0.0
        self.max_speed, self.acceleration = MAX_SPEED, ACCELERATION
        self.brake_deceleration, self.friction = BRAKE_DECELERATION, FRICTION
        self.turn_speed = TURN_SPEED
        self.accelerating, self.braking, self.turning_left, self.turning_right = False, False, False, False
        self.checkpoints_hit = set()
        self.lap_time = 0.0
        self.race_finished = False
        self.next_checkpoint_index = 0
        self.distance_to_next_checkpoint = float('inf')
        self.on_finish_line = True
        self.image = self.original_image
        self.rect = self.image.get_rect(center=self.pos)

    def set_controls(self, accelerate, brake, steer_left, steer_right):
        self.accelerating, self.braking, self.turning_left, self.turning_right = accelerate, brake, steer_left, steer_right

    def cast_rays(self, track_boundaries):
        num_rays, ray_distances = 9, [1000.0] * 9
        inner_boundary, outer_boundary = track_boundaries
        for i, angle_offset in enumerate(np.linspace(-90, 90, num_rays)):
            angle = math.radians(self.angle + angle_offset)
            # Create a very long ray
            ray_end = (self.pos.x + 2000 * math.cos(angle), self.pos.y + 2000 * math.sin(angle))
            ray_line = LineString([self.pos, ray_end])
            
            min_dist = 1000.0
            for boundary in [inner_boundary, outer_boundary]:
                intersection = ray_line.intersection(boundary)
                if not intersection.is_empty:
                    # Handle single or multiple intersection points
                    points = intersection.geoms if hasattr(intersection, "geoms") else [intersection]
                    for point in points:
                        # Ensure the intersection is a Point before accessing coords
                        if isinstance(point, Point):
                            dist = self.pos.distance_to(point.coords[0])
                            min_dist = min(min_dist, dist)
            ray_distances[i] = min_dist
        return ray_distances

    def update_checkpoints(self, track):
        if not track.checkpoints or self.race_finished: 
            return

        next_checkpoint_pos = track.checkpoints[self.next_checkpoint_index][:2]
        if self.pos.distance_to(next_checkpoint_pos) < track.track_width / 1.5:
            self.checkpoints_hit.add(self.next_checkpoint_index)
            
            if len(self.checkpoints_hit) >= len(track.checkpoints):
                self.race_finished = True
                return
            else:
                while self.next_checkpoint_index in self.checkpoints_hit:
                    self.next_checkpoint_index = (self.next_checkpoint_index + 1) % len(track.checkpoints)
        
        # Update distance to next checkpoint
        if len(self.checkpoints_hit) < len(track.checkpoints):
            self.distance_to_next_checkpoint = self.pos.distance_to(track.checkpoints[self.next_checkpoint_index][:2])
        else:
            self.distance_to_next_checkpoint = 0

    # Physics
    def update(self, track):
        if self.accelerating: 
            self.speed += self.acceleration
        elif self.braking: 
            self.speed -= self.brake_deceleration
                
        # No reversing
        self.speed = max(0, min(self.max_speed, self.speed))

        if self.speed > 0.1:
            # Increased base turning and less speed penalty for tighter turns
            turn_effect = 1.0 - (self.speed / self.max_speed * 0.3) 
            turn_multiplier = 1.25 
            if self.turning_left: self.angle -= self.turn_speed * turn_effect * turn_multiplier
            if self.turning_right: self.angle += self.turn_speed * turn_effect * turn_multiplier
        
        angle_rad = math.radians(self.angle)
        self.vel.x = self.speed * math.cos(angle_rad)
        self.vel.y = self.speed * math.sin(angle_rad)
        
        # Apply friction based on surface
        if not track.is_on_track(self.pos):
            self.speed *= OFF_ROAD_FRICTION
        else:
            self.speed *= self.friction
        
        self.pos += self.vel
        
        # Lap Logic
        if self.on_finish_line and not track.is_near_finish_line(self.pos):
             self.on_finish_line = False

        if not self.race_finished and not self.on_finish_line and track.is_near_finish_line(self.pos) and len(self.checkpoints_hit) >= len(track.checkpoints):
            self.race_finished = True
        
        # Graphics
        self.image = pygame.transform.rotate(self.original_image, -self.angle)
        self.rect = self.image.get_rect(center=self.pos)
        
        if not self.race_finished: 
            self.lap_time += 1/FPS

class TrainingState:    
    def __init__(self, save_file='configs/training_state.json'):
        self.save_file = save_file
        self.current_episode = 0
        self.max_checkpoint_reached = 0
        self.best_lap_time = float('inf')
        self.episode_rewards = []
        self.successful_laps = 0
        self.total_episodes = 5000
        self.training_start_time = datetime.now()
        self.last_save_time = datetime.now()
        self.epsilon_history = []
        self.checkpoint_progress = []
        
    def save_state(self, agent=None):
        state_data = {
            'current_episode': self.current_episode,
            'max_checkpoint_reached': self.max_checkpoint_reached,
            'best_lap_time': self.best_lap_time if self.best_lap_time != float('inf') else None,
            'episode_rewards': self.episode_rewards,
            'successful_laps': self.successful_laps,
            'total_episodes': self.total_episodes,
            'training_start_time': self.training_start_time.isoformat(),
            'last_save_time': datetime.now().isoformat(),
            'epsilon_history': self.epsilon_history,
            'checkpoint_progress': self.checkpoint_progress
        }
        
        # Save agent epsilon if provided
        if agent:
            state_data['current_epsilon'] = agent.epsilon
            
        try:
            with open(self.save_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            print(f"Training state saved to {self.save_file}")
        except Exception as e:
            print(f"Failed to save training state: {e}")
            
    def load_state(self):
        if not os.path.exists(self.save_file):
            print(f"No existing training state found. Starting fresh.")
            return False
            
        try:
            with open(self.save_file, 'r') as f:
                state_data = json.load(f)
                
            self.current_episode = state_data.get('current_episode', 0)
            self.max_checkpoint_reached = state_data.get('max_checkpoint_reached', 0)
            self.best_lap_time = state_data.get('best_lap_time', float('inf'))
            if self.best_lap_time is None:
                self.best_lap_time = float('inf')
            self.episode_rewards = state_data.get('episode_rewards', [])
            self.successful_laps = state_data.get('successful_laps', 0)
            self.total_episodes = state_data.get('total_episodes', 5000)
            self.epsilon_history = state_data.get('epsilon_history', [])
            self.checkpoint_progress = state_data.get('checkpoint_progress', [])
            
            # Parse timestamps
            start_time_str = state_data.get('training_start_time')
            if start_time_str:
                self.training_start_time = datetime.fromisoformat(start_time_str)
            
            last_save_str = state_data.get('last_save_time')
            if last_save_str:
                self.last_save_time = datetime.fromisoformat(last_save_str)
                
            print(f"Training state loaded from {self.save_file}")
            print(f"Resuming from episode {self.current_episode + 1}")
            print(f"Max checkpoint reached: {self.max_checkpoint_reached + 1}")            
            return True
            
        except Exception as e:
            print(f"Failed to load training state: {e}")
            print("Starting fresh training...")
            return False
    
    def get_current_epsilon(self):
        if not os.path.exists(self.save_file):
            return None
            
        try:
            with open(self.save_file, 'r') as f:
                state_data = json.load(f)
            return state_data.get('current_epsilon')
        except:
            return None
    
    def print_session_summary(self):
        print(f"\n=== Training Session Summary ===")
        print(f"Total episodes completed: {self.current_episode}")
        
        if len(self.episode_rewards) > 0:
            recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards
            print(f"Average reward (recent): {np.mean(recent_rewards):.1f}")
            
        if len(self.epsilon_history) > 0:
            print(f"Current exploration rate: {self.epsilon_history[-1]:.3f}")