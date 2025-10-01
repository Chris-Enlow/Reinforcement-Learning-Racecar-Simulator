import pygame
import math

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)

class TrainingRenderer:
    """Handles all rendering for the training visualization."""
    
    def __init__(self, screen, font, large_font, screen_width=1200, screen_height=800):
        self.screen = screen
        self.font = font
        self.large_font = large_font
        self.screen_width = screen_width
        self.screen_height = screen_height
    
    def render_frame(self, car, track, camera_offset, training_info, paused):
        """Render complete frame with all elements."""
        try:
            if not pygame.get_init() or pygame.display.get_surface() is None:
                return False
            
            # Draw track
            track.draw(self.screen, camera_offset)
            
            # Draw checkpoints
            self._draw_checkpoints(track, car, camera_offset)
            
            # Draw finish line
            self._draw_finish_line(track, camera_offset)
            
            # Draw car
            self._draw_car(car, camera_offset)
            
            # Draw UI
            self._draw_ui(training_info)
            
            # Draw progress bars
            self._draw_progress_bars(training_info)
            
            # Draw pause overlay if paused
            if paused:
                self._draw_pause_overlay()
            
            pygame.display.flip()
            return True
            
        except pygame.error as e:
            if "display Surface quit" in str(e):
                print("Display surface quit during rendering")
                return False
            else:
                print(f"Pygame error during rendering: {e}")
                return True
        except Exception as e:
            print(f"General error during rendering: {e}")
            return True
    
    def _draw_checkpoints(self, track, car, camera_offset):
        """Draw all checkpoints with status indicators."""
        try:
            if not hasattr(track, 'checkpoints') or not track.checkpoints:
                return
            
            for i, checkpoint in enumerate(track.checkpoints):
                if len(checkpoint) < 2:
                    continue
                
                checkpoint_screen_pos = (
                    checkpoint[0] - camera_offset.x,
                    checkpoint[1] - camera_offset.y
                )
                
                car_checkpoints_hit = getattr(car, 'checkpoints_hit', set())
                car_next_checkpoint_index = getattr(car, 'next_checkpoint_index', 0)
                
                if i in car_checkpoints_hit:
                    # Completed - green
                    color = GREEN
                    pygame.draw.circle(self.screen, color, (int(checkpoint_screen_pos[0]), int(checkpoint_screen_pos[1])), 15, 3)
                    pygame.draw.circle(self.screen, WHITE, (int(checkpoint_screen_pos[0]), int(checkpoint_screen_pos[1])), 8)
                elif i == car_next_checkpoint_index:
                    # Next - pulsing yellow
                    pulse = int(128 + 127 * math.sin(pygame.time.get_ticks() * 0.01))
                    color = (255, 255, pulse)
                    pygame.draw.circle(self.screen, color, (int(checkpoint_screen_pos[0]), int(checkpoint_screen_pos[1])), 18, 4)
                    
                    font_small = pygame.font.SysFont(None, 24)
                    text = font_small.render("NEXT", True, color)
                    self.screen.blit(text, (checkpoint_screen_pos[0] - 20, checkpoint_screen_pos[1] - 35))
                else:
                    # Future - gray
                    color = GRAY
                    pygame.draw.circle(self.screen, color, (int(checkpoint_screen_pos[0]), int(checkpoint_screen_pos[1])), 12, 2)
                
                # Checkpoint number
                small_font = pygame.font.SysFont(None, 20)
                text = small_font.render(str(i + 1), True, WHITE)
                text_rect = text.get_rect(center=(int(checkpoint_screen_pos[0]), int(checkpoint_screen_pos[1])))
                self.screen.blit(text, text_rect)
                
        except Exception as e:
            print(f"Warning: Error drawing checkpoints: {e}")
    
    def _draw_finish_line(self, track, camera_offset):
        """Draw the finish line."""
        try:
            if not hasattr(track, 'finish_line_center') or not track.finish_line_center:
                return
            
            finish_screen_pos = (
                track.finish_line_center[0] - camera_offset.x,
                track.finish_line_center[1] - camera_offset.y
            )
            
            pygame.draw.circle(self.screen, WHITE, (int(finish_screen_pos[0]), int(finish_screen_pos[1])), 30, 4)
            pygame.draw.circle(self.screen, BLACK, (int(finish_screen_pos[0]), int(finish_screen_pos[1])), 25, 2)
            
            finish_font = pygame.font.SysFont(None, 24)
            finish_text = finish_font.render("FINISH", True, WHITE)
            text_rect = finish_text.get_rect(center=(finish_screen_pos[0], finish_screen_pos[1] - 45))
            
            bg_rect = text_rect.inflate(10, 5)
            pygame.draw.rect(self.screen, BLACK, bg_rect)
            pygame.draw.rect(self.screen, WHITE, bg_rect, 1)
            self.screen.blit(finish_text, text_rect)
            
        except Exception as e:
            pass  # Silently skip if error
    
    def _draw_car(self, car, camera_offset):
        """Draw the car."""
        try:
            car_screen_pos = car.pos - camera_offset
            
            # Validate position
            if (hasattr(car_screen_pos, 'x') and hasattr(car_screen_pos, 'y') and
                not math.isnan(car_screen_pos.x) and not math.isnan(car_screen_pos.y) and
                abs(car_screen_pos.x) < 10000 and abs(car_screen_pos.y) < 10000):
                
                if hasattr(car, 'image') and car.image is not None:
                    car_rect = car.image.get_rect()
                    car_rect.center = (int(car_screen_pos.x), int(car_screen_pos.y))
                    self.screen.blit(car.image, car_rect)
                else:
                    pygame.draw.circle(self.screen, RED, (int(car_screen_pos.x), int(car_screen_pos.y)), 10)
            else:
                print(f"WARNING: Invalid car position: {car_screen_pos}")
                pygame.draw.circle(self.screen, RED, (self.screen_width//2, self.screen_height//2), 10)
                
        except Exception as e:
            print(f"Warning: Error drawing car: {e}")
            try:
                pygame.draw.circle(self.screen, RED, (self.screen_width//2, self.screen_height//2), 10)
            except:
                pass
    
    def _draw_ui(self, training_info):
        """Draw training information UI."""
        try:
            if not pygame.get_init() or pygame.display.get_surface() is None:
                return
            
            info_lines = training_info.get('info_lines', [])
            
            info_bg = pygame.Rect(5, 5, 280, len(info_lines) * 25 + 10)
            s = pygame.Surface((info_bg.width, info_bg.height), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            self.screen.blit(s, (info_bg.x, info_bg.y))
            pygame.draw.rect(self.screen, WHITE, info_bg, 1)
            
            for i, line in enumerate(info_lines):
                color = WHITE
                episode_reward = training_info.get('episode_reward', 0)
                
                if "Reward" in line:
                    if episode_reward > 100:
                        color = GREEN
                    elif episode_reward < 0:
                        color = RED
                elif "Session" in line or "Total Time" in line:
                    color = YELLOW
                
                text_surface = self.font.render(line, True, color)
                self.screen.blit(text_surface, (10, 10 + i * 25))
                
        except pygame.error as e:
            if "display Surface quit" not in str(e):
                print(f"Pygame error during UI rendering: {e}")
        except Exception as e:
            print(f"General error during UI rendering: {e}")
    
    def _draw_progress_bars(self, training_info):
        """Draw episode and overall progress bars."""
        try:
            if not pygame.get_init() or pygame.display.get_surface() is None:
                return
            
            progress_width = 200
            progress_height = 10
            progress_x = self.screen_width - progress_width - 10
            progress_y = 10
            
            # Episode progress
            steps = training_info.get('steps', 0)
            progress = min(steps / 2400, 1.0)
            pygame.draw.rect(self.screen, GRAY, (progress_x, progress_y, progress_width, progress_height))
            pygame.draw.rect(self.screen, GREEN, (progress_x, progress_y, progress_width * progress, progress_height))
            pygame.draw.rect(self.screen, WHITE, (progress_x, progress_y, progress_width, progress_height), 1)
            
            # Overall progress
            overall_progress = training_info.get('overall_progress', 0)
            overall_progress_y = progress_y + 20
            pygame.draw.rect(self.screen, GRAY, (progress_x, overall_progress_y, progress_width, progress_height))
            pygame.draw.rect(self.screen, BLUE, (progress_x, overall_progress_y, progress_width * overall_progress, progress_height))
            pygame.draw.rect(self.screen, WHITE, (progress_x, overall_progress_y, progress_width, progress_height), 1)
            
            # Labels
            episode_text = self.font.render("Episode", True, WHITE)
            overall_text = self.font.render("Overall", True, WHITE)
            self.screen.blit(episode_text, (progress_x - 60, progress_y - 5))
            self.screen.blit(overall_text, (progress_x - 60, overall_progress_y - 5))
            
        except pygame.error as e:
            if "display Surface quit" not in str(e):
                print(f"Pygame error during progress bar rendering: {e}")
        except Exception as e:
            print(f"General error during progress bar rendering: {e}")
    
    def _draw_pause_overlay(self):
        """Draw pause overlay."""
        try:
            pause_text = self.large_font.render("PAUSED", True, YELLOW)
            self.screen.blit(pause_text, (self.screen_width//2 - 60, self.screen_height//2))
        except Exception as e:
            print(f"Error drawing pause overlay: {e}")