import pygame
import os
import torch
from datetime import datetime

# Import modules
from ai_agent import DQNAgent
from game_classes import Track, Car, TrainingState
from utils import *
from event_handler import EventHandler
from episode_manager import EpisodeManager
from rendering import TrainingRenderer

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60

# Create dictionary of training information for rendering
def create_training_info(episode, episode_mgr, agent, car, track, num_episodes):
    stats = episode_mgr.get_episode_stats()
    
    try:
        car_checkpoints_hit = getattr(car, 'checkpoints_hit', set())
        car_speed = getattr(car, 'speed', 0.0)
        track_checkpoint_count = len(getattr(track, 'checkpoints', []))
        
        info_lines = [
            f"Episode: {episode+1}/{num_episodes}",
            f"Step: {stats['steps']}",
            f"Reward: {stats['reward']:.1f}",
            f"Epsilon: {agent.epsilon:.3f}",
            f"Checkpoints: {len(car_checkpoints_hit)}/{track_checkpoint_count}",
            f"Speed: {car_speed:.1f}/8.0",
            f"Off-track: {stats['off_track_timer']/60:.1f}s",
            f"Memory: {len(agent.memory)}/10000",
        ]
    except Exception as e:
        print(f"Warning: Error creating info display: {e}")
        info_lines = [
            f"Episode: {episode+1}/{num_episodes}",
            f"Step: {stats['steps']}",
            f"Error in display"
        ]
    
    return {
        'info_lines': info_lines,
        'episode_reward': stats['reward'],
        'steps': stats['steps'],
        'overall_progress': (episode + 1) / num_episodes
    }

# Run a single training episode. Returns (episode_reward, max_checkpoint, should_quit)
def run_episode(car, track, agent, episode_mgr, renderer, event_handler, episode_num, num_episodes, max_checkpoint_reached):
    # Initialize episode
    episode_mgr.initialize_episode(car, max_checkpoint_reached)
    camera_offset = pygame.math.Vector2(0, 0)
    track_boundaries = track.get_track_boundaries()
    
    while True:
        # Handle events
        should_quit, _ = event_handler.process_events()
        if should_quit:
            return episode_mgr.episode_reward, episode_mgr.episode_max_checkpoint, True
        
        # Handle pause
        if event_handler.handle_pause():
            continue
        
        state = get_state_vector(car, track, track_boundaries)
        
        # AI selects and applies action
        action_tensor = agent.select_action(state)
        action = action_tensor.item()
        apply_action(car, action)
        
        # Store previous distance for reward calculation
        prev_distance = episode_mgr.last_distance_to_checkpoint
        
        # Update car state
        success, reason = episode_mgr.update_car_state(car)
        if not success:
            # Episode ended due to update error
            done = True
            termination_reason = reason
        else:
            # Update timers and progress
            episode_mgr.update_timers(car)
            checkpoint_reached = episode_mgr.update_progress(car)
            
            # Update max checkpoint reached globally
            if checkpoint_reached:
                max_checkpoint_reached = max(max_checkpoint_reached, episode_mgr.last_checkpoint_count)
            
            # Calculate reward
            try:
                reward = calculate_reward(
                    car, track, prev_distance, episode_mgr.stuck_timer,
                    episode_mgr.last_checkpoint_count, episode_mgr.steps,
                    episode_mgr.off_track_timer, action
                )
                episode_mgr.episode_reward += reward
            except Exception as e:
                print(f"Warning: Error calculating reward: {e}")
                reward = -1.0
                episode_mgr.episode_reward += reward
            
            # Check termination
            done, termination_reason = episode_mgr.check_termination(car)
            
            # Handle lap completion
            if done and termination_reason == "Lap completed":
                return episode_mgr.episode_reward, episode_mgr.episode_max_checkpoint, False
        
        # Get next state
        next_state = get_state_vector(car, track, track_boundaries)
        
        # Store experience and train
        try:
            reward_tensor = torch.tensor([reward], dtype=torch.float32)
            done_tensor = torch.tensor([done], dtype=torch.bool)
            agent.memory.push(state, action_tensor, reward_tensor, next_state, done_tensor)
            agent.learn()
        except Exception as e:
            print(f"Warning: Error in agent training: {e}")
        
        # Update camera
        camera_offset = calculate_camera_offset(car, camera_offset, SCREEN_WIDTH, SCREEN_HEIGHT)
        
        # Render
        training_info = create_training_info(episode_num, episode_mgr, agent, car, track, num_episodes)
        if not renderer.render_frame(car, track, camera_offset, training_info, event_handler.paused):
            # Display closed during rendering
            return episode_mgr.episode_reward, episode_mgr.episode_max_checkpoint, True
        
        # Increment step counter
        episode_mgr.steps += 1
        
        # Episode ended
        if done:
            print(f"Episode {episode_num+1} finished: {termination_reason}, "
                  f"Steps: {episode_mgr.steps}, Reward: {episode_mgr.episode_reward:.1f}")
            return episode_mgr.episode_reward, episode_mgr.episode_max_checkpoint, False
    
    return episode_mgr.episode_reward, episode_mgr.episode_max_checkpoint, False


def main():
    print("=== AI Racecar Training Started ===")
    
    # Load training state
    training_state = TrainingState()
    training_state.load_state()
    
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("AI Racecar Training")
    font = pygame.font.SysFont(None, 24)
    large_font = pygame.font.SysFont(None, 36)
    
    # Create game objects
    track = Track()
    car = Car(*track.track_points[0])
    
    # Initialize AI
    n_actions = 5  # Accel, Accel+Left, Accel+Right, Brake, Coast
    n_observations = 12  # 9 rays + speed + angle_to_checkpoint + center_distance
    agent = DQNAgent(n_observations, n_actions)
    
    # Load model if exists
    model_path = 'models/car_dqn_model.pth'
    if os.path.exists(model_path):
        try:
            agent.policy_net.load_state_dict(torch.load(model_path))
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            print(f"✓ Model loaded from {model_path}")
            agent.epsilon = prompt_epsilon(training_state)
        except Exception as e:
            print(f"✗ Could not load model: {e}")
            print("Starting fresh training...")
    
    # Create managers
    renderer = TrainingRenderer(screen, font, large_font, SCREEN_WIDTH, SCREEN_HEIGHT)
    episode_mgr = EpisodeManager(track, training_state)
    event_handler = EventHandler()
    
    # Training state variables
    num_episodes = training_state.total_episodes
    max_checkpoint_reached = training_state.max_checkpoint_reached
    best_lap_time = training_state.best_lap_time
    episode_rewards = training_state.episode_rewards
    successful_laps = training_state.successful_laps
    start_episode = training_state.current_episode
    
    # Main training loop
    for i_episode in range(start_episode, num_episodes):
        training_state.current_episode = i_episode
        
        # Run episode
        episode_reward, episode_max_checkpoint, should_quit = run_episode(
            car, track, agent, episode_mgr, renderer, event_handler,
            i_episode, num_episodes, max_checkpoint_reached
        )
        
        # Check for quit
        if should_quit:
            training_state.max_checkpoint_reached = max_checkpoint_reached
            training_state.best_lap_time = best_lap_time
            training_state.episode_rewards = episode_rewards
            training_state.successful_laps = successful_laps
            training_state.epsilon_history.append(agent.epsilon)
            training_state.checkpoint_progress.append(max_checkpoint_reached)
            save_and_exit(training_state, agent)
            pygame.quit()
            return
        
        # Update tracking
        episode_rewards.append(episode_reward)
        training_state.epsilon_history.append(agent.epsilon)
        training_state.checkpoint_progress.append(episode_max_checkpoint)
        max_checkpoint_reached = max(max_checkpoint_reached, episode_max_checkpoint)
        
        # Check for lap completion
        if hasattr(car, 'race_finished') and car.race_finished:
            if hasattr(car, 'lap_time') and car.lap_time < best_lap_time:
                best_lap_time = car.lap_time
            successful_laps += 1
        
        # Periodic saves
        if (i_episode + 1) % 10 == 0:
            agent.update_target_net()
            training_state.max_checkpoint_reached = max_checkpoint_reached
            training_state.best_lap_time = best_lap_time
            training_state.episode_rewards = episode_rewards
            training_state.successful_laps = successful_laps
            save_checkpoint(agent, training_state, i_episode)
        
        # Milestone backups
        if (i_episode + 1) % 100 == 0:
            backup_path = f'models/car_dqn_model_episode_{i_episode+1}.pth'
            torch.save(agent.policy_net.state_dict(), backup_path)
            print(f"Backup model saved: {backup_path}")
            
            backup_state_path = f'configs/training_state_episode_{i_episode+1}.json'
            training_state.save_file = backup_state_path
            training_state.max_checkpoint_reached = max_checkpoint_reached
            training_state.best_lap_time = best_lap_time
            training_state.episode_rewards = episode_rewards
            training_state.successful_laps = successful_laps
            training_state.save_state(agent)
            training_state.save_file = 'configs/training_state.json'
            print(f"Backup training state saved: {backup_state_path}")
    
    # Training complete
    training_state.current_episode = num_episodes
    training_state.max_checkpoint_reached = max_checkpoint_reached
    training_state.best_lap_time = best_lap_time
    training_state.episode_rewards = episode_rewards
    training_state.successful_laps = successful_laps
    training_state.save_state(agent)
    
    print("\n=== Training Complete! ===")
    torch.save(agent.policy_net.state_dict(), model_path)
    print(f"Final model saved as '{model_path}'")
    
    training_state.print_session_summary()
    
    final_backup_path = f'models/car_dqn_model_FINAL.pth'
    torch.save(agent.policy_net.state_dict(), final_backup_path)
    print(f"Final backup saved as '{final_backup_path}'")
    
    pygame.quit()


if __name__ == "__main__":
    main()