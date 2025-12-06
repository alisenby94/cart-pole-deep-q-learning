import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from agent import DQNAgent
import os
from datetime import datetime


def train_agent(
    n_episodes: int = 1000,
    max_steps: int = 500,
    learning_rate: float = 0.001,
    discount_factor: float = 0.99,
    epsilon: float = 1.0,
    epsilon_decay: float = 0.995,
    epsilon_min: float = 0.01,
    buffer_capacity: int = 10000,
    batch_size: int = 64,
    target_update_freq: int = 100,
    render: bool = False,
    save_path: str = 'models'
):
    # Create environment
    if render:
        env = gym.make('CartPole-v1', render_mode='human')
    else:
        env = gym.make('CartPole-v1')
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    # Create DQN agent
    agent = DQNAgent(
        state_dim=state_dim,
        n_actions=n_actions,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        target_update_freq=target_update_freq
    )
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    losses = []
    
    print("Starting DQN training...")
    print(f"Device: {agent.device}")
    print(f"Episodes: {n_episodes}, Max Steps: {max_steps}")
    print(f"Learning Rate: {learning_rate}, Discount Factor: {discount_factor}")
    print(f"Initial Epsilon: {epsilon}, Epsilon Decay: {epsilon_decay}, Min Epsilon: {epsilon_min}")
    print(f"Buffer Capacity: {buffer_capacity}, Batch Size: {batch_size}")
    print(f"Target Update Frequency: {target_update_freq}")
    print("-" * 80)
    
    last_episode_data = []  # Store data for last episode to get final 5 timesteps
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        
        total_reward = 0
        steps = 0
        episode_loss = []
        episode_data = []  # Store all timesteps for this episode
        
        for step in range(max_steps):
            # Select action
            action = agent.get_action(state, training=True)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store episode data (we'll save last episode's last 5 timesteps at the end)
            episode_data.append({
                'step': step,
                'state': state.copy(),
                'action': action,
                'reward': reward
            })
            
            # Save Q-value snapshot for first 5 timesteps of first episode
            if episode == 0 and step < 5:
                agent.save_q_value_snapshot(episode, step, state, action, reward)
            
            # Store transition in replay buffer
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update agent (train on batch from replay buffer)
            loss = agent.update()
            if loss is not None:
                episode_loss.append(loss)
            
            # Update state
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Store this episode's data as potential last episode
        last_episode_data = episode_data
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Record metrics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        if episode_loss:
            losses.append(np.mean(episode_loss))
        
        # Print progress
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_length = np.mean(episode_lengths[-50:])
            avg_loss = np.mean(losses[-50:]) if losses else 0
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Avg Reward (last 50): {avg_reward:.2f} | "
                  f"Avg Length (last 50): {avg_length:.2f} | "
                  f"Avg Loss: {avg_loss:.4f} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Buffer: {len(agent.replay_buffer)}/{buffer_capacity}")
        
        # Check if solved (average reward of 195 over 100 consecutive episodes)
        if len(episode_rewards) >= 100:
            avg_reward_100 = np.mean(episode_rewards[-100:])
            if avg_reward_100 >= 195.0:
                print(f"\nEnvironment solved in {episode + 1} episodes!")
                print(f"Average reward over last 100 episodes: {avg_reward_100:.2f}")
                break
    
    env.close()
    
    # Save Q-value snapshots for last 5 timesteps of final episode
    if len(last_episode_data) >= 5:
        final_episode = len(episode_rewards) - 1
        for i in range(max(0, len(last_episode_data) - 5), len(last_episode_data)):
            data = last_episode_data[i]
            agent.save_q_value_snapshot(
                final_episode, 
                data['step'], 
                data['state'], 
                data['action'], 
                data['reward']
            )
    
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Save agent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(save_path, f'cartpole_dqn_agent_{timestamp}.pth')
    agent.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Plot training progress
    plot_training_progress(episode_rewards, episode_lengths, losses, save_path, timestamp)
    
    # Save Q-value snapshots to file for report
    save_q_value_snapshots(agent.q_value_history, save_path, timestamp)
    
    return agent, episode_rewards, episode_lengths


def save_q_value_snapshots(q_value_history, save_path, timestamp):
    """Save Q-value snapshots from DQN agent."""
    snapshot_path = os.path.join(save_path, f'q_value_snapshots_{timestamp}.txt')
    
    with open(snapshot_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DQN Q-VALUE SNAPSHOTS FOR REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # First 5 timesteps of episode 1
        f.write("FIRST 5 TIMESTEPS OF EPISODE 1:\n")
        f.write("-" * 80 + "\n")
        first_episode_snapshots = [s for s in q_value_history if s['episode'] == 0]
        for snapshot in first_episode_snapshots[:5]:
            f.write(f"\nTimestep {snapshot['timestep']}:\n")
            f.write(f"  State (continuous): {snapshot['state']}\n")
            f.write(f"  Action taken: {snapshot['action']}\n")
            f.write(f"  Reward: {snapshot['reward']}\n")
            f.write(f"  Q-values for this state: {snapshot['q_values_for_state']}\n")
            f.write(f"  Epsilon: {snapshot['epsilon']:.4f}\n")
        
        # Last 5 timesteps of last episode
        f.write("\n" + "=" * 80 + "\n")
        f.write("LAST 5 TIMESTEPS OF FINAL EPISODE:\n")
        f.write("-" * 80 + "\n")
        if q_value_history:
            last_episode_num = max([s['episode'] for s in q_value_history])
            last_episode_snapshots = [s for s in q_value_history if s['episode'] == last_episode_num]
            for snapshot in last_episode_snapshots[-5:]:
                f.write(f"\nTimestep {snapshot['timestep']}:\n")
                f.write(f"  State (continuous): {snapshot['state']}\n")
                f.write(f"  Action taken: {snapshot['action']}\n")
                f.write(f"  Reward: {snapshot['reward']}\n")
                f.write(f"  Q-values for this state: {snapshot['q_values_for_state']}\n")
                f.write(f"  Epsilon: {snapshot['epsilon']:.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"Q-value snapshots saved to: {snapshot_path}")
    
    # Also save as JSON for easier programmatic access
    import json
    json_path = os.path.join(save_path, f'q_value_snapshots_{timestamp}.json')
    
    with open(json_path, 'w') as f:
        json.dump(q_value_history, f, indent=2)
    print(f"Q-value snapshots (JSON) saved to: {json_path}")


def plot_training_progress(rewards, lengths, losses, save_path, timestamp):
    """Plot training metrics for DQN agent."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14))
    
    # Plot rewards
    ax1.plot(rewards, alpha=0.6, color='blue', linewidth=0.5)
    if len(rewards) >= 100:
        # Plot moving average
        moving_avg = np.convolve(rewards, np.ones(100)/100, mode='valid')
        ax1.plot(range(99, len(rewards)), moving_avg, color='red', linewidth=2, label='100-episode moving average')
        ax1.axhline(y=195, color='green', linestyle='--', label='Solved threshold (195)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('DQN Training Progress: Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot episode lengths
    ax2.plot(lengths, alpha=0.6, color='blue', linewidth=0.5)
    if len(lengths) >= 100:
        moving_avg_length = np.convolve(lengths, np.ones(100)/100, mode='valid')
        ax2.plot(range(99, len(lengths)), moving_avg_length, color='red', linewidth=2, label='100-episode moving average')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Length')
    ax2.set_title('DQN Training Progress: Episode Lengths')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot loss
    if losses:
        ax3.plot(losses, alpha=0.6, color='purple', linewidth=0.5)
        if len(losses) >= 50:
            moving_avg_loss = np.convolve(losses, np.ones(50)/50, mode='valid')
            ax3.plot(range(49, len(losses)), moving_avg_loss, color='red', linewidth=2, label='50-episode moving average')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Average Loss')
        ax3.set_title('DQN Training Progress: Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_path, f'dqn_training_progress_{timestamp}.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Training plot saved to: {plot_path}")
    plt.close()


if __name__ == "__main__":
    # Train DQN agent with default parameters
    agent, rewards, lengths = train_agent(
        n_episodes=1000,
        max_steps=500,
        learning_rate=0.001,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        buffer_capacity=10000,
        batch_size=64,
        target_update_freq=100,
        render=False,
        save_path='models'
    )
    
    print("\nDQN Training completed!")
    print(f"Total episodes: {len(rewards)}")
    print(f"Final average reward (last 100 episodes): {np.mean(rewards[-100:]):.2f}")
    print(f"Best episode reward: {max(rewards):.2f}")
