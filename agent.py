import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Tuple, Optional


class DQNNetwork(nn.Module):
    """Deep Q-Network with fully connected layers."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Tuple[int, ...] = (128, 128)):
        super(DQNNetwork, self).__init__()
        
        layers = []
        prev_dim = state_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(state)


class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions."""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        """Add a transition to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample a batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.bool_)
        )
    
    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network agent for reinforcement learning."""
    
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        learning_rate: float = 0.001,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        hidden_dims: Tuple[int, ...] = (128, 128),
        device: Optional[str] = None
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.hidden_dims = hidden_dims
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Create policy and target networks
        self.policy_net = DQNNetwork(state_dim, n_actions, hidden_dims).to(self.device)
        self.target_net = DQNNetwork(state_dim, n_actions, hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer and loss function
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # Training statistics
        self.update_count = 0
        self.q_value_history = []
    
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select an action using epsilon-greedy policy."""
        if training and np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploitation: best action from policy network
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool) -> None:
        """Store a transition in the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Optional[float]:
        """
        Update the policy network using a batch from replay buffer.
        Returns the loss value if update was performed, None otherwise.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values using target network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones.float()) * self.discount_factor * next_q_values
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        return loss.item()
    
    def save_q_value_snapshot(self, episode: int, timestep: int, state: np.ndarray,
                              action: int, reward: float) -> None:
        """Save Q-values for a given state for analysis."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]
        
        snapshot = {
            'episode': episode,
            'timestep': timestep,
            'state': state.tolist(),
            'action': action,
            'reward': reward,
            'q_values_for_state': q_values.tolist(),
            'epsilon': self.epsilon
        }
        self.q_value_history.append(snapshot)
    
    def decay_epsilon(self) -> None:
        """Decay epsilon for exploration-exploitation trade-off."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str) -> None:
        """Save the agent's state to a file."""
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'state_dim': self.state_dim,
            'n_actions': self.n_actions,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min,
            'batch_size': self.batch_size,
            'target_update_freq': self.target_update_freq,
            'hidden_dims': self.hidden_dims,
            'update_count': self.update_count
        }
        torch.save(checkpoint, filepath)
    
    def load(self, filepath: str) -> None:
        """Load the agent's state from a file."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        # Restore parameters
        self.state_dim = checkpoint['state_dim']
        self.n_actions = checkpoint['n_actions']
        self.learning_rate = checkpoint['learning_rate']
        self.discount_factor = checkpoint['discount_factor']
        self.epsilon = checkpoint['epsilon']
        self.epsilon_decay = checkpoint['epsilon_decay']
        self.epsilon_min = checkpoint['epsilon_min']
        self.batch_size = checkpoint['batch_size']
        self.target_update_freq = checkpoint['target_update_freq']
        self.hidden_dims = checkpoint['hidden_dims']
        self.update_count = checkpoint['update_count']
        
        # Recreate networks with correct dimensions
        self.policy_net = DQNNetwork(self.state_dim, self.n_actions, self.hidden_dims).to(self.device)
        self.target_net = DQNNetwork(self.state_dim, self.n_actions, self.hidden_dims).to(self.device)
        
        # Load state dictionaries
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        
        # Recreate optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.target_net.eval()
