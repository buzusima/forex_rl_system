# src/rl/ppo_agent.py - PPO Agent สำหรับ Forex Trading
"""
ไฟล์นี้สร้าง PPO Agent สำหรับเทรด XAUUSD/Forex
ใช้ประสบการณ์จากการเทรดจริงมากกว่า 15 ปี
แก้ไขไฟล์นี้เมื่อ: ต้องการปรับ network architecture หรือ hyperparameters
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import pickle
import os
from collections import deque

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PPOConfig:
    """PPO Hyperparameters - ปรับจากประสบการณ์จริงในตลาด Forex"""
    
    # Network Architecture
    hidden_sizes: List[int] = (256, 128, 64)  # ลดขนาดเพื่อป้องกัน overfitting
    activation: str = "tanh"  # tanh ทำงานดีกว่า ReLU ในตลาดการเงิน
    use_lstm: bool = True     # LSTM สำคัญสำหรับ sequential data
    lstm_hidden: int = 128    # LSTM hidden size
    
    # PPO Parameters
    lr_actor: float = 3e-4    # Learning rate สำหรับ actor
    lr_critic: float = 1e-3   # Learning rate สำหรับ critic (สูงกว่า actor)
    gamma: float = 0.99       # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    clip_ratio: float = 0.2   # PPO clip ratio
    entropy_coef: float = 0.01 # Entropy coefficient (encourage exploration)
    value_coef: float = 0.5   # Value function coefficient
    max_grad_norm: float = 0.5 # Gradient clipping
    
    # Training Parameters
    batch_size: int = 64      # Batch size (เล็กสำหรับ Forex)
    n_epochs: int = 10        # PPO epochs per update
    buffer_size: int = 2048   # Experience buffer size
    normalize_advantages: bool = True
    
    # Market-specific Parameters
    use_market_regime: bool = True    # ใช้ market regime detection
    volatility_adjustment: bool = True # ปรับ learning rate ตาม volatility
    
class ActorCriticNetwork(nn.Module):
    """Actor-Critic Network สำหรับ PPO"""
    
    def __init__(self, 
                 obs_dim: int, 
                 action_dim: int, 
                 config: PPOConfig,
                 device: torch.device):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        self.device = device
        
        # Activation function
        if config.activation == "relu":
            self.activation = nn.ReLU()
        elif config.activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ELU()
        
        # Input normalization
        self.input_norm = nn.LayerNorm(obs_dim)
        
        # LSTM layer (ถ้าใช้)
        if config.use_lstm:
            self.lstm = nn.LSTM(
                input_size=obs_dim,
                hidden_size=config.lstm_hidden,
                batch_first=True
            )
            current_size = config.lstm_hidden
        else:
            self.lstm = None
            current_size = obs_dim
        
        # Shared layers
        self.shared_layers = nn.ModuleList()
        for hidden_size in config.hidden_sizes[:-1]:
            self.shared_layers.append(nn.Linear(current_size, hidden_size))
            self.shared_layers.append(nn.LayerNorm(hidden_size))
            current_size = hidden_size
        
        # Actor head (policy network)
        self.actor_layers = nn.ModuleList([
            nn.Linear(current_size, config.hidden_sizes[-1]),
            nn.LayerNorm(config.hidden_sizes[-1]),
            nn.Linear(config.hidden_sizes[-1], action_dim)
        ])
        
        # Critic head (value network)
        self.critic_layers = nn.ModuleList([
            nn.Linear(current_size, config.hidden_sizes[-1]),
            nn.LayerNorm(config.hidden_sizes[-1]),
            nn.Linear(config.hidden_sizes[-1], 1)
        ])
        
        # Initialize weights
        self._init_weights()
        
        # LSTM hidden state
        self.lstm_hidden = None
        
    def _init_weights(self):
        """Initialize network weights - สำคัญสำหรับการเรียนรู้ที่เสถียร"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization สำหรับ tanh, He initialization สำหรับ ReLU
                if self.config.activation == "tanh":
                    nn.init.xavier_uniform_(module.weight)
                else:
                    nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(self, obs: torch.Tensor, hidden: Optional[Tuple] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """Forward pass"""
        batch_size = obs.shape[0]
        
        # Input normalization
        x = self.input_norm(obs)
        
        # LSTM layer
        if self.lstm is not None:
            if len(x.shape) == 2:
                x = x.unsqueeze(1)  # Add sequence dimension
            
            if hidden is None:
                h_0 = torch.zeros(1, batch_size, self.config.lstm_hidden, device=self.device)
                c_0 = torch.zeros(1, batch_size, self.config.lstm_hidden, device=self.device)
                hidden = (h_0, c_0)
            
            x, new_hidden = self.lstm(x, hidden)
            x = x.squeeze(1)  # Remove sequence dimension
        else:
            new_hidden = None
        
        # Shared layers
        for i in range(0, len(self.shared_layers), 2):
            x = self.shared_layers[i](x)
            x = self.shared_layers[i+1](x)
            x = self.activation(x)
        
        # Actor branch
        actor_x = x
        for i in range(0, len(self.actor_layers) - 1, 2):
            actor_x = self.actor_layers[i](actor_x)
            actor_x = self.actor_layers[i+1](actor_x)
            actor_x = self.activation(actor_x)
        
        logits = self.actor_layers[-1](actor_x)
        
        # Critic branch
        critic_x = x
        for i in range(0, len(self.critic_layers) - 1, 2):
            critic_x = self.critic_layers[i](critic_x)
            critic_x = self.critic_layers[i+1](critic_x)
            critic_x = self.activation(critic_x)
        
        value = self.critic_layers[-1](critic_x)
        
        return logits, value.squeeze(-1), new_hidden
    
    def get_action_and_value(self, obs: torch.Tensor, action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log prob, entropy, and value"""
        logits, value, _ = self.forward(obs)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
        
        log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        
        return action, log_prob, entropy, value

class ExperienceBuffer:
    """Experience buffer สำหรับ PPO"""
    
    def __init__(self, buffer_size: int, obs_dim: int, device: torch.device):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.device = device
        self.clear()
    
    def clear(self):
        """Clear buffer"""
        self.observations = torch.zeros((self.buffer_size, self.obs_dim), device=self.device)
        self.actions = torch.zeros(self.buffer_size, dtype=torch.long, device=self.device)
        self.rewards = torch.zeros(self.buffer_size, device=self.device)
        self.values = torch.zeros(self.buffer_size, device=self.device)
        self.log_probs = torch.zeros(self.buffer_size, device=self.device)
        self.dones = torch.zeros(self.buffer_size, dtype=torch.bool, device=self.device)
        self.advantages = torch.zeros(self.buffer_size, device=self.device)
        self.returns = torch.zeros(self.buffer_size, device=self.device)
        self.ptr = 0
        self.size = 0
    
    def store(self, obs: np.ndarray, action: int, reward: float, value: float, log_prob: float, done: bool):
        """Store experience"""
        if self.ptr >= self.buffer_size:
            logger.warning("Buffer overflow, clearing buffer")
            self.clear()
        
        self.observations[self.ptr] = torch.tensor(obs, device=self.device, dtype=torch.float32)
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr += 1
        self.size = min(self.size + 1, self.buffer_size)
    
    def finish_path(self, last_value: float, config: PPOConfig):
        """Calculate advantages and returns using GAE"""
        path_slice = slice(0, self.ptr)
        
        # Calculate returns and advantages
        rewards = self.rewards[path_slice]
        values = self.values[path_slice]
        dones = self.dones[path_slice]
        
        # Add last value
        values_with_last = torch.cat([values, torch.tensor([last_value], device=self.device)])
        
        # Calculate GAE
        gae = 0
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_non_terminal = 1.0 - float(dones[i])
                next_value = last_value
            else:
                next_non_terminal = 1.0 - float(dones[i])
                next_value = values_with_last[i + 1]
            
            delta = rewards[i] + config.gamma * next_value * next_non_terminal - values[i]
            gae = delta + config.gamma * config.gae_lambda * next_non_terminal * gae
            self.advantages[i] = gae
        
        # Calculate returns
        self.returns[path_slice] = self.advantages[path_slice] + values
    
    def get_batch(self, config: PPOConfig) -> Dict[str, torch.Tensor]:
        """Get training batch"""
        assert self.ptr == self.buffer_size, "Buffer not full"
        
        # Normalize advantages
        if config.normalize_advantages:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)
        
        return {
            'observations': self.observations,
            'actions': self.actions,
            'returns': self.returns,
            'advantages': self.advantages,
            'log_probs': self.log_probs,
            'values': self.values
        }

class PPOAgent:
    """PPO Agent สำหรับ Forex Trading"""
    
    def __init__(self, 
                 obs_dim: int, 
                 action_dim: int, 
                 config: PPOConfig = None,
                 device: str = "auto"):
        
        self.config = config or PPOConfig()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Network
        self.network = ActorCriticNetwork(
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=self.config,
            device=self.device
        ).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            [p for n, p in self.network.named_parameters() if 'actor' in n or 'shared' in n or 'lstm' in n],
            lr=self.config.lr_actor
        )
        self.critic_optimizer = optim.Adam(
            [p for n, p in self.network.named_parameters() if 'critic' in n or 'shared' in n],
            lr=self.config.lr_critic
        )
        
        # Experience buffer
        self.buffer = ExperienceBuffer(
            buffer_size=self.config.buffer_size,
            obs_dim=obs_dim,
            device=self.device
        )
        
        # Training statistics
        self.training_stats = {
            'episodes': 0,
            'total_steps': 0,
            'actor_loss': deque(maxlen=100),
            'critic_loss': deque(maxlen=100),
            'entropy': deque(maxlen=100),
            'kl_divergence': deque(maxlen=100),
            'explained_variance': deque(maxlen=100)
        }
    
    def get_action(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        """Get action from observation"""
        self.network.eval()
        
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, device=self.device, dtype=torch.float32).unsqueeze(0)
            action, log_prob, entropy, value = self.network.get_action_and_value(obs_tensor)
            
            if deterministic:
                # Use argmax for deterministic action
                logits, _, _ = self.network.forward(obs_tensor)
                action = torch.argmax(logits, dim=-1)
                log_prob = torch.log(F.softmax(logits, dim=-1)[0, action])
        
        return action.item(), log_prob.item(), value.item()
    
    def store_experience(self, obs: np.ndarray, action: int, reward: float, value: float, log_prob: float, done: bool):
        """Store experience in buffer"""
        self.buffer.store(obs, action, reward, value, log_prob, done)
    
    def update(self) -> Dict[str, float]:
        """Update policy using PPO"""
        if self.buffer.ptr < self.config.buffer_size:
            logger.warning(f"Buffer not full: {self.buffer.ptr}/{self.config.buffer_size}")
            return {}
        
        # Get the last value for GAE calculation
        with torch.no_grad():
            last_obs = self.buffer.observations[self.buffer.ptr - 1].unsqueeze(0)
            _, _, _, last_value = self.network.get_action_and_value(last_obs)
        
        # Finish path and calculate advantages
        self.buffer.finish_path(last_value.item(), self.config)
        
        # Get training batch
        batch = self.buffer.get_batch(self.config)
        
        # Training statistics
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        total_kl = 0
        
        self.network.train()
        
        # PPO training epochs
        for epoch in range(self.config.n_epochs):
            # Create mini-batches
            indices = torch.randperm(self.config.buffer_size, device=self.device)
            
            for start in range(0, self.config.buffer_size, self.config.batch_size):
                end = start + self.config.batch_size
                if end > self.config.buffer_size:
                    continue
                
                batch_indices = indices[start:end]
                
                # Get current policy outputs
                _, new_log_probs, entropy, new_values = self.network.get_action_and_value(
                    batch['observations'][batch_indices],
                    batch['actions'][batch_indices]
                )
                
                # Calculate ratio
                log_ratio = new_log_probs - batch['log_probs'][batch_indices]
                ratio = torch.exp(log_ratio)
                
                # Actor loss (PPO clip)
                advantages = batch['advantages'][batch_indices]
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio)
                actor_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                
                # Critic loss
                returns = batch['returns'][batch_indices]
                critic_loss = F.mse_loss(new_values, returns)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = actor_loss + self.config.value_coef * critic_loss + self.config.entropy_coef * entropy_loss
                
                # Update networks
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                # Statistics
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                
                # KL divergence (for monitoring)
                with torch.no_grad():
                    kl = (batch['log_probs'][batch_indices] - new_log_probs).mean()
                    total_kl += kl.item()
        
        # Calculate explained variance
        with torch.no_grad():
            explained_var = 1 - torch.var(batch['returns'] - batch['values']) / torch.var(batch['returns'])
        
        # Update statistics
        n_updates = self.config.n_epochs * (self.config.buffer_size // self.config.batch_size)
        stats = {
            'actor_loss': total_actor_loss / n_updates,
            'critic_loss': total_critic_loss / n_updates,
            'entropy': total_entropy / n_updates,
            'kl_divergence': total_kl / n_updates,
            'explained_variance': explained_var.item()
        }
        
        # Store statistics
        for key, value in stats.items():
            self.training_stats[key].append(value)
        
        # Clear buffer
        self.buffer.clear()
        
        return stats
    
    def save(self, path: str):
        """Save model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'network_state_dict': self.network.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'config': self.config,
            'training_stats': {k: list(v) if hasattr(v, '__iter__') and not isinstance(v, str) else v 
                   for k, v in self.training_stats.items()},
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)        
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        # Load statistics
        for key, value in checkpoint['training_stats'].items():
            if isinstance(value, list):
                self.training_stats[key] = deque(value, maxlen=100)
            else:
                self.training_stats[key] = value        
                logger.info(f"Model loaded from {path}")
    
    def get_training_stats(self) -> Dict[str, float]:
        """Get recent training statistics"""
        stats = {}
        for key, values in self.training_stats.items():
            if values and key != 'episodes' and key != 'total_steps':
                stats[f'{key}_mean'] = np.mean(values)
                stats[f'{key}_recent'] = values[-1] if values else 0.0
        
        stats['episodes'] = self.training_stats['episodes']
        stats['total_steps'] = self.training_stats['total_steps']
        
        return stats

def create_ppo_agent(obs_dim: int, action_dim: int, config: PPOConfig = None) -> PPOAgent:
    """Create PPO Agent instance"""
    return PPOAgent(obs_dim, action_dim, config)

if __name__ == "__main__":
    # Test PPO Agent
    print("🧪 Testing PPO Agent...")
    
    # Create dummy environment dimensions
    obs_dim = 243  # From your environment
    action_dim = 3  # Hold, Buy, Sell
    
    # Create agent
    agent = create_ppo_agent(obs_dim, action_dim)
    
    print(f"✅ PPO Agent created")
    print(f"   Observation dim: {obs_dim}")
    print(f"   Action dim: {action_dim}")
    print(f"   Network parameters: {sum(p.numel() for p in agent.network.parameters()):,}")
    print(f"   Device: {agent.device}")
    
    # Test action selection
    dummy_obs = np.random.randn(obs_dim)
    action, log_prob, value = agent.get_action(dummy_obs)
    
    print(f"\n🎯 Action test:")
    print(f"   Action: {action}")
    print(f"   Log prob: {log_prob:.4f}")
    print(f"   Value: {value:.4f}")
    
    print(f"\n🎉 PPO Agent test completed!")