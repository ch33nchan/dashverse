"""Reinforcement Learning component for improving attribute extraction accuracy."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from collections import deque, defaultdict
import random
import json
from pathlib import Path

from .base import PipelineStage, CharacterAttributes

class AttributeQNetwork(nn.Module):
    """Q-Network for learning optimal attribute extraction strategies."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class ExperienceReplay:
    """Experience replay buffer for RL training."""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample batch of experiences."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)

class RLOptimizer(PipelineStage):
    """Reinforcement Learning optimizer for attribute extraction."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("RLOptimizer", config)
        
        # Action space: different extraction strategies
        self.actions = {
            0: 'conservative_clip',  # High confidence threshold for CLIP
            1: 'aggressive_clip',    # Low confidence threshold for CLIP
            2: 'tag_priority',       # Prioritize tag-based extraction
            3: 'visual_priority',    # Prioritize visual extraction
            4: 'ensemble_weighted',  # Weighted ensemble of methods
            5: 'uncertainty_aware',  # Focus on uncertain predictions
        }
        
        # RL Configuration
        if config:
            self.state_dim = config.get('state_dim', 128)
            self.action_dim = len(self.actions)
            self.hidden_dim = config.get('hidden_dim', 256)
            self.learning_rate = config.get('learning_rate', 0.001)
            self.epsilon = config.get('epsilon', 0.1)
            self.epsilon_decay = config.get('epsilon_decay', 0.995)
            self.epsilon_min = config.get('epsilon_min', 0.01)
            self.gamma = config.get('gamma', 0.95)
            self.batch_size = config.get('batch_size', 32)
            self.update_frequency = config.get('update_frequency', 100)
            self.model_path = Path(config.get('model_path', './models/rl_optimizer.pth'))
        else:
            self.state_dim = 128
            self.action_dim = len(self.actions)
            self.hidden_dim = 256
            self.learning_rate = 0.001
            self.epsilon = 0.1
            self.epsilon_decay = 0.995
            self.epsilon_min = 0.01
            self.gamma = 0.95
            self.batch_size = 32
            self.update_frequency = 100
            self.model_path = Path('./models/rl_optimizer.pth')

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize networks
        self.q_network = AttributeQNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.target_network = AttributeQNetwork(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Experience replay
        self.memory = ExperienceReplay()
        
        # Training metrics
        self.training_step = 0
        self.episode_rewards = []
        self.attribute_accuracy = defaultdict(list)
        
        # Load saved model if exists
        self.load_model()
    
    def _create_state_vector(self, clip_results: CharacterAttributes, tag_results: CharacterAttributes, 
                           clip_confidences: Dict[str, float], tag_confidences: Dict[str, float]) -> torch.Tensor:
        """Create state vector from extraction results."""
        state_features = []
        
        # CLIP confidence features
        clip_conf_values = list(clip_confidences.values())
        state_features.extend(clip_conf_values[:16])  # Pad/truncate to 16
        if len(clip_conf_values) < 16:
            state_features.extend([0.0] * (16 - len(clip_conf_values)))
        
        # Tag confidence features
        tag_conf_values = list(tag_confidences.values())
        state_features.extend(tag_conf_values[:16])  # Pad/truncate to 16
        if len(tag_conf_values) < 16:
            state_features.extend([0.0] * (16 - len(tag_conf_values)))
        
        # Agreement features (how much CLIP and tags agree)
        agreement_features = []
        attributes = ['age', 'gender', 'hair_color', 'hair_length', 'hair_style', 
                     'eye_color', 'body_type', 'dress', 'facial_expression']
        
        for attr in attributes:
            clip_val = getattr(clip_results, attr, None)
            tag_val = getattr(tag_results, attr, None)
            
            if clip_val and tag_val:
                agreement = 1.0 if clip_val == tag_val else 0.0
            elif clip_val or tag_val:
                agreement = 0.5  # Partial information
            else:
                agreement = 0.0  # No information
            
            agreement_features.append(agreement)
        
        state_features.extend(agreement_features)
        
        # Pad to state_dim
        while len(state_features) < self.state_dim:
            state_features.append(0.0)
        
        return torch.tensor(state_features[:self.state_dim], dtype=torch.float32).to(self.device)
    
    def _calculate_reward(self, predicted_attrs: CharacterAttributes, ground_truth: Optional[CharacterAttributes] = None) -> float:
        """Calculate reward based on prediction quality."""
        if ground_truth is None:
            # Use heuristic reward based on confidence and completeness
            completeness = sum(1 for attr in ['age', 'gender', 'hair_color', 'hair_length', 
                             'hair_style', 'eye_color', 'body_type', 'dress'] 
                             if getattr(predicted_attrs, attr, None) is not None)
            
            confidence = predicted_attrs.confidence_score or 0.0
            
            # Reward completeness and confidence
            reward = (completeness / 8.0) * 0.7 + confidence * 0.3
            
            # Bonus for having multiple attributes
            if completeness >= 5:
                reward += 0.2
            
            return reward
        else:
            # Calculate accuracy-based reward
            correct = 0
            total = 0
            
            attributes = ['age', 'gender', 'hair_color', 'hair_length', 'hair_style', 
                         'eye_color', 'body_type', 'dress', 'facial_expression']
            
            for attr in attributes:
                pred_val = getattr(predicted_attrs, attr, None)
                true_val = getattr(ground_truth, attr, None)
                
                if true_val is not None:
                    total += 1
                    if pred_val == true_val:
                        correct += 1
            
            accuracy = correct / total if total > 0 else 0.0
            return accuracy
    
    def _apply_action(self, action: int, clip_results: CharacterAttributes, tag_results: CharacterAttributes,
                     clip_confidences: Dict[str, float], tag_confidences: Dict[str, float]) -> CharacterAttributes:
        """Apply the selected action to combine extraction results."""
        action_name = self.actions[action]
        final_attrs = CharacterAttributes()
        
        attributes = ['age', 'gender', 'hair_color', 'hair_length', 'hair_style', 
                     'eye_color', 'body_type', 'dress', 'facial_expression']
        
        if action_name == 'conservative_clip':
            # Use CLIP only with high confidence
            for attr in attributes:
                clip_val = getattr(clip_results, attr, None)
                clip_conf = clip_confidences.get(attr, 0.0)
                
                if clip_conf > 0.7:
                    setattr(final_attrs, attr, clip_val)
                else:
                    setattr(final_attrs, attr, getattr(tag_results, attr, None))
        
        elif action_name == 'aggressive_clip':
            # Use CLIP with lower confidence threshold
            for attr in attributes:
                clip_val = getattr(clip_results, attr, None)
                clip_conf = clip_confidences.get(attr, 0.0)
                
                if clip_conf > 0.3:
                    setattr(final_attrs, attr, clip_val)
                else:
                    setattr(final_attrs, attr, getattr(tag_results, attr, None))
        
        elif action_name == 'tag_priority':
            # Prioritize tag-based results
            for attr in attributes:
                tag_val = getattr(tag_results, attr, None)
                if tag_val:
                    setattr(final_attrs, attr, tag_val)
                else:
                    setattr(final_attrs, attr, getattr(clip_results, attr, None))
        
        elif action_name == 'visual_priority':
            # Prioritize visual (CLIP) results
            for attr in attributes:
                clip_val = getattr(clip_results, attr, None)
                if clip_val:
                    setattr(final_attrs, attr, clip_val)
                else:
                    setattr(final_attrs, attr, getattr(tag_results, attr, None))
        
        elif action_name == 'ensemble_weighted':
            # Weighted combination based on confidence
            for attr in attributes:
                clip_val = getattr(clip_results, attr, None)
                tag_val = getattr(tag_results, attr, None)
                clip_conf = clip_confidences.get(attr, 0.0)
                tag_conf = tag_confidences.get(attr, 0.0)
                
                if clip_val and tag_val:
                    # Choose based on confidence
                    if clip_conf > tag_conf:
                        setattr(final_attrs, attr, clip_val)
                    else:
                        setattr(final_attrs, attr, tag_val)
                elif clip_val:
                    setattr(final_attrs, attr, clip_val)
                elif tag_val:
                    setattr(final_attrs, attr, tag_val)
        
        elif action_name == 'uncertainty_aware':
            # Focus on attributes with high uncertainty
            for attr in attributes:
                clip_val = getattr(clip_results, attr, None)
                tag_val = getattr(tag_results, attr, None)
                clip_conf = clip_confidences.get(attr, 0.0)
                tag_conf = tag_confidences.get(attr, 0.0)
                
                # If both methods agree, use the result
                if clip_val == tag_val and clip_val is not None:
                    setattr(final_attrs, attr, clip_val)
                # If they disagree, use the more confident one
                elif clip_conf > tag_conf:
                    setattr(final_attrs, attr, clip_val)
                else:
                    setattr(final_attrs, attr, tag_val)
        
        # Calculate combined confidence
        all_confidences = list(clip_confidences.values()) + list(tag_confidences.values())
        final_attrs.confidence_score = np.mean(all_confidences) if all_confidences else 0.0
        
        return final_attrs
    
    def select_action(self, state: torch.Tensor, training: bool = True) -> int:
        """Select action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            q_values = self.q_network(state.unsqueeze(0))
            return q_values.argmax().item()
    
    def train_step(self):
        """Perform one training step."""
        if len(self.memory) < self.batch_size:
            return
        
        batch = self.memory.sample(self.batch_size)
        states = torch.stack([exp[0] for exp in batch])
        actions = torch.tensor([exp[1] for exp in batch], dtype=torch.long).to(self.device)
        rewards = torch.tensor([exp[2] for exp in batch], dtype=torch.float32).to(self.device)
        next_states = torch.stack([exp[3] for exp in batch])
        dones = torch.tensor([exp[4] for exp in batch], dtype=torch.bool).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Update target network
        if self.training_step % self.update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.training_step += 1
    
    def process(self, input_data: Any) -> CharacterAttributes:
        """Process input using RL-optimized strategy."""
        if not isinstance(input_data, dict):
            raise ValueError("RLOptimizer expects dict input with 'clip_results', 'tag_results', etc.")
        
        clip_results = input_data.get('clip_results', CharacterAttributes())
        tag_results = input_data.get('tag_results', CharacterAttributes())
        clip_confidences = input_data.get('clip_confidences', {})
        tag_confidences = input_data.get('tag_confidences', {})
        ground_truth = input_data.get('ground_truth', None)
        
        # Create state vector
        state = self._create_state_vector(clip_results, tag_results, clip_confidences, tag_confidences)
        
        # Select action
        action = self.select_action(state, training=True)
        
        # Apply action to get final result
        final_result = self._apply_action(action, clip_results, tag_results, clip_confidences, tag_confidences)
        
        # Calculate reward
        reward = self._calculate_reward(final_result, ground_truth)
        
        # Store experience for training (simplified - in practice you'd need next state)
        # For now, we'll just train on immediate rewards
        if hasattr(self, '_last_state') and hasattr(self, '_last_action'):
            self.memory.push(self._last_state, self._last_action, reward, state, False)
            self.train_step()
        
        self._last_state = state
        self._last_action = action
        
        return final_result
    
    def save_model(self):
        """Save the trained model."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, self.model_path)
        self.logger.info(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load a saved model."""
        if self.model_path.exists():
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.q_network.load_state_dict(checkpoint['q_network'])
                self.target_network.load_state_dict(checkpoint['target_network'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.epsilon = checkpoint.get('epsilon', self.epsilon)
                self.training_step = checkpoint.get('training_step', 0)
                self.logger.info(f"Model loaded from {self.model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load model: {e}")
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data."""
        if not isinstance(input_data, dict):
            return False
        return 'clip_results' in input_data and 'tag_results' in input_data