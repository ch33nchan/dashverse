import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sqlite3
import pickle
from typing import List, Tuple, Dict, Any
from .rl_orchestrator import DecisionTransformer, StateVector, RLOrchestrator
import asyncio
from pathlib import Path
import json

class TrajectoryDataset(Dataset):
    def __init__(self, db_path: str, sequence_length: int = 10):
        self.db_path = db_path
        self.sequence_length = sequence_length
        self.trajectories = self._load_trajectories()
    
    def _load_trajectories(self) -> List[List[Tuple]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            SELECT episode_id, step, state, action, reward, next_state, done
            FROM trajectories
            ORDER BY episode_id, step
        """)
        
        trajectories = {}
        for row in cursor.fetchall():
            episode_id, step, state_blob, action, reward, next_state_blob, done = row
            state = pickle.loads(state_blob)
            next_state = pickle.loads(next_state_blob)
            
            if episode_id not in trajectories:
                trajectories[episode_id] = []
            
            trajectories[episode_id].append((state, action, reward, next_state, bool(done)))
        
        conn.close()
        return list(trajectories.values())
    
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        trajectory = self.trajectories[idx]
        
        if len(trajectory) <= self.sequence_length:
            padded_trajectory = trajectory + [(trajectory[-1][0], 0, 0.0, trajectory[-1][3], True)] * (self.sequence_length - len(trajectory))
        else:
            start_idx = np.random.randint(0, len(trajectory) - self.sequence_length + 1)
            padded_trajectory = trajectory[start_idx:start_idx + self.sequence_length]
        
        states = []
        actions = []
        rewards = []
        returns_to_go = []
        
        total_return = sum([step[2] for step in padded_trajectory])
        
        for i, (state, action, reward, next_state, done) in enumerate(padded_trajectory):
            states.append(state.to_tensor())
            actions.append(action)
            rewards.append(reward)
            
            remaining_return = sum([step[2] for step in padded_trajectory[i:]])
            returns_to_go.append(remaining_return)
        
        return {
            'states': torch.stack(states),
            'actions': torch.tensor(actions, dtype=torch.long),
            'rewards': torch.tensor(rewards, dtype=torch.float32),
            'returns_to_go': torch.tensor(returns_to_go, dtype=torch.float32)
        }

class ExpertPolicyGenerator:
    def __init__(self, orchestrator: RLOrchestrator):
        self.orchestrator = orchestrator
    
    async def generate_cheap_first_policy(self, samples: List[Tuple[Any, str, Dict]]) -> List[str]:
        episode_ids = []
        
        for i, (image_data, text_data, ground_truth) in enumerate(samples):
            image_embedding = np.random.randn(768)
            text_embedding = self.orchestrator.state_manager.text_encoder.encode(text_data) if text_data else np.zeros(384)
            
            state = self.orchestrator.state_manager.initialize_state(image_embedding, text_embedding)
            trajectory = []
            extracted_attributes = {}
            total_cost = 0.0
            confidences = []
            
            cheap_actions = [0, 3, 4, 5, 6, 7, 8]  # detectors and classifiers first
            expensive_actions = [1, 2]  # VLM and LLM last
            
            action_sequence = cheap_actions + expensive_actions
            
            for action in action_sequence:
                if state.remaining_budget <= 0.05:
                    break
                
                if state.action_history_mask[action] == 1:
                    continue
                
                result = await self.orchestrator.action_toolbox.execute_action(action, image_data, text_data)
                
                if result.success:
                    extracted_attributes.update(result.extracted_data)
                    confidences.append(result.confidence)
                    total_cost += result.cost
                
                next_state = self.orchestrator.state_manager.update_state(state, action, result)
                
                reward = self.orchestrator._calculate_reward(extracted_attributes, ground_truth, total_cost, confidences)
                trajectory.append((state, action, reward, next_state, False))
                state = next_state
            
            if trajectory:
                final_reward = self.orchestrator._calculate_reward(extracted_attributes, ground_truth, total_cost, confidences)
                trajectory[-1] = (trajectory[-1][0], trajectory[-1][1], final_reward, trajectory[-1][3], True)
                
                episode_id = f"cheap_first_{i}"
                self.orchestrator.trajectory_collector.save_trajectory(episode_id, trajectory)
                episode_ids.append(episode_id)
        
        return episode_ids
    
    async def generate_text_first_policy(self, samples: List[Tuple[Any, str, Dict]]) -> List[str]:
        episode_ids = []
        
        for i, (image_data, text_data, ground_truth) in enumerate(samples):
            if not text_data:
                continue
            
            image_embedding = np.random.randn(768)
            text_embedding = self.orchestrator.state_manager.text_encoder.encode(text_data)
            
            state = self.orchestrator.state_manager.initialize_state(image_embedding, text_embedding)
            trajectory = []
            extracted_attributes = {}
            total_cost = 0.0
            confidences = []
            
            text_actions = [2]  # text parser first
            other_actions = [0, 1, 3, 4, 5, 6, 7, 8]
            
            action_sequence = text_actions + other_actions
            
            for action in action_sequence:
                if state.remaining_budget <= 0.05:
                    break
                
                if state.action_history_mask[action] == 1:
                    continue
                
                result = await self.orchestrator.action_toolbox.execute_action(action, image_data, text_data)
                
                if result.success:
                    extracted_attributes.update(result.extracted_data)
                    confidences.append(result.confidence)
                    total_cost += result.cost
                
                next_state = self.orchestrator.state_manager.update_state(state, action, result)
                
                reward = self.orchestrator._calculate_reward(extracted_attributes, ground_truth, total_cost, confidences)
                trajectory.append((state, action, reward, next_state, False))
                state = next_state
            
            if trajectory:
                final_reward = self.orchestrator._calculate_reward(extracted_attributes, ground_truth, total_cost, confidences)
                trajectory[-1] = (trajectory[-1][0], trajectory[-1][1], final_reward, trajectory[-1][3], True)
                
                episode_id = f"text_first_{i}"
                self.orchestrator.trajectory_collector.save_trajectory(episode_id, trajectory)
                episode_ids.append(episode_id)
        
        return episode_ids
    
    async def generate_comprehensive_policy(self, samples: List[Tuple[Any, str, Dict]]) -> List[str]:
        episode_ids = []
        
        for i, (image_data, text_data, ground_truth) in enumerate(samples):
            image_embedding = np.random.randn(768)
            text_embedding = self.orchestrator.state_manager.text_encoder.encode(text_data) if text_data else np.zeros(384)
            
            state = self.orchestrator.state_manager.initialize_state(image_embedding, text_embedding)
            trajectory = []
            extracted_attributes = {}
            total_cost = 0.0
            confidences = []
            
            all_actions = list(range(9))  # all tools except flag_ambiguous and finalize
            
            for action in all_actions:
                if state.remaining_budget <= 0.05:
                    break
                
                result = await self.orchestrator.action_toolbox.execute_action(action, image_data, text_data)
                
                if result.success:
                    extracted_attributes.update(result.extracted_data)
                    confidences.append(result.confidence)
                    total_cost += result.cost
                
                next_state = self.orchestrator.state_manager.update_state(state, action, result)
                
                reward = self.orchestrator._calculate_reward(extracted_attributes, ground_truth, total_cost, confidences)
                trajectory.append((state, action, reward, next_state, False))
                state = next_state
            
            if trajectory:
                final_reward = self.orchestrator._calculate_reward(extracted_attributes, ground_truth, total_cost, confidences)
                trajectory[-1] = (trajectory[-1][0], trajectory[-1][1], final_reward, trajectory[-1][3], True)
                
                episode_id = f"comprehensive_{i}"
                self.orchestrator.trajectory_collector.save_trajectory(episode_id, trajectory)
                episode_ids.append(episode_id)
        
        return episode_ids

class DecisionTransformerTrainer:
    def __init__(self, state_dim: int = 1239, action_dim: int = 11, hidden_dim: int = 256, n_layers: int = 3):
        self.model = DecisionTransformer(state_dim, action_dim, hidden_dim, n_layers)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            states = batch['states']
            actions = batch['actions']
            returns_to_go = batch['returns_to_go']
            
            self.optimizer.zero_grad()
            
            action_preds = self.model(states, returns_to_go, actions[:, :-1])
            
            loss = self.criterion(
                action_preds.reshape(-1, action_preds.size(-1)),
                actions[:, 1:].reshape(-1)
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path))

class RLTrainingPipeline:
    def __init__(self, data_samples: List[Tuple[Any, str, Dict]], model_save_path: str = "decision_transformer.pth"):
        self.data_samples = data_samples
        self.model_save_path = model_save_path
        self.orchestrator = RLOrchestrator()
        self.expert_generator = ExpertPolicyGenerator(self.orchestrator)
        self.trainer = DecisionTransformerTrainer()
    
    async def generate_expert_trajectories(self):
        print("Generating expert trajectories...")
        
        cheap_episodes = await self.expert_generator.generate_cheap_first_policy(self.data_samples[:100])
        print(f"Generated {len(cheap_episodes)} cheap-first episodes")
        
        text_episodes = await self.expert_generator.generate_text_first_policy(
            [(img, txt, gt) for img, txt, gt in self.data_samples[:100] if txt]
        )
        print(f"Generated {len(text_episodes)} text-first episodes")
        
        comprehensive_episodes = await self.expert_generator.generate_comprehensive_policy(self.data_samples[:50])
        print(f"Generated {len(comprehensive_episodes)} comprehensive episodes")
        
        total_episodes = len(cheap_episodes) + len(text_episodes) + len(comprehensive_episodes)
        print(f"Total expert trajectories: {total_episodes}")
    
    def train_decision_transformer(self, epochs: int = 100, batch_size: int = 32):
        print("Training Decision Transformer...")
        
        dataset = TrajectoryDataset("trajectories.db")
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            avg_loss = self.trainer.train_epoch(dataloader)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
        
        self.trainer.save_model(self.model_save_path)
        print(f"Model saved to {self.model_save_path}")
    
    async def run_full_training(self, epochs: int = 100):
        await self.generate_expert_trajectories()
        self.train_decision_transformer(epochs)
        print("Training completed!")

async def train_rl_pipeline(data_samples: List[Tuple[Any, str, Dict]]):
    pipeline = RLTrainingPipeline(data_samples)
    await pipeline.run_full_training()
    return pipeline.model_save_path