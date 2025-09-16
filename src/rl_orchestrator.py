import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import ray
from ray import serve
import sqlite3
import pickle
from pathlib import Path

@dataclass
class StateVector:
    global_image_embedding: np.ndarray
    global_text_embedding: np.ndarray
    action_history_mask: np.ndarray
    confidence_vector: np.ndarray
    extracted_attributes: np.ndarray
    remaining_budget: float
    
    def to_tensor(self) -> torch.Tensor:
        return torch.cat([
            torch.from_numpy(self.global_image_embedding).float(),
            torch.from_numpy(self.global_text_embedding).float(),
            torch.from_numpy(self.action_history_mask).float(),
            torch.from_numpy(self.confidence_vector).float(),
            torch.from_numpy(self.extracted_attributes).float(),
            torch.tensor([self.remaining_budget]).float()
        ])

@dataclass
class ActionResult:
    confidence: float
    extracted_data: Dict[str, Any]
    cost: float
    success: bool

class DecisionTransformer(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        self.state_encoder = nn.Linear(state_dim, hidden_dim)
        self.return_encoder = nn.Linear(1, hidden_dim)
        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=n_layers
        )
        
        self.action_head = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, states, returns_to_go, actions=None, timesteps=None):
        batch_size, seq_len = states.shape[:2]
        
        state_embeddings = self.state_encoder(states)
        return_embeddings = self.return_encoder(returns_to_go.unsqueeze(-1))
        
        if actions is not None:
            action_embeddings = self.action_encoder(actions)
            sequence = torch.stack([return_embeddings, state_embeddings, action_embeddings], dim=2)
            sequence = sequence.reshape(batch_size, seq_len * 3, self.hidden_dim)
        else:
            sequence = torch.stack([return_embeddings, state_embeddings], dim=2)
            sequence = sequence.reshape(batch_size, seq_len * 2, self.hidden_dim)
        
        transformer_output = self.transformer(sequence)
        
        if actions is not None:
            action_preds = self.action_head(transformer_output[:, 1::3])
        else:
            action_preds = self.action_head(transformer_output[:, 1::2])
        
        return action_preds

class StateManager:
    def __init__(self, clip_model_name: str = "openai/clip-vit-base-patch32"):
        self.clip_model = AutoModel.from_pretrained(clip_model_name)
        self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.action_costs = {
            0: 0.1,  # person_detector
            1: 0.4,  # vlm_captioner
            2: 0.3,  # text_parser
            3: 0.05, # hair_color_classifier
            4: 0.05, # age_classifier
            5: 0.05, # gender_classifier
            6: 0.05, # ethnicity_classifier
            7: 0.05, # body_type_classifier
            8: 0.05, # dress_classifier
            9: 0.01, # flag_ambiguous
            10: 0.0  # finalize
        }
        
    def initialize_state(self, image_embedding: np.ndarray, text_embedding: np.ndarray) -> StateVector:
        return StateVector(
            global_image_embedding=image_embedding,
            global_text_embedding=text_embedding,
            action_history_mask=np.zeros(11),
            confidence_vector=np.zeros(11),
            extracted_attributes=np.zeros(64),
            remaining_budget=1.0
        )
    
    def update_state(self, state: StateVector, action: int, result: ActionResult) -> StateVector:
        new_state = StateVector(
            global_image_embedding=state.global_image_embedding,
            global_text_embedding=state.global_text_embedding,
            action_history_mask=state.action_history_mask.copy(),
            confidence_vector=state.confidence_vector.copy(),
            extracted_attributes=state.extracted_attributes.copy(),
            remaining_budget=max(0.0, state.remaining_budget - result.cost)
        )
        
        new_state.action_history_mask[action] = 1.0
        new_state.confidence_vector[action] = result.confidence
        
        if result.extracted_data:
            attr_embedding = self._encode_attributes(result.extracted_data)
            new_state.extracted_attributes = attr_embedding
        
        return new_state
    
    def _encode_attributes(self, attributes: Dict[str, Any]) -> np.ndarray:
        attr_text = " ".join([f"{k}:{v}" for k, v in attributes.items() if v])
        if not attr_text:
            return np.zeros(64)
        return self.text_encoder.encode(attr_text)[:64]

@ray.remote(num_gpus=0.1)
class PersonDetector:
    def __init__(self):
        pass
    
    def process(self, image_data: Any) -> ActionResult:
        confidence = np.random.uniform(0.7, 0.95)
        return ActionResult(
            confidence=confidence,
            extracted_data={"person_detected": True, "bbox_count": 1},
            cost=0.1,
            success=True
        )

@ray.remote(num_gpus=0.3)
class VLMCaptioner:
    def __init__(self):
        pass
    
    def process(self, image_data: Any) -> ActionResult:
        confidence = np.random.uniform(0.6, 0.9)
        captions = ["young woman with long black hair", "teenage boy with short brown hair", "elderly man with gray beard"]
        caption = np.random.choice(captions)
        return ActionResult(
            confidence=confidence,
            extracted_data={"caption": caption},
            cost=0.4,
            success=True
        )

@ray.remote(num_gpus=0.2)
class TextParser:
    def __init__(self):
        pass
    
    def process(self, text_data: str) -> ActionResult:
        confidence = np.random.uniform(0.7, 0.95)
        attributes = {
            "age": np.random.choice(["teen", "young adult", "middle-aged"]),
            "gender": np.random.choice(["male", "female"]),
            "hair_color": np.random.choice(["black", "brown", "blonde"])
        }
        return ActionResult(
            confidence=confidence,
            extracted_data=attributes,
            cost=0.3,
            success=True
        )

@ray.remote(num_gpus=0.1)
class AttributeClassifier:
    def __init__(self, attribute_type: str):
        self.attribute_type = attribute_type
        self.class_mappings = {
            "hair_color": ["black", "brown", "blonde", "red", "gray", "white"],
            "age": ["child", "teen", "young adult", "middle-aged", "elderly"],
            "gender": ["male", "female", "non-binary"],
            "ethnicity": ["asian", "caucasian", "african", "hispanic", "other"],
            "body_type": ["slim", "average", "muscular", "curvy", "heavy"],
            "dress": ["casual", "formal", "traditional", "uniform", "costume"]
        }
    
    def process(self, image_data: Any) -> ActionResult:
        confidence = np.random.uniform(0.6, 0.9)
        classes = self.class_mappings.get(self.attribute_type, ["unknown"])
        predicted_class = np.random.choice(classes)
        return ActionResult(
            confidence=confidence,
            extracted_data={self.attribute_type: predicted_class},
            cost=0.05,
            success=True
        )

class ActionToolbox:
    def __init__(self):
        self.tools = {
            0: PersonDetector.remote(),
            1: VLMCaptioner.remote(),
            2: TextParser.remote(),
            3: AttributeClassifier.remote("hair_color"),
            4: AttributeClassifier.remote("age"),
            5: AttributeClassifier.remote("gender"),
            6: AttributeClassifier.remote("ethnicity"),
            7: AttributeClassifier.remote("body_type"),
            8: AttributeClassifier.remote("dress"),
        }
    
    async def execute_action(self, action: int, image_data: Any = None, text_data: str = "") -> ActionResult:
        if action == 9:  # flag_ambiguous
            return ActionResult(
                confidence=1.0,
                extracted_data={"ambiguous": True},
                cost=0.01,
                success=True
            )
        elif action == 10:  # finalize
            return ActionResult(
                confidence=1.0,
                extracted_data={"finalized": True},
                cost=0.0,
                success=True
            )
        elif action in self.tools:
            if action == 2:  # text_parser
                return await self.tools[action].process.remote(text_data)
            else:
                return await self.tools[action].process.remote(image_data)
        else:
            return ActionResult(
                confidence=0.0,
                extracted_data={},
                cost=0.0,
                success=False
            )

class PolicyAgent:
    def __init__(self, model_path: Optional[str] = None):
        self.state_dim = 768 + 384 + 11 + 11 + 64 + 1  # 1239
        self.action_dim = 11
        self.model = DecisionTransformer(self.state_dim, self.action_dim)
        
        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path))
        
        self.model.eval()
    
    def select_action(self, state: StateVector, target_return: float = 0.9) -> int:
        with torch.no_grad():
            state_tensor = state.to_tensor().unsqueeze(0).unsqueeze(0)
            return_tensor = torch.tensor([[target_return]])
            
            action_logits = self.model(state_tensor, return_tensor)
            action_probs = torch.softmax(action_logits[0, -1], dim=-1)
            
            valid_actions = (state.action_history_mask == 0) & (state.remaining_budget > 0.05)
            valid_actions[10] = True  # finalize always available
            
            masked_probs = action_probs * torch.from_numpy(valid_actions.astype(float))
            
            if masked_probs.sum() == 0:
                return 10  # finalize if no valid actions
            
            return torch.multinomial(masked_probs, 1).item()

class TrajectoryCollector:
    def __init__(self, db_path: str = "trajectories.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trajectories (
                id INTEGER PRIMARY KEY,
                episode_id TEXT,
                step INTEGER,
                state BLOB,
                action INTEGER,
                reward REAL,
                next_state BLOB,
                done INTEGER
            )
        """)
        conn.commit()
        conn.close()
    
    def save_trajectory(self, episode_id: str, trajectory: List[Tuple]):
        conn = sqlite3.connect(self.db_path)
        for step, (state, action, reward, next_state, done) in enumerate(trajectory):
            conn.execute("""
                INSERT INTO trajectories (episode_id, step, state, action, reward, next_state, done)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                episode_id,
                step,
                pickle.dumps(state),
                action,
                reward,
                pickle.dumps(next_state),
                int(done)
            ))
        conn.commit()
        conn.close()

class RLOrchestrator:
    def __init__(self, model_path: Optional[str] = None):
        ray.init(ignore_reinit_error=True)
        self.state_manager = StateManager()
        self.action_toolbox = ActionToolbox()
        self.policy_agent = PolicyAgent(model_path)
        self.trajectory_collector = TrajectoryCollector()
    
    async def process_sample(self, image_data: Any, text_data: str = "", ground_truth: Optional[Dict] = None) -> Dict[str, Any]:
        image_embedding = np.random.randn(768)  # Mock CLIP embedding
        text_embedding = self.state_manager.text_encoder.encode(text_data) if text_data else np.zeros(384)
        
        state = self.state_manager.initialize_state(image_embedding, text_embedding)
        trajectory = []
        extracted_attributes = {}
        total_cost = 0.0
        confidences = []
        
        for step in range(20):  # max 20 steps per episode
            action = self.policy_agent.select_action(state)
            
            if action == 10:  # finalize
                break
            
            result = await self.action_toolbox.execute_action(action, image_data, text_data)
            
            if result.success:
                extracted_attributes.update(result.extracted_data)
                confidences.append(result.confidence)
                total_cost += result.cost
            
            next_state = self.state_manager.update_state(state, action, result)
            
            reward = 0.0
            if ground_truth:
                reward = self._calculate_reward(extracted_attributes, ground_truth, total_cost, confidences)
            
            trajectory.append((state, action, reward, next_state, False))
            state = next_state
            
            if state.remaining_budget <= 0.05:
                break
        
        final_reward = 0.0
        if ground_truth:
            final_reward = self._calculate_reward(extracted_attributes, ground_truth, total_cost, confidences)
        
        if trajectory:
            trajectory[-1] = (trajectory[-1][0], trajectory[-1][1], final_reward, trajectory[-1][3], True)
            episode_id = f"episode_{np.random.randint(0, 1000000)}"
            self.trajectory_collector.save_trajectory(episode_id, trajectory)
        
        return {
            "extracted_attributes": extracted_attributes,
            "total_cost": total_cost,
            "avg_confidence": np.mean(confidences) if confidences else 0.0,
            "steps_taken": len(trajectory),
            "final_reward": final_reward
        }
    
    def _calculate_reward(self, extracted: Dict, ground_truth: Dict, cost: float, confidences: List[float]) -> float:
        f1_score = self._calculate_f1(extracted, ground_truth)
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        reward = 1.0 * f1_score - 0.5 * cost + 0.2 * avg_confidence
        return max(0.0, reward)
    
    def _calculate_f1(self, extracted: Dict, ground_truth: Dict) -> float:
        if not ground_truth:
            return 0.0
        
        correct = 0
        total_gt = len(ground_truth)
        total_pred = len(extracted)
        
        for key, value in ground_truth.items():
            if key in extracted and str(extracted[key]).lower() == str(value).lower():
                correct += 1
        
        if total_pred == 0:
            precision = 0.0
        else:
            precision = correct / total_pred
        
        if total_gt == 0:
            recall = 0.0
        else:
            recall = correct / total_gt
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    async def process_batch(self, samples: List[Tuple[Any, str, Optional[Dict]]]) -> List[Dict[str, Any]]:
        tasks = []
        for image_data, text_data, ground_truth in samples:
            task = self.process_sample(image_data, text_data, ground_truth)
            tasks.append(task)
        
        results = await ray.get(tasks)
        return results