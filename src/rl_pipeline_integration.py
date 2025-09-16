import asyncio
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from PIL import Image
import torch
from pathlib import Path
import json
import time
from .rl_orchestrator import RLOrchestrator, StateVector
from .rl_trainer import train_rl_pipeline
from .pipeline.base import CharacterAttributes, ProcessingResult
from .pipeline.input_loader import DatasetItem
import ray

class ProductionRLPipeline:
    def __init__(self, model_path: Optional[str] = None, enable_training: bool = False):
        self.orchestrator = RLOrchestrator(model_path)
        self.enable_training = enable_training
        self.training_data = []
        self.performance_metrics = {
            "total_processed": 0,
            "avg_processing_time": 0.0,
            "avg_confidence": 0.0,
            "avg_cost": 0.0,
            "success_rate": 0.0
        }
    
    async def extract_attributes_rl(self, image: Union[str, Path, Image.Image], 
                                   tags: Optional[str] = None,
                                   ground_truth: Optional[Dict] = None) -> CharacterAttributes:
        start_time = time.time()
        
        try:
            if isinstance(image, (str, Path)):
                image_data = Image.open(image).convert('RGB')
            else:
                image_data = image
            
            text_data = tags or ""
            
            result = await self.orchestrator.process_sample(image_data, text_data, ground_truth)
            
            processing_time = time.time() - start_time
            
            attributes = self._convert_to_character_attributes(result["extracted_attributes"])
            attributes.confidence_score = result["avg_confidence"]
            
            self._update_metrics(result, processing_time, success=True)
            
            if self.enable_training and ground_truth:
                self.training_data.append((image_data, text_data, ground_truth))
            
            return attributes
            
        except Exception as e:
            processing_time = time.time() - start_time
            self._update_metrics({"avg_confidence": 0.0, "total_cost": 0.0}, processing_time, success=False)
            
            return CharacterAttributes()
    
    def _convert_to_character_attributes(self, extracted_data: Dict[str, Any]) -> CharacterAttributes:
        attributes = CharacterAttributes()
        
        mapping = {
            "age": "age",
            "gender": "gender", 
            "ethnicity": "ethnicity",
            "hair_color": "hair_color",
            "hair_style": "hair_style",
            "hair_length": "hair_length",
            "eye_color": "eye_color",
            "body_type": "body_type",
            "dress": "dress",
            "facial_expression": "facial_expression",
            "accessories": "accessories",
            "scars": "scars",
            "tattoos": "tattoos"
        }
        
        for key, attr_name in mapping.items():
            if key in extracted_data and extracted_data[key]:
                setattr(attributes, attr_name, str(extracted_data[key]).title())
        
        if "caption" in extracted_data:
            self._parse_caption_to_attributes(extracted_data["caption"], attributes)
        
        return attributes
    
    def _parse_caption_to_attributes(self, caption: str, attributes: CharacterAttributes):
        caption_lower = caption.lower()
        
        age_keywords = {
            "child": ["child", "kid", "young child"],
            "teen": ["teen", "teenager", "adolescent"],
            "young adult": ["young", "young adult", "young woman", "young man"],
            "middle-aged": ["middle-aged", "adult"],
            "elderly": ["elderly", "old", "senior"]
        }
        
        hair_colors = {
            "black": ["black hair", "dark hair"],
            "brown": ["brown hair", "brunette"],
            "blonde": ["blonde", "blond hair", "golden hair"],
            "red": ["red hair", "ginger", "auburn"],
            "gray": ["gray hair", "grey hair"],
            "white": ["white hair", "silver hair"]
        }
        
        hair_lengths = {
            "short": ["short hair"],
            "medium": ["medium hair", "shoulder-length"],
            "long": ["long hair"]
        }
        
        for age, keywords in age_keywords.items():
            if any(keyword in caption_lower for keyword in keywords):
                if not attributes.age:
                    attributes.age = age.title()
                break
        
        for color, keywords in hair_colors.items():
            if any(keyword in caption_lower for keyword in keywords):
                if not attributes.hair_color:
                    attributes.hair_color = color.title()
                break
        
        for length, keywords in hair_lengths.items():
            if any(keyword in caption_lower for keyword in keywords):
                if not attributes.hair_length:
                    attributes.hair_length = length.title()
                break
        
        if "woman" in caption_lower or "female" in caption_lower:
            if not attributes.gender:
                attributes.gender = "Female"
        elif "man" in caption_lower or "male" in caption_lower or "boy" in caption_lower:
            if not attributes.gender:
                attributes.gender = "Male"
    
    def _update_metrics(self, result: Dict, processing_time: float, success: bool):
        self.performance_metrics["total_processed"] += 1
        
        total = self.performance_metrics["total_processed"]
        
        self.performance_metrics["avg_processing_time"] = (
            (self.performance_metrics["avg_processing_time"] * (total - 1) + processing_time) / total
        )
        
        if success:
            self.performance_metrics["avg_confidence"] = (
                (self.performance_metrics["avg_confidence"] * (total - 1) + result["avg_confidence"]) / total
            )
            
            self.performance_metrics["avg_cost"] = (
                (self.performance_metrics["avg_cost"] * (total - 1) + result["total_cost"]) / total
            )
        
        success_count = self.performance_metrics["success_rate"] * (total - 1) + (1 if success else 0)
        self.performance_metrics["success_rate"] = success_count / total
    
    async def process_batch_rl(self, items: List[DatasetItem]) -> List[ProcessingResult]:
        batch_data = []
        for item in items:
            try:
                if hasattr(item, 'image_path') and item.image_path:
                    image_data = Image.open(item.image_path).convert('RGB')
                else:
                    image_data = item.image if hasattr(item, 'image') else None
                
                text_data = item.tags if hasattr(item, 'tags') else ""
                ground_truth = getattr(item, 'ground_truth', None)
                
                batch_data.append((image_data, text_data, ground_truth))
            except Exception:
                batch_data.append((None, "", None))
        
        batch_results = await self.orchestrator.process_batch(batch_data)
        
        results = []
        for i, (item, rl_result) in enumerate(zip(items, batch_results)):
            try:
                attributes = self._convert_to_character_attributes(rl_result["extracted_attributes"])
                attributes.confidence_score = rl_result["avg_confidence"]
                
                result = ProcessingResult(
                    item_id=item.item_id,
                    attributes=attributes,
                    success=True,
                    processing_time=rl_result.get("processing_time", 0.0)
                )
            except Exception as e:
                result = ProcessingResult(
                    item_id=item.item_id,
                    attributes=CharacterAttributes(),
                    success=False,
                    error_message=str(e),
                    processing_time=0.0
                )
            
            results.append(result)
        
        return results
    
    async def retrain_model(self, min_samples: int = 100) -> bool:
        if not self.enable_training or len(self.training_data) < min_samples:
            return False
        
        try:
            model_path = await train_rl_pipeline(self.training_data)
            
            self.orchestrator = RLOrchestrator(model_path)
            
            self.training_data = []
            
            return True
        except Exception:
            return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        return self.performance_metrics.copy()
    
    def save_training_data(self, path: str):
        if self.training_data:
            training_export = []
            for image_data, text_data, ground_truth in self.training_data:
                training_export.append({
                    "text_data": text_data,
                    "ground_truth": ground_truth,
                    "image_shape": list(image_data.size) if image_data else None
                })
            
            with open(path, 'w') as f:
                json.dump(training_export, f, indent=2)
    
    def load_training_data(self, path: str):
        try:
            with open(path, 'r') as f:
                training_export = json.load(f)
            
            print(f"Loaded {len(training_export)} training samples metadata")
        except Exception:
            pass

class HybridPipeline:
    def __init__(self, fallback_pipeline, rl_model_path: Optional[str] = None, use_rl_primary: bool = True):
        self.fallback_pipeline = fallback_pipeline
        self.rl_pipeline = ProductionRLPipeline(rl_model_path, enable_training=True)
        self.use_rl_primary = use_rl_primary
        self.rl_failure_count = 0
        self.fallback_threshold = 5
    
    async def extract_from_image(self, image: Union[str, Path, Image.Image], 
                               tags: Optional[str] = None) -> CharacterAttributes:
        if self.use_rl_primary and self.rl_failure_count < self.fallback_threshold:
            try:
                result = await self.rl_pipeline.extract_attributes_rl(image, tags)
                
                if result.confidence_score and result.confidence_score > 0.3:
                    self.rl_failure_count = max(0, self.rl_failure_count - 1)
                    return result
                else:
                    self.rl_failure_count += 1
            except Exception:
                self.rl_failure_count += 1
        
        return self.fallback_pipeline.extract_from_image(image, tags)
    
    async def process_batch(self, items: List[DatasetItem]) -> List[ProcessingResult]:
        if self.use_rl_primary and self.rl_failure_count < self.fallback_threshold:
            try:
                results = await self.rl_pipeline.process_batch_rl(items)
                
                success_rate = sum(1 for r in results if r.success) / len(results) if results else 0
                
                if success_rate > 0.7:
                    self.rl_failure_count = max(0, self.rl_failure_count - 1)
                    return results
                else:
                    self.rl_failure_count += 1
            except Exception:
                self.rl_failure_count += 1
        
        return self.fallback_pipeline.process_batch(items)
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "using_rl_primary": self.use_rl_primary,
            "rl_failure_count": self.rl_failure_count,
            "fallback_threshold": self.fallback_threshold,
            "rl_metrics": self.rl_pipeline.get_performance_metrics()
        }
    
    async def trigger_retraining(self) -> bool:
        return await self.rl_pipeline.retrain_model()

def create_rl_enhanced_pipeline(fallback_pipeline, rl_model_path: Optional[str] = None) -> HybridPipeline:
    return HybridPipeline(fallback_pipeline, rl_model_path, use_rl_primary=True)