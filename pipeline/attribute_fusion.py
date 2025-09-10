"""Attribute fusion stage for combining results from multiple extractors."""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from .base import PipelineStage, CharacterAttributes

class AttributeFusion(PipelineStage):
    """Fuses attributes from multiple extraction methods."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("AttributeFusion", config)
        
        # Fusion strategy
        self.fusion_strategy = config.get('fusion_strategy', 'confidence_weighted') if config else 'confidence_weighted'
        self.confidence_threshold = config.get('confidence_threshold', 0.1) if config else 0.1
        self.agreement_bonus = config.get('agreement_bonus', 0.1) if config else 0.1
        
        # Attribute weights for different methods
        self.method_weights = config.get('method_weights', {
            'clip': 0.6,
            'tags': 0.4,
            'rl': 0.8,  # RL optimizer gets higher weight when available
            'blip2': 0.7  # BLIP2 gets high weight for visual understanding
        }) if config else {
            'clip': 0.6,
            'tags': 0.4,
            'rl': 0.8,
            'blip2': 0.7
        }
    
    def _extract_confidences(self, results: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Extract confidence scores for each method and attribute."""
        confidences = defaultdict(dict)
        
        # CLIP confidences
        if 'clip_results' in results:
            clip_attrs = results['clip_results']
            if hasattr(clip_attrs, 'confidence_score') and clip_attrs.confidence_score:
                # Distribute overall confidence to individual attributes
                base_conf = clip_attrs.confidence_score
                attributes = ['age', 'gender', 'hair_color', 'hair_length', 'hair_style', 
                             'eye_color', 'body_type', 'dress', 'facial_expression']
                
                for attr in attributes:
                    if getattr(clip_attrs, attr, None):
                        confidences['clip'][attr] = base_conf
        
        # Tag confidences
        if 'tag_results' in results:
            tag_attrs = results['tag_results']
            if hasattr(tag_attrs, 'confidence_score') and tag_attrs.confidence_score:
                base_conf = tag_attrs.confidence_score
                attributes = ['age', 'gender', 'hair_color', 'hair_length', 'hair_style', 
                             'eye_color', 'body_type', 'dress', 'facial_expression']
                
                for attr in attributes:
                    if getattr(tag_attrs, attr, None):
                        confidences['tags'][attr] = base_conf
        
        # RL confidences (if available)
        if 'rl_results' in results:
            rl_attrs = results['rl_results']
            if hasattr(rl_attrs, 'confidence_score') and rl_attrs.confidence_score:
                base_conf = rl_attrs.confidence_score
                attributes = ['age', 'gender', 'hair_color', 'hair_length', 'hair_style', 
                             'eye_color', 'body_type', 'dress', 'facial_expression']
                
                for attr in attributes:
                    if getattr(rl_attrs, attr, None):
                        confidences['rl'][attr] = base_conf
        
        # BLIP2 confidences (if available)
        if 'blip2_results' in results:
            blip2_attrs = results['blip2_results']
            if hasattr(blip2_attrs, 'confidence_score') and blip2_attrs.confidence_score:
                base_conf = blip2_attrs.confidence_score
                attributes = ['age', 'gender', 'hair_color', 'hair_length', 'hair_style', 
                             'eye_color', 'body_type', 'dress', 'facial_expression']
                
                for attr in attributes:
                    if getattr(blip2_attrs, attr, None):
                        confidences['blip2'][attr] = base_conf
        
        return confidences
    
    def _confidence_weighted_fusion(self, results: Dict[str, Any], confidences: Dict[str, Dict[str, float]]) -> CharacterAttributes:
        """Fuse attributes using confidence-weighted voting."""
        fused_attrs = CharacterAttributes()
        attributes = ['age', 'gender', 'ethnicity', 'hair_color', 'hair_length', 'hair_style', 
                     'eye_color', 'body_type', 'dress', 'facial_expression']
        
        overall_confidences = []
        
        for attr in attributes:
            candidates = []
            
            # Collect candidates from each method
            for method in ['clip', 'tags', 'rl', 'blip2']:
                if f'{method}_results' in results:
                    method_attrs = results[f'{method}_results']
                    value = getattr(method_attrs, attr, None)
                    
                    if value:
                        confidence = confidences.get(method, {}).get(attr, 0.0)
                        weight = self.method_weights.get(method, 1.0)
                        weighted_confidence = confidence * weight
                        
                        candidates.append((value, weighted_confidence, method))
            
            if not candidates:
                continue
            
            # Group by value and sum confidences
            value_scores = defaultdict(list)
            for value, conf, method in candidates:
                value_scores[value].append((conf, method))
            
            # Calculate final scores
            final_scores = {}
            for value, conf_list in value_scores.items():
                total_conf = sum(conf for conf, _ in conf_list)
                methods = [method for _, method in conf_list]
                
                # Bonus for agreement between methods
                if len(set(methods)) > 1:
                    total_conf += self.agreement_bonus
                
                final_scores[value] = total_conf
            
            # Select best value
            if final_scores:
                best_value = max(final_scores.keys(), key=lambda x: final_scores[x])
                best_confidence = final_scores[best_value]
                
                if best_confidence >= self.confidence_threshold:
                    setattr(fused_attrs, attr, best_value)
                    overall_confidences.append(best_confidence)
                elif not overall_confidences and final_scores:  # Fallback: include best value even if below threshold
                    setattr(fused_attrs, attr, best_value)
                    overall_confidences.append(best_confidence * 0.5)  # Reduced confidence for fallback
        
        # Set overall confidence with minimum fallback
        if overall_confidences:
            fused_attrs.confidence_score = np.mean(overall_confidences)
        else:
            # Fallback: try to extract at least one attribute with very low threshold
            for attr in attributes:
                candidates = []
                for method in ['clip', 'tags', 'rl', 'blip2']:
                    if f'{method}_results' in results:
                        method_attrs = results[f'{method}_results']
                        value = getattr(method_attrs, attr, None)
                        if value:
                            candidates.append(value)
                if candidates:
                    setattr(fused_attrs, attr, candidates[0])  # Take first available
                    fused_attrs.confidence_score = 0.1  # Very low confidence
                    break
        
        return fused_attrs
    
    def _majority_voting_fusion(self, results: Dict[str, Any]) -> CharacterAttributes:
        """Fuse attributes using majority voting."""
        fused_attrs = CharacterAttributes()
        attributes = ['age', 'gender', 'hair_color', 'hair_length', 'hair_style', 
                     'eye_color', 'body_type', 'dress', 'facial_expression']
        
        for attr in attributes:
            votes = defaultdict(int)
            
            # Collect votes from each method
            for method in ['clip', 'tags', 'rl', 'blip2']:
                if f'{method}_results' in results:
                    method_attrs = results[f'{method}_results']
                    value = getattr(method_attrs, attr, None)
                    
                    if value:
                        votes[value] += 1
            
            # Select majority vote
            if votes:
                best_value = max(votes.keys(), key=lambda x: votes[x])
                setattr(fused_attrs, attr, best_value)
        
        return fused_attrs
    
    def _hierarchical_fusion(self, results: Dict[str, Any]) -> CharacterAttributes:
        """Fuse attributes using hierarchical priority (RL > CLIP > Tags)."""
        fused_attrs = CharacterAttributes()
        attributes = ['age', 'gender', 'hair_color', 'hair_length', 'hair_style', 
                     'eye_color', 'body_type', 'dress', 'facial_expression']
        
        # Priority order: RL > BLIP2 > CLIP > Tags
        method_priority = ['rl', 'blip2', 'clip', 'tags']
        
        for attr in attributes:
            for method in method_priority:
                if f'{method}_results' in results:
                    method_attrs = results[f'{method}_results']
                    value = getattr(method_attrs, attr, None)
                    
                    if value:
                        setattr(fused_attrs, attr, value)
                        break  # Use first available value in priority order
        
        return fused_attrs
    
    def _ensemble_fusion(self, results: Dict[str, Any], confidences: Dict[str, Dict[str, float]]) -> CharacterAttributes:
        """Advanced ensemble fusion with uncertainty estimation."""
        fused_attrs = CharacterAttributes()
        attributes = ['age', 'gender', 'hair_color', 'hair_length', 'hair_style', 
                     'eye_color', 'body_type', 'dress', 'facial_expression']
        
        overall_confidences = []
        
        for attr in attributes:
            # Collect all predictions with their uncertainties
            predictions = []
            
            for method in ['clip', 'tags', 'rl', 'blip2']:
                if f'{method}_results' in results:
                    method_attrs = results[f'{method}_results']
                    value = getattr(method_attrs, attr, None)
                    
                    if value:
                        confidence = confidences.get(method, {}).get(attr, 0.0)
                        uncertainty = 1.0 - confidence
                        weight = self.method_weights.get(method, 1.0)
                        
                        predictions.append({
                            'value': value,
                            'confidence': confidence,
                            'uncertainty': uncertainty,
                            'weight': weight,
                            'method': method
                        })
            
            if not predictions:
                continue
            
            # Calculate weighted ensemble
            value_weights = defaultdict(float)
            value_uncertainties = defaultdict(list)
            
            for pred in predictions:
                # Weight by confidence and method weight
                ensemble_weight = pred['confidence'] * pred['weight']
                value_weights[pred['value']] += ensemble_weight
                value_uncertainties[pred['value']].append(pred['uncertainty'])
            
            # Select best value considering both weight and uncertainty
            best_value = None
            best_score = -1
            
            for value, weight in value_weights.items():
                # Average uncertainty for this value
                avg_uncertainty = np.mean(value_uncertainties[value])
                
                # Score combines weight and low uncertainty
                score = weight * (1.0 - avg_uncertainty)
                
                if score > best_score:
                    best_score = score
                    best_value = value
            
            if best_value and best_score >= self.confidence_threshold:
                setattr(fused_attrs, attr, best_value)
                overall_confidences.append(best_score)
        
        # Set overall confidence
        fused_attrs.confidence_score = np.mean(overall_confidences) if overall_confidences else 0.0
        
        return fused_attrs
    
    def _merge_accessories(self, results: Dict[str, Any]) -> List[str]:
        """Merge accessories from all methods."""
        all_accessories = set()
        
        for method in ['clip', 'tags', 'rl', 'blip2']:
            if f'{method}_results' in results:
                method_attrs = results[f'{method}_results']
                accessories = getattr(method_attrs, 'accessories', None)
                
                if accessories:
                    all_accessories.update(accessories)
        
        return list(all_accessories) if all_accessories else None
    
    def process(self, input_data: Any) -> CharacterAttributes:
        """Fuse attributes from multiple extraction methods."""
        if not isinstance(input_data, dict):
            raise ValueError("AttributeFusion expects dict input with extraction results")
        
        # Extract confidences
        confidences = self._extract_confidences(input_data)
        
        # Apply fusion strategy
        if self.fusion_strategy == 'confidence_weighted':
            fused_attrs = self._confidence_weighted_fusion(input_data, confidences)
        elif self.fusion_strategy == 'majority_voting':
            fused_attrs = self._majority_voting_fusion(input_data)
        elif self.fusion_strategy == 'hierarchical':
            fused_attrs = self._hierarchical_fusion(input_data)
        elif self.fusion_strategy == 'ensemble':
            fused_attrs = self._ensemble_fusion(input_data, confidences)
        else:
            self.logger.warning(f"Unknown fusion strategy: {self.fusion_strategy}, using confidence_weighted")
            fused_attrs = self._confidence_weighted_fusion(input_data, confidences)
        
        # Merge accessories
        fused_attrs.accessories = self._merge_accessories(input_data)
        
        # Preserve source information
        all_tags = []
        for method in ['clip', 'tags', 'rl', 'blip2']:
            if f'{method}_results' in input_data:
                method_attrs = input_data[f'{method}_results']
                source_tags = getattr(method_attrs, 'source_tags', None)
                if source_tags:
                    all_tags.extend(source_tags)
        
        fused_attrs.source_tags = list(set(all_tags)) if all_tags else None
        
        return fused_attrs
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data."""
        if not isinstance(input_data, dict):
            return False
        
        # At least one extraction result should be present
        required_keys = ['clip_results', 'tag_results', 'rl_results', 'blip2_results']
        return any(key in input_data for key in required_keys)