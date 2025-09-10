"""CLIP-based image analysis for character attribute extraction."""

import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .base import PipelineStage, CharacterAttributes

class CLIPAnalyzer(PipelineStage):
    """Uses CLIP for zero-shot classification of character attributes from images."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("CLIPAnalyzer", config)
        
        # Model configuration
        self.model_name = config.get('model_name', 'openai/clip-vit-base-patch32') if config else 'openai/clip-vit-base-patch32'
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu') if config else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.confidence_threshold = config.get('confidence_threshold', 0.3) if config else 0.3
        
        # Initialize model and processor
        self.model = None
        self.processor = None
        self._initialize_model()
        self._initialize_prompts()
    
    def _initialize_model(self):
        """Initialize CLIP model and processor."""
        try:
            self.logger.info(f"Loading CLIP model: {self.model_name}")
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self.logger.info(f"CLIP model loaded successfully on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load CLIP model: {e}")
            raise
    
    def _initialize_prompts(self):
        """Initialize text prompts for zero-shot classification."""
        
        self.age_prompts = [
            "a child character",
            "a teenage character", 
            "a young adult character",
            "a middle-aged character",
            "an elderly character"
        ]
        
        self.gender_prompts = [
            "a male character",
            "a female character",
            "an androgynous character"
        ]
        
        self.ethnicity_prompts = [
            "an Asian character",
            "an African character", 
            "a Caucasian character",
            "a Hispanic character",
            "a Middle Eastern character",
            "a Native American character",
            "a mixed ethnicity character"
        ]
        
        self.hair_color_prompts = [
            "a character with black hair",
            "a character with brown hair",
            "a character with blonde hair",
            "a character with red hair",
            "a character with blue hair",
            "a character with green hair",
            "a character with purple hair",
            "a character with pink hair",
            "a character with white hair",
            "a character with multicolored hair"
        ]
        
        self.hair_length_prompts = [
            "a character with short hair",
            "a character with medium length hair",
            "a character with long hair"
        ]
        
        self.hair_style_prompts = [
            "a character with a ponytail",
            "a character with twintails",
            "a character with a hair bun",
            "a character with braided hair",
            "a character with curly hair",
            "a character with straight hair",
            "a character with messy hair",
            "a character with spiky hair"
        ]
        
        self.eye_color_prompts = [
            "a character with brown eyes",
            "a character with blue eyes",
            "a character with green eyes",
            "a character with red eyes",
            "a character with purple eyes",
            "a character with yellow eyes",
            "a character with pink eyes",
            "a character with black eyes",
            "a character with grey eyes"
        ]
        
        self.body_type_prompts = [
            "a slim character",
            "a muscular character",
            "a curvy character",
            "a chubby character",
            "a tall character",
            "a short character"
        ]
        
        self.dress_prompts = [
            "a character in casual clothes",
            "a character in formal attire",
            "a character in traditional clothing",
            "a character in school uniform",
            "a character in swimwear",
            "a character in cosplay costume",
            "a character in military uniform",
            "a character in maid outfit",
            "a character in gothic clothing"
        ]
        
        self.expression_prompts = [
            "a happy character",
            "a sad character",
            "an angry character",
            "a surprised character",
            "a neutral expression character",
            "an embarrassed character",
            "a serious character"
        ]
        
        # Map prompts to attribute values
        self.prompt_mappings = {
            'age': dict(zip(self.age_prompts, ['child', 'teen', 'young adult', 'middle-aged', 'elderly'])),
            'gender': dict(zip(self.gender_prompts, ['male', 'female', 'non-binary'])),
            'ethnicity': dict(zip(self.ethnicity_prompts, ['Asian', 'African', 'Caucasian', 'Hispanic', 'Middle Eastern', 'Native American', 'Mixed'])),
            'hair_color': dict(zip(self.hair_color_prompts, ['black', 'brown', 'blonde', 'red', 'blue', 'green', 'purple', 'pink', 'white', 'multicolored'])),
            'hair_length': dict(zip(self.hair_length_prompts, ['short', 'medium', 'long'])),
            'hair_style': dict(zip(self.hair_style_prompts, ['ponytail', 'twintails', 'bun', 'braided', 'curly', 'straight', 'messy', 'spiky'])),
            'eye_color': dict(zip(self.eye_color_prompts, ['brown', 'blue', 'green', 'red', 'purple', 'yellow', 'pink', 'black', 'grey'])),
            'body_type': dict(zip(self.body_type_prompts, ['slim', 'muscular', 'curvy', 'chubby', 'tall', 'short'])),
            'dress': dict(zip(self.dress_prompts, ['casual', 'formal', 'traditional', 'school uniform', 'swimwear', 'cosplay', 'military', 'maid', 'gothic'])),
            'facial_expression': dict(zip(self.expression_prompts, ['happy', 'sad', 'angry', 'surprised', 'neutral', 'embarrassed', 'serious']))
        }
    
    def _classify_attribute(self, image: Image.Image, prompts: List[str]) -> Tuple[str, float]:
        """Classify a single attribute using zero-shot classification."""
        try:
            # Prepare inputs
            inputs = self.processor(
                text=prompts,
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = F.softmax(logits_per_image, dim=1)
            
            # Get best prediction
            best_idx = torch.argmax(probs, dim=1).item()
            confidence = probs[0, best_idx].item()
            
            return prompts[best_idx], confidence
            
        except Exception as e:
            self.logger.error(f"Error in CLIP classification: {e}")
            return "", 0.0
    
    def _extract_visual_attributes(self, image: Image.Image) -> Dict[str, Tuple[str, float]]:
        """Extract all visual attributes from image using CLIP."""
        results = {}
        
        # Classify each attribute type
        attribute_prompts = {
            'age': self.age_prompts,
            'gender': self.gender_prompts,
            'ethnicity': self.ethnicity_prompts,
            'hair_color': self.hair_color_prompts,
            'hair_length': self.hair_length_prompts,
            'hair_style': self.hair_style_prompts,
            'eye_color': self.eye_color_prompts,
            'body_type': self.body_type_prompts,
            'dress': self.dress_prompts,
            'facial_expression': self.expression_prompts
        }
        
        for attr_name, prompts in attribute_prompts.items():
            best_prompt, confidence = self._classify_attribute(image, prompts)
            
            if confidence >= self.confidence_threshold:
                # Map prompt back to attribute value
                attr_value = self.prompt_mappings[attr_name].get(best_prompt, "")
                results[attr_name] = (attr_value, confidence)
                self.logger.debug(f"{attr_name}: {attr_value} (confidence: {confidence:.3f})")
            else:
                results[attr_name] = ("", confidence)
        
        return results
    
    def process(self, input_data: Any) -> CharacterAttributes:
        """Process image and extract visual attributes using CLIP."""
        if isinstance(input_data, dict):
            image = input_data.get('image')
            if image is None:
                self.logger.warning("No image found in input data")
                return CharacterAttributes()
        elif isinstance(input_data, Image.Image):
            image = input_data
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
        
        if not isinstance(image, Image.Image):
            self.logger.error("Invalid image format")
            return CharacterAttributes()
        
        # Extract visual attributes
        visual_results = self._extract_visual_attributes(image)
        
        # Create CharacterAttributes object
        attributes = CharacterAttributes()
        
        # Map results to attributes
        if 'age' in visual_results and visual_results['age'][0]:
            attributes.age = visual_results['age'][0]
        
        if 'gender' in visual_results and visual_results['gender'][0]:
            attributes.gender = visual_results['gender'][0]
        
        if 'ethnicity' in visual_results and visual_results['ethnicity'][0]:
            attributes.ethnicity = visual_results['ethnicity'][0]
        
        if 'hair_color' in visual_results and visual_results['hair_color'][0]:
            attributes.hair_color = visual_results['hair_color'][0]
        
        if 'hair_length' in visual_results and visual_results['hair_length'][0]:
            attributes.hair_length = visual_results['hair_length'][0]
        
        if 'hair_style' in visual_results and visual_results['hair_style'][0]:
            attributes.hair_style = visual_results['hair_style'][0]
        
        if 'eye_color' in visual_results and visual_results['eye_color'][0]:
            attributes.eye_color = visual_results['eye_color'][0]
        
        if 'body_type' in visual_results and visual_results['body_type'][0]:
            attributes.body_type = visual_results['body_type'][0]
        
        if 'dress' in visual_results and visual_results['dress'][0]:
            attributes.dress = visual_results['dress'][0]
        
        if 'facial_expression' in visual_results and visual_results['facial_expression'][0]:
            attributes.facial_expression = visual_results['facial_expression'][0]
        
        # Calculate overall confidence as average of individual confidences
        confidences = [result[1] for result in visual_results.values() if result[1] > 0]
        attributes.confidence_score = np.mean(confidences) if confidences else 0.0
        
        return attributes
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data."""
        if isinstance(input_data, dict):
            return 'image' in input_data and isinstance(input_data['image'], Image.Image)
        elif isinstance(input_data, Image.Image):
            return True
        return False
    
    def get_embeddings(self, image: Image.Image) -> torch.Tensor:
        """Get CLIP image embeddings for caching/similarity search."""
        try:
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                # Normalize embeddings
                image_features = F.normalize(image_features, p=2, dim=1)
            
            return image_features.cpu()
            
        except Exception as e:
            self.logger.error(f"Error getting embeddings: {e}")
            return torch.empty(0)