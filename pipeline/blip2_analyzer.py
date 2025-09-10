"""BLIP2-based image analysis for enhanced character attribute extraction."""

import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
from typing import Any, Dict, List, Optional, Tuple
import re

from .base import PipelineStage, CharacterAttributes

class BLIP2Analyzer(PipelineStage):
    """Uses BLIP2 for image captioning and attribute extraction through natural language."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("BLIP2Analyzer", config)
        
        # Model configuration
        self.model_name = config.get('model_name', 'Salesforce/blip2-opt-2.7b') if config else 'Salesforce/blip2-opt-2.7b'
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu') if config else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = config.get('max_length', 50) if config else 50
        
        # Initialize model and processor
        self.model = None
        self.processor = None
        self._initialize_model()
        self._initialize_prompts()
    
    def _initialize_model(self):
        """Initialize BLIP2 model and processor."""
        try:
            self.logger.info(f"Loading BLIP2 model: {self.model_name}")
            self.processor = Blip2Processor.from_pretrained(self.model_name)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            )
            self.model.to(self.device)
            self.model.eval()
            self.logger.info(f"BLIP2 model loaded successfully on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load BLIP2 model: {e}")
            raise
    
    def _initialize_prompts(self):
        """Initialize prompts for targeted attribute extraction."""
        
        self.attribute_prompts = {
            'general_description': "Describe this character in detail:",
            'age': "What is the apparent age of this character?",
            'gender': "What is the gender of this character?",
            'hair': "Describe the hair of this character:",
            'eyes': "Describe the eyes of this character:",
            'clothing': "Describe what this character is wearing:",
            'expression': "What is the facial expression of this character?",
            'accessories': "What accessories is this character wearing?"
        }
        
        # Keyword mappings for parsing BLIP2 responses
        self.age_keywords = {
            'child': ['child', 'kid', 'young child', 'little', 'small child'],
            'teen': ['teen', 'teenager', 'adolescent', 'young person', 'high school'],
            'young adult': ['young adult', 'young woman', 'young man', 'college', 'university'],
            'middle-aged': ['middle-aged', 'adult', 'mature', 'middle age'],
            'elderly': ['elderly', 'old', 'senior', 'aged', 'grandmother', 'grandfather']
        }
        
        self.gender_keywords = {
            'male': ['male', 'man', 'boy', 'masculine', 'he', 'his', 'him'],
            'female': ['female', 'woman', 'girl', 'feminine', 'she', 'her'],
            'non-binary': ['androgynous', 'non-binary', 'ambiguous']
        }
        
        self.hair_color_keywords = {
            'black': ['black hair', 'dark hair', 'black'],
            'brown': ['brown hair', 'brunette', 'brown'],
            'blonde': ['blonde hair', 'yellow hair', 'golden hair', 'blonde'],
            'red': ['red hair', 'redhead', 'orange hair', 'red'],
            'blue': ['blue hair', 'blue'],
            'green': ['green hair', 'green'],
            'purple': ['purple hair', 'violet hair', 'purple'],
            'pink': ['pink hair', 'pink'],
            'white': ['white hair', 'silver hair', 'grey hair', 'gray hair', 'white']
        }
        
        self.hair_style_keywords = {
            'ponytail': ['ponytail', 'pony tail'],
            'twintails': ['twintails', 'twin tails', 'pigtails'],
            'bun': ['bun', 'hair bun'],
            'braided': ['braid', 'braided', 'braids'],
            'curly': ['curly', 'wavy', 'curls'],
            'straight': ['straight', 'long straight'],
            'short': ['short hair', 'short'],
            'long': ['long hair', 'long']
        }
        
        self.expression_keywords = {
            'happy': ['smiling', 'happy', 'cheerful', 'joyful', 'smile'],
            'sad': ['sad', 'crying', 'tears', 'melancholy'],
            'angry': ['angry', 'mad', 'furious', 'scowling'],
            'surprised': ['surprised', 'shocked', 'amazed'],
            'neutral': ['neutral', 'calm', 'expressionless'],
            'serious': ['serious', 'stern', 'focused']
        }
    
    def _generate_caption(self, image: Image.Image, prompt: str = None) -> str:
        """Generate caption for image using BLIP2."""
        try:
            if prompt:
                inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
            else:
                inputs = self.processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    num_beams=5,
                    temperature=0.7,
                    do_sample=True
                )
            
            caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return caption.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating caption: {e}")
            return ""
    
    def _extract_attribute_from_text(self, text: str, keywords_dict: Dict[str, List[str]]) -> Optional[str]:
        """Extract attribute from text using keyword matching."""
        text_lower = text.lower()
        
        for attribute, keywords in keywords_dict.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return attribute
        
        return None
    
    def _parse_hair_description(self, hair_text: str) -> Dict[str, Optional[str]]:
        """Parse hair description to extract color, style, and length."""
        hair_info = {
            'color': None,
            'style': None,
            'length': None
        }
        
        # Extract hair color
        hair_info['color'] = self._extract_attribute_from_text(hair_text, self.hair_color_keywords)
        
        # Extract hair style
        hair_info['style'] = self._extract_attribute_from_text(hair_text, self.hair_style_keywords)
        
        # Extract hair length
        if 'short' in hair_text.lower():
            hair_info['length'] = 'short'
        elif 'long' in hair_text.lower():
            hair_info['length'] = 'long'
        elif 'medium' in hair_text.lower() or 'shoulder' in hair_text.lower():
            hair_info['length'] = 'medium'
        
        return hair_info
    
    def _extract_eye_color(self, eye_text: str) -> Optional[str]:
        """Extract eye color from eye description."""
        eye_colors = {
            'brown': ['brown', 'dark brown'],
            'blue': ['blue', 'light blue'],
            'green': ['green', 'emerald'],
            'red': ['red', 'crimson'],
            'purple': ['purple', 'violet'],
            'yellow': ['yellow', 'golden'],
            'pink': ['pink'],
            'black': ['black'],
            'grey': ['grey', 'gray', 'silver']
        }
        
        return self._extract_attribute_from_text(eye_text, eye_colors)
    
    def _extract_clothing_style(self, clothing_text: str) -> Optional[str]:
        """Extract clothing style from clothing description."""
        clothing_styles = {
            'casual': ['casual', 't-shirt', 'jeans', 'hoodie'],
            'formal': ['formal', 'suit', 'dress shirt', 'tie'],
            'traditional': ['kimono', 'yukata', 'traditional'],
            'school uniform': ['school uniform', 'uniform', 'blazer'],
            'swimwear': ['bikini', 'swimsuit', 'swimming'],
            'military': ['military', 'uniform', 'camouflage'],
            'maid': ['maid outfit', 'maid dress', 'apron']
        }
        
        return self._extract_attribute_from_text(clothing_text, clothing_styles)
    
    def process(self, input_data: Any) -> CharacterAttributes:
        """Process image and extract attributes using BLIP2."""
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
        
        # Generate captions for different aspects
        captions = {}
        for aspect, prompt in self.attribute_prompts.items():
            caption = self._generate_caption(image, prompt)
            captions[aspect] = caption
            self.logger.debug(f"{aspect}: {caption}")
        
        # Extract attributes from captions
        attributes = CharacterAttributes()
        
        # Extract age
        if captions['age']:
            attributes.age = self._extract_attribute_from_text(captions['age'], self.age_keywords)
        
        # Extract gender
        if captions['gender']:
            attributes.gender = self._extract_attribute_from_text(captions['gender'], self.gender_keywords)
        
        # Extract hair information
        if captions['hair']:
            hair_info = self._parse_hair_description(captions['hair'])
            attributes.hair_color = hair_info['color']
            attributes.hair_style = hair_info['style']
            attributes.hair_length = hair_info['length']
        
        # Extract eye color
        if captions['eyes']:
            attributes.eye_color = self._extract_eye_color(captions['eyes'])
        
        # Extract clothing style
        if captions['clothing']:
            attributes.dress = self._extract_clothing_style(captions['clothing'])
        
        # Extract facial expression
        if captions['expression']:
            attributes.facial_expression = self._extract_attribute_from_text(captions['expression'], self.expression_keywords)
        
        # Extract accessories
        if captions['accessories']:
            accessories = []
            acc_text = captions['accessories'].lower()
            if 'glasses' in acc_text:
                accessories.append('glasses')
            if 'hat' in acc_text or 'cap' in acc_text:
                accessories.append('hat')
            if 'necklace' in acc_text or 'jewelry' in acc_text:
                accessories.append('jewelry')
            if 'bow' in acc_text:
                accessories.append('bow')
            
            attributes.accessories = accessories if accessories else None
        
        # Set confidence score based on how many attributes were extracted
        extracted_count = sum(1 for attr in [attributes.age, attributes.gender, attributes.hair_color, 
                                            attributes.hair_style, attributes.eye_color, attributes.dress] 
                            if attr is not None)
        attributes.confidence_score = min(0.9, extracted_count / 6.0 + 0.1)
        
        return attributes
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data."""
        if isinstance(input_data, dict):
            return 'image' in input_data and isinstance(input_data['image'], Image.Image)
        elif isinstance(input_data, Image.Image):
            return True
        return False
    
    def get_detailed_description(self, image: Image.Image) -> str:
        """Get a detailed description of the character in the image."""
        return self._generate_caption(image, self.attribute_prompts['general_description'])