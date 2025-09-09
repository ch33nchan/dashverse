"""Tag parser for extracting structured attributes from Danbooru tags."""

import re
from typing import Any, Dict, List, Optional, Set
from .base import PipelineStage, CharacterAttributes

class TagParser(PipelineStage):
    """Parses Danbooru tags and extracts structured character attributes."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("TagParser", config)
        self._initialize_tag_mappings()
    
    def _initialize_tag_mappings(self):
        """Initialize tag mappings for different attributes."""
        
        # Age-related tags
        self.age_tags = {
            'child': {'child', 'loli', 'shota', 'young', 'kid', 'toddler'},
            'teen': {'teen', 'teenager', 'adolescent', 'high school', 'school uniform'},
            'young adult': {'young adult', 'college', 'university', '20s', 'twenties'},
            'middle-aged': {'middle-aged', 'mature', 'adult', '30s', '40s', 'thirties', 'forties'},
            'elderly': {'elderly', 'old', 'senior', 'grandmother', 'grandfather', 'granny'}
        }
        
        # Gender-related tags
        self.gender_tags = {
            'male': {'1boy', 'male', 'man', 'guy', 'masculine', 'male focus'},
            'female': {'1girl', 'female', 'woman', 'girl', 'feminine', 'female focus'},
            'non-binary': {'androgynous', 'non-binary', 'genderless', 'ambiguous gender'}
        }
        
        # Hair style tags
        self.hair_style_tags = {
            'ponytail': {'ponytail', 'side ponytail', 'high ponytail', 'low ponytail'},
            'twintails': {'twintails', 'twin tails', 'pigtails'},
            'bun': {'hair bun', 'double bun', 'side bun', 'top bun'},
            'braided': {'braid', 'braided hair', 'side braid', 'french braid'},
            'curly': {'curly hair', 'wavy hair', 'ringlets'},
            'straight': {'straight hair'},
            'messy': {'messy hair', 'disheveled hair'},
            'spiky': {'spiky hair', 'spiked hair'},
            'bob': {'bob cut', 'bob hair'},
            'hime cut': {'hime cut'}
        }
        
        # Hair color tags
        self.hair_color_tags = {
            'black': {'black hair'},
            'brown': {'brown hair', 'dark brown hair', 'light brown hair'},
            'blonde': {'blonde hair', 'yellow hair', 'golden hair'},
            'red': {'red hair', 'redhead', 'orange hair'},
            'blue': {'blue hair', 'dark blue hair', 'light blue hair'},
            'green': {'green hair', 'dark green hair', 'light green hair'},
            'purple': {'purple hair', 'violet hair'},
            'pink': {'pink hair'},
            'white': {'white hair', 'silver hair', 'grey hair', 'gray hair'},
            'multicolored': {'multicolored hair', 'two-tone hair', 'gradient hair'}
        }
        
        # Hair length tags
        self.hair_length_tags = {
            'short': {'short hair', 'very short hair'},
            'medium': {'medium hair', 'shoulder-length hair'},
            'long': {'long hair', 'very long hair', 'floor-length hair'}
        }
        
        # Eye color tags
        self.eye_color_tags = {
            'brown': {'brown eyes', 'dark brown eyes'},
            'blue': {'blue eyes', 'light blue eyes', 'dark blue eyes'},
            'green': {'green eyes', 'light green eyes'},
            'red': {'red eyes'},
            'purple': {'purple eyes', 'violet eyes'},
            'yellow': {'yellow eyes', 'golden eyes'},
            'pink': {'pink eyes'},
            'black': {'black eyes'},
            'grey': {'grey eyes', 'gray eyes'},
            'heterochromia': {'heterochromia', 'odd eyes'}
        }
        
        # Body type tags
        self.body_type_tags = {
            'slim': {'slim', 'thin', 'skinny', 'slender', 'petite'},
            'muscular': {'muscular', 'toned', 'athletic', 'abs', 'muscle'},
            'curvy': {'curvy', 'voluptuous', 'hourglass figure'},
            'chubby': {'chubby', 'plump', 'thick'},
            'tall': {'tall', 'height'},
            'short': {'short stature', 'small'}
        }
        
        # Dress/clothing style tags
        self.dress_tags = {
            'casual': {'casual', 'everyday clothes', 't-shirt', 'jeans', 'hoodie'},
            'formal': {'formal', 'suit', 'dress shirt', 'tie', 'formal dress'},
            'traditional': {'kimono', 'yukata', 'traditional clothes', 'hanfu', 'qipao'},
            'school uniform': {'school uniform', 'serafuku', 'blazer', 'sailor uniform'},
            'swimwear': {'bikini', 'swimsuit', 'one-piece swimsuit'},
            'lingerie': {'lingerie', 'underwear', 'bra', 'panties'},
            'cosplay': {'cosplay', 'costume'},
            'military': {'military uniform', 'camouflage'},
            'maid': {'maid outfit', 'maid dress', 'apron'},
            'gothic': {'gothic', 'goth', 'dark clothing'}
        }
        
        # Facial expression tags
        self.expression_tags = {
            'happy': {'smile', 'smiling', 'happy', 'cheerful', 'grin'},
            'sad': {'sad', 'crying', 'tears', 'depressed'},
            'angry': {'angry', 'mad', 'furious', 'scowl'},
            'surprised': {'surprised', 'shock', 'amazed'},
            'neutral': {'neutral', 'expressionless', 'blank stare'},
            'embarrassed': {'blush', 'blushing', 'embarrassed', 'shy'},
            'serious': {'serious', 'stern', 'focused'}
        }
        
        # Accessories tags
        self.accessories_tags = {
            'glasses': {'glasses', 'eyewear', 'sunglasses'},
            'hat': {'hat', 'cap', 'headwear', 'beret'},
            'jewelry': {'necklace', 'earrings', 'bracelet', 'ring', 'jewelry'},
            'headband': {'headband', 'hair ornament'},
            'bow': {'hair bow', 'bow', 'ribbon'}
        }
    
    def _extract_tags_from_text(self, text: str) -> List[str]:
        """Extract individual tags from the text."""
        # Split by commas and clean up
        tags = [tag.strip().lower() for tag in text.split(',')]
        
        # Remove empty tags and special markers
        tags = [tag for tag in tags if tag and not tag.startswith('|||')]
        
        return tags
    
    def _find_attribute_match(self, tags: List[str], attribute_mappings: Dict[str, Set[str]]) -> Optional[str]:
        """Find the best matching attribute from tags."""
        tag_set = set(tags)
        
        for attribute, keywords in attribute_mappings.items():
            if any(keyword in tag_set for keyword in keywords):
                return attribute
        
        # Try partial matching for compound tags
        for tag in tags:
            for attribute, keywords in attribute_mappings.items():
                if any(keyword in tag for keyword in keywords):
                    return attribute
        
        return None
    
    def _extract_accessories(self, tags: List[str]) -> List[str]:
        """Extract all accessories mentioned in tags."""
        accessories = []
        tag_set = set(tags)
        
        for accessory, keywords in self.accessories_tags.items():
            if any(keyword in tag_set for keyword in keywords):
                accessories.append(accessory)
        
        # Also check for partial matches
        for tag in tags:
            for accessory, keywords in self.accessories_tags.items():
                if any(keyword in tag for keyword in keywords) and accessory not in accessories:
                    accessories.append(accessory)
        
        return accessories
    
    def _calculate_confidence(self, attributes: CharacterAttributes, tags: List[str]) -> float:
        """Calculate confidence score based on how many attributes were extracted."""
        total_possible = 9  # Main attributes
        extracted = 0
        
        if attributes.age: extracted += 1
        if attributes.gender: extracted += 1
        if attributes.ethnicity: extracted += 1
        if attributes.hair_style: extracted += 1
        if attributes.hair_color: extracted += 1
        if attributes.hair_length: extracted += 1
        if attributes.eye_color: extracted += 1
        if attributes.body_type: extracted += 1
        if attributes.dress: extracted += 1
        
        return extracted / total_possible
    
    def process(self, input_data: Any) -> CharacterAttributes:
        """Parse tags and extract character attributes."""
        if isinstance(input_data, dict):
            tags_text = input_data.get('tags', '')
        elif isinstance(input_data, str):
            tags_text = input_data
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")
        
        if not tags_text:
            return CharacterAttributes()
        
        # Extract individual tags
        tags = self._extract_tags_from_text(tags_text)
        
        # Extract attributes
        attributes = CharacterAttributes()
        
        attributes.age = self._find_attribute_match(tags, self.age_tags)
        attributes.gender = self._find_attribute_match(tags, self.gender_tags)
        attributes.hair_style = self._find_attribute_match(tags, self.hair_style_tags)
        attributes.hair_color = self._find_attribute_match(tags, self.hair_color_tags)
        attributes.hair_length = self._find_attribute_match(tags, self.hair_length_tags)
        attributes.eye_color = self._find_attribute_match(tags, self.eye_color_tags)
        attributes.body_type = self._find_attribute_match(tags, self.body_type_tags)
        attributes.dress = self._find_attribute_match(tags, self.dress_tags)
        attributes.facial_expression = self._find_attribute_match(tags, self.expression_tags)
        
        # Extract accessories
        accessories = self._extract_accessories(tags)
        attributes.accessories = accessories if accessories else None
        
        # Store source tags and calculate confidence
        attributes.source_tags = tags
        attributes.confidence_score = self._calculate_confidence(attributes, tags)
        
        # Note: Ethnicity is difficult to extract from anime/manga tags
        # This would typically require visual analysis
        
        return attributes
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data."""
        if isinstance(input_data, dict):
            return 'tags' in input_data
        elif isinstance(input_data, str):
            return len(input_data.strip()) > 0
        return False