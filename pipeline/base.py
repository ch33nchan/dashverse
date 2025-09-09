"""Base classes for the character attribute extraction pipeline."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CharacterAttributes:
    """Structured character attributes following the problem statement schema."""
    age: Optional[str] = None  # child, teen, young adult, middle-aged, elderly
    gender: Optional[str] = None  # male, female, non-binary
    ethnicity: Optional[str] = None  # Asian, African, Caucasian, etc.
    hair_style: Optional[str] = None  # ponytail, curly, bun, etc.
    hair_color: Optional[str] = None  # black, blonde, red, etc.
    hair_length: Optional[str] = None  # short, medium, long
    eye_color: Optional[str] = None  # brown, blue, green, etc.
    body_type: Optional[str] = None  # slim, muscular, curvy, etc.
    dress: Optional[str] = None  # casual, traditional, formal, etc.
    
    # Optional attributes
    facial_expression: Optional[str] = None
    accessories: Optional[List[str]] = None
    scars: Optional[str] = None
    tattoos: Optional[str] = None
    
    # Metadata
    confidence_score: Optional[float] = None
    source_tags: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format as specified in problem statement."""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if field_value is not None and not field_name.startswith('_'):
                # Convert field names to title case for output
                if field_name == 'hair_style':
                    key = 'Hair Style'
                elif field_name == 'hair_color':
                    key = 'Hair Color'
                elif field_name == 'hair_length':
                    key = 'Hair Length'
                elif field_name == 'eye_color':
                    key = 'Eye Color'
                elif field_name == 'body_type':
                    key = 'Body Type'
                elif field_name == 'facial_expression':
                    key = 'Facial Expression'
                else:
                    key = field_name.replace('_', ' ').title()
                result[key] = field_value
        return result

@dataclass
class ProcessingResult:
    """Result of processing a single item through the pipeline."""
    item_id: str
    attributes: CharacterAttributes
    success: bool
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    intermediate_outputs: Optional[Dict[str, Any]] = None

class PipelineStage(ABC):
    """Abstract base class for all pipeline stages."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Process input data and return output."""
        pass
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data format."""
        return True
    
    def handle_error(self, error: Exception, input_data: Any) -> Any:
        """Handle processing errors gracefully."""
        self.logger.error(f"Error in {self.name}: {str(error)}")
        return None

class Pipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, stages: List[PipelineStage], config: Optional[Dict[str, Any]] = None):
        self.stages = stages
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
    def process_single(self, input_data: Any, item_id: str) -> ProcessingResult:
        """Process a single item through all pipeline stages."""
        import time
        start_time = time.time()
        
        try:
            current_data = input_data
            intermediate_outputs = {}
            
            for stage in self.stages:
                if not stage.validate_input(current_data):
                    raise ValueError(f"Invalid input for stage {stage.name}")
                
                stage_output = stage.process(current_data)
                intermediate_outputs[stage.name] = stage_output
                current_data = stage_output
            
            # Final output should be CharacterAttributes
            if isinstance(current_data, CharacterAttributes):
                attributes = current_data
            else:
                # Convert dict to CharacterAttributes if needed
                attributes = CharacterAttributes(**current_data) if isinstance(current_data, dict) else CharacterAttributes()
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                item_id=item_id,
                attributes=attributes,
                success=True,
                processing_time=processing_time,
                intermediate_outputs=intermediate_outputs
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Pipeline failed for item {item_id}: {str(e)}")
            
            return ProcessingResult(
                item_id=item_id,
                attributes=CharacterAttributes(),
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
    
    def process_batch(self, input_batch: List[tuple], batch_size: int = 32) -> List[ProcessingResult]:
        """Process a batch of items."""
        results = []
        
        for i in range(0, len(input_batch), batch_size):
            batch = input_batch[i:i + batch_size]
            
            for input_data, item_id in batch:
                result = self.process_single(input_data, item_id)
                results.append(result)
                
                if result.success:
                    self.logger.info(f"Successfully processed {item_id}")
                else:
                    self.logger.warning(f"Failed to process {item_id}: {result.error_message}")
        
        return results