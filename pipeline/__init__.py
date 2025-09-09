"""Character Attribute Extraction Pipeline."""

from .base import PipelineStage, CharacterAttributes, ProcessingResult, Pipeline
from .input_loader import InputLoader, DatasetItem
from .tag_parser import TagParser
from .clip_analyzer import CLIPAnalyzer
from .rl_optimizer import RLOptimizer
from .attribute_fusion import AttributeFusion
from .database import DatabaseStorage

__all__ = [
    'PipelineStage',
    'CharacterAttributes', 
    'ProcessingResult',
    'Pipeline',
    'InputLoader',
    'DatasetItem',
    'TagParser',
    'CLIPAnalyzer',
    'RLOptimizer',
    'AttributeFusion',
    'DatabaseStorage'
]