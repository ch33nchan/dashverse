"""Main Character Attribute Extraction Pipeline."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image

from pipeline import (
    Pipeline, PipelineStage, CharacterAttributes, ProcessingResult,
    InputLoader, TagParser, CLIPAnalyzer, BLIP2Analyzer, RLOptimizer, 
    AttributeFusion, DatabaseStorage, DatasetItem
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CharacterExtractionPipeline:
    """Complete pipeline for character attribute extraction."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
        
        # Create pipeline stages
        self._create_pipeline()
        
        # Initialize database
        self.db = DatabaseStorage(self.config.get('database', {}))
        
        self.logger.info("Character extraction pipeline initialized")
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        try:
            # Input loader
            self.input_loader = InputLoader(self.config.get('input_loader', {
                'dataset_path': './continued/sensitive'
            }))
            
            # Tag parser
            self.tag_parser = TagParser(self.config.get('tag_parser', {}))
            
            # CLIP analyzer
            self.clip_analyzer = CLIPAnalyzer(self.config.get('clip_analyzer', {
                'model_name': 'openai/clip-vit-base-patch32',
                'confidence_threshold': 0.3
            }))
            
            # BLIP2 analyzer (optional)
            self.use_blip2 = self.config.get('use_blip2', False)
            if self.use_blip2:
                self.blip2_analyzer = BLIP2Analyzer(self.config.get('blip2_analyzer', {}))
            
            # RL optimizer (optional)
            self.use_rl = self.config.get('use_rl', True)
            if self.use_rl:
                self.rl_optimizer = RLOptimizer(self.config.get('rl_optimizer', {}))
            
            # Attribute fusion
            self.attribute_fusion = AttributeFusion(self.config.get('attribute_fusion', {
                'fusion_strategy': 'confidence_weighted'
            }))
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _create_pipeline(self):
        """Create the processing pipeline."""
        # For now, we'll handle the pipeline manually for better control
        # In a production system, you might want to use the Pipeline class
        pass
    
    def extract_from_image(self, image: Union[str, Path, Image.Image], 
                          tags: Optional[str] = None) -> CharacterAttributes:
        """Extract character attributes from a single image."""
        try:
            # Prepare input data
            if isinstance(image, (str, Path)):
                image_path = str(image)
                if tags is None:
                    # Try to find corresponding text file
                    text_path = Path(image_path).with_suffix('.txt')
                    if text_path.exists():
                        with open(text_path, 'r', encoding='utf-8') as f:
                            tags = f.read().strip()
                
                # Load image
                pil_image = Image.open(image_path).convert('RGB')
            else:
                pil_image = image
                image_path = None
            
            # Create input data
            input_data = {
                'image': pil_image,
                'tags': tags or '',
                'image_path': image_path
            }
            
            # Extract using tags
            tag_results = self.tag_parser.process(input_data)
            
            # Extract using CLIP
            clip_results = self.clip_analyzer.process(input_data)
            
            # Extract using BLIP2 if available
            blip2_results = None
            if self.use_blip2 and hasattr(self, 'blip2_analyzer'):
                try:
                    blip2_results = self.blip2_analyzer.process(input_data)
                except Exception as e:
                    self.logger.warning(f"BLIP2 analysis failed: {e}")
            
            # Prepare fusion input
            fusion_input = {
                'tag_results': tag_results,
                'clip_results': clip_results,
                'blip2_results': blip2_results,
                'clip_confidences': {'overall': clip_results.confidence_score or 0.0},
                'tag_confidences': {'overall': tag_results.confidence_score or 0.0},
                'blip2_confidences': {'overall': blip2_results.confidence_score or 0.0} if blip2_results else {'overall': 0.0}
            }
            
            # Apply RL optimization if enabled
            if self.use_rl and hasattr(self, 'rl_optimizer'):
                try:
                    rl_results = self.rl_optimizer.process(fusion_input)
                    fusion_input['rl_results'] = rl_results
                except Exception as e:
                    self.logger.warning(f"RL optimization failed: {e}")
            
            # Fuse results
            final_attributes = self.attribute_fusion.process(fusion_input)
            
            return final_attributes
            
        except Exception as e:
            self.logger.error(f"Failed to extract attributes: {e}")
            return CharacterAttributes()
    
    def extract_from_dataset_item(self, item: DatasetItem) -> ProcessingResult:
        """Extract attributes from a dataset item."""
        start_time = time.time()
        
        try:
            # Check cache first
            cached_result = self.db.get_result(item.item_id)
            if cached_result:
                self.logger.debug(f"Using cached result for {item.item_id}")
                return cached_result
            
            # Load data
            input_data = self.input_loader.process(item)
            
            # Extract attributes
            attributes = self.extract_from_image(
                input_data['image'], 
                input_data['tags']
            )
            
            processing_time = time.time() - start_time
            
            # Create result
            result = ProcessingResult(
                item_id=item.item_id,
                attributes=attributes,
                success=True,
                processing_time=processing_time
            )
            
            # Store in database
            self.db.store_result(result)
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            
            result = ProcessingResult(
                item_id=item.item_id,
                attributes=CharacterAttributes(),
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
            
            # Store failed result too
            self.db.store_result(result)
            
            return result
    
    def process_batch(self, items: List[DatasetItem], 
                     batch_size: int = 8) -> List[ProcessingResult]:
        """Process a batch of dataset items."""
        results = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(items) + batch_size - 1)//batch_size}")
            
            for item in batch:
                result = self.extract_from_dataset_item(item)
                results.append(result)
                
                if result.success:
                    self.logger.info(f"✓ {item.item_id}: {len([attr for attr in result.attributes.__dict__.values() if attr])} attributes")
                else:
                    self.logger.warning(f"✗ {item.item_id}: {result.error_message}")
        
        return results
    
    def process_dataset(self, limit: Optional[int] = None) -> List[ProcessingResult]:
        """Process the entire dataset."""
        # Discover dataset items
        all_items = self.input_loader.discover_dataset_items()
        
        if limit:
            all_items = all_items[:limit]
        
        self.logger.info(f"Processing {len(all_items)} items from dataset")
        
        return self.process_batch(all_items)
    
    def get_sample_results(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get sample results for demonstration."""
        sample_items = self.input_loader.get_sample_items(n)
        results = []
        
        for item in sample_items:
            result = self.extract_from_dataset_item(item)
            
            if result.success:
                results.append({
                    'item_id': result.item_id,
                    'image_path': item.image_path,
                    'attributes': result.attributes.to_dict(),
                    'confidence': result.attributes.confidence_score,
                    'processing_time': result.processing_time
                })
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        db_stats = self.db.get_statistics()
        
        # Add pipeline-specific stats
        stats = {
            'pipeline_config': {
                'use_rl': self.use_rl,
                'fusion_strategy': self.config.get('attribute_fusion', {}).get('fusion_strategy', 'confidence_weighted'),
                'clip_model': self.config.get('clip_analyzer', {}).get('model_name', 'openai/clip-vit-base-patch32')
            },
            'database_stats': db_stats
        }
        
        return stats
    
    def save_models(self):
        """Save trained models."""
        if self.use_rl and hasattr(self, 'rl_optimizer'):
            self.rl_optimizer.save_model()
            self.logger.info("RL model saved")
    
    def benchmark_performance(self, n_samples: int = 50) -> Dict[str, Any]:
        """Benchmark pipeline performance."""
        sample_items = self.input_loader.get_sample_items(n_samples)
        
        start_time = time.time()
        results = self.process_batch(sample_items)
        total_time = time.time() - start_time
        
        successful_results = [r for r in results if r.success]
        
        benchmark = {
            'total_items': len(results),
            'successful_items': len(successful_results),
            'success_rate': len(successful_results) / len(results),
            'total_time': total_time,
            'avg_time_per_item': total_time / len(results),
            'throughput_items_per_second': len(results) / total_time,
            'avg_confidence': sum(r.attributes.confidence_score or 0 for r in successful_results) / len(successful_results) if successful_results else 0
        }
        
        return benchmark

# Factory function for easy pipeline creation
def create_pipeline(config: Optional[Dict[str, Any]] = None) -> CharacterExtractionPipeline:
    """Create a character extraction pipeline with default configuration."""
    default_config = {
        'input_loader': {
            'dataset_path': './continued/sensitive'
        },
        'clip_analyzer': {
            'model_name': 'openai/clip-vit-base-patch32',
            'confidence_threshold': 0.3
        },
        'attribute_fusion': {
            'fusion_strategy': 'confidence_weighted'
        },
        'use_rl': True,
        'database': {
            'db_path': './data/character_attributes.db',
            'enable_caching': True
        }
    }
    
    if config:
        # Merge configs
        for key, value in config.items():
            if isinstance(value, dict) and key in default_config:
                default_config[key].update(value)
            else:
                default_config[key] = value
    
    return CharacterExtractionPipeline(default_config)