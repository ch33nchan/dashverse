"""Main Character Attribute Extraction Pipeline."""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image
import asyncio

from .pipeline import (
    Pipeline,
    PipelineStage,
    CharacterAttributes,
    ProcessingResult,
    InputLoader,
    TagParser,
    CLIPAnalyzer,
    BLIP2Analyzer,
    RLOptimizer,
    AttributeFusion,
    DatabaseStorage,
    DatasetItem,
    EdgeCaseHandler,
    ImagePreprocessor,
    DistributedProcessor,
    AdvancedCacheManager,
)
from .rl_pipeline_integration import create_rl_enhanced_pipeline, ProductionRLPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CharacterExtractionPipeline:
    """Complete pipeline for character attribute extraction."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        self._initialize_components()
        self.db = DatabaseStorage(self.config.get('database', {}))
        
        rl_model_path = self.config.get('rl_model_path')
        self.rl_pipeline = create_rl_enhanced_pipeline(self, rl_model_path)
        
        self.logger.info("Character extraction pipeline initialized")
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        try:
            self.input_loader = InputLoader(self.config.get('input_loader'))
            self.tag_parser = TagParser(self.config.get('tag_parser'))
            self.clip_analyzer = CLIPAnalyzer(self.config.get('clip_analyzer'))
            
            self.use_blip2 = self.config.get('use_blip2', False)
            if self.use_blip2:
                self.blip2_analyzer = BLIP2Analyzer(self.config.get('blip2_analyzer'))
            
            self.use_rl = self.config.get('use_rl', True)
            if self.use_rl:
                self.rl_optimizer = RLOptimizer(self.config.get('rl_optimizer'))
            
            self.attribute_fusion = AttributeFusion(self.config.get('attribute_fusion'))
            self.edge_case_handler = EdgeCaseHandler(self.config.get('edge_case_handler'))
            self.preprocessor = ImagePreprocessor(self.config.get('image_preprocessor'))
            self.cache_manager = AdvancedCacheManager(self.config.get('cache_manager'))
            
            try:
                self.distributed_processor = DistributedProcessor(self.config.get('distributed_processor'))
                self.distributed_available = True
            except ImportError:
                self.distributed_processor = None
                self.distributed_available = False
                self.logger.info("Distributed processing not available")
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _create_pipeline(self):
        """Create the processing pipeline."""
        pass
    
    def extract_from_image(self, image: Union[str, Path, Image.Image], 
                          tags: Optional[str] = None) -> CharacterAttributes:
        """Extract character attributes from a single image."""
        try:
            if self.config.get('use_rl_primary', True):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        self.rl_pipeline.extract_from_image(image, tags)
                    )
                    if result.confidence_score and result.confidence_score > 0.2:
                        return result
                finally:
                    loop.close()
            
            return self._extract_from_image_fallback(image, tags)
            
        except Exception as e:
            self.logger.error(f"Failed to extract attributes: {e}")
            return CharacterAttributes()
    
    def _extract_from_image_fallback(self, image: Union[str, Path, Image.Image], 
                                   tags: Optional[str] = None) -> CharacterAttributes:
        """Fallback extraction method using traditional pipeline."""
        try:
            if isinstance(image, (str, Path)):
                image_path = str(image)
                if tags is None:
                    text_path = Path(image_path).with_suffix('.txt')
                    if text_path.exists():
                        with open(text_path, 'r', encoding='utf-8') as f:
                            tags = f.read().strip()
                
                pil_image = Image.open(image_path).convert('RGB')
            else:
                pil_image = image
                image_path = None
            
            preprocess_result = self.preprocessor.preprocess_image(pil_image)
            processed_image = preprocess_result.get('processed_image', pil_image)
            
            edge_analysis = self.edge_case_handler.analyze_image_content(processed_image)
            
            quality_info = {
                'edge_cases': edge_analysis.get('edge_cases', []),
                'quality_score': edge_analysis.get('confidence', 0.0),
                'recommendation': edge_analysis.get('recommendation', 'unknown'),
                'is_good_quality': edge_analysis.get('recommendation') == 'process'
            }
            
            if edge_analysis.get('edge_cases'):
                self.logger.info(f"Edge cases detected: {edge_analysis.get('edge_cases')} - continuing with extraction")
            
            input_data = {
                'image': processed_image,
                'tags': tags or '',
                'image_path': image_path
            }
            
            tag_results = self.tag_parser.process(input_data)
            clip_results = self.clip_analyzer.process(input_data)
            
            blip2_results = None
            if self.use_blip2 and hasattr(self, 'blip2_analyzer'):
                try:
                    blip2_results = self.blip2_analyzer.process(input_data)
                except Exception as e:
                    self.logger.warning(f"BLIP2 analysis failed: {e}")
            
            fusion_input = {
                'tag_results': tag_results,
                'clip_results': clip_results,
                'blip2_results': blip2_results,
                'clip_confidences': {'overall': clip_results.confidence_score or 0.0},
                'tag_confidences': {'overall': tag_results.confidence_score or 0.0},
                'blip2_confidences': {'overall': blip2_results.confidence_score or 0.0} if blip2_results else {'overall': 0.0}
            }
            
            if self.use_rl and hasattr(self, 'rl_optimizer'):
                try:
                    rl_results = self.rl_optimizer.process(fusion_input)
                    fusion_input['rl_results'] = rl_results
                except Exception as e:
                    self.logger.warning(f"RL optimization failed: {e}")
            
            final_attributes = self.attribute_fusion.process(fusion_input)
            
            if hasattr(final_attributes, 'metadata'):
                final_attributes.metadata = {
                    'edge_cases': edge_analysis['edge_cases'],
                    'preprocessing_applied': preprocess_result['preprocessing_info']['steps_applied'],
                    'confidence_adjustment': edge_analysis['confidence'],
                    'quality_info': quality_info
                }
            
            return final_attributes
            
        except Exception as e:
            self.logger.error(f"Fallback extraction failed: {e}")
            return CharacterAttributes()
    
    def extract_from_dataset_item(self, item: DatasetItem) -> ProcessingResult:
        """Extract attributes from a dataset item."""
        start_time = time.time()
        
        try:
            cached_result = self.db.get_result(item.item_id)
            if cached_result:
                self.logger.debug(f"Using cached result for {item.item_id}")
                return cached_result
            
            input_data = self.input_loader.process(item)
            
            attributes = self.extract_from_image(
                input_data['image'], 
                input_data['tags']
            )
            
            processing_time = time.time() - start_time
            
            result = ProcessingResult(
                item_id=item.item_id,
                attributes=attributes,
                success=True,
                processing_time=processing_time
            )
            
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
            
            self.db.store_result(result)
            
            return result
    
    def process_batch(self, items: List[DatasetItem], 
                     batch_size: int = 8) -> List[ProcessingResult]:
        """Process a batch of dataset items."""
        if self.config.get('use_rl_primary', True):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    results = loop.run_until_complete(
                        self.rl_pipeline.process_batch(items)
                    )
                    
                    success_rate = sum(1 for r in results if r.success) / len(results) if results else 0
                    if success_rate > 0.5:
                        return results
                finally:
                    loop.close()
            except Exception as e:
                self.logger.warning(f"RL batch processing failed: {e}, falling back to traditional")
        
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
        items = self.input_loader.get_sample_items(limit)
        return self.process_batch(items)
    
    def get_sample_results(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get sample results for demonstration."""
        items = self.input_loader.get_sample_items(n)
        results = self.process_batch(items)
        
        sample_results = []
        for result in results:
            sample_results.append({
                'item_id': result.item_id,
                'success': result.success,
                'attributes': result.attributes.to_dict(),
                'processing_time': result.processing_time,
                'error': result.error_message if not result.success else None
            })
        
        return sample_results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats = self.db.get_statistics()
        
        rl_status = self.rl_pipeline.get_status() if hasattr(self, 'rl_pipeline') else {}
        
        return {
            'total_processed': stats.get('total_count', 0),
            'success_rate': stats.get('success_rate', 0.0),
            'avg_processing_time': stats.get('avg_processing_time', 0.0),
            'avg_confidence': stats.get('avg_confidence', 0.0),
            'rl_status': rl_status
        }
    
    def save_models(self):
        """Save trained models."""
        if hasattr(self, 'rl_pipeline'):
            self.rl_pipeline.rl_pipeline.save_training_data("rl_training_data.json")
    
    def benchmark_performance(self, n_samples: int = 50) -> Dict[str, Any]:
        """Benchmark pipeline performance."""
        start_time = time.time()
        
        items = self.input_loader.get_sample_items(n_samples)
        results = self.process_batch(items)
        
        total_time = time.time() - start_time
        successful_results = [r for r in results if r.success]
        
        return {
            'total_samples': n_samples,
            'successful_extractions': len(successful_results),
            'success_rate': len(successful_results) / n_samples,
            'total_time': total_time,
            'avg_time_per_sample': total_time / n_samples,
            'avg_confidence': sum(r.attributes.confidence_score or 0 for r in successful_results) / len(successful_results) if successful_results else 0
        }
    
    async def trigger_rl_retraining(self) -> bool:
        """Trigger RL model retraining."""
        if hasattr(self, 'rl_pipeline'):
            return await self.rl_pipeline.trigger_retraining()
        return False

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
        'use_rl_primary': True,
        'rl_model_path': None,
        'database': {
            'db_path': './data/character_attributes.db',
            'enable_caching': True
        }
    }
    
    if config:
        for key, value in config.items():
            if isinstance(value, dict) and key in default_config:
                default_config[key].update(value)
            else:
                default_config[key] = value
    
    return CharacterExtractionPipeline(default_config)