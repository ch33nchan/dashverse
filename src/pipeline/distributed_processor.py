"""Distributed processing module for scaling to 5M+ samples using Ray."""

import logging
import time
import hashlib
from typing import List, Dict, Any, Optional, Iterator
from pathlib import Path
import numpy as np
from PIL import Image
import io
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    ray = None

from .base import PipelineStage, CharacterAttributes, ProcessingResult
from .input_loader import DatasetItem

logger = logging.getLogger(__name__)

if RAY_AVAILABLE:
    @ray.remote
    class DistributedWorker:
        """Ray actor for distributed character attribute extraction."""
        
        def __init__(self, pipeline_config: Dict[str, Any]):
            """Initialize worker with pipeline configuration."""
            self.pipeline_config = pipeline_config
            self.pipeline = None
            self._initialize_pipeline()
        
        def _initialize_pipeline(self):
            """Initialize the character extraction pipeline on worker."""
            try:
                from character_pipeline import create_pipeline
                self.pipeline = create_pipeline(self.pipeline_config)
                logger.info(f"Worker {ray.get_runtime_context().get_worker_id()} initialized")
            except Exception as e:
                logger.error(f"Failed to initialize pipeline on worker: {e}")
                raise
        
        def process_batch(self, items: List[DatasetItem]) -> List[ProcessingResult]:
            """Process a batch of items on this worker."""
            results = []
            for item in items:
                try:
                    start_time = time.time()
                    result = self.pipeline.extract_from_image(item.image_path)
                    processing_time = time.time() - start_time
                    
                    results.append(ProcessingResult(
                        item_id=item.item_id,
                        attributes=result,
                        success=True,
                        processing_time=processing_time
                    ))
                except Exception as e:
                    logger.warning(f"Failed to process {item.item_id}: {e}")
                    results.append(ProcessingResult(
                        item_id=item.item_id,
                        attributes=CharacterAttributes(),
                        success=False,
                        error_message=str(e)
                    ))
            return results
        
        def get_worker_stats(self) -> Dict[str, Any]:
            """Get worker statistics and health status."""
            return {
                'worker_id': ray.get_runtime_context().get_worker_id(),
                'node_id': ray.get_runtime_context().get_node_id(),
                'memory_usage': ray.cluster_resources().get('memory', 0),
                'cpu_usage': ray.cluster_resources().get('CPU', 0)
            }
else:
    class DistributedWorker:
        """Fallback worker when Ray is not available."""
        def __init__(self, pipeline_config: Dict[str, Any]):
            raise ImportError("Ray is not available. Install with: pip install ray[default]")

class DistributedProcessor(PipelineStage):
    """Distributed processor for handling large-scale character extraction."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("DistributedProcessor", config)
        
        # Configuration
        self.num_workers = config.get('num_workers', 4) if config else 4
        self.batch_size = config.get('batch_size', 32) if config else 32
        self.max_retries = config.get('max_retries', 3) if config else 3
        self.checkpoint_interval = config.get('checkpoint_interval', 1000) if config else 1000
        
        # Ray configuration
        self.ray_config = config.get('ray_config', {}) if config else {}
        
        # State
        self.workers = []
        self.is_initialized = False
        
    def initialize_cluster(self) -> bool:
        """Initialize Ray cluster and workers."""
        if not RAY_AVAILABLE:
            self.logger.error("Ray is not available. Install with: pip install ray[default]")
            return False
            
        try:
            if not ray.is_initialized():
                ray.init(**self.ray_config)
            
            # Create distributed workers
            pipeline_config = self.config.get('pipeline_config', {})
            self.workers = [
                DistributedWorker.remote(pipeline_config) 
                for _ in range(self.num_workers)
            ]
            
            # Test worker initialization
            health_checks = ray.get([worker.get_worker_stats.remote() for worker in self.workers])
            self.logger.info(f"Initialized {len(health_checks)} workers successfully")
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Ray cluster: {e}")
            return False
    
    def process_large_dataset(self, dataset_path: str, output_path: str) -> Dict[str, Any]:
        """Process large dataset with distributed workers."""
        if not self.is_initialized:
            if not self.initialize_cluster():
                raise RuntimeError("Failed to initialize distributed cluster")
        
        start_time = time.time()
        total_processed = 0
        total_successful = 0
        checkpoints = []
        
        try:
            # Load dataset in streaming fashion
            dataset_stream = self._create_dataset_stream(dataset_path)
            
            # Process in distributed batches
            batch_futures = []
            current_batch = []
            
            for item in dataset_stream:
                current_batch.append(item)
                
                if len(current_batch) >= self.batch_size:
                    # Distribute batch to available worker
                    worker = self.workers[len(batch_futures) % len(self.workers)]
                    future = worker.process_batch.remote(current_batch.copy())
                    batch_futures.append(future)
                    current_batch = []
                    
                    # Process completed batches
                    if len(batch_futures) >= self.num_workers * 2:
                        completed_results = self._collect_completed_batches(batch_futures)
                        total_processed += sum(len(results) for results in completed_results)
                        total_successful += sum(
                            sum(1 for r in results if r.success) 
                            for results in completed_results
                        )
                        
                        # Save checkpoint
                        if total_processed % self.checkpoint_interval == 0:
                            checkpoint = self._save_checkpoint(completed_results, output_path, total_processed)
                            checkpoints.append(checkpoint)
                            self.logger.info(f"Checkpoint saved: {total_processed} items processed")
            
            # Process remaining batch
            if current_batch:
                worker = self.workers[0]
                future = worker.process_batch.remote(current_batch)
                batch_futures.append(future)
            
            # Collect all remaining results
            if batch_futures:
                remaining_results = ray.get(batch_futures)
                total_processed += sum(len(results) for results in remaining_results)
                total_successful += sum(
                    sum(1 for r in results if r.success) 
                    for results in remaining_results
                )
                
                # Final checkpoint
                final_checkpoint = self._save_checkpoint(remaining_results, output_path, total_processed)
                checkpoints.append(final_checkpoint)
            
            processing_time = time.time() - start_time
            
            return {
                'total_processed': total_processed,
                'total_successful': total_successful,
                'success_rate': total_successful / total_processed if total_processed > 0 else 0,
                'processing_time': processing_time,
                'throughput': total_processed / processing_time if processing_time > 0 else 0,
                'checkpoints': checkpoints,
                'num_workers': len(self.workers)
            }
            
        except Exception as e:
            self.logger.error(f"Distributed processing failed: {e}")
            raise
    
    def _create_dataset_stream(self, dataset_path: str) -> Iterator[DatasetItem]:
        """Create streaming iterator for large dataset."""
        dataset_path = Path(dataset_path)
        
        if dataset_path.is_file():
            # Single file - assume it's a list of image paths
            with open(dataset_path, 'r') as f:
                for line_num, line in enumerate(f):
                    image_path = line.strip()
                    if image_path and Path(image_path).exists():
                        yield DatasetItem(
                            item_id=f"item_{line_num}",
                            image_path=image_path
                        )
        else:
            # Directory - iterate through image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
            for idx, image_path in enumerate(dataset_path.rglob('*')):
                if image_path.suffix.lower() in image_extensions:
                    yield DatasetItem(
                        item_id=f"item_{idx}",
                        image_path=str(image_path)
                    )
    
    def _collect_completed_batches(self, batch_futures: List) -> List[List[ProcessingResult]]:
        """Collect completed batch results and remove from futures list."""
        completed_results = []
        ready_futures, batch_futures[:] = ray.wait(batch_futures, num_returns=len(batch_futures), timeout=0)
        
        if ready_futures:
            completed_results = ray.get(ready_futures)
        
        return completed_results
    
    def _save_checkpoint(self, results: List[List[ProcessingResult]], output_path: str, total_processed: int) -> str:
        """Save processing checkpoint."""
        checkpoint_path = f"{output_path}_checkpoint_{total_processed}.pkl"
        
        # Flatten results
        flat_results = []
        for batch_results in results:
            flat_results.extend(batch_results)
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(flat_results, f)
        
        return checkpoint_path
    
    def estimate_scalability(self, sample_size: int = 1000) -> Dict[str, Any]:
        """Estimate processing capabilities for 5M scale."""
        if not self.is_initialized:
            if not self.initialize_cluster():
                raise RuntimeError("Failed to initialize distributed cluster")
        
        # Create sample dataset
        sample_items = [
            DatasetItem(item_id=f"sample_{i}", image_path="sample_image.jpg")
            for i in range(sample_size)
        ]
        
        # Measure processing time
        start_time = time.time()
        
        # Distribute sample processing
        batch_size = min(self.batch_size, sample_size // self.num_workers)
        futures = []
        
        for i in range(0, sample_size, batch_size):
            batch = sample_items[i:i + batch_size]
            worker = self.workers[i // batch_size % len(self.workers)]
            future = worker.process_batch.remote(batch)
            futures.append(future)
        
        # Collect results
        results = ray.get(futures)
        processing_time = time.time() - start_time
        
        # Calculate metrics
        total_items = sum(len(batch_results) for batch_results in results)
        throughput = total_items / processing_time if processing_time > 0 else 0
        
        # Estimate 5M scale
        estimated_5m_time = 5_000_000 / throughput if throughput > 0 else float('inf')
        estimated_5m_hours = estimated_5m_time / 3600
        
        # Memory estimation
        cluster_resources = ray.cluster_resources()
        total_memory_gb = cluster_resources.get('memory', 0) / (1024**3)
        total_cpus = cluster_resources.get('CPU', 0)
        
        return {
            'sample_size': sample_size,
            'processing_time': processing_time,
            'throughput_per_second': throughput,
            'estimated_5m_processing_time_hours': estimated_5m_hours,
            'estimated_5m_processing_time_days': estimated_5m_hours / 24,
            'cluster_resources': {
                'total_memory_gb': total_memory_gb,
                'total_cpus': total_cpus,
                'num_workers': len(self.workers)
            },
            'scalability_recommendations': self._generate_scalability_recommendations(throughput, total_memory_gb, total_cpus)
        }
    
    def _generate_scalability_recommendations(self, throughput: float, memory_gb: float, cpus: int) -> List[str]:
        """Generate recommendations for scaling to 5M samples."""
        recommendations = []
        
        if throughput < 100:  # Less than 100 items/second
            recommendations.append("Consider increasing worker count or optimizing pipeline")
        
        if memory_gb < 32:
            recommendations.append("Increase cluster memory for better caching and model loading")
        
        if cpus < 16:
            recommendations.append("Add more CPU cores for parallel processing")
        
        recommendations.extend([
            "Implement data sharding across multiple storage systems",
            "Use Redis cluster for distributed caching",
            "Consider GPU acceleration for CLIP model inference",
            "Implement progressive loading to reduce memory footprint",
            "Use Apache Beam or Spark for even larger scale processing"
        ])
        
        return recommendations
    
    def shutdown(self):
        """Shutdown distributed cluster."""
        if self.is_initialized and RAY_AVAILABLE:
            ray.shutdown()
            self.is_initialized = False
            self.logger.info("Distributed cluster shutdown complete")
    
    def process(self, input_data: Any) -> Any:
        """Process input using distributed workers."""
        if not RAY_AVAILABLE:
            raise RuntimeError("Ray is not available. Install with: pip install ray[default]")
            
        if isinstance(input_data, str):
            # Assume it's a dataset path
            return self.process_large_dataset(input_data, "distributed_output")
        else:
            raise ValueError("DistributedProcessor expects dataset path as input")
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data."""
        return isinstance(input_data, str) and Path(input_data).exists()