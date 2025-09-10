"""Streaming data processor for memory-efficient handling of large datasets."""

import logging
import time
import json
import csv
from typing import Iterator, Dict, Any, Optional, List, Callable
from pathlib import Path
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
import psutil
from dataclasses import asdict

from .base import PipelineStage, CharacterAttributes, ProcessingResult
from .input_loader import DatasetItem

logger = logging.getLogger(__name__)

class MemoryMonitor:
    """Monitor memory usage and trigger cleanup when needed."""
    
    def __init__(self, max_memory_gb: float = 8.0):
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        self.process = psutil.Process()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory_info = self.process.memory_info()
        return {
            'rss_gb': memory_info.rss / (1024**3),
            'vms_gb': memory_info.vms / (1024**3),
            'percent': self.process.memory_percent()
        }
    
    def should_cleanup(self) -> bool:
        """Check if memory cleanup is needed."""
        return self.process.memory_info().rss > self.max_memory_bytes
    
    def force_cleanup(self):
        """Force garbage collection and memory cleanup."""
        gc.collect()
        logger.info(f"Memory cleanup performed. Current usage: {self.get_memory_usage()}")

class StreamingDataLoader:
    """Memory-efficient data loader for large datasets."""
    
    def __init__(self, batch_size: int = 32, prefetch_size: int = 2):
        self.batch_size = batch_size
        self.prefetch_size = prefetch_size
    
    def load_from_directory(self, directory_path: str, 
                          file_extensions: List[str] = None) -> Iterator[DatasetItem]:
        """Stream dataset items from directory."""
        if file_extensions is None:
            file_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        directory = Path(directory_path)
        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory_path}")
        
        # Use generator to avoid loading all paths into memory
        for idx, file_path in enumerate(directory.rglob('*')):
            if file_path.suffix.lower() in file_extensions:
                # Check for corresponding text file
                text_path = file_path.with_suffix('.txt')
                text_file = str(text_path) if text_path.exists() else None
                
                yield DatasetItem(
                    item_id=f"item_{idx}_{file_path.stem}",
                    image_path=str(file_path),
                    text_path=text_file
                )
    
    def load_from_manifest(self, manifest_path: str) -> Iterator[DatasetItem]:
        """Stream dataset items from manifest file."""
        manifest_file = Path(manifest_path)
        
        if manifest_file.suffix.lower() == '.json':
            yield from self._load_from_json_manifest(manifest_file)
        elif manifest_file.suffix.lower() == '.csv':
            yield from self._load_from_csv_manifest(manifest_file)
        elif manifest_file.suffix.lower() == '.txt':
            yield from self._load_from_text_manifest(manifest_file)
        else:
            raise ValueError(f"Unsupported manifest format: {manifest_file.suffix}")
    
    def _load_from_json_manifest(self, manifest_path: Path) -> Iterator[DatasetItem]:
        """Load from JSON Lines manifest."""
        with open(manifest_path, 'r') as f:
            for line_num, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    yield DatasetItem(
                        item_id=data.get('id', f"item_{line_num}"),
                        image_path=data['image_path'],
                        text_path=data.get('text_path')
                    )
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Skipping invalid line {line_num} in manifest: {e}")
    
    def _load_from_csv_manifest(self, manifest_path: Path) -> Iterator[DatasetItem]:
        """Load from CSV manifest."""
        with open(manifest_path, 'r') as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader):
                try:
                    yield DatasetItem(
                        item_id=row.get('id', f"item_{row_num}"),
                        image_path=row['image_path'],
                        text_path=row.get('text_path')
                    )
                except KeyError as e:
                    logger.warning(f"Skipping invalid row {row_num} in CSV: missing {e}")
    
    def _load_from_text_manifest(self, manifest_path: Path) -> Iterator[DatasetItem]:
        """Load from simple text file (one image path per line)."""
        with open(manifest_path, 'r') as f:
            for line_num, line in enumerate(f):
                image_path = line.strip()
                if image_path and Path(image_path).exists():
                    yield DatasetItem(
                        item_id=f"item_{line_num}",
                        image_path=image_path
                    )
    
    def create_batches(self, data_stream: Iterator[DatasetItem]) -> Iterator[List[DatasetItem]]:
        """Create batches from data stream."""
        batch = []
        for item in data_stream:
            batch.append(item)
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        
        # Yield remaining items
        if batch:
            yield batch

class StreamingProcessor(PipelineStage):
    """Streaming processor for memory-efficient large-scale processing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("StreamingProcessor", config)
        
        # Configuration
        self.batch_size = config.get('batch_size', 32) if config else 32
        self.num_workers = config.get('num_workers', 4) if config else 4
        self.max_memory_gb = config.get('max_memory_gb', 8.0) if config else 8.0
        self.checkpoint_interval = config.get('checkpoint_interval', 1000) if config else 1000
        self.output_format = config.get('output_format', 'jsonl') if config else 'jsonl'
        
        # Components
        self.data_loader = StreamingDataLoader(self.batch_size)
        self.memory_monitor = MemoryMonitor(self.max_memory_gb)
        
        # State
        self.processed_count = 0
        self.success_count = 0
        self.error_count = 0
        
        # Pipeline reference
        self.pipeline = None
    
    def set_pipeline(self, pipeline):
        """Set the character extraction pipeline."""
        self.pipeline = pipeline
    
    def process_stream(self, data_source: str, output_path: str, 
                      progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Process data stream with memory-efficient streaming."""
        if self.pipeline is None:
            raise ValueError("Pipeline not set. Call set_pipeline() first.")
        
        start_time = time.time()
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine data source type
        source_path = Path(data_source)
        if source_path.is_dir():
            data_stream = self.data_loader.load_from_directory(data_source)
        elif source_path.is_file():
            data_stream = self.data_loader.load_from_manifest(data_source)
        else:
            raise ValueError(f"Invalid data source: {data_source}")
        
        # Create batch stream
        batch_stream = self.data_loader.create_batches(data_stream)
        
        # Process batches
        with open(output_file, 'w') as output_f:
            self._write_output_header(output_f)
            
            for batch_num, batch in enumerate(batch_stream):
                try:
                    # Process batch
                    batch_results = self._process_batch(batch)
                    
                    # Write results
                    for result in batch_results:
                        self._write_result(output_f, result)
                        
                        if result.success:
                            self.success_count += 1
                        else:
                            self.error_count += 1
                        
                        self.processed_count += 1
                    
                    # Progress callback
                    if progress_callback:
                        progress_callback({
                            'processed': self.processed_count,
                            'success': self.success_count,
                            'errors': self.error_count,
                            'batch_num': batch_num
                        })
                    
                    # Memory management
                    if self.memory_monitor.should_cleanup():
                        self.memory_monitor.force_cleanup()
                    
                    # Checkpoint
                    if self.processed_count % self.checkpoint_interval == 0:
                        self._create_checkpoint(output_path, batch_num)
                        logger.info(f"Checkpoint: {self.processed_count} items processed")
                
                except Exception as e:
                    logger.error(f"Failed to process batch {batch_num}: {e}")
                    self.error_count += len(batch)
                    self.processed_count += len(batch)
        
        processing_time = time.time() - start_time
        
        return {
            'total_processed': self.processed_count,
            'successful': self.success_count,
            'errors': self.error_count,
            'success_rate': self.success_count / self.processed_count if self.processed_count > 0 else 0,
            'processing_time': processing_time,
            'throughput': self.processed_count / processing_time if processing_time > 0 else 0,
            'memory_usage': self.memory_monitor.get_memory_usage(),
            'output_file': str(output_file)
        }
    
    def _process_batch(self, batch: List[DatasetItem]) -> List[ProcessingResult]:
        """Process a batch of items using thread pool."""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all items in batch
            future_to_item = {
                executor.submit(self._process_single_item, item): item 
                for item in batch
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process {item.item_id}: {e}")
                    results.append(ProcessingResult(
                        item_id=item.item_id,
                        attributes=CharacterAttributes(),
                        success=False,
                        error_message=str(e)
                    ))
        
        return results
    
    def _process_single_item(self, item: DatasetItem) -> ProcessingResult:
        """Process a single dataset item."""
        start_time = time.time()
        
        try:
            # Extract attributes using pipeline
            attributes = self.pipeline.extract_from_image(item.image_path)
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                item_id=item.item_id,
                attributes=attributes,
                success=True,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            return ProcessingResult(
                item_id=item.item_id,
                attributes=CharacterAttributes(),
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
    
    def _write_output_header(self, file_handle):
        """Write output file header based on format."""
        if self.output_format == 'csv':
            writer = csv.writer(file_handle)
            writer.writerow([
                'item_id', 'success', 'age', 'gender', 'ethnicity',
                'hair_style', 'hair_color', 'hair_length', 'eye_color',
                'body_type', 'dress', 'confidence_score', 'processing_time', 'error_message'
            ])
    
    def _write_result(self, file_handle, result: ProcessingResult):
        """Write processing result to output file."""
        if self.output_format == 'jsonl':
            # JSON Lines format
            result_dict = {
                'item_id': result.item_id,
                'success': result.success,
                'attributes': asdict(result.attributes),
                'processing_time': result.processing_time,
                'error_message': result.error_message
            }
            file_handle.write(json.dumps(result_dict) + '\n')
            
        elif self.output_format == 'csv':
            # CSV format
            writer = csv.writer(file_handle)
            attrs = result.attributes
            writer.writerow([
                result.item_id, result.success, attrs.age, attrs.gender, attrs.ethnicity,
                attrs.hair_style, attrs.hair_color, attrs.hair_length, attrs.eye_color,
                attrs.body_type, attrs.dress, attrs.confidence_score,
                result.processing_time, result.error_message
            ])
    
    def _create_checkpoint(self, output_path: str, batch_num: int):
        """Create processing checkpoint."""
        checkpoint_data = {
            'processed_count': self.processed_count,
            'success_count': self.success_count,
            'error_count': self.error_count,
            'batch_num': batch_num,
            'timestamp': time.time(),
            'memory_usage': self.memory_monitor.get_memory_usage()
        }
        
        checkpoint_path = f"{output_path}.checkpoint_{self.processed_count}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
    
    def estimate_processing_time(self, data_source: str, sample_size: int = 100) -> Dict[str, Any]:
        """Estimate processing time for full dataset based on sample."""
        if self.pipeline is None:
            raise ValueError("Pipeline not set. Call set_pipeline() first.")
        
        # Load sample data
        source_path = Path(data_source)
        if source_path.is_dir():
            data_stream = self.data_loader.load_from_directory(data_source)
        elif source_path.is_file():
            data_stream = self.data_loader.load_from_manifest(data_source)
        else:
            raise ValueError(f"Invalid data source: {data_source}")
        
        # Process sample
        sample_items = []
        for i, item in enumerate(data_stream):
            if i >= sample_size:
                break
            sample_items.append(item)
        
        if not sample_items:
            return {'error': 'No items found in data source'}
        
        # Time sample processing
        start_time = time.time()
        sample_results = self._process_batch(sample_items)
        sample_time = time.time() - start_time
        
        # Calculate metrics
        successful_results = [r for r in sample_results if r.success]
        avg_time_per_item = sample_time / len(sample_items)
        success_rate = len(successful_results) / len(sample_items)
        
        # Estimate full dataset size
        if source_path.is_dir():
            # Count files in directory
            file_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            total_files = sum(1 for f in source_path.rglob('*') if f.suffix.lower() in file_extensions)
        else:
            # Estimate from manifest file size
            with open(source_path, 'r') as f:
                total_files = sum(1 for _ in f)
        
        # Projections
        estimated_total_time = total_files * avg_time_per_item
        estimated_total_hours = estimated_total_time / 3600
        estimated_total_days = estimated_total_hours / 24
        
        return {
            'sample_size': len(sample_items),
            'sample_processing_time': sample_time,
            'avg_time_per_item': avg_time_per_item,
            'success_rate': success_rate,
            'estimated_total_files': total_files,
            'estimated_total_time_seconds': estimated_total_time,
            'estimated_total_time_hours': estimated_total_hours,
            'estimated_total_time_days': estimated_total_days,
            'throughput_items_per_second': 1 / avg_time_per_item if avg_time_per_item > 0 else 0,
            'memory_usage': self.memory_monitor.get_memory_usage(),
            'recommendations': self._generate_processing_recommendations(total_files, avg_time_per_item)
        }
    
    def _generate_processing_recommendations(self, total_files: int, avg_time_per_item: float) -> List[str]:
        """Generate recommendations for processing optimization."""
        recommendations = []
        
        if total_files > 1_000_000:
            recommendations.append("Consider distributed processing with Ray for datasets > 1M items")
        
        if avg_time_per_item > 1.0:
            recommendations.append("Processing time > 1s per item - consider GPU acceleration")
        
        if total_files > 100_000:
            recommendations.append("Enable Redis caching for large datasets")
            recommendations.append("Use database sharding for better performance")
        
        memory_usage = self.memory_monitor.get_memory_usage()
        if memory_usage['percent'] > 80:
            recommendations.append("High memory usage detected - reduce batch size")
        
        recommendations.extend([
            "Monitor memory usage during processing",
            "Use checkpointing for long-running jobs",
            "Consider preprocessing to filter out edge cases"
        ])
        
        return recommendations
    
    def process(self, input_data: Any) -> Any:
        """Process streaming data."""
        if isinstance(input_data, dict):
            operation = input_data.get('operation')
            
            if operation == 'process_stream':
                return self.process_stream(
                    input_data['data_source'],
                    input_data['output_path'],
                    input_data.get('progress_callback')
                )
            elif operation == 'estimate':
                return self.estimate_processing_time(
                    input_data['data_source'],
                    input_data.get('sample_size', 100)
                )
        
        raise ValueError("StreamingProcessor expects operation dict as input")
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data."""
        return isinstance(input_data, dict) and 'operation' in input_data