"""Celery tasks for background processing of character attribute extraction."""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime

try:
    from celery import Celery
    from celery.result import AsyncResult
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    Celery = None
    AsyncResult = None

from character_pipeline import create_pipeline
from pipeline.input_loader import DatasetItem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Celery configuration
if CELERY_AVAILABLE:
    # Redis broker configuration
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    
    celery_app = Celery(
        'character_extraction',
        broker=REDIS_URL,
        backend=REDIS_URL,
        include=['celery_tasks']
    )
    
    # Celery configuration
    celery_app.conf.update(
        task_serializer='json',
        accept_content=['json'],
        result_serializer='json',
        timezone='UTC',
        enable_utc=True,
        task_track_started=True,
        task_time_limit=30 * 60,  # 30 minutes
        task_soft_time_limit=25 * 60,  # 25 minutes
        worker_prefetch_multiplier=1,
        worker_max_tasks_per_child=1000,
    )
else:
    celery_app = None
    logger.warning("Celery not available. Install with: pip install celery[redis]")

# Global pipeline instance (initialized per worker)
pipeline = None

def get_pipeline():
    """Get or create pipeline instance."""
    global pipeline
    if pipeline is None:
        logger.info("Initializing pipeline in worker...")
        pipeline = create_pipeline()
        logger.info("Pipeline initialized successfully")
    return pipeline

if CELERY_AVAILABLE:
    @celery_app.task(bind=True, name='extract_single_image')
    def extract_single_image(self, image_path: str, tags: Optional[str] = None):
        """Extract attributes from a single image."""
        try:
            self.update_state(state='PROGRESS', meta={'progress': 0, 'status': 'Loading pipeline...'})
            
            pipeline = get_pipeline()
            
            self.update_state(state='PROGRESS', meta={'progress': 25, 'status': 'Processing image...'})
            
            # Extract attributes
            attributes = pipeline.extract_from_image(image_path)
            
            self.update_state(state='PROGRESS', meta={'progress': 75, 'status': 'Formatting results...'})
            
            # Format result
            result = {
                'success': True,
                'image_path': image_path,
                'attributes': {
                    'age': getattr(attributes, 'age', None),
                    'gender': getattr(attributes, 'gender', None),
                    'ethnicity': getattr(attributes, 'ethnicity', None),
                    'hair_style': getattr(attributes, 'hair_style', None),
                    'hair_color': getattr(attributes, 'hair_color', None),
                    'hair_length': getattr(attributes, 'hair_length', None),
                    'eye_color': getattr(attributes, 'eye_color', None),
                    'body_type': getattr(attributes, 'body_type', None),
                    'dress': getattr(attributes, 'dress', None)
                },
                'confidence': getattr(attributes, 'confidence_score', 0.0),
                'processed_at': datetime.now().isoformat()
            }
            
            self.update_state(state='PROGRESS', meta={'progress': 100, 'status': 'Completed'})
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            self.update_state(
                state='FAILURE',
                meta={'error': str(e), 'image_path': image_path}
            )
            raise
    
    @celery_app.task(bind=True, name='batch_extract_images')
    def batch_extract_images(self, image_paths: List[str], batch_size: int = 32, 
                           use_hf_datasets: bool = True):
        """Extract attributes from multiple images in batch."""
        try:
            self.update_state(state='PROGRESS', meta={
                'progress': 0, 
                'status': 'Initializing batch processing...',
                'total_images': len(image_paths)
            })
            
            pipeline = get_pipeline()
            
            # Create dataset items
            items = [DatasetItem(image_path=path) for path in image_paths]
            
            self.update_state(state='PROGRESS', meta={
                'progress': 10, 
                'status': 'Created dataset items',
                'total_images': len(items)
            })
            
            results = []
            
            if use_hf_datasets and len(items) > batch_size:
                # Use HuggingFace datasets for large batches
                self.update_state(state='PROGRESS', meta={
                    'progress': 15, 
                    'status': 'Using HuggingFace datasets for processing...'
                })
                
                def process_batch_hf(batch):
                    batch_results = []
                    for item_id, image_path in zip(batch['item_id'], batch['image_path']):
                        try:
                            attributes = pipeline.extract_from_image(image_path)
                            result = {
                                'success': True,
                                'item_id': item_id,
                                'image_path': image_path,
                                'attributes': {
                                    'age': getattr(attributes, 'age', None),
                                    'gender': getattr(attributes, 'gender', None),
                                    'ethnicity': getattr(attributes, 'ethnicity', None),
                                    'hair_style': getattr(attributes, 'hair_style', None),
                                    'hair_color': getattr(attributes, 'hair_color', None),
                                    'hair_length': getattr(attributes, 'hair_length', None),
                                    'eye_color': getattr(attributes, 'eye_color', None),
                                    'body_type': getattr(attributes, 'body_type', None),
                                    'dress': getattr(attributes, 'dress', None)
                                },
                                'confidence': getattr(attributes, 'confidence_score', 0.0)
                            }
                        except Exception as e:
                            result = {
                                'success': False,
                                'item_id': item_id,
                                'image_path': image_path,
                                'error': str(e)
                            }
                        batch_results.append(result)
                    
                    return {'results': batch_results}
                
                # Process using HuggingFace datasets
                processed_dataset = pipeline.input_loader.process_with_hf_map(
                    process_batch_hf,
                    items=items,
                    batch_size=batch_size,
                    num_proc=4
                )
                
                if processed_dataset:
                    for item in processed_dataset:
                        results.extend(item['results'])
                        
                        # Update progress
                        progress = min(20 + (len(results) / len(items)) * 70, 90)
                        self.update_state(state='PROGRESS', meta={
                            'progress': progress,
                            'status': f'Processed {len(results)}/{len(items)} images',
                            'processed': len(results)
                        })
            
            else:
                # Use PyTorch DataLoader for smaller batches
                self.update_state(state='PROGRESS', meta={
                    'progress': 15, 
                    'status': 'Using PyTorch DataLoader for processing...'
                })
                
                dataloader = pipeline.input_loader.create_dataloader(
                    items=items,
                    batch_size=batch_size,
                    shuffle=False
                )
                
                for batch_idx, batch in enumerate(dataloader):
                    batch_results = []
                    
                    for i, (item_id, image_path) in enumerate(zip(batch['item_ids'], batch['image_paths'])):
                        try:
                            attributes = pipeline.extract_from_image(image_path)
                            result = {
                                'success': True,
                                'item_id': item_id,
                                'image_path': image_path,
                                'attributes': {
                                    'age': getattr(attributes, 'age', None),
                                    'gender': getattr(attributes, 'gender', None),
                                    'ethnicity': getattr(attributes, 'ethnicity', None),
                                    'hair_style': getattr(attributes, 'hair_style', None),
                                    'hair_color': getattr(attributes, 'hair_color', None),
                                    'hair_length': getattr(attributes, 'hair_length', None),
                                    'eye_color': getattr(attributes, 'eye_color', None),
                                    'body_type': getattr(attributes, 'body_type', None),
                                    'dress': getattr(attributes, 'dress', None)
                                },
                                'confidence': getattr(attributes, 'confidence_score', 0.0)
                            }
                        except Exception as e:
                            result = {
                                'success': False,
                                'item_id': item_id,
                                'image_path': image_path,
                                'error': str(e)
                            }
                        
                        batch_results.append(result)
                    
                    results.extend(batch_results)
                    
                    # Update progress
                    progress = min(20 + (len(results) / len(items)) * 70, 90)
                    self.update_state(state='PROGRESS', meta={
                        'progress': progress,
                        'status': f'Processed batch {batch_idx + 1}/{len(dataloader)}',
                        'processed': len(results)
                    })
            
            # Final processing
            self.update_state(state='PROGRESS', meta={
                'progress': 95, 
                'status': 'Finalizing results...'
            })
            
            # Calculate summary statistics
            successful_results = [r for r in results if r.get('success', False)]
            failed_results = [r for r in results if not r.get('success', False)]
            
            avg_confidence = 0.0
            if successful_results:
                avg_confidence = sum(r['confidence'] for r in successful_results) / len(successful_results)
            
            final_result = {
                'success': True,
                'total_processed': len(results),
                'successful': len(successful_results),
                'failed': len(failed_results),
                'success_rate': len(successful_results) / len(results) * 100 if results else 0,
                'average_confidence': avg_confidence,
                'results': results,
                'processed_at': datetime.now().isoformat(),
                'processing_method': 'huggingface_datasets' if use_hf_datasets and len(items) > batch_size else 'pytorch_dataloader'
            }
            
            return final_result
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            self.update_state(
                state='FAILURE',
                meta={'error': str(e), 'total_images': len(image_paths)}
            )
            raise
    
    @celery_app.task(bind=True, name='process_dataset_directory')
    def process_dataset_directory(self, dataset_path: str, batch_size: int = 32, 
                                max_images: Optional[int] = None):
        """Process all images in a dataset directory."""
        try:
            self.update_state(state='PROGRESS', meta={
                'progress': 0, 
                'status': 'Discovering images in dataset...'
            })
            
            pipeline = get_pipeline()
            
            # Configure input loader with dataset path
            pipeline.input_loader.dataset_path = dataset_path
            items = pipeline.input_loader.discover_dataset_items()
            
            if max_images:
                items = items[:max_images]
            
            self.update_state(state='PROGRESS', meta={
                'progress': 10, 
                'status': f'Found {len(items)} images',
                'total_images': len(items)
            })
            
            if not items:
                return {
                    'success': True,
                    'message': 'No images found in dataset directory',
                    'total_processed': 0
                }
            
            # Extract image paths
            image_paths = [item.image_path for item in items]
            
            # Use the batch processing task
            return batch_extract_images.apply(
                args=[image_paths, batch_size, True],
                task_id=self.request.id
            ).get()
            
        except Exception as e:
            logger.error(f"Error processing dataset directory {dataset_path}: {e}")
            self.update_state(
                state='FAILURE',
                meta={'error': str(e), 'dataset_path': dataset_path}
            )
            raise

else:
    # Dummy functions when Celery is not available
    def extract_single_image(*args, **kwargs):
        raise NotImplementedError("Celery not available")
    
    def batch_extract_images(*args, **kwargs):
        raise NotImplementedError("Celery not available")
    
    def process_dataset_directory(*args, **kwargs):
        raise NotImplementedError("Celery not available")

# Utility functions for task management
def get_task_status(task_id: str) -> Dict[str, Any]:
    """Get the status of a Celery task."""
    if not CELERY_AVAILABLE:
        return {'error': 'Celery not available'}
    
    result = AsyncResult(task_id, app=celery_app)
    
    return {
        'task_id': task_id,
        'status': result.status,
        'result': result.result,
        'info': result.info,
        'successful': result.successful(),
        'failed': result.failed()
    }

def cancel_task(task_id: str) -> Dict[str, Any]:
    """Cancel a Celery task."""
    if not CELERY_AVAILABLE:
        return {'error': 'Celery not available'}
    
    celery_app.control.revoke(task_id, terminate=True)
    
    return {
        'task_id': task_id,
        'status': 'cancelled'
    }

def get_active_tasks() -> List[Dict[str, Any]]:
    """Get list of active tasks."""
    if not CELERY_AVAILABLE:
        return []
    
    inspect = celery_app.control.inspect()
    active_tasks = inspect.active()
    
    if not active_tasks:
        return []
    
    tasks = []
    for worker, task_list in active_tasks.items():
        for task in task_list:
            tasks.append({
                'worker': worker,
                'task_id': task['id'],
                'name': task['name'],
                'args': task['args'],
                'kwargs': task['kwargs']
            })
    
    return tasks

if __name__ == '__main__':
    if CELERY_AVAILABLE:
        # Start Celery worker
        celery_app.start()
    else:
        print("Celery not available. Install with: pip install celery[redis]")