"""Graceful failure handling and partial processing recovery system."""

import logging
import time
import json
import traceback
from typing import Any, Dict, List, Optional, Callable, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from contextlib import contextmanager

from .base import PipelineStage, CharacterAttributes, ProcessingResult

logger = logging.getLogger(__name__)

class FailureType(Enum):
    """Types of failures that can occur during processing."""
    NETWORK_ERROR = "network_error"
    MODEL_ERROR = "model_error"
    MEMORY_ERROR = "memory_error"
    IO_ERROR = "io_error"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN_ERROR = "unknown_error"

@dataclass
class FailureRecord:
    """Record of a processing failure."""
    item_id: str
    failure_type: FailureType
    error_message: str
    timestamp: float
    retry_count: int
    stack_trace: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class CircuitBreaker:
    """Circuit breaker pattern for handling cascading failures."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.Lock()
    
    @contextmanager
    def call(self):
        """Execute operation with circuit breaker protection."""
        with self.lock:
            if self.state == 'OPEN':
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = 'HALF_OPEN'
                    logger.info("Circuit breaker entering HALF_OPEN state")
                else:
                    raise Exception("Circuit breaker is OPEN - operation blocked")
        
        try:
            yield
            # Success - reset if in HALF_OPEN
            with self.lock:
                if self.state == 'HALF_OPEN':
                    self.state = 'CLOSED'
                    self.failure_count = 0
                    logger.info("Circuit breaker reset to CLOSED state")
        
        except Exception as e:
            with self.lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = 'OPEN'
                    logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
            
            raise e

class RetryManager:
    """Manages retry logic with exponential backoff."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
    
    def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with exponential backoff retry."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    logger.warning(f"Attempt {attempt + 1} failed, retrying in {delay}s: {e}")
                    time.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed: {e}")
        
        raise last_exception

class FailureHandler(PipelineStage):
    """Comprehensive failure handling and recovery system."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("FailureHandler", config)
        
        # Configuration
        self.max_retries = config.get('max_retries', 3) if config else 3
        self.circuit_breaker_threshold = config.get('circuit_breaker_threshold', 10) if config else 10
        self.recovery_timeout = config.get('recovery_timeout', 300) if config else 300  # 5 minutes
        self.failure_log_path = config.get('failure_log_path', './failures.jsonl') if config else './failures.jsonl'
        
        # Components
        self.retry_manager = RetryManager(self.max_retries)
        self.circuit_breaker = CircuitBreaker(self.circuit_breaker_threshold, self.recovery_timeout)
        
        # Failure tracking
        self.failure_records: List[FailureRecord] = []
        self.failure_stats = {
            FailureType.NETWORK_ERROR: 0,
            FailureType.MODEL_ERROR: 0,
            FailureType.MEMORY_ERROR: 0,
            FailureType.IO_ERROR: 0,
            FailureType.VALIDATION_ERROR: 0,
            FailureType.TIMEOUT_ERROR: 0,
            FailureType.UNKNOWN_ERROR: 0
        }
        
        # Recovery strategies
        self.recovery_strategies = {
            FailureType.MEMORY_ERROR: self._handle_memory_error,
            FailureType.MODEL_ERROR: self._handle_model_error,
            FailureType.NETWORK_ERROR: self._handle_network_error,
            FailureType.IO_ERROR: self._handle_io_error,
            FailureType.TIMEOUT_ERROR: self._handle_timeout_error
        }
    
    def _classify_error(self, error: Exception) -> FailureType:
        """Classify error type for appropriate handling."""
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        if 'memory' in error_str or 'oom' in error_str or isinstance(error, MemoryError):
            return FailureType.MEMORY_ERROR
        elif 'network' in error_str or 'connection' in error_str or 'timeout' in error_str:
            return FailureType.NETWORK_ERROR
        elif 'model' in error_str or 'cuda' in error_str or 'tensor' in error_str:
            return FailureType.MODEL_ERROR
        elif 'file' in error_str or 'io' in error_str or isinstance(error, (IOError, FileNotFoundError)):
            return FailureType.IO_ERROR
        elif 'validation' in error_str or 'schema' in error_str or isinstance(error, ValueError):
            return FailureType.VALIDATION_ERROR
        elif 'timeout' in error_str or isinstance(error, TimeoutError):
            return FailureType.TIMEOUT_ERROR
        else:
            return FailureType.UNKNOWN_ERROR
    
    def _handle_memory_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle memory-related errors."""
        import gc
        gc.collect()
        
        return {
            'strategy': 'memory_cleanup',
            'action': 'garbage_collection_performed',
            'recommendation': 'reduce_batch_size',
            'fallback': 'process_individually'
        }
    
    def _handle_model_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model-related errors."""
        return {
            'strategy': 'model_fallback',
            'action': 'switch_to_backup_model',
            'recommendation': 'check_model_compatibility',
            'fallback': 'use_simplified_extraction'
        }
    
    def _handle_network_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle network-related errors."""
        return {
            'strategy': 'network_retry',
            'action': 'exponential_backoff_retry',
            'recommendation': 'check_network_connectivity',
            'fallback': 'use_local_models_only'
        }
    
    def _handle_io_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle I/O related errors."""
        return {
            'strategy': 'io_recovery',
            'action': 'verify_file_permissions',
            'recommendation': 'check_disk_space',
            'fallback': 'skip_corrupted_files'
        }
    
    def _handle_timeout_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle timeout errors."""
        return {
            'strategy': 'timeout_recovery',
            'action': 'increase_timeout_duration',
            'recommendation': 'optimize_processing_pipeline',
            'fallback': 'process_with_reduced_quality'
        }
    
    def record_failure(self, item_id: str, error: Exception, context: Optional[Dict[str, Any]] = None):
        """Record failure for analysis and recovery."""
        failure_type = self._classify_error(error)
        
        failure_record = FailureRecord(
            item_id=item_id,
            failure_type=failure_type,
            error_message=str(error),
            timestamp=time.time(),
            retry_count=context.get('retry_count', 0) if context else 0,
            stack_trace=traceback.format_exc(),
            context=context
        )
        
        self.failure_records.append(failure_record)
        self.failure_stats[failure_type] += 1
        
        # Log to file for persistence
        self._log_failure_to_file(failure_record)
        
        logger.error(f"Recorded failure for {item_id}: {failure_type.value} - {error}")
    
    def _log_failure_to_file(self, failure_record: FailureRecord):
        """Log failure record to persistent file."""
        try:
            failure_data = asdict(failure_record)
            failure_data['failure_type'] = failure_record.failure_type.value
            
            with open(self.failure_log_path, 'a') as f:
                f.write(json.dumps(failure_data) + '\n')
        except Exception as e:
            logger.error(f"Failed to log failure record: {e}")
    
    def handle_failure(self, item_id: str, error: Exception, 
                      context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle failure with appropriate recovery strategy."""
        failure_type = self._classify_error(error)
        
        # Record the failure
        self.record_failure(item_id, error, context)
        
        # Apply recovery strategy
        recovery_info = {'strategy': 'none', 'action': 'skip'}
        if failure_type in self.recovery_strategies:
            try:
                recovery_info = self.recovery_strategies[failure_type](error, context or {})
            except Exception as recovery_error:
                logger.error(f"Recovery strategy failed: {recovery_error}")
        
        return {
            'item_id': item_id,
            'failure_type': failure_type.value,
            'error_message': str(error),
            'recovery': recovery_info,
            'should_retry': recovery_info.get('action') != 'skip',
            'fallback_available': 'fallback' in recovery_info
        }
    
    def execute_with_protection(self, func: Callable, item_id: str, 
                               *args, **kwargs) -> ProcessingResult:
        """Execute function with comprehensive failure protection."""
        context = {
            'item_id': item_id,
            'function': func.__name__,
            'retry_count': 0
        }
        
        try:
            with self.circuit_breaker.call():
                # Execute with retry logic
                result = self.retry_manager.execute_with_retry(
                    self._execute_with_monitoring, func, item_id, context, *args, **kwargs
                )
                return result
                
        except Exception as e:
            # Handle failure
            failure_info = self.handle_failure(item_id, e, context)
            
            # Try fallback if available
            if failure_info['fallback_available']:
                try:
                    fallback_result = self._execute_fallback(item_id, failure_info, *args, **kwargs)
                    logger.info(f"Fallback successful for {item_id}")
                    return fallback_result
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed for {item_id}: {fallback_error}")
            
            # Return failed result
            return ProcessingResult(
                item_id=item_id,
                attributes=CharacterAttributes(),
                success=False,
                error_message=str(e),
                processing_time=0.0
            )
    
    def _execute_with_monitoring(self, func: Callable, item_id: str, 
                                context: Dict[str, Any], *args, **kwargs) -> ProcessingResult:
        """Execute function with monitoring and context tracking."""
        start_time = time.time()
        
        try:
            # Update retry count
            context['retry_count'] = context.get('retry_count', 0) + 1
            
            # Execute function
            result = func(*args, **kwargs)
            processing_time = time.time() - start_time
            
            # Validate result
            if not self._validate_result(result):
                raise ValueError(f"Invalid result returned for {item_id}")
            
            return ProcessingResult(
                item_id=item_id,
                attributes=result,
                success=True,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            context['processing_time'] = processing_time
            raise e
    
    def _validate_result(self, result: Any) -> bool:
        """Validate processing result."""
        if not isinstance(result, CharacterAttributes):
            return False
        
        # Check if result has meaningful content
        attrs_dict = asdict(result)
        non_none_attrs = {k: v for k, v in attrs_dict.items() if v is not None and v != ""}
        
        # Must have at least some attributes
        return len(non_none_attrs) >= 3
    
    def _execute_fallback(self, item_id: str, failure_info: Dict[str, Any], 
                         *args, **kwargs) -> ProcessingResult:
        """Execute fallback processing strategy."""
        fallback_strategy = failure_info['recovery']['fallback']
        
        if fallback_strategy == 'use_simplified_extraction':
            # Return basic attributes with low confidence
            return ProcessingResult(
                item_id=item_id,
                attributes=CharacterAttributes(
                    age="unknown",
                    gender="unknown",
                    ethnicity="unknown",
                    hair_style="unknown",
                    hair_color="unknown",
                    hair_length="unknown",
                    eye_color="unknown",
                    body_type="unknown",
                    dress="unknown",
                    confidence_score=0.1
                ),
                success=True,
                processing_time=0.001,
                metadata={'fallback': True, 'strategy': fallback_strategy}
            )
        
        elif fallback_strategy == 'skip_corrupted_files':
            # Return empty result but mark as successful skip
            return ProcessingResult(
                item_id=item_id,
                attributes=CharacterAttributes(),
                success=True,
                processing_time=0.001,
                metadata={'skipped': True, 'reason': 'corrupted_file'}
            )
        
        else:
            # Default fallback
            raise Exception(f"Unknown fallback strategy: {fallback_strategy}")
    
    def get_failure_analysis(self) -> Dict[str, Any]:
        """Analyze failure patterns and provide recommendations."""
        total_failures = len(self.failure_records)
        
        if total_failures == 0:
            return {
                'total_failures': 0,
                'failure_rate': 0.0,
                'recommendations': ['No failures detected - system running smoothly']
            }
        
        # Analyze failure patterns
        failure_by_type = {ft.value: count for ft, count in self.failure_stats.items()}
        most_common_failure = max(failure_by_type.items(), key=lambda x: x[1])
        
        # Recent failures (last hour)
        recent_threshold = time.time() - 3600
        recent_failures = [f for f in self.failure_records if f.timestamp > recent_threshold]
        
        # Generate recommendations
        recommendations = self._generate_failure_recommendations(failure_by_type, recent_failures)
        
        return {
            'total_failures': total_failures,
            'failure_by_type': failure_by_type,
            'most_common_failure': most_common_failure,
            'recent_failures_count': len(recent_failures),
            'circuit_breaker_state': self.circuit_breaker.state,
            'recommendations': recommendations,
            'failure_trends': self._analyze_failure_trends()
        }
    
    def _generate_failure_recommendations(self, failure_by_type: Dict[str, int], 
                                        recent_failures: List[FailureRecord]) -> List[str]:
        """Generate recommendations based on failure patterns."""
        recommendations = []
        
        # Memory error recommendations
        if failure_by_type.get('memory_error', 0) > 5:
            recommendations.extend([
                "High memory errors detected - reduce batch size",
                "Consider implementing memory pooling",
                "Enable more aggressive garbage collection"
            ])
        
        # Model error recommendations
        if failure_by_type.get('model_error', 0) > 3:
            recommendations.extend([
                "Model errors detected - verify model compatibility",
                "Consider model warm-up strategies",
                "Implement model health checks"
            ])
        
        # Network error recommendations
        if failure_by_type.get('network_error', 0) > 5:
            recommendations.extend([
                "Network instability detected - implement offline mode",
                "Use local model caching",
                "Increase network timeout values"
            ])
        
        # Recent failure spike
        if len(recent_failures) > 10:
            recommendations.append("Recent failure spike detected - investigate system health")
        
        # General recommendations
        recommendations.extend([
            "Monitor system resources during processing",
            "Implement progressive quality degradation",
            "Use health checks before processing batches"
        ])
        
        return recommendations
    
    def _analyze_failure_trends(self) -> Dict[str, Any]:
        """Analyze failure trends over time."""
        if not self.failure_records:
            return {'trend': 'stable', 'analysis': 'No failures to analyze'}
        
        # Group failures by hour
        hourly_failures = {}
        for failure in self.failure_records:
            hour = int(failure.timestamp // 3600)
            hourly_failures[hour] = hourly_failures.get(hour, 0) + 1
        
        # Analyze trend
        if len(hourly_failures) < 2:
            trend = 'insufficient_data'
        else:
            hours = sorted(hourly_failures.keys())
            recent_avg = sum(hourly_failures[h] for h in hours[-3:]) / min(3, len(hours))
            older_avg = sum(hourly_failures[h] for h in hours[:-3]) / max(1, len(hours) - 3)
            
            if recent_avg > older_avg * 1.5:
                trend = 'increasing'
            elif recent_avg < older_avg * 0.5:
                trend = 'decreasing'
            else:
                trend = 'stable'
        
        return {
            'trend': trend,
            'hourly_failures': hourly_failures,
            'analysis': f"Failure trend is {trend} based on recent patterns"
        }
    
    def create_recovery_checkpoint(self, processed_items: List[str], 
                                  failed_items: List[str], output_path: str) -> str:
        """Create recovery checkpoint for resuming processing."""
        checkpoint_data = {
            'timestamp': time.time(),
            'processed_items': processed_items,
            'failed_items': failed_items,
            'failure_stats': {ft.value: count for ft, count in self.failure_stats.items()},
            'circuit_breaker_state': self.circuit_breaker.state,
            'total_processed': len(processed_items),
            'total_failed': len(failed_items)
        }
        
        checkpoint_path = f"{output_path}.recovery_checkpoint.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logger.info(f"Recovery checkpoint created: {checkpoint_path}")
        return checkpoint_path
    
    def load_recovery_checkpoint(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """Load recovery checkpoint to resume processing."""
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            logger.info(f"Loaded recovery checkpoint: {len(checkpoint_data['processed_items'])} processed, {len(checkpoint_data['failed_items'])} failed")
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Failed to load recovery checkpoint: {e}")
            return None
    
    def process(self, input_data: Any) -> Any:
        """Process failure handling operations."""
        if isinstance(input_data, dict):
            operation = input_data.get('operation')
            
            if operation == 'execute_protected':
                return self.execute_with_protection(
                    input_data['function'],
                    input_data['item_id'],
                    *input_data.get('args', []),
                    **input_data.get('kwargs', {})
                )
            elif operation == 'analyze_failures':
                return self.get_failure_analysis()
            elif operation == 'create_checkpoint':
                return self.create_recovery_checkpoint(
                    input_data['processed_items'],
                    input_data['failed_items'],
                    input_data['output_path']
                )
            elif operation == 'load_checkpoint':
                return self.load_recovery_checkpoint(input_data['checkpoint_path'])
        
        raise ValueError("FailureHandler expects operation dict as input")
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data."""
        return isinstance(input_data, dict) and 'operation' in input_data