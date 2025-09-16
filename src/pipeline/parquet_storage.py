"""Parquet storage module for efficient large-scale data export and storage."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import json

try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False
    pd = None
    pa = None
    pq = None

from .base import PipelineStage, CharacterAttributes, ProcessingResult

logger = logging.getLogger(__name__)

class ParquetStorage(PipelineStage):
    """Parquet storage for efficient large-scale data export and analytics."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("ParquetStorage", config)
        
        if not PARQUET_AVAILABLE:
            logger.warning("Parquet dependencies not available. Install with: pip install pandas pyarrow")
            return
        
        if config:
            self.output_dir = Path(config.get('output_dir', './parquet_exports'))
            self.compression = config.get('compression', 'snappy')
            self.row_group_size = config.get('row_group_size', 50000)
            self.partition_cols = config.get('partition_cols', ['processing_date'])
        else:
            self.output_dir = Path('./parquet_exports')
            self.compression = 'snappy'
            self.row_group_size = 50000
            self.partition_cols = ['processing_date']

        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"ParquetStorage initialized with output directory: {self.output_dir}")
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        """Process and store data in Parquet format."""
        if not PARQUET_AVAILABLE:
            return {'success': False, 'error': 'Parquet dependencies not available'}
        
        try:
            if isinstance(input_data, list):
                # Batch of results
                return self.store_batch_results(input_data)
            elif isinstance(input_data, ProcessingResult):
                # Single result
                return self.store_single_result(input_data)
            elif isinstance(input_data, dict):
                # Dictionary data
                return self.store_dict_data(input_data)
            else:
                raise ValueError(f"Unsupported input type: {type(input_data)}")
        
        except Exception as e:
            logger.error(f"Error storing data in Parquet: {e}")
            return {'success': False, 'error': str(e)}
    
    def store_single_result(self, result: ProcessingResult) -> Dict[str, Any]:
        """Store a single processing result in Parquet format."""
        try:
            # Convert to dictionary format
            data = self._result_to_dict(result)
            
            # Create DataFrame
            df = pd.DataFrame([data])
            
            # Add metadata
            df['processing_date'] = datetime.now().strftime('%Y-%m-%d')
            df['processing_timestamp'] = datetime.now().isoformat()
            
            # Generate filename
            filename = f"single_result_{result.item_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            filepath = self.output_dir / filename
            
            # Write to Parquet
            df.to_parquet(
                filepath,
                compression=self.compression,
                index=False
            )
            
            return {
                'success': True,
                'filepath': str(filepath),
                'records_written': 1,
                'file_size_mb': filepath.stat().st_size / (1024 * 1024)
            }
        
        except Exception as e:
            logger.error(f"Error storing single result: {e}")
            return {'success': False, 'error': str(e)}
    
    def store_batch_results(self, results: List[Union[ProcessingResult, Dict[str, Any]]]) -> Dict[str, Any]:
        """Store batch processing results in Parquet format with partitioning."""
        try:
            # Convert all results to dictionary format
            data_list = []
            for result in results:
                if isinstance(result, ProcessingResult):
                    data_list.append(self._result_to_dict(result))
                elif isinstance(result, dict):
                    data_list.append(self._normalize_dict_result(result))
                else:
                    logger.warning(f"Skipping unsupported result type: {type(result)}")
            
            if not data_list:
                return {'success': False, 'error': 'No valid results to store'}
            
            # Create DataFrame
            df = pd.DataFrame(data_list)
            
            # Add metadata columns
            df['processing_date'] = datetime.now().strftime('%Y-%m-%d')
            df['processing_timestamp'] = datetime.now().isoformat()
            df['batch_id'] = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Generate filename
            filename = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            filepath = self.output_dir / filename
            
            # Write to Parquet with partitioning if enabled
            if self.partition_cols and len(df) > 1000:
                # Use partitioned dataset for large batches
                table = pa.Table.from_pandas(df)
                pq.write_to_dataset(
                    table,
                    root_path=self.output_dir / "partitioned",
                    partition_cols=self.partition_cols,
                    compression=self.compression,
                    row_group_size=self.row_group_size
                )
                
                return {
                    'success': True,
                    'storage_type': 'partitioned_dataset',
                    'output_dir': str(self.output_dir / "partitioned"),
                    'records_written': len(df),
                    'partitions': df[self.partition_cols].drop_duplicates().to_dict('records')
                }
            else:
                # Single file for smaller batches
                df.to_parquet(
                    filepath,
                    compression=self.compression,
                    index=False,
                    row_group_size=self.row_group_size
                )
                
                return {
                    'success': True,
                    'storage_type': 'single_file',
                    'filepath': str(filepath),
                    'records_written': len(df),
                    'file_size_mb': filepath.stat().st_size / (1024 * 1024)
                }
        
        except Exception as e:
            logger.error(f"Error storing batch results: {e}")
            return {'success': False, 'error': str(e)}
    
    def store_dict_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Store dictionary data in Parquet format."""
        try:
            # Normalize the dictionary
            normalized_data = self._normalize_dict_result(data)
            
            # Create DataFrame
            df = pd.DataFrame([normalized_data])
            
            # Add metadata
            df['processing_date'] = datetime.now().strftime('%Y-%m-%d')
            df['processing_timestamp'] = datetime.now().isoformat()
            
            # Generate filename
            filename = f"dict_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            filepath = self.output_dir / filename
            
            # Write to Parquet
            df.to_parquet(
                filepath,
                compression=self.compression,
                index=False
            )
            
            return {
                'success': True,
                'filepath': str(filepath),
                'records_written': 1,
                'file_size_mb': filepath.stat().st_size / (1024 * 1024)
            }
        
        except Exception as e:
            logger.error(f"Error storing dict data: {e}")
            return {'success': False, 'error': str(e)}
    
    def read_parquet_file(self, filepath: Union[str, Path]) -> Optional[pd.DataFrame]:
        """Read a Parquet file and return as DataFrame."""
        if not PARQUET_AVAILABLE:
            logger.error("Parquet dependencies not available")
            return None
        
        try:
            return pd.read_parquet(filepath)
        except Exception as e:
            logger.error(f"Error reading Parquet file {filepath}: {e}")
            return None
    
    def read_partitioned_dataset(self, dataset_path: Optional[Union[str, Path]] = None) -> Optional[pd.DataFrame]:
        """Read a partitioned Parquet dataset."""
        if not PARQUET_AVAILABLE:
            logger.error("Parquet dependencies not available")
            return None
        
        try:
            dataset_path = dataset_path or (self.output_dir / "partitioned")
            
            if not Path(dataset_path).exists():
                logger.warning(f"Dataset path does not exist: {dataset_path}")
                return None
            
            dataset = pq.ParquetDataset(dataset_path)
            return dataset.read().to_pandas()
        
        except Exception as e:
            logger.error(f"Error reading partitioned dataset {dataset_path}: {e}")
            return None
    
    def get_dataset_info(self, dataset_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Get information about a Parquet dataset."""
        if not PARQUET_AVAILABLE:
            return {'error': 'Parquet dependencies not available'}
        
        try:
            dataset_path = dataset_path or (self.output_dir / "partitioned")
            
            if not Path(dataset_path).exists():
                return {'error': f'Dataset path does not exist: {dataset_path}'}
            
            dataset = pq.ParquetDataset(dataset_path)
            
            # Get schema information
            schema = dataset.schema.to_arrow_schema()
            
            # Get file information
            files = list(dataset.pieces)
            total_size = sum(piece.get_metadata().serialized_size for piece in files)
            
            return {
                'dataset_path': str(dataset_path),
                'num_files': len(files),
                'total_size_mb': total_size / (1024 * 1024),
                'schema': {
                    'columns': [field.name for field in schema],
                    'types': [str(field.type) for field in schema]
                },
                'partitions': dataset.partitions.partition_names if hasattr(dataset.partitions, 'partition_names') else []
            }
        
        except Exception as e:
            logger.error(f"Error getting dataset info: {e}")
            return {'error': str(e)}
    
    def export_to_csv(self, parquet_path: Union[str, Path], csv_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """Export Parquet data to CSV format."""
        if not PARQUET_AVAILABLE:
            return {'success': False, 'error': 'Parquet dependencies not available'}
        
        try:
            # Read Parquet data
            if Path(parquet_path).is_dir():
                df = self.read_partitioned_dataset(parquet_path)
            else:
                df = self.read_parquet_file(parquet_path)
            
            if df is None:
                return {'success': False, 'error': 'Failed to read Parquet data'}
            
            # Generate CSV path if not provided
            if csv_path is None:
                csv_path = Path(parquet_path).with_suffix('.csv')
            
            # Export to CSV
            df.to_csv(csv_path, index=False)
            
            return {
                'success': True,
                'csv_path': str(csv_path),
                'records_exported': len(df),
                'file_size_mb': Path(csv_path).stat().st_size / (1024 * 1024)
            }
        
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            return {'success': False, 'error': str(e)}
    
    def _result_to_dict(self, result: ProcessingResult) -> Dict[str, Any]:
        """Convert ProcessingResult to dictionary format."""
        data = {
            'item_id': result.item_id,
            'success': result.success,
            'processing_time': result.processing_time,
            'error_message': result.error_message
        }
        
        if result.attributes:
            # Add all character attributes
            for attr_name in ['age', 'gender', 'ethnicity', 'hair_style', 'hair_color', 
                             'hair_length', 'eye_color', 'body_type', 'dress', 'facial_expression']:
                data[attr_name] = getattr(result.attributes, attr_name, None)
            
            data['confidence_score'] = getattr(result.attributes, 'confidence_score', 0.0)
        
        return data
    
    def _normalize_dict_result(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize dictionary result to consistent format."""
        normalized = {
            'item_id': data.get('item_id', ''),
            'success': data.get('success', True),
            'processing_time': data.get('processing_time', 0.0),
            'error_message': data.get('error_message', None),
            'confidence_score': data.get('confidence', 0.0)
        }
        
        # Handle nested attributes
        if 'attributes' in data and isinstance(data['attributes'], dict):
            for attr_name in ['age', 'gender', 'ethnicity', 'hair_style', 'hair_color', 
                             'hair_length', 'eye_color', 'body_type', 'dress', 'facial_expression']:
                normalized[attr_name] = data['attributes'].get(attr_name, None)
        else:
            # Attributes at top level
            for attr_name in ['age', 'gender', 'ethnicity', 'hair_style', 'hair_color', 
                             'hair_length', 'eye_color', 'body_type', 'dress', 'facial_expression']:
                normalized[attr_name] = data.get(attr_name, None)
        
        return normalized
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data for Parquet storage."""
        if not PARQUET_AVAILABLE:
            return False
        
        if isinstance(input_data, (ProcessingResult, dict, list)):
            return True
        
        return False
    
    def cleanup_old_files(self, days_old: int = 30) -> Dict[str, Any]:
        """Clean up old Parquet files."""
        try:
            from datetime import timedelta
            
            cutoff_date = datetime.now() - timedelta(days=days_old)
            deleted_files = []
            total_size_freed = 0
            
            for file_path in self.output_dir.rglob("*.parquet"):
                if file_path.stat().st_mtime < cutoff_date.timestamp():
                    file_size = file_path.stat().st_size
                    file_path.unlink()
                    deleted_files.append(str(file_path))
                    total_size_freed += file_size
            
            return {
                'success': True,
                'deleted_files': len(deleted_files),
                'size_freed_mb': total_size_freed / (1024 * 1024),
                'files': deleted_files
            }
        
        except Exception as e:
            logger.error(f"Error cleaning up old files: {e}")
            return {'success': False, 'error': str(e)}