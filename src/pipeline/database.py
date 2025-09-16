"""Database storage for caching and storing extraction results."""

import sqlite3
import json
import pickle
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import logging

from .base import PipelineStage, CharacterAttributes, ProcessingResult

logger = logging.getLogger(__name__)

class DatabaseStorage(PipelineStage):
    """SQLite database for storing and caching extraction results."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("DatabaseStorage", config)
        
        self.db_path = Path(config.get('db_path', './data/character_attributes.db') if config else './data/character_attributes.db')
        self.enable_caching = config.get('enable_caching', True) if config else True
        self.cache_embeddings = config.get('cache_embeddings', True) if config else True
        
        # Create database directory
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Main results table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS extraction_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        item_id TEXT UNIQUE NOT NULL,
                        image_path TEXT,
                        text_path TEXT,
                        attributes_json TEXT NOT NULL,
                        confidence_score REAL,
                        processing_time REAL,
                        success BOOLEAN,
                        error_message TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Individual attributes table for easier querying
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS character_attributes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        item_id TEXT NOT NULL,
                        attribute_name TEXT NOT NULL,
                        attribute_value TEXT,
                        confidence REAL,
                        extraction_method TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (item_id) REFERENCES extraction_results (item_id)
                    )
                ''')
                
                # Embeddings cache table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS embeddings_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        item_id TEXT NOT NULL,
                        embedding_type TEXT NOT NULL,
                        embedding_hash TEXT NOT NULL,
                        embedding_data BLOB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(item_id, embedding_type)
                    )
                ''')
                
                # Processing statistics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS processing_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date DATE NOT NULL,
                        total_processed INTEGER DEFAULT 0,
                        successful_extractions INTEGER DEFAULT 0,
                        failed_extractions INTEGER DEFAULT 0,
                        avg_processing_time REAL,
                        avg_confidence REAL,
                        UNIQUE(date)
                    )
                ''')
                
                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_item_id ON extraction_results(item_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_attributes_item ON character_attributes(item_id)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_attributes_name ON character_attributes(attribute_name)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_embeddings_item ON embeddings_cache(item_id)')
                
                conn.commit()
                self.logger.info(f"Database initialized at {self.db_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _hash_data(self, data: Any) -> str:
        """Create hash for data caching."""
        if isinstance(data, str):
            return hashlib.md5(data.encode()).hexdigest()
        else:
            return hashlib.md5(str(data).encode()).hexdigest()
    
    def store_result(self, result: ProcessingResult) -> bool:
        """Store processing result in database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Store main result
                attributes_json = json.dumps(result.attributes.to_dict())
                
                cursor.execute('''
                    INSERT OR REPLACE INTO extraction_results 
                    (item_id, attributes_json, confidence_score, processing_time, 
                     success, error_message, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result.item_id,
                    attributes_json,
                    result.attributes.confidence_score,
                    result.processing_time,
                    result.success,
                    result.error_message,
                    datetime.now()
                ))
                
                # Store individual attributes
                if result.success and result.attributes:
                    # Clear existing attributes for this item
                    cursor.execute('DELETE FROM character_attributes WHERE item_id = ?', (result.item_id,))
                    
                    # Insert new attributes
                    attributes_dict = result.attributes.__dict__
                    for attr_name, attr_value in attributes_dict.items():
                        if attr_value is not None and not attr_name.startswith('_'):
                            cursor.execute('''
                                INSERT INTO character_attributes 
                                (item_id, attribute_name, attribute_value, confidence)
                                VALUES (?, ?, ?, ?)
                            ''', (
                                result.item_id,
                                attr_name,
                                str(attr_value) if not isinstance(attr_value, list) else json.dumps(attr_value),
                                result.attributes.confidence_score
                            ))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to store result for {result.item_id}: {e}")
            return False
    
    def get_result(self, item_id: str) -> Optional[ProcessingResult]:
        """Retrieve processing result from database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT attributes_json, confidence_score, processing_time, 
                           success, error_message
                    FROM extraction_results 
                    WHERE item_id = ?
                ''', (item_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                attributes_dict = json.loads(row[0])
                attributes = CharacterAttributes()
                
                # Reconstruct attributes object
                for key, value in attributes_dict.items():
                    # Convert back to snake_case
                    snake_key = key.lower().replace(' ', '_')
                    if hasattr(attributes, snake_key):
                        setattr(attributes, snake_key, value)
                
                attributes.confidence_score = row[1]
                
                return ProcessingResult(
                    item_id=item_id,
                    attributes=attributes,
                    success=row[3],
                    error_message=row[4],
                    processing_time=row[2]
                )
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve result for {item_id}: {e}")
            return None
    
    def cache_embedding(self, item_id: str, embedding_type: str, embedding_data: Any) -> bool:
        """Cache embedding data."""
        if not self.cache_embeddings:
            return False
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Serialize embedding data
                embedding_blob = pickle.dumps(embedding_data)
                embedding_hash = self._hash_data(embedding_blob)
                
                cursor.execute('''
                    INSERT OR REPLACE INTO embeddings_cache 
                    (item_id, embedding_type, embedding_hash, embedding_data)
                    VALUES (?, ?, ?, ?)
                ''', (item_id, embedding_type, embedding_hash, embedding_blob))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to cache embedding for {item_id}: {e}")
            return False
    
    def get_cached_embedding(self, item_id: str, embedding_type: str) -> Optional[Any]:
        """Retrieve cached embedding data."""
        if not self.cache_embeddings:
            return None
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT embedding_data 
                    FROM embeddings_cache 
                    WHERE item_id = ? AND embedding_type = ?
                ''', (item_id, embedding_type))
                
                row = cursor.fetchone()
                if row:
                    return pickle.loads(row[0])
                
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to retrieve cached embedding for {item_id}: {e}")
            return None
    
    def update_processing_stats(self, date: str, stats: Dict[str, Any]) -> bool:
        """Update daily processing statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO processing_stats 
                    (date, total_processed, successful_extractions, failed_extractions, 
                     avg_processing_time, avg_confidence)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    date,
                    stats.get('total_processed', 0),
                    stats.get('successful_extractions', 0),
                    stats.get('failed_extractions', 0),
                    stats.get('avg_processing_time', 0.0),
                    stats.get('avg_confidence', 0.0)
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to update processing stats: {e}")
            return False
    
    def query_attributes(self, filters: Dict[str, Any], limit: int = 100) -> List[Dict[str, Any]]:
        """Query characters by attributes."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build query dynamically based on filters
                where_clauses = []
                params = []
                
                for attr_name, attr_value in filters.items():
                    where_clauses.append('(attribute_name = ? AND attribute_value = ?)')
                    params.extend([attr_name, attr_value])
                
                if where_clauses:
                    where_sql = ' OR '.join(where_clauses)
                    query = f'''
                        SELECT DISTINCT item_id 
                        FROM character_attributes 
                        WHERE {where_sql}
                        LIMIT ?
                    '''
                    params.append(limit)
                else:
                    query = 'SELECT DISTINCT item_id FROM character_attributes LIMIT ?'
                    params = [limit]
                
                cursor.execute(query, params)
                item_ids = [row[0] for row in cursor.fetchall()]
                
                # Get full results for these items
                results = []
                for item_id in item_ids:
                    result = self.get_result(item_id)
                    if result:
                        results.append({
                            'item_id': item_id,
                            'attributes': result.attributes.to_dict(),
                            'confidence': result.attributes.confidence_score
                        })
                
                return results
                
        except Exception as e:
            self.logger.error(f"Failed to query attributes: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total records
                cursor.execute('SELECT COUNT(*) FROM extraction_results')
                total_records = cursor.fetchone()[0]
                
                # Successful extractions
                cursor.execute('SELECT COUNT(*) FROM extraction_results WHERE success = 1')
                successful = cursor.fetchone()[0]
                
                # Average confidence
                cursor.execute('SELECT AVG(confidence_score) FROM extraction_results WHERE success = 1')
                avg_confidence = cursor.fetchone()[0] or 0.0
                
                # Average processing time
                cursor.execute('SELECT AVG(processing_time) FROM extraction_results WHERE success = 1')
                avg_time = cursor.fetchone()[0] or 0.0
                
                # Most common attributes
                cursor.execute('''
                    SELECT attribute_name, attribute_value, COUNT(*) as count
                    FROM character_attributes 
                    GROUP BY attribute_name, attribute_value 
                    ORDER BY count DESC 
                    LIMIT 10
                ''')
                common_attributes = cursor.fetchall()
                
                return {
                    'total_records': total_records,
                    'successful_extractions': successful,
                    'success_rate': successful / total_records if total_records > 0 else 0.0,
                    'average_confidence': avg_confidence,
                    'average_processing_time': avg_time,
                    'common_attributes': [
                        {'name': row[0], 'value': row[1], 'count': row[2]} 
                        for row in common_attributes
                    ]
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def process(self, input_data: Any) -> Any:
        """Process and store data (passthrough with storage)."""
        if isinstance(input_data, ProcessingResult):
            self.store_result(input_data)
            return input_data
        else:
            # For other data types, just pass through
            return input_data
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data."""
        return True  # Database storage accepts any input
    
    def cleanup_old_records(self, days: int = 30) -> int:
        """Clean up old records older than specified days."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    DELETE FROM extraction_results 
                    WHERE created_at < datetime('now', '-{} days')
                '''.format(days))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                self.logger.info(f"Cleaned up {deleted_count} old records")
                return deleted_count
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup old records: {e}")
            return 0