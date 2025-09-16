"""Advanced caching system with Redis and database sharding for 5M+ scale."""

import redis
import sqlite3
import hashlib
import pickle
import json
import logging
import time
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import threading
from contextlib import contextmanager
from dataclasses import asdict

from .base import PipelineStage, CharacterAttributes, ProcessingResult

logger = logging.getLogger(__name__)

class ShardedDatabase:
    """Database sharding implementation for horizontal scaling."""
    
    def __init__(self, base_path: str, num_shards: int = 16):
        self.base_path = Path(base_path)
        self.num_shards = num_shards
        self.connections = {}
        self.locks = {i: threading.Lock() for i in range(num_shards)}
        
        # Create shard directories
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize shards
        for shard_id in range(num_shards):
            self._initialize_shard(shard_id)
    
    def _initialize_shard(self, shard_id: int):
        """Initialize a database shard."""
        shard_path = self.base_path / f"shard_{shard_id}.db"
        
        conn = sqlite3.connect(str(shard_path), check_same_thread=False)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS character_cache (
                key TEXT PRIMARY KEY,
                value BLOB,
                timestamp REAL,
                access_count INTEGER DEFAULT 0,
                ttl REAL
            )
        ''')
        
        # Create indexes for performance
        conn.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON character_cache(timestamp)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_ttl ON character_cache(ttl)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_access_count ON character_cache(access_count)')
        
        conn.commit()
        self.connections[shard_id] = conn
    
    def _get_shard_id(self, key: str) -> int:
        """Determine shard ID for a given key using consistent hashing."""
        return hash(key) % self.num_shards
    
    @contextmanager
    def _get_connection(self, shard_id: int):
        """Get database connection with proper locking."""
        with self.locks[shard_id]:
            yield self.connections[shard_id]
    
    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Store value in appropriate shard."""
        try:
            shard_id = self._get_shard_id(key)
            serialized_value = pickle.dumps(value)
            current_time = time.time()
            expiry_time = current_time + ttl if ttl else None
            
            with self._get_connection(shard_id) as conn:
                conn.execute(
                    'INSERT OR REPLACE INTO character_cache (key, value, timestamp, ttl) VALUES (?, ?, ?, ?)',
                    (key, serialized_value, current_time, expiry_time)
                )
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to set cache key {key}: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from appropriate shard."""
        try:
            shard_id = self._get_shard_id(key)
            current_time = time.time()
            
            with self._get_connection(shard_id) as conn:
                cursor = conn.execute(
                    'SELECT value, ttl FROM character_cache WHERE key = ?',
                    (key,)
                )
                result = cursor.fetchone()
                
                if result is None:
                    return None
                
                value_blob, ttl = result
                
                # Check TTL
                if ttl and current_time > ttl:
                    # Expired, remove it
                    conn.execute('DELETE FROM character_cache WHERE key = ?', (key,))
                    conn.commit()
                    return None
                
                # Update access count
                conn.execute(
                    'UPDATE character_cache SET access_count = access_count + 1 WHERE key = ?',
                    (key,)
                )
                conn.commit()
                
                return pickle.loads(value_blob)
                
        except Exception as e:
            logger.error(f"Failed to get cache key {key}: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete key from appropriate shard."""
        try:
            shard_id = self._get_shard_id(key)
            
            with self._get_connection(shard_id) as conn:
                cursor = conn.execute('DELETE FROM character_cache WHERE key = ?', (key,))
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Failed to delete cache key {key}: {e}")
            return False
    
    def cleanup_expired(self) -> int:
        """Remove expired entries from all shards."""
        total_removed = 0
        current_time = time.time()
        
        for shard_id in range(self.num_shards):
            try:
                with self._get_connection(shard_id) as conn:
                    cursor = conn.execute(
                        'DELETE FROM character_cache WHERE ttl IS NOT NULL AND ttl < ?',
                        (current_time,)
                    )
                    conn.commit()
                    total_removed += cursor.rowcount
            except Exception as e:
                logger.error(f"Failed to cleanup shard {shard_id}: {e}")
        
        return total_removed
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics across all shards."""
        stats = {
            'total_entries': 0,
            'total_size_mb': 0,
            'shard_stats': []
        }
        
        for shard_id in range(self.num_shards):
            try:
                with self._get_connection(shard_id) as conn:
                    cursor = conn.execute('SELECT COUNT(*), SUM(LENGTH(value)) FROM character_cache')
                    count, size_bytes = cursor.fetchone()
                    
                    shard_stat = {
                        'shard_id': shard_id,
                        'entries': count or 0,
                        'size_mb': (size_bytes or 0) / (1024 * 1024)
                    }
                    
                    stats['shard_stats'].append(shard_stat)
                    stats['total_entries'] += shard_stat['entries']
                    stats['total_size_mb'] += shard_stat['size_mb']
                    
            except Exception as e:
                logger.error(f"Failed to get stats for shard {shard_id}: {e}")
        
        return stats

class RedisCache:
    """Redis-based distributed cache for high-performance caching."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = None
        self.is_available = False
        
        self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis connection."""
        try:
            redis_config = {
                'host': self.config.get('redis_host', 'localhost'),
                'port': self.config.get('redis_port', 6379),
                'db': self.config.get('redis_db', 0),
                'decode_responses': False,
                'socket_timeout': self.config.get('socket_timeout', 5),
                'socket_connect_timeout': self.config.get('connect_timeout', 5)
            }
            
            if 'redis_password' in self.config:
                redis_config['password'] = self.config['redis_password']
            
            self.redis_client = redis.Redis(**redis_config)
            
            # Test connection
            self.redis_client.ping()
            self.is_available = True
            logger.info("Redis cache initialized successfully")
            
        except Exception as e:
            logger.warning(f"Redis not available, falling back to local cache: {e}")
            self.is_available = False
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Redis cache."""
        if not self.is_available:
            return False
        
        try:
            serialized_value = pickle.dumps(value)
            if ttl:
                return self.redis_client.setex(key, ttl, serialized_value)
            else:
                return self.redis_client.set(key, serialized_value)
        except Exception as e:
            logger.error(f"Redis set failed for key {key}: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        if not self.is_available:
            return None
        
        try:
            value = self.redis_client.get(key)
            if value is None:
                return None
            return pickle.loads(value)
        except Exception as e:
            logger.error(f"Redis get failed for key {key}: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete key from Redis cache."""
        if not self.is_available:
            return False
        
        try:
            return bool(self.redis_client.delete(key))
        except Exception as e:
            logger.error(f"Redis delete failed for key {key}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics."""
        if not self.is_available:
            return {'available': False}
        
        try:
            info = self.redis_client.info()
            return {
                'available': True,
                'used_memory_mb': info.get('used_memory', 0) / (1024 * 1024),
                'connected_clients': info.get('connected_clients', 0),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0)
            }
        except Exception as e:
            logger.error(f"Failed to get Redis stats: {e}")
            return {'available': False, 'error': str(e)}

class AdvancedCacheManager(PipelineStage):
    """Advanced multi-tier caching system for 5M+ scale processing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("AdvancedCacheManager", config)
        
        # Configuration
        self.cache_dir = config.get('cache_dir', './cache') if config else './cache'
        self.num_shards = config.get('num_shards', 16) if config else 16
        self.default_ttl = config.get('default_ttl', 3600) if config else 3600  # 1 hour
        self.max_memory_mb = config.get('max_memory_mb', 1024) if config else 1024  # 1GB
        
        # Initialize cache layers
        self.redis_cache = RedisCache(config.get('redis', {}) if config else {})
        self.sharded_db = ShardedDatabase(self.cache_dir, self.num_shards)
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0
        }
    
    def _generate_cache_key(self, image_path: str, pipeline_version: str = "v1") -> str:
        """Generate consistent cache key for image."""
        # Use image path and file modification time for cache invalidation
        try:
            file_path = Path(image_path)
            if file_path.exists():
                mtime = file_path.stat().st_mtime
                content = f"{image_path}:{mtime}:{pipeline_version}"
            else:
                content = f"{image_path}:{pipeline_version}"
            
            return hashlib.sha256(content.encode()).hexdigest()
        except Exception:
            # Fallback to simple hash
            return hashlib.sha256(f"{image_path}:{pipeline_version}".encode()).hexdigest()
    
    def get_cached_result(self, image_path: str) -> Optional[CharacterAttributes]:
        """Retrieve cached character attributes for image."""
        cache_key = self._generate_cache_key(image_path)
        
        # Try Redis first (fastest)
        if self.redis_cache.is_available:
            result = self.redis_cache.get(cache_key)
            if result is not None:
                self.stats['hits'] += 1
                logger.debug(f"Cache hit (Redis): {cache_key}")
                return result
        
        # Try sharded database
        result = self.sharded_db.get(cache_key)
        if result is not None:
            self.stats['hits'] += 1
            logger.debug(f"Cache hit (DB): {cache_key}")
            
            # Promote to Redis for faster future access
            if self.redis_cache.is_available:
                self.redis_cache.set(cache_key, result, self.default_ttl)
            
            return result
        
        self.stats['misses'] += 1
        logger.debug(f"Cache miss: {cache_key}")
        return None
    
    def cache_result(self, image_path: str, attributes: CharacterAttributes) -> bool:
        """Cache character attributes for image."""
        cache_key = self._generate_cache_key(image_path)
        
        success = False
        
        # Store in Redis (fast access)
        if self.redis_cache.is_available:
            if self.redis_cache.set(cache_key, attributes, self.default_ttl):
                success = True
        
        # Store in sharded database (persistent)
        if self.sharded_db.set(cache_key, attributes, self.default_ttl):
            success = True
        
        if success:
            self.stats['sets'] += 1
            logger.debug(f"Cached result: {cache_key}")
        
        return success
    
    def invalidate_cache(self, image_path: str) -> bool:
        """Invalidate cached result for image."""
        cache_key = self._generate_cache_key(image_path)
        
        success = False
        
        # Remove from Redis
        if self.redis_cache.is_available:
            if self.redis_cache.delete(cache_key):
                success = True
        
        # Remove from sharded database
        if self.sharded_db.delete(cache_key):
            success = True
        
        if success:
            self.stats['deletes'] += 1
            logger.debug(f"Invalidated cache: {cache_key}")
        
        return success
    
    def cleanup_expired(self) -> Dict[str, int]:
        """Clean up expired cache entries."""
        db_removed = self.sharded_db.cleanup_expired()
        
        return {
            'database_removed': db_removed,
            'redis_available': self.redis_cache.is_available
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        db_stats = self.sharded_db.get_stats()
        redis_stats = self.redis_cache.get_stats()
        
        hit_rate = self.stats['hits'] / (self.stats['hits'] + self.stats['misses']) if (self.stats['hits'] + self.stats['misses']) > 0 else 0
        
        return {
            'performance': {
                'hit_rate': hit_rate,
                'total_hits': self.stats['hits'],
                'total_misses': self.stats['misses'],
                'total_sets': self.stats['sets'],
                'total_deletes': self.stats['deletes']
            },
            'database': db_stats,
            'redis': redis_stats,
            'configuration': {
                'num_shards': self.num_shards,
                'default_ttl': self.default_ttl,
                'max_memory_mb': self.max_memory_mb
            }
        }
    
    def estimate_5m_capacity(self) -> Dict[str, Any]:
        """Estimate cache capacity for 5M samples."""
        # Estimate average size per cached result
        sample_attributes = CharacterAttributes(
            age="young adult",
            gender="female",
            ethnicity="Asian",
            hair_style="ponytail",
            hair_color="black",
            hair_length="long",
            eye_color="brown",
            body_type="slim",
            dress="casual",
            confidence_score=0.85
        )
        
        sample_size_bytes = len(pickle.dumps(sample_attributes))
        
        # Calculate storage requirements for 5M samples
        total_size_gb = (sample_size_bytes * 5_000_000) / (1024**3)
        
        # Estimate shard distribution
        samples_per_shard = 5_000_000 // self.num_shards
        size_per_shard_gb = total_size_gb / self.num_shards
        
        return {
            'sample_size_bytes': sample_size_bytes,
            'total_storage_gb': total_size_gb,
            'samples_per_shard': samples_per_shard,
            'storage_per_shard_gb': size_per_shard_gb,
            'recommended_shards': max(16, int(total_size_gb / 10)),  # 10GB per shard max
            'recommended_redis_memory_gb': min(32, total_size_gb * 0.1),  # 10% in Redis
            'scalability_notes': [
                f"Each cached result requires ~{sample_size_bytes} bytes",
                f"5M samples would require ~{total_size_gb:.1f}GB storage",
                f"Current {self.num_shards} shards can handle ~{samples_per_shard:,} samples each",
                "Consider Redis cluster for distributed memory caching",
                "Implement cache warming strategies for frequently accessed data"
            ]
        }
    
    def process(self, input_data: Any) -> Any:
        """Process cache operations."""
        if isinstance(input_data, dict):
            operation = input_data.get('operation')
            
            if operation == 'get':
                return self.get_cached_result(input_data['image_path'])
            elif operation == 'set':
                return self.cache_result(input_data['image_path'], input_data['attributes'])
            elif operation == 'invalidate':
                return self.invalidate_cache(input_data['image_path'])
            elif operation == 'stats':
                return self.get_cache_stats()
            elif operation == 'cleanup':
                return self.cleanup_expired()
            elif operation == 'estimate':
                return self.estimate_5m_capacity()
        
        raise ValueError("AdvancedCacheManager expects operation dict as input")
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data."""
        return isinstance(input_data, dict) and 'operation' in input_data