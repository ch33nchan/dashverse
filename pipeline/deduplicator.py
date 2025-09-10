"""Deduplication system using perceptual hashing and clustering."""

import hashlib
import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Set
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3
import json
from dataclasses import dataclass, asdict

from .base import PipelineStage

logger = logging.getLogger(__name__)

@dataclass
class ImageHash:
    """Perceptual hash representation of an image."""
    item_id: str
    image_path: str
    dhash: str
    phash: str
    ahash: str
    whash: str
    feature_vector: List[float]
    timestamp: float
    file_size: int
    image_dimensions: Tuple[int, int]

class PerceptualHasher:
    """Generate perceptual hashes for image similarity detection."""
    
    def __init__(self, hash_size: int = 8):
        self.hash_size = hash_size
    
    def compute_dhash(self, image: Image.Image) -> str:
        """Compute difference hash (dHash)."""
        # Resize and convert to grayscale
        resized = image.resize((self.hash_size + 1, self.hash_size), Image.Resampling.LANCZOS)
        gray = resized.convert('L')
        
        # Convert to numpy array
        pixels = np.array(gray)
        
        # Compute differences between adjacent pixels
        diff = pixels[:, 1:] > pixels[:, :-1]
        
        # Convert to hash string
        return ''.join(['1' if d else '0' for row in diff for d in row])
    
    def compute_phash(self, image: Image.Image) -> str:
        """Compute perceptual hash (pHash) using DCT."""
        # Resize and convert to grayscale
        resized = image.resize((32, 32), Image.Resampling.LANCZOS)
        gray = np.array(resized.convert('L'), dtype=np.float32)
        
        # Apply DCT
        dct = cv2.dct(gray)
        
        # Extract top-left 8x8 region (low frequencies)
        dct_low = dct[:self.hash_size, :self.hash_size]
        
        # Compute median
        median = np.median(dct_low)
        
        # Generate hash
        hash_bits = dct_low > median
        return ''.join(['1' if bit else '0' for row in hash_bits for bit in row])
    
    def compute_ahash(self, image: Image.Image) -> str:
        """Compute average hash (aHash)."""
        # Resize and convert to grayscale
        resized = image.resize((self.hash_size, self.hash_size), Image.Resampling.LANCZOS)
        gray = np.array(resized.convert('L'))
        
        # Compute average
        avg = np.mean(gray)
        
        # Generate hash
        hash_bits = gray > avg
        return ''.join(['1' if bit else '0' for row in hash_bits for bit in row])
    
    def compute_whash(self, image: Image.Image) -> str:
        """Compute wavelet hash (wHash)."""
        # Simplified wavelet hash using Haar-like transform
        resized = image.resize((self.hash_size, self.hash_size), Image.Resampling.LANCZOS)
        gray = np.array(resized.convert('L'), dtype=np.float32)
        
        # Simple 2D Haar transform approximation
        # Horizontal differences
        h_diff = gray[:, ::2] - gray[:, 1::2]
        
        # Vertical differences
        v_diff = gray[::2, :] - gray[1::2, :]
        
        # Combine and threshold
        combined = np.concatenate([h_diff.flatten(), v_diff.flatten()])
        median = np.median(combined)
        
        hash_bits = combined > median
        return ''.join(['1' if bit else '0' for bit in hash_bits[:self.hash_size*self.hash_size]])
    
    def compute_feature_vector(self, image: Image.Image) -> List[float]:
        """Compute feature vector for clustering."""
        try:
            # Resize image
            resized = image.resize((64, 64), Image.Resampling.LANCZOS)
            
            # Color histogram features
            hist_r = cv2.calcHist([np.array(resized)[:,:,0]], [0], None, [16], [0, 256])
            hist_g = cv2.calcHist([np.array(resized)[:,:,1]], [0], None, [16], [0, 256])
            hist_b = cv2.calcHist([np.array(resized)[:,:,2]], [0], None, [16], [0, 256])
            
            # Texture features (LBP-like)
            gray = cv2.cvtColor(np.array(resized), cv2.COLOR_RGB2GRAY)
            texture_features = self._compute_texture_features(gray)
            
            # Combine features
            features = np.concatenate([
                hist_r.flatten(),
                hist_g.flatten(), 
                hist_b.flatten(),
                texture_features
            ])
            
            # Normalize
            features = features / (np.linalg.norm(features) + 1e-8)
            
            return features.tolist()
            
        except Exception as e:
            logger.error(f"Feature vector computation failed: {e}")
            return [0.0] * 64  # Return zero vector as fallback
    
    def _compute_texture_features(self, gray_image: np.ndarray) -> np.ndarray:
        """Compute simple texture features."""
        # Gradient magnitude
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Texture statistics
        features = [
            np.mean(magnitude),
            np.std(magnitude),
            np.mean(gray_image),
            np.std(gray_image)
        ]
        
        return np.array(features)
    
    def compute_all_hashes(self, image: Image.Image, item_id: str, image_path: str) -> ImageHash:
        """Compute all perceptual hashes for an image."""
        try:
            # Get file info
            path_obj = Path(image_path)
            file_size = path_obj.stat().st_size if path_obj.exists() else 0
            
            return ImageHash(
                item_id=item_id,
                image_path=image_path,
                dhash=self.compute_dhash(image),
                phash=self.compute_phash(image),
                ahash=self.compute_ahash(image),
                whash=self.compute_whash(image),
                feature_vector=self.compute_feature_vector(image),
                timestamp=time.time(),
                file_size=file_size,
                image_dimensions=image.size
            )
            
        except Exception as e:
            logger.error(f"Hash computation failed for {item_id}: {e}")
            # Return empty hash as fallback
            return ImageHash(
                item_id=item_id,
                image_path=image_path,
                dhash="0" * (self.hash_size * self.hash_size),
                phash="0" * (self.hash_size * self.hash_size),
                ahash="0" * (self.hash_size * self.hash_size),
                whash="0" * (self.hash_size * self.hash_size),
                feature_vector=[0.0] * 64,
                timestamp=time.time(),
                file_size=0,
                image_dimensions=(0, 0)
            )

class DuplicateDetector:
    """Detect duplicate and near-duplicate images."""
    
    def __init__(self, config: Dict[str, Any]):
        self.similarity_threshold = config.get('similarity_threshold', 0.85)
        self.hash_distance_threshold = config.get('hash_distance_threshold', 5)
        self.clustering_eps = config.get('clustering_eps', 0.15)
        self.clustering_min_samples = config.get('clustering_min_samples', 2)
        
        # Hash database
        self.db_path = config.get('hash_db_path', './image_hashes.db')
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize hash storage database."""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS image_hashes (
                    item_id TEXT PRIMARY KEY,
                    image_path TEXT,
                    dhash TEXT,
                    phash TEXT,
                    ahash TEXT,
                    whash TEXT,
                    feature_vector TEXT,
                    timestamp REAL,
                    file_size INTEGER,
                    width INTEGER,
                    height INTEGER
                )
            ''')
            
            # Create indexes for fast similarity search
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_dhash ON image_hashes(dhash)')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_phash ON image_hashes(phash)')
            self.conn.execute('CREATE INDEX IF NOT EXISTS idx_file_size ON image_hashes(file_size)')
            
            self.conn.commit()
            logger.info("Hash database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize hash database: {e}")
            raise
    
    def store_hash(self, image_hash: ImageHash) -> bool:
        """Store image hash in database."""
        try:
            feature_vector_json = json.dumps(image_hash.feature_vector)
            
            self.conn.execute('''
                INSERT OR REPLACE INTO image_hashes 
                (item_id, image_path, dhash, phash, ahash, whash, feature_vector, 
                 timestamp, file_size, width, height)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                image_hash.item_id, image_hash.image_path,
                image_hash.dhash, image_hash.phash, image_hash.ahash, image_hash.whash,
                feature_vector_json, image_hash.timestamp, image_hash.file_size,
                image_hash.image_dimensions[0], image_hash.image_dimensions[1]
            ))
            
            self.conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to store hash for {image_hash.item_id}: {e}")
            return False
    
    def find_similar_images(self, target_hash: ImageHash) -> List[Dict[str, Any]]:
        """Find similar images using multiple hash comparison methods."""
        try:
            # Query database for potential matches
            cursor = self.conn.execute('''
                SELECT item_id, image_path, dhash, phash, ahash, whash, feature_vector, file_size
                FROM image_hashes 
                WHERE item_id != ?
            ''', (target_hash.item_id,))
            
            candidates = cursor.fetchall()
            similar_images = []
            
            for candidate in candidates:
                item_id, image_path, dhash, phash, ahash, whash, feature_vector_json, file_size = candidate
                
                # Parse feature vector
                try:
                    feature_vector = json.loads(feature_vector_json)
                except:
                    continue
                
                # Calculate similarities
                similarities = self._calculate_similarities(target_hash, {
                    'dhash': dhash,
                    'phash': phash,
                    'ahash': ahash,
                    'whash': whash,
                    'feature_vector': feature_vector
                })
                
                # Check if similar enough
                max_similarity = max(similarities.values())
                if max_similarity > self.similarity_threshold:
                    similar_images.append({
                        'item_id': item_id,
                        'image_path': image_path,
                        'similarities': similarities,
                        'max_similarity': max_similarity,
                        'file_size': file_size,
                        'size_difference': abs(target_hash.file_size - file_size)
                    })
            
            # Sort by similarity
            similar_images.sort(key=lambda x: x['max_similarity'], reverse=True)
            
            return similar_images
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    def _calculate_similarities(self, target_hash: ImageHash, candidate: Dict[str, Any]) -> Dict[str, float]:
        """Calculate various similarity metrics."""
        similarities = {}
        
        # Hash distance similarities
        for hash_type in ['dhash', 'phash', 'ahash', 'whash']:
            target_bits = target_hash.__dict__[hash_type]
            candidate_bits = candidate[hash_type]
            
            # Hamming distance
            hamming_distance = sum(t != c for t, c in zip(target_bits, candidate_bits))
            max_distance = len(target_bits)
            similarity = 1.0 - (hamming_distance / max_distance)
            similarities[f'{hash_type}_similarity'] = similarity
        
        # Feature vector cosine similarity
        try:
            target_features = np.array(target_hash.feature_vector).reshape(1, -1)
            candidate_features = np.array(candidate['feature_vector']).reshape(1, -1)
            
            cosine_sim = cosine_similarity(target_features, candidate_features)[0][0]
            similarities['feature_similarity'] = max(0.0, cosine_sim)  # Ensure non-negative
            
        except Exception as e:
            logger.warning(f"Feature similarity calculation failed: {e}")
            similarities['feature_similarity'] = 0.0
        
        return similarities
    
    def cluster_similar_images(self, image_hashes: List[ImageHash]) -> List[List[str]]:
        """Cluster similar images using feature vectors."""
        if len(image_hashes) < 2:
            return [[h.item_id] for h in image_hashes]
        
        try:
            # Extract feature vectors
            features = np.array([h.feature_vector for h in image_hashes])
            
            # Perform clustering
            clustering = DBSCAN(
                eps=self.clustering_eps,
                min_samples=self.clustering_min_samples,
                metric='cosine'
            )
            
            cluster_labels = clustering.fit_predict(features)
            
            # Group by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(image_hashes[i].item_id)
            
            # Return clusters as list (excluding noise points with label -1)
            return [cluster for label, cluster in clusters.items() if label != -1]
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return [[h.item_id] for h in image_hashes]

class Deduplicator(PipelineStage):
    """Main deduplication system for handling near-duplicate images."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("Deduplicator", config)
        
        # Configuration
        self.similarity_threshold = config.get('similarity_threshold', 0.85) if config else 0.85
        self.dedup_strategy = config.get('dedup_strategy', 'keep_highest_quality') if config else 'keep_highest_quality'
        self.batch_size = config.get('batch_size', 1000) if config else 1000
        
        # Components
        self.hasher = PerceptualHasher(config.get('hash_size', 8) if config else 8)
        self.detector = DuplicateDetector(config or {})
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'duplicates_found': 0,
            'clusters_created': 0,
            'items_removed': 0
        }
    
    def process_image_for_deduplication(self, image: Image.Image, item_id: str, 
                                       image_path: str) -> Dict[str, Any]:
        """Process single image for deduplication."""
        try:
            # Compute hashes
            image_hash = self.hasher.compute_all_hashes(image, item_id, image_path)
            
            # Store in database
            self.detector.store_hash(image_hash)
            
            # Find similar images
            similar_images = self.detector.find_similar_images(image_hash)
            
            # Determine if this is a duplicate
            is_duplicate = len(similar_images) > 0
            
            if is_duplicate:
                self.stats['duplicates_found'] += 1
                
                # Apply deduplication strategy
                action = self._determine_dedup_action(image_hash, similar_images)
            else:
                action = {'action': 'keep', 'reason': 'unique_image'}
            
            self.stats['total_processed'] += 1
            
            return {
                'item_id': item_id,
                'is_duplicate': is_duplicate,
                'similar_count': len(similar_images),
                'similar_images': similar_images[:5],  # Top 5 most similar
                'action': action,
                'hash_info': asdict(image_hash)
            }
            
        except Exception as e:
            logger.error(f"Deduplication processing failed for {item_id}: {e}")
            return {
                'item_id': item_id,
                'is_duplicate': False,
                'similar_count': 0,
                'similar_images': [],
                'action': {'action': 'keep', 'reason': 'processing_failed'},
                'error': str(e)
            }
    
    def _determine_dedup_action(self, target_hash: ImageHash, 
                               similar_images: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Determine what action to take for duplicate image."""
        if self.dedup_strategy == 'keep_highest_quality':
            # Keep image with largest file size (proxy for quality)
            max_size = max(sim['file_size'] for sim in similar_images)
            
            if target_hash.file_size >= max_size:
                return {'action': 'keep', 'reason': 'highest_quality'}
            else:
                return {
                    'action': 'remove', 
                    'reason': 'lower_quality',
                    'better_alternative': max(similar_images, key=lambda x: x['file_size'])['item_id']
                }
        
        elif self.dedup_strategy == 'keep_first':
            # Keep the first processed image (by timestamp)
            return {'action': 'remove', 'reason': 'duplicate_of_earlier_image'}
        
        elif self.dedup_strategy == 'keep_all':
            # Keep all but mark as duplicate
            return {'action': 'keep', 'reason': 'keep_all_strategy', 'mark_duplicate': True}
        
        else:
            return {'action': 'keep', 'reason': 'unknown_strategy'}
    
    def batch_deduplicate(self, image_paths: List[str]) -> Dict[str, Any]:
        """Perform batch deduplication on a list of image paths."""
        start_time = time.time()
        
        # Process images in batches
        all_hashes = []
        dedup_results = []
        
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i:i + self.batch_size]
            
            for j, image_path in enumerate(batch_paths):
                try:
                    item_id = f"batch_{i//self.batch_size}_{j}"
                    image = Image.open(image_path)
                    
                    result = self.process_image_for_deduplication(image, item_id, image_path)
                    dedup_results.append(result)
                    
                    if 'hash_info' in result:
                        all_hashes.append(ImageHash(**result['hash_info']))
                    
                except Exception as e:
                    logger.error(f"Failed to process {image_path}: {e}")
                    dedup_results.append({
                        'item_id': f"batch_{i//self.batch_size}_{j}",
                        'is_duplicate': False,
                        'action': {'action': 'keep', 'reason': 'processing_failed'},
                        'error': str(e)
                    })
        
        # Perform clustering for additional duplicate detection
        clusters = self.detector.cluster_similar_images(all_hashes)
        
        # Update results based on clustering
        cluster_updates = self._apply_clustering_results(dedup_results, clusters)
        
        processing_time = time.time() - start_time
        
        # Generate summary
        summary = self._generate_dedup_summary(dedup_results, clusters, processing_time)
        
        return {
            'results': dedup_results,
            'clusters': clusters,
            'cluster_updates': cluster_updates,
            'summary': summary
        }
    
    def _apply_clustering_results(self, dedup_results: List[Dict[str, Any]], 
                                 clusters: List[List[str]]) -> List[Dict[str, Any]]:
        """Apply clustering results to update deduplication decisions."""
        cluster_updates = []
        
        for cluster in clusters:
            if len(cluster) > 1:
                # Find the best representative for this cluster
                cluster_items = [r for r in dedup_results if r['item_id'] in cluster]
                
                if cluster_items:
                    # Keep the one with highest quality (largest file size)
                    best_item = max(cluster_items, 
                                  key=lambda x: x.get('hash_info', {}).get('file_size', 0))
                    
                    # Mark others for removal
                    for item in cluster_items:
                        if item['item_id'] != best_item['item_id']:
                            item['action'] = {
                                'action': 'remove',
                                'reason': 'clustered_duplicate',
                                'cluster_representative': best_item['item_id']
                            }
                            cluster_updates.append({
                                'item_id': item['item_id'],
                                'action': 'removed_from_cluster',
                                'representative': best_item['item_id']
                            })
        
        return cluster_updates
    
    def _generate_dedup_summary(self, results: List[Dict[str, Any]], 
                               clusters: List[List[str]], processing_time: float) -> Dict[str, Any]:
        """Generate deduplication summary."""
        total_items = len(results)
        duplicates = [r for r in results if r['is_duplicate']]
        removed_items = [r for r in results if r['action']['action'] == 'remove']
        
        return {
            'total_items': total_items,
            'duplicates_detected': len(duplicates),
            'items_removed': len(removed_items),
            'clusters_found': len(clusters),
            'duplicate_rate': len(duplicates) / total_items if total_items > 0 else 0,
            'removal_rate': len(removed_items) / total_items if total_items > 0 else 0,
            'processing_time': processing_time,
            'throughput': total_items / processing_time if processing_time > 0 else 0,
            'space_saved_estimate': sum(r.get('hash_info', {}).get('file_size', 0) for r in removed_items),
            'recommendations': self._generate_dedup_recommendations(duplicates, clusters)
        }
    
    def _generate_dedup_recommendations(self, duplicates: List[Dict[str, Any]], 
                                       clusters: List[List[str]]) -> List[str]:
        """Generate recommendations based on deduplication results."""
        recommendations = []
        
        duplicate_rate = len(duplicates) / max(1, self.stats['total_processed'])
        
        if duplicate_rate > 0.2:
            recommendations.append("High duplicate rate detected - consider data source cleanup")
        
        if len(clusters) > len(duplicates) * 0.5:
            recommendations.append("Many clusters found - consider stricter similarity thresholds")
        
        recommendations.extend([
            "Implement periodic deduplication for ongoing datasets",
            "Consider using content-based hashing for exact duplicates",
            "Monitor storage savings from deduplication"
        ])
        
        return recommendations
    
    def get_deduplication_stats(self) -> Dict[str, Any]:
        """Get comprehensive deduplication statistics."""
        # Database statistics
        cursor = self.conn.execute('SELECT COUNT(*) FROM image_hashes')
        total_hashes = cursor.fetchone()[0]
        
        cursor = self.conn.execute('SELECT AVG(file_size), SUM(file_size) FROM image_hashes')
        avg_size, total_size = cursor.fetchone()
        
        return {
            'processing_stats': self.stats,
            'database_stats': {
                'total_hashes_stored': total_hashes,
                'average_file_size_bytes': avg_size or 0,
                'total_storage_bytes': total_size or 0
            },
            'efficiency_metrics': {
                'duplicate_detection_rate': self.stats['duplicates_found'] / max(1, self.stats['total_processed']),
                'removal_rate': self.stats['items_removed'] / max(1, self.stats['total_processed'])
            }
        }
    
    def process(self, input_data: Any) -> Any:
        """Process deduplication operations."""
        if isinstance(input_data, dict):
            operation = input_data.get('operation')
            
            if operation == 'process_image':
                return self.process_image_for_deduplication(
                    input_data['image'],
                    input_data['item_id'],
                    input_data['image_path']
                )
            elif operation == 'batch_deduplicate':
                return self.batch_deduplicate(input_data['image_paths'])
            elif operation == 'stats':
                return self.get_deduplication_stats()
            elif operation == 'find_similar':
                return self.detector.find_similar_images(input_data['target_hash'])
        
        raise ValueError("Deduplicator expects operation dict as input")
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data."""
        return isinstance(input_data, dict) and 'operation' in input_data