"""Preprocessing pipeline for style normalization and occlusion handling."""

import cv2
import numpy as np
import logging
import time
from typing import Dict, Any, Optional, Tuple, List, Union
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torchvision.transforms as transforms
from sklearn.cluster import KMeans

from .base import PipelineStage

logger = logging.getLogger(__name__)

class StyleNormalizer:
    """Normalizes different art styles for consistent processing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.target_size = config.get('target_size', (512, 512))
        self.normalize_brightness = config.get('normalize_brightness', True)
        self.normalize_contrast = config.get('normalize_contrast', True)
        self.normalize_saturation = config.get('normalize_saturation', True)
        
        # Style detection thresholds
        self.anime_threshold = config.get('anime_threshold', 0.7)
        self.realistic_threshold = config.get('realistic_threshold', 0.6)
    
    def detect_art_style(self, image: Image.Image) -> Dict[str, Any]:
        """Detect the art style of the image."""
        try:
            # Convert to numpy for analysis
            img_array = np.array(image)
            
            # Color analysis
            color_variance = np.var(img_array, axis=(0, 1))
            avg_color_variance = np.mean(color_variance)
            
            # Edge analysis
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Saturation analysis
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            saturation = hsv[:, :, 1]
            avg_saturation = np.mean(saturation)
            
            # Style classification heuristics
            anime_score = 0.0
            realistic_score = 0.0
            
            # High saturation + low color variance = anime-like
            if avg_saturation > 100 and avg_color_variance < 2000:
                anime_score += 0.4
            
            # Sharp edges = anime-like
            if edge_density > 0.1:
                anime_score += 0.3
            
            # High color variance + moderate saturation = realistic
            if avg_color_variance > 3000 and 50 < avg_saturation < 150:
                realistic_score += 0.5
            
            # Determine primary style
            if anime_score > self.anime_threshold:
                style = 'anime'
                confidence = anime_score
            elif realistic_score > self.realistic_threshold:
                style = 'realistic'
                confidence = realistic_score
            else:
                style = 'mixed'
                confidence = max(anime_score, realistic_score)
            
            return {
                'style': style,
                'confidence': confidence,
                'metrics': {
                    'color_variance': avg_color_variance,
                    'edge_density': edge_density,
                    'saturation': avg_saturation,
                    'anime_score': anime_score,
                    'realistic_score': realistic_score
                }
            }
            
        except Exception as e:
            logger.error(f"Style detection failed: {e}")
            return {
                'style': 'unknown',
                'confidence': 0.0,
                'error': str(e)
            }
    
    def normalize_image(self, image: Image.Image, style_info: Dict[str, Any]) -> Image.Image:
        """Normalize image based on detected style."""
        try:
            normalized = image.copy()
            
            # Resize to target size
            normalized = normalized.resize(self.target_size, Image.Resampling.LANCZOS)
            
            # Style-specific normalization
            style = style_info.get('style', 'unknown')
            
            if style == 'anime':
                normalized = self._normalize_anime_style(normalized)
            elif style == 'realistic':
                normalized = self._normalize_realistic_style(normalized)
            else:
                normalized = self._normalize_generic_style(normalized)
            
            return normalized
            
        except Exception as e:
            logger.error(f"Image normalization failed: {e}")
            return image.resize(self.target_size, Image.Resampling.LANCZOS)
    
    def _normalize_anime_style(self, image: Image.Image) -> Image.Image:
        """Normalize anime-style images."""
        # Anime images often have high saturation and sharp edges
        # Slightly reduce saturation for better CLIP processing
        if self.normalize_saturation:
            enhancer = ImageEnhance.Color(image)
            image = enhancer.enhance(0.9)  # Reduce saturation slightly
        
        # Enhance contrast for better feature detection
        if self.normalize_contrast:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.1)
        
        return image
    
    def _normalize_realistic_style(self, image: Image.Image) -> Image.Image:
        """Normalize realistic-style images."""
        # Realistic images may need brightness and contrast adjustment
        if self.normalize_brightness:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.05)  # Slight brightness boost
        
        if self.normalize_contrast:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.15)  # Enhance contrast
        
        # Slight sharpening for better feature detection
        image = image.filter(ImageFilter.UnsharpMask(radius=1, percent=110, threshold=3))
        
        return image
    
    def _normalize_generic_style(self, image: Image.Image) -> Image.Image:
        """Generic normalization for unknown styles."""
        # Conservative normalization
        if self.normalize_brightness:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.02)
        
        if self.normalize_contrast:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.05)
        
        return image

class OcclusionHandler:
    """Handles occluded or partially visible characters."""
    
    def __init__(self, config: Dict[str, Any]):
        self.min_visible_ratio = config.get('min_visible_ratio', 0.3)
        self.inpainting_enabled = config.get('inpainting_enabled', False)
        
        # Face detection for occlusion analysis
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    
    def detect_occlusion_regions(self, image: Image.Image) -> Dict[str, Any]:
        """Detect occluded regions in the image."""
        try:
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
            
            if len(faces) == 0:
                return {
                    'has_occlusion': True,
                    'occlusion_type': 'no_face_detected',
                    'visible_ratio': 0.0,
                    'recommendation': 'skip'
                }
            
            # Analyze largest face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            
            # Extract face region
            face_region = gray[y:y+h, x:x+w]
            
            # Detect occlusion using edge density and uniformity
            edges = cv2.Canny(face_region, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Check for uniform regions (potential occlusion)
            uniform_threshold = 10
            uniform_regions = np.std(face_region) < uniform_threshold
            
            # Estimate visible ratio
            visible_ratio = edge_density * (1.0 if not uniform_regions else 0.5)
            
            has_occlusion = visible_ratio < self.min_visible_ratio
            
            return {
                'has_occlusion': has_occlusion,
                'occlusion_type': 'partial' if has_occlusion else 'none',
                'visible_ratio': visible_ratio,
                'face_location': largest_face.tolist(),
                'edge_density': edge_density,
                'recommendation': 'process_with_caution' if has_occlusion else 'process'
            }
            
        except Exception as e:
            logger.error(f"Occlusion detection failed: {e}")
            return {
                'has_occlusion': False,
                'occlusion_type': 'unknown',
                'visible_ratio': 1.0,
                'recommendation': 'process',
                'error': str(e)
            }
    
    def enhance_occluded_image(self, image: Image.Image, 
                              occlusion_info: Dict[str, Any]) -> Image.Image:
        """Enhance occluded images for better attribute extraction."""
        try:
            enhanced = image.copy()
            
            if occlusion_info.get('has_occlusion', False):
                # Apply enhancement based on occlusion type
                occlusion_type = occlusion_info.get('occlusion_type', 'unknown')
                
                if occlusion_type == 'partial':
                    # Enhance contrast and sharpness for partially occluded images
                    enhancer = ImageEnhance.Contrast(enhanced)
                    enhanced = enhancer.enhance(1.3)
                    
                    enhancer = ImageEnhance.Sharpness(enhanced)
                    enhanced = enhancer.enhance(1.2)
                    
                    # Apply unsharp mask
                    enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
                
                elif occlusion_type == 'no_face_detected':
                    # Try to enhance overall image quality
                    enhancer = ImageEnhance.Brightness(enhanced)
                    enhanced = enhancer.enhance(1.1)
                    
                    enhancer = ImageEnhance.Contrast(enhanced)
                    enhanced = enhancer.enhance(1.2)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Image enhancement failed: {e}")
            return image

class ImagePreprocessor(PipelineStage):
    """Comprehensive image preprocessing for improved attribute extraction."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("ImagePreprocessor", config)
        
        # Configuration
        self.enable_style_normalization = config.get('enable_style_normalization', True) if config else True
        self.enable_occlusion_handling = config.get('enable_occlusion_handling', True) if config else True
        self.enable_quality_enhancement = config.get('enable_quality_enhancement', True) if config else True
        self.target_size = config.get('target_size', (512, 512)) if config else (512, 512)
        
        # Components
        self.style_normalizer = StyleNormalizer(config or {})
        self.occlusion_handler = OcclusionHandler(config or {})
        
        # Preprocessing statistics
        self.stats = {
            'processed': 0,
            'style_normalized': 0,
            'occlusion_handled': 0,
            'quality_enhanced': 0,
            'skipped': 0
        }
    
    def preprocess_image(self, image: Image.Image) -> Dict[str, Any]:
        """Comprehensive image preprocessing."""
        try:
            start_time = time.time()
            processed_image = image.copy()
            preprocessing_info = {
                'original_size': image.size,
                'steps_applied': [],
                'style_info': {},
                'occlusion_info': {},
                'quality_info': {}
            }
            
            # Step 1: Style detection and normalization
            if self.enable_style_normalization:
                style_info = self.style_normalizer.detect_art_style(processed_image)
                processed_image = self.style_normalizer.normalize_image(processed_image, style_info)
                preprocessing_info['style_info'] = style_info
                preprocessing_info['steps_applied'].append('style_normalization')
                self.stats['style_normalized'] += 1
            
            # Step 2: Occlusion detection and handling
            if self.enable_occlusion_handling:
                occlusion_info = self.occlusion_handler.detect_occlusion_regions(processed_image)
                
                if occlusion_info.get('has_occlusion', False):
                    processed_image = self.occlusion_handler.enhance_occluded_image(
                        processed_image, occlusion_info
                    )
                    preprocessing_info['steps_applied'].append('occlusion_handling')
                    self.stats['occlusion_handled'] += 1
                
                preprocessing_info['occlusion_info'] = occlusion_info
            
            # Step 3: Quality enhancement
            if self.enable_quality_enhancement:
                quality_info = self._assess_and_enhance_quality(processed_image)
                if quality_info.get('enhanced', False):
                    processed_image = quality_info['enhanced_image']
                    preprocessing_info['steps_applied'].append('quality_enhancement')
                    self.stats['quality_enhanced'] += 1
                
                preprocessing_info['quality_info'] = quality_info
            
            # Final validation
            should_skip = self._should_skip_image(preprocessing_info)
            if should_skip:
                self.stats['skipped'] += 1
                preprocessing_info['recommendation'] = 'skip'
                preprocessing_info['skip_reason'] = should_skip
            else:
                preprocessing_info['recommendation'] = 'process'
            
            processing_time = time.time() - start_time
            preprocessing_info['processing_time'] = processing_time
            
            self.stats['processed'] += 1
            
            return {
                'processed_image': processed_image,
                'preprocessing_info': preprocessing_info,
                'should_skip': should_skip is not False
            }
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            return {
                'processed_image': image,
                'preprocessing_info': {'error': str(e)},
                'should_skip': False
            }
    
    def _assess_and_enhance_quality(self, image: Image.Image) -> Dict[str, Any]:
        """Assess and enhance image quality."""
        try:
            # Convert to numpy for analysis
            img_array = np.array(image)
            
            # Quality metrics
            brightness = np.mean(img_array)
            contrast = np.std(img_array)
            
            # Blur detection
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            quality_issues = []
            enhanced_image = image.copy()
            
            # Brightness correction
            if brightness < 80:
                quality_issues.append('too_dark')
                enhancer = ImageEnhance.Brightness(enhanced_image)
                enhanced_image = enhancer.enhance(1.2)
            elif brightness > 180:
                quality_issues.append('too_bright')
                enhancer = ImageEnhance.Brightness(enhanced_image)
                enhanced_image = enhancer.enhance(0.9)
            
            # Contrast enhancement
            if contrast < 40:
                quality_issues.append('low_contrast')
                enhancer = ImageEnhance.Contrast(enhanced_image)
                enhanced_image = enhancer.enhance(1.3)
            
            # Blur correction
            if blur_score < 100:
                quality_issues.append('blurry')
                enhanced_image = enhanced_image.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
            
            return {
                'quality_issues': quality_issues,
                'enhanced': len(quality_issues) > 0,
                'enhanced_image': enhanced_image if quality_issues else image,
                'metrics': {
                    'brightness': brightness,
                    'contrast': contrast,
                    'blur_score': blur_score
                }
            }
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return {
                'quality_issues': [],
                'enhanced': False,
                'enhanced_image': image,
                'error': str(e)
            }
    
    def _should_skip_image(self, preprocessing_info: Dict[str, Any]) -> Union[str, bool]:
        """Determine if image should be skipped based on preprocessing results."""
        # Check occlusion
        occlusion_info = preprocessing_info.get('occlusion_info', {})
        if occlusion_info.get('recommendation') == 'skip':
            return 'severe_occlusion'
        
        # Check style confidence
        style_info = preprocessing_info.get('style_info', {})
        if style_info.get('confidence', 1.0) < 0.2:
            return 'unrecognizable_style'
        
        # Check quality issues
        quality_info = preprocessing_info.get('quality_info', {})
        quality_issues = quality_info.get('quality_issues', [])
        if len(quality_issues) >= 3:
            return 'poor_quality'
        
        return False
    
    def batch_preprocess(self, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """Preprocess a batch of images efficiently."""
        results = []
        
        for i, image in enumerate(images):
            try:
                result = self.preprocess_image(image)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to preprocess image {i}: {e}")
                results.append({
                    'processed_image': image,
                    'preprocessing_info': {'error': str(e)},
                    'should_skip': True
                })
        
        return results
    
    def get_preprocessing_stats(self) -> Dict[str, Any]:
        """Get preprocessing statistics."""
        total = self.stats['processed']
        
        return {
            'total_processed': total,
            'style_normalized': self.stats['style_normalized'],
            'occlusion_handled': self.stats['occlusion_handled'],
            'quality_enhanced': self.stats['quality_enhanced'],
            'skipped': self.stats['skipped'],
            'skip_rate': self.stats['skipped'] / total if total > 0 else 0,
            'enhancement_rate': (self.stats['style_normalized'] + self.stats['quality_enhanced']) / total if total > 0 else 0
        }
    
    def process(self, input_data: Any) -> Any:
        """Process image preprocessing."""
        if isinstance(input_data, Image.Image):
            return self.preprocess_image(input_data)
        elif isinstance(input_data, list) and all(isinstance(img, Image.Image) for img in input_data):
            return self.batch_preprocess(input_data)
        elif isinstance(input_data, dict) and input_data.get('operation') == 'stats':
            return self.get_preprocessing_stats()
        else:
            raise ValueError("ImagePreprocessor expects PIL Image, list of images, or stats operation")
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data."""
        return (
            isinstance(input_data, Image.Image) or
            (isinstance(input_data, list) and all(isinstance(img, Image.Image) for img in input_data)) or
            (isinstance(input_data, dict) and 'operation' in input_data)
        )