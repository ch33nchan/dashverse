"""Edge case handling for multi-character detection and ambiguous images."""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import torch
from transformers import pipeline

from .base import PipelineStage, CharacterAttributes

logger = logging.getLogger(__name__)

class EdgeCaseHandler(PipelineStage):
    """Handles edge cases like multi-character images and ambiguous content."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("EdgeCaseHandler", config)
        
        # Configuration
        self.face_detection_threshold = config.get('face_detection_threshold', 0.5) if config else 0.5
        self.multi_character_threshold = config.get('multi_character_threshold', 2) if config else 2
        self.min_face_size = config.get('min_face_size', 50) if config else 50
        self.occlusion_threshold = config.get('occlusion_threshold', 0.7) if config else 0.7
        
        # Models
        self.face_cascade = None
        self.person_detector = None
        self.quality_assessor = None
        
        self._initialize_detectors()
    
    def _initialize_detectors(self):
        """Initialize detection models."""
        try:
            # OpenCV face detector
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Person detection using YOLO-like approach
            try:
                self.person_detector = pipeline(
                    "object-detection",
                    model="facebook/detr-resnet-50",
                    device=0 if torch.cuda.is_available() else -1
                )
            except Exception as e:
                logger.warning(f"Could not load person detector: {e}")
                self.person_detector = None
            
            logger.info("Edge case detectors initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize detectors: {e}")
    
    def detect_multiple_characters(self, image: Image.Image) -> Dict[str, Any]:
        """Detect if image contains multiple characters."""
        try:
            # Convert PIL to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Face detection
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self.min_face_size, self.min_face_size)
            )
            
            face_count = len(faces)
            
            # Person detection using transformer model
            person_count = 0
            person_confidence = 0.0
            
            if self.person_detector:
                try:
                    detections = self.person_detector(image)
                    persons = [d for d in detections if d['label'] == 'person' and d['score'] > self.face_detection_threshold]
                    person_count = len(persons)
                    person_confidence = np.mean([p['score'] for p in persons]) if persons else 0.0
                except Exception as e:
                    logger.warning(f"Person detection failed: {e}")
            
            # Determine if multiple characters
            is_multi_character = (
                face_count >= self.multi_character_threshold or 
                person_count >= self.multi_character_threshold
            )
            
            # Calculate confidence
            detection_confidence = max(
                face_count / self.multi_character_threshold,
                person_confidence
            )
            
            return {
                'is_multi_character': is_multi_character,
                'face_count': face_count,
                'person_count': person_count,
                'detection_confidence': min(detection_confidence, 1.0),
                'face_locations': faces.tolist() if len(faces) > 0 else [],
                'recommendation': 'skip' if is_multi_character else 'process'
            }
            
        except Exception as e:
            logger.error(f"Multi-character detection failed: {e}")
            return {
                'is_multi_character': False,
                'face_count': 0,
                'person_count': 0,
                'detection_confidence': 0.0,
                'face_locations': [],
                'recommendation': 'process',
                'error': str(e)
            }
    
    def assess_image_quality(self, image: Image.Image) -> Dict[str, Any]:
        """Assess image quality and detect potential issues."""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Basic quality metrics
            quality_metrics = {
                'resolution': img_array.shape[:2],
                'aspect_ratio': img_array.shape[1] / img_array.shape[0],
                'channels': img_array.shape[2] if len(img_array.shape) > 2 else 1
            }
            
            # Convert to grayscale for analysis
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Blur detection (Laplacian variance)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            is_blurry = blur_score < 100  # Threshold for blur detection
            
            # Brightness analysis
            brightness = np.mean(gray)
            is_too_dark = brightness < 50
            is_too_bright = brightness > 200
            
            # Contrast analysis
            contrast = gray.std()
            is_low_contrast = contrast < 30
            
            # Noise detection (using standard deviation in small patches)
            noise_score = self._estimate_noise(gray)
            is_noisy = noise_score > 20
            
            # Overall quality score
            quality_score = self._calculate_quality_score(
                blur_score, brightness, contrast, noise_score
            )
            
            # Determine recommendation
            issues = []
            if is_blurry:
                issues.append('blurry')
            if is_too_dark:
                issues.append('too_dark')
            if is_too_bright:
                issues.append('too_bright')
            if is_low_contrast:
                issues.append('low_contrast')
            if is_noisy:
                issues.append('noisy')
            
            recommendation = 'skip' if len(issues) >= 3 or quality_score < 0.3 else 'process'
            
            return {
                'quality_score': quality_score,
                'blur_score': blur_score,
                'brightness': brightness,
                'contrast': contrast,
                'noise_score': noise_score,
                'issues': issues,
                'is_acceptable': quality_score >= 0.3,
                'recommendation': recommendation,
                'metrics': quality_metrics
            }
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return {
                'quality_score': 0.5,
                'issues': ['assessment_failed'],
                'is_acceptable': True,
                'recommendation': 'process',
                'error': str(e)
            }
    
    def _estimate_noise(self, gray_image: np.ndarray) -> float:
        """Estimate noise level in image."""
        try:
            # Use Laplacian to detect edges, then measure noise in non-edge regions
            laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
            edges = np.abs(laplacian) > np.percentile(np.abs(laplacian), 90)
            
            # Calculate noise in non-edge regions
            non_edge_regions = gray_image[~edges]
            if len(non_edge_regions) > 0:
                return np.std(non_edge_regions)
            else:
                return np.std(gray_image)
        except:
            return np.std(gray_image)
    
    def _calculate_quality_score(self, blur_score: float, brightness: float, 
                                contrast: float, noise_score: float) -> float:
        """Calculate overall quality score from individual metrics."""
        # Normalize scores to 0-1 range
        blur_norm = min(blur_score / 500, 1.0)  # Higher is better
        brightness_norm = 1.0 - abs(brightness - 128) / 128  # Closer to 128 is better
        contrast_norm = min(contrast / 100, 1.0)  # Higher is better
        noise_norm = max(0, 1.0 - noise_score / 50)  # Lower is better
        
        # Weighted average
        weights = [0.3, 0.2, 0.3, 0.2]  # blur, brightness, contrast, noise
        scores = [blur_norm, brightness_norm, contrast_norm, noise_norm]
        
        return sum(w * s for w, s in zip(weights, scores))
    
    def detect_occlusion(self, image: Image.Image) -> Dict[str, Any]:
        """Detect if character is significantly occluded."""
        try:
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces to estimate visible character area
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
            
            if len(faces) == 0:
                # No face detected - might be occluded or not a character
                return {
                    'is_occluded': True,
                    'occlusion_ratio': 1.0,
                    'visible_face_area': 0,
                    'recommendation': 'skip',
                    'reason': 'no_face_detected'
                }
            
            # Calculate largest face area
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            face_area = largest_face[2] * largest_face[3]
            image_area = image.width * image.height
            
            # Estimate occlusion based on face size relative to image
            expected_face_ratio = 0.1  # Expected minimum face area ratio
            actual_face_ratio = face_area / image_area
            
            occlusion_ratio = max(0, 1 - (actual_face_ratio / expected_face_ratio))
            is_occluded = occlusion_ratio > self.occlusion_threshold
            
            return {
                'is_occluded': is_occluded,
                'occlusion_ratio': occlusion_ratio,
                'visible_face_area': face_area,
                'face_ratio': actual_face_ratio,
                'recommendation': 'skip' if is_occluded else 'process',
                'face_location': largest_face.tolist()
            }
            
        except Exception as e:
            logger.error(f"Occlusion detection failed: {e}")
            return {
                'is_occluded': False,
                'occlusion_ratio': 0.0,
                'visible_face_area': 0,
                'recommendation': 'process',
                'error': str(e)
            }
    
    def analyze_image_content(self, image: Image.Image) -> Dict[str, Any]:
        """Comprehensive analysis of image content for edge cases."""
        # Run all detection methods
        multi_char_result = self.detect_multiple_characters(image)
        quality_result = self.assess_image_quality(image)
        occlusion_result = self.detect_occlusion(image)
        
        # Combine results
        edge_cases = []
        if multi_char_result['is_multi_character']:
            edge_cases.append('multiple_characters')
        if not quality_result['is_acceptable']:
            edge_cases.append('poor_quality')
        if occlusion_result['is_occluded']:
            edge_cases.append('occluded')
        
        # Overall recommendation
        recommendations = [
            multi_char_result['recommendation'],
            quality_result['recommendation'],
            occlusion_result['recommendation']
        ]
        
        skip_count = recommendations.count('skip')
        overall_recommendation = 'skip' if skip_count >= 2 else 'process'
        
        # Confidence score for the recommendation
        confidence_scores = [
            multi_char_result.get('detection_confidence', 0.5),
            quality_result.get('quality_score', 0.5),
            1.0 - occlusion_result.get('occlusion_ratio', 0.5)
        ]
        overall_confidence = np.mean(confidence_scores)
        
        return {
            'edge_cases': edge_cases,
            'has_edge_cases': len(edge_cases) > 0,
            'recommendation': overall_recommendation,
            'confidence': overall_confidence,
            'details': {
                'multi_character': multi_char_result,
                'quality': quality_result,
                'occlusion': occlusion_result
            },
            'processing_advice': self._generate_processing_advice(edge_cases, overall_recommendation)
        }
    
    def _generate_processing_advice(self, edge_cases: List[str], recommendation: str) -> List[str]:
        """Generate advice for handling detected edge cases."""
        advice = []
        
        if 'multiple_characters' in edge_cases:
            advice.append("Image contains multiple characters - consider cropping to single character")
        
        if 'poor_quality' in edge_cases:
            advice.append("Image quality is poor - consider preprocessing or skipping")
        
        if 'occluded' in edge_cases:
            advice.append("Character appears occluded - extraction may be incomplete")
        
        if recommendation == 'skip':
            advice.append("Recommend skipping this image due to multiple edge cases")
        elif len(edge_cases) > 0:
            advice.append("Proceed with caution - edge cases detected but may be processable")
        else:
            advice.append("Image appears suitable for character extraction")
        
        return advice
    
    def process(self, input_data: Any) -> Dict[str, Any]:
        """Process image and return edge case analysis."""
        if isinstance(input_data, Image.Image):
            return self.analyze_image_content(input_data)
        elif isinstance(input_data, str):
            # Assume it's an image path
            try:
                image = Image.open(input_data)
                return self.analyze_image_content(image)
            except Exception as e:
                return {
                    'edge_cases': ['invalid_image'],
                    'has_edge_cases': True,
                    'recommendation': 'skip',
                    'confidence': 0.0,
                    'error': str(e)
                }
        else:
            raise ValueError("EdgeCaseHandler expects PIL Image or image path")
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data."""
        return isinstance(input_data, (Image.Image, str))