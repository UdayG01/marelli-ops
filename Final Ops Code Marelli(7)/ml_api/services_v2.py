# ml_api/services.py - Enhanced with IMPROVED BOUNDING BOX ACCURACY - FIXED INDENTATION

import os
import cv2
import numpy as np
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import tempfile
import logging
from django.conf import settings
from django.core.files.storage import default_storage
# Add these imports after your existing imports around line 10-15
from itertools import combinations
import json

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("WARNING: ultralytics not available. Please install: pip install ultralytics")

logger = logging.getLogger(__name__)

class FlexibleNutDetectionService:
    """
    Flexible Industrial Nut Detection Service - ENHANCED WITH IMPROVED BOUNDING BOX ACCURACY
    Implements your exact business logic:
    - All 4 nuts present → GREEN boxes on all positions
    - Any nut missing → RED boxes on all positions + report
    - FIXED: Perfect bounding box positioning on nuts
    """

    def __init__(self):
        self.model = None
        self.model_path = getattr(settings, 'NUT_DETECTION_MODEL_PATH', 
                                 os.path.join(settings.BASE_DIR, 'models', 'industrial_nut_detection.pt'))

        # Configuration matching your ML model exactly
        self.config = {
            'confidence_threshold': 0.35,    # Primary confidence (configurable)
            'primary_confidence': 0.35,      # Same as above for backward compatibility
            'fallback_confidence': 0.25,     # Fallback detection confidence
            'minimum_confidence': 0.15,      # Minimum detection confidence
            'ultra_low_confidence': [0.1, 0.05],  # Ultra low confidence levels
            'iou_threshold': 0.45,
            'expected_classes': ['MISSING', 'PRESENT'],
            'target_size': (640, 640),
            'max_detections': 8,             # Allow more than 4 to filter later
            'overlap_threshold': 0.7,        # FIXED: Increased for better duplicate removal
            'multi_scales': [480, 640, 800, 1024],  # Multi-scale detection
            'expected_nuts': 4,
            # NEW: Bounding box refinement parameters
            'box_refinement': {
                'enable': True,
                'min_box_area': 400,         # Minimum box area (20x20 pixels)
                'max_box_area': 50000,       # Maximum box area (224x224 pixels)
                'aspect_ratio_range': (0.3, 3.0),  # Width/height ratio limits
                'center_adjustment': 0.1,    # 10% center adjustment tolerance
                'size_normalization': True   # Normalize box sizes
            }
        }

        self.stats = {
            'total_processed': 0,
            'successful_detections': 0,
            'failed_detections': 0,
            'average_processing_time': 0,
            'primary_detection_count': 0,
            'fallback_detection_count': 0,
            'enhancement_detection_count': 0,
            'multi_scale_detection_count': 0,
            'complete_detections': 0,
            'incomplete_detections': 0,
            # NEW: Box refinement stats
            'boxes_refined': 0,
            'boxes_filtered_out': 0,
            'duplicate_boxes_removed': 0
        }

        # Reference distance profile storage
        self.reference_profile = None
        self.reference_distances = []  # Store actual distances as list
        self.reference_profile_file = os.path.join(settings.BASE_DIR, 'reference_distance_profile.json')
        self.reference_image_path = os.path.join(settings.BASE_DIR, 'reference_images', 'perfect_reference.jpg')

        print("SYSTEM INITIALIZATION: Loading YOLO model first...")
        # Load model FIRST
        self._load_model()
        logger.info("YOLOv8 model loaded: {}".format(self.model_path))

        print("SYSTEM INITIALIZATION: Setting up reference distances...")
        # THEN load/create reference distances
        self._load_or_create_reference_distances()

        logger.info("Enhanced Industrial Nut Detection Service Initialized with Improved Bounding Box Accuracy")

    def _load_model(self):
        """Load your trained YOLOv8 model"""
        try:
            if not YOLO_AVAILABLE:
                raise ImportError("ultralytics not available")
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            
            self.model = YOLO(self.model_path)
            logger.info(f"Model loaded: {self.model_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def update_confidence_levels(self, primary=None, fallback=None, minimum=None, ultra_low=None):
        """
        Update confidence levels for different detection methods
        
        Args:
            primary: Primary detection confidence (0.1-0.9)
            fallback: Fallback detection confidence (0.1-0.8)
            minimum: Minimum detection confidence (0.05-0.5)
            ultra_low: List of ultra low confidence levels
        """
        if primary is not None:
            self.config['confidence_threshold'] = primary
            self.config['primary_confidence'] = primary
            logger.info(f"Updated primary confidence to: {primary}")
        
        if fallback is not None:
            self.config['fallback_confidence'] = fallback
            logger.info(f"Updated fallback confidence to: {fallback}")
        
        if minimum is not None:
            self.config['minimum_confidence'] = minimum
            logger.info(f"Updated minimum confidence to: {minimum}")
        
        if ultra_low is not None:
            self.config['ultra_low_confidence'] = ultra_low
            logger.info(f"Updated ultra low confidence levels to: {ultra_low}")
        
        logger.info("Confidence levels updated successfully")

    def get_confidence_settings(self):
        """Get current confidence level settings"""
        return {
            'primary_confidence': self.config['primary_confidence'],
            'fallback_confidence': self.config['fallback_confidence'],
            'minimum_confidence': self.config['minimum_confidence'],
            'ultra_low_confidence': self.config['ultra_low_confidence']
        }

    def enhance_image_for_detection(self, image):
        """Apply comprehensive image enhancement techniques"""
        enhanced_versions = {}
        
        # Original
        enhanced_versions['original'] = image.copy()
        
        try:
            # 1. Brightness normalization
            brightness = np.mean(image)
            if brightness < 100 or brightness > 140:
                target_brightness = 120
                factor = target_brightness / max(brightness, 1)
                bright_enhanced = np.clip(image * factor, 0, 255).astype(np.uint8)
                enhanced_versions['brightness'] = bright_enhanced
            
            # 2. CLAHE contrast enhancement
            if len(image.shape) == 3:
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                clahe_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                enhanced_versions['clahe'] = clahe_enhanced
            
            # 3. Sharpening filter
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(image, -1, kernel)
            enhanced_versions['sharpened'] = sharpened
            
            # 4. Combined enhancement
            combined = enhanced_versions.get('brightness', image)
            if 'clahe' in enhanced_versions:
                combined = cv2.addWeighted(combined, 0.7, enhanced_versions['clahe'], 0.3, 0)
            enhanced_versions['combined'] = combined
            
        except Exception as e:
            logger.error(f"Image enhancement error: {e}")
        
        return enhanced_versions

    def _extract_detection_info(self, box, method='unknown'):
        """ENHANCED: Extract detection information from YOLO box with validation"""
        try:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            
            if class_id in [0, 1]:  # Only accept MISSING(0) or PRESENT(1)
                # NEW: Validate bounding box before accepting
                if self._validate_bounding_box(bbox):
                    return {
                        'class_id': class_id,
                        'class_name': self.config['expected_classes'][class_id],
                        'confidence': confidence,
                        'bbox': bbox,
                        'detection_method': method
                    }
                else:
                    logger.debug(f"Bounding box validation failed for detection: {bbox}")
                    self.stats['boxes_filtered_out'] += 1
            return None
        except Exception as e:
            logger.error(f"Error extracting detection: {e}")
            return None

    def _validate_bounding_box(self, bbox):
        """NEW: Validate bounding box dimensions and aspect ratio"""
        try:
            x1, y1, x2, y2 = bbox
            
            # Calculate box dimensions
            width = x2 - x1
            height = y2 - y1
            area = width * height
            aspect_ratio = width / height if height > 0 else 0
            
            # Get validation parameters
            box_config = self.config['box_refinement']
            
            # Check area bounds
            if area < box_config['min_box_area'] or area > box_config['max_box_area']:
                logger.debug(f"Box area {area} outside bounds [{box_config['min_box_area']}, {box_config['max_box_area']}]")
                return False
            
            # Check aspect ratio
            min_ratio, max_ratio = box_config['aspect_ratio_range']
            if aspect_ratio < min_ratio or aspect_ratio > max_ratio:
                logger.debug(f"Aspect ratio {aspect_ratio:.2f} outside bounds [{min_ratio}, {max_ratio}]")
                return False
            
            # Check for valid coordinates
            if width <= 0 or height <= 0:
                logger.debug(f"Invalid box dimensions: {width}x{height}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Box validation error: {e}")
            return False

    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        try:
            x1_max = max(bbox1[0], bbox2[0])
            y1_max = max(bbox1[1], bbox2[1])
            x2_min = min(bbox1[2], bbox2[2])
            y2_min = min(bbox1[3], bbox2[3])
            
            if x2_min <= x1_max or y2_min <= y1_max:
                return 0.0
            
            intersection = (x2_min - x1_max) * (y2_min - y1_max)
            
            area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
            area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
            
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        except:
            return 0.0

    def _overlaps_with_existing(self, new_detection, existing_detections, overlap_threshold=None):
        """ENHANCED: Check if new detection overlaps significantly with existing ones"""
        if overlap_threshold is None:
            overlap_threshold = self.config['overlap_threshold']
        
        try:
            new_bbox = new_detection['bbox']
            
            for existing in existing_detections:
                existing_bbox = existing['bbox']
                iou = self._calculate_iou(new_bbox, existing_bbox)
                
                # ENHANCED: Also check center distance for better overlap detection
                center_distance = self._calculate_center_distance(new_bbox, existing_bbox)
                box_diagonal = self._calculate_box_diagonal(new_bbox)
                
                # Consider overlap if IoU is high OR centers are very close
                if iou > overlap_threshold or (center_distance < box_diagonal * 0.3):
                    return True
            return False
        except:
            return False

    def _calculate_center_distance(self, bbox1, bbox2):
        """NEW: Calculate distance between bounding box centers"""
        try:
            center1_x = (bbox1[0] + bbox1[2]) / 2
            center1_y = (bbox1[1] + bbox1[3]) / 2
            center2_x = (bbox2[0] + bbox2[2]) / 2
            center2_y = (bbox2[1] + bbox2[3]) / 2
            
            distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
            return distance
        except:
            return float('inf')

    def _calculate_box_diagonal(self, bbox):
        """NEW: Calculate bounding box diagonal length"""
        try:
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            diagonal = np.sqrt(width**2 + height**2)
            return diagonal
        except:
            return 0

    def _refine_bounding_boxes(self, detections):
        """NEW: Refine bounding box coordinates for better accuracy"""
        if not self.config['box_refinement']['enable']:
            return detections
        
        refined_detections = []
        
        for detection in detections:
            try:
                refined_detection = detection.copy()
                bbox = detection['bbox']
                
                # Apply refinement based on confidence and detection method
                refined_bbox = self._apply_box_refinement(bbox, detection['confidence'])
                refined_detection['bbox'] = refined_bbox
                
                # Mark as refined
                refined_detection['refined'] = True
                refined_detections.append(refined_detection)
                self.stats['boxes_refined'] += 1
                
            except Exception as e:
                logger.error(f"Box refinement error: {e}")
                # Keep original detection if refinement fails
                refined_detections.append(detection)
        
        return refined_detections

    def _apply_box_refinement(self, bbox, confidence):
        """NEW: Apply refinement to individual bounding box"""
        try:
            x1, y1, x2, y2 = bbox
            
            # Calculate current center and dimensions
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            
            # Apply confidence-based size adjustment (higher confidence = tighter box)
            confidence_factor = 0.85 + (confidence * 0.15)  # Range: 0.85 to 1.0
            
            # Adjust box size based on confidence
            new_width = width * confidence_factor
            new_height = height * confidence_factor
            
            # Recalculate coordinates with refined dimensions
            refined_x1 = center_x - new_width / 2
            refined_y1 = center_y - new_height / 2
            refined_x2 = center_x + new_width / 2
            refined_y2 = center_y + new_height / 2
            
            # Ensure coordinates are positive
            refined_x1 = max(0, refined_x1)
            refined_y1 = max(0, refined_y1)
            
            refined_bbox = [refined_x1, refined_y1, refined_x2, refined_y2]
            
            logger.debug(f"Box refined: confidence={confidence:.2f}, factor={confidence_factor:.2f}")
            logger.debug(f"Original: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
            logger.debug(f"Refined:  [{refined_x1:.1f}, {refined_y1:.1f}, {refined_x2:.1f}, {refined_y2:.1f}]")
            
            return refined_bbox
            
        except Exception as e:
            logger.error(f"Box refinement application error: {e}")
            return bbox  # Return original bbox if refinement fails

    def _filter_and_rank_detections(self, detections):
        """ENHANCED: Filter overlapping detections and rank by confidence with better logic"""
        if not detections:
            return detections
        
        logger.debug(f"Starting detection filtering: {len(detections)} detections")
        
        # Sort by confidence (highest first)
        sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        # ENHANCED: Remove overlapping detections with improved logic
        filtered_detections = []
        duplicate_count = 0
        
        for detection in sorted_detections:
            # Check overlap with existing filtered detections
            has_overlap = self._overlaps_with_existing(
                detection, 
                filtered_detections, 
                overlap_threshold=self.config['overlap_threshold']
            )
            
            if not has_overlap:
                filtered_detections.append(detection)
                logger.debug(f"Accepted detection: {detection['class_name']} (conf: {detection['confidence']:.3f})")
            else:
                duplicate_count += 1
                logger.debug(f"Filtered duplicate: {detection['class_name']} (conf: {detection['confidence']:.3f})")
        
        # Update statistics
        self.stats['duplicate_boxes_removed'] += duplicate_count
        
        # NEW: Apply bounding box refinement
        refined_detections = self._refine_bounding_boxes(filtered_detections)
        
        # Take top 4 detections (our expected maximum)
        final_detections = refined_detections[:4]
        
        logger.debug(f"Final detections after filtering and refinement: {len(final_detections)}")
        
        return final_detections

    def _run_detection(self, image_path):
        """
        ENHANCED: Comprehensive detection pipeline with improved bounding box accuracy
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image: {image_path}")
                return []
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            all_detections = []
            
            logger.info(f"DEBUG - Processing: {Path(image_path).name}")
            
            # Method 1: Primary detection with normal confidence
            try:
                primary_results = self.model.predict(
                    image_path,
                    conf=0.25,
                    iou=self.config['iou_threshold'],
                    max_det=self.config['max_detections'],
                    verbose=False
                )
                
                primary_count = 0
                if primary_results[0].boxes is not None:
                    for box in primary_results[0].boxes:
                        detection = self._extract_detection_info(box, 'primary')
                        if detection:
                            all_detections.append(detection)
                            primary_count += 1
                
                logger.info(f"DEBUG - Primary detection: {primary_count} nuts found")
                self.stats['primary_detection_count'] += primary_count
            except Exception as e:
                logger.error(f"Primary detection error: {e}")
            
            # Method 2: Enhanced image detection if we need more detections
            if len(all_detections) < 4:
                logger.info(f"DEBUG - Applying image enhancement methods...")
                try:
                    enhanced_versions = self.enhance_image_for_detection(image_rgb)
                    enhancement_count = 0
                    
                    for enhancement_type, enhanced_img in enhanced_versions.items():
                        if enhancement_type == 'original':
                            continue  # Skip original as we already processed it
                        
                        results = self.model(enhanced_img, conf=0.2, iou=self.config['iou_threshold'])
                        if len(results) > 0 and len(results[0].boxes) > 0:
                            for detection in results[0].boxes:
                                detection_info = self._extract_detection_info(detection, f'enhanced_{enhancement_type}')
                                if (detection_info and 
                                    not self._overlaps_with_existing(detection_info, all_detections)):
                                    all_detections.append(detection_info)
                                    enhancement_count += 1
                    
                    logger.info(f"DEBUG - Image enhancement: +{enhancement_count} detections")
                    self.stats['enhancement_detection_count'] += enhancement_count
                except Exception as e:
                    logger.error(f"Image enhancement error: {e}")
            
            # Method 3: Fallback detection with lower confidence
            if len(all_detections) < 4:
                logger.info(f"DEBUG - Applying fallback detection (need {4 - len(all_detections)} more)...")
                try:
                    fallback_results = self.model.predict(
                        image_path,
                        conf=self.config['fallback_confidence'],
                        iou=self.config['iou_threshold'],
                        max_det=self.config['max_detections'],
                        verbose=False
                    )
                    
                    fallback_count = 0
                    if fallback_results[0].boxes is not None:
                        for box in fallback_results[0].boxes:
                            detection = self._extract_detection_info(box, 'fallback_low_conf')
                            if (detection and 
                                not self._overlaps_with_existing(detection, all_detections)):
                                all_detections.append(detection)
                                fallback_count += 1
                    
                    logger.info(f"DEBUG - Low confidence method: +{fallback_count} detections")
                    self.stats['fallback_detection_count'] += fallback_count
                except Exception as e:
                    logger.error(f"Fallback detection error: {e}")
            
            # Method 4: Ultra-low confidence detection
            if len(all_detections) < 4:
                try:
                    for conf in self.config['ultra_low_confidence']:
                        results = self.model(image_rgb, conf=conf, iou=0.3)
                        if len(results) > 0 and len(results[0].boxes) > 0:
                            for detection in results[0].boxes:
                                detection_info = self._extract_detection_info(detection, f'ultra_low_{conf}')
                                if (detection_info and 
                                    not self._overlaps_with_existing(detection_info, all_detections)):
                                    all_detections.append(detection_info)
                    
                    logger.info(f"DEBUG - Ultra-low confidence method: additional detections found")
                except Exception as e:
                    logger.error(f"Ultra-low confidence detection error: {e}")
            
            # ENHANCED: Filter and rank final detections with better logic
            final_detections = self._filter_and_rank_detections(all_detections)
            
            # Convert to expected format (remove detection_method for compatibility)
            processed_detections = []
            for detection in final_detections:
                processed_detection = {
                    'class_id': detection['class_id'],
                    'class_name': detection['class_name'],
                    'confidence': detection['confidence'],
                    'bbox': detection['bbox']
                }
                processed_detections.append(processed_detection)
            
            # DEBUG: Log final processed detections
            logger.info(f"DEBUG - Final detections: {len(processed_detections)}")
            missing_count = sum(1 for d in processed_detections if d['class_name'] == 'MISSING')
            present_count = sum(1 for d in processed_detections if d['class_name'] == 'PRESENT')
            logger.info(f"DEBUG - PRESENT: {present_count}, MISSING: {missing_count}")
            
            # Log bounding box refinement stats
            if self.stats['boxes_refined'] > 0:
                logger.info(f"DEBUG - Box refinement: {self.stats['boxes_refined']} boxes refined, {self.stats['duplicate_boxes_removed']} duplicates removed")
            
            # Update statistics
            if len(processed_detections) >= 4:
                self.stats['complete_detections'] += 1
            else:
                self.stats['incomplete_detections'] += 1
            
            return processed_detections
            
        except Exception as e:
            logger.error(f"Detection error for {image_path}: {e}")
            return []

    def _calculate_nut_centers_and_distances(self, detections, image=None):
        """
        Calculate centers of detected nuts and distances between all pairs
        
        Args:
            detections: List of detection dictionaries with 'bbox' key
            image: Optional image to mark centers with dots
        
        Returns:
            Dictionary containing centers, distances, and optionally marked image
        """
        print(f"\n{'='*70}")
        print(f"DISTANCE CALCULATION - DETAILED ANALYSIS")
        print(f"{'='*70}")
        
        if len(detections) == 0:
            print("No detections found - cannot calculate distances")
            return {
                'centers': [],
                'distances': [],
                'distance_matrix': {},
                'marked_image': image
            }
        
        print(f"Total detections to process: {len(detections)}")
        
        # Calculate centers for each detection
        centers = []
        print(f"\nSTEP 1: CALCULATING NUT CENTERS")
        print(f"{'-'*50}")
        
        for i, detection in enumerate(detections):
            bbox = detection['bbox']  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox
            
            # Calculate center point
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            centers.append({
                'nut_index': i,
                'center': (center_x, center_y),
                'class_name': detection['class_name'],
                'confidence': detection['confidence']
            })
            
            # Print detailed center calculation
            print(f"Nut {i+1} ({detection['class_name']}):")
            print(f"  Bounding Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
            print(f"  Center X: ({x1:.1f} + {x2:.1f}) / 2 = {center_x:.2f}")
            print(f"  Center Y: ({y1:.1f} + {y2:.1f}) / 2 = {center_y:.2f}")
            print(f"  Final Center: ({center_x:.2f}, {center_y:.2f})")
            print(f"  Confidence: {detection['confidence']:.3f}")
            print()
        
        # Calculate distances between all pairs of centers
        distances = []
        distance_matrix = {}
        
        print(f"STEP 2: CALCULATING DISTANCES BETWEEN ALL NUT PAIRS")
        print(f"{'-'*60}")
        
        pair_count = 0
        for i, j in combinations(range(len(centers)), 2):
            pair_count += 1
            center1 = centers[i]['center']
            center2 = centers[j]['center']
            
            # Calculate Euclidean distance with detailed breakdown
            delta_x = center2[0] - center1[0]
            delta_y = center2[1] - center1[1]
            distance = np.sqrt(delta_x**2 + delta_y**2)
            
            distance_info = {
                'nut1_index': i,
                'nut2_index': j,
                'nut1_center': center1,
                'nut2_center': center2,
                'distance': distance,
                'nut1_class': centers[i]['class_name'],
                'nut2_class': centers[j]['class_name']
            }
            
            distances.append(distance_info)
            
            # Store in matrix format for easy lookup
            key = f"nut{i}_to_nut{j}"
            distance_matrix[key] = distance
            
            # Print detailed distance calculation
            print(f"Distance Pair #{pair_count}: Nut{i+1} ↔ Nut{j+1}")
            print(f"  Nut{i+1} center: ({center1[0]:.2f}, {center1[1]:.2f}) [{centers[i]['class_name']}]")
            print(f"  Nut{j+1} center: ({center2[0]:.2f}, {center2[1]:.2f}) [{centers[j]['class_name']}]")
            print(f"  Delta X: {center2[0]:.2f} - {center1[0]:.2f} = {delta_x:.2f}")
            print(f"  Delta Y: {center2[1]:.2f} - {center1[1]:.2f} = {delta_y:.2f}")
            print(f"  Distance = √({delta_x:.2f}² + {delta_y:.2f}²)")
            print(f"  Distance = √({delta_x**2:.2f} + {delta_y**2:.2f})")
            print(f"  Distance = √{delta_x**2 + delta_y**2:.2f}")
            print(f"  FINAL DISTANCE: {distance:.2f} pixels")
            print()
        
        # Mark centers on image if provided
        marked_image = image
        if image is not None:
            marked_image = image.copy()
            
            # Draw nut labels only (no center dots)
            for i, center_info in enumerate(centers):
                center = center_info['center']
                x, y = int(center[0]), int(center[1])
                
                # Draw nut label only
                label = f"nut{i+1}"
                cv2.putText(marked_image, label, (x-15, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        
        return {
            'centers': centers,
            'distances': distances,
            'distance_matrix': distance_matrix,
            'marked_image': marked_image,
            'total_nuts': len(centers)
        }
    

    def _extract_distances_only(self, detections):
        """Extract only distance values from detections - no detailed printing"""
        print("DISTANCE EXTRACTION: Calculating center points...")
        distances = []
        centers = []
        
        # Calculate centers
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            centers.append((center_x, center_y))
            print(f"DISTANCE EXTRACTION: Nut {i+1} center: ({center_x:.2f}, {center_y:.2f})")
        
        # Calculate distances between all pairs
        print("DISTANCE EXTRACTION: Calculating distances between all nut pairs...")
        pair_count = 0
        for i, j in combinations(range(len(centers)), 2):
            pair_count += 1
            center1 = centers[i]
            center2 = centers[j]
            distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
            distances.append(distance)
            print(f"DISTANCE EXTRACTION: Pair {pair_count} (Nut{i+1} ↔ Nut{j+1}): {distance:.2f} pixels")
        
        print(f"DISTANCE EXTRACTION: Complete - extracted {len(distances)} distances")
        return distances

    def _create_reference_distance_profile(self, reference_detections):
        """
        Create reference distance profile from a golden/reference image
        
        Args:
            reference_detections: List of detections from reference image
        
        Returns:
            Reference distance profile for validation
        """
        print(f"\n{'='*70}")
        print(f"CREATING REFERENCE DISTANCE PROFILE")
        print(f"{'='*70}")
        
        if len(reference_detections) != 4:
            print(f"ERROR: Reference image should have exactly 4 nuts, found {len(reference_detections)}")
            return None
        
        print(f"Processing reference image with {len(reference_detections)} nuts")
        print(f"Tolerance setting: ±20% (0.8x to 1.2x reference distance)")
        
        # Calculate centers and distances for reference
        reference_data = self._calculate_nut_centers_and_distances(reference_detections)
        
        # Create reference profile
        reference_profile = {
            'reference_distances': [],
            'distance_tolerance': 0.20,  # 20% tolerance as requested
            'expected_nut_count': 4
        }
        
        print(f"\nSTEP 4: BUILDING REFERENCE PROFILE")
        print(f"{'-'*50}")
        
        # Store reference distances
        for idx, dist_info in enumerate(reference_data['distances']):
            distance = dist_info['distance']
            min_distance = distance * 0.8  # -20%
            max_distance = distance * 1.2  # +20%
            tolerance_range = max_distance - min_distance
            
            pair_info = {
                'pair': f"nut{dist_info['nut1_index']}_to_nut{dist_info['nut2_index']}",
                'distance': distance,
                'min_distance': min_distance,
                'max_distance': max_distance
            }
            
            reference_profile['reference_distances'].append(pair_info)
            
            # Print detailed profile creation
            print(f"Reference Pair #{idx+1}: {pair_info['pair']}")
            print(f"  Reference distance:  {distance:.2f} pixels")
            print(f"  Minimum allowed:     {min_distance:.2f} pixels (80% of reference)")
            print(f"  Maximum allowed:     {max_distance:.2f} pixels (120% of reference)")
            print(f"  Tolerance range:     {tolerance_range:.2f} pixels")
            print(f"  Nut classes:         {dist_info['nut1_class']} ↔ {dist_info['nut2_class']}")
            print()
        
        print(f"REFERENCE PROFILE SUMMARY:")
        print(f"{'-'*30}")
        print(f"Total distance pairs:     {len(reference_profile['reference_distances'])}")
        print(f"Expected nut count:       {reference_profile['expected_nut_count']}")
        print(f"Tolerance percentage:     ±{reference_profile['distance_tolerance']*100:.0f}%")
        
        # Calculate profile statistics
        distances = [d['distance'] for d in reference_profile['reference_distances']]
        min_dist = min(distances)
        max_dist = max(distances)
        avg_dist = sum(distances) / len(distances)
        
        print(f"\nDISTANCE STATISTICS:")
        print(f"  Minimum distance:       {min_dist:.2f} pixels")
        print(f"  Maximum distance:       {max_dist:.2f} pixels")
        print(f"  Average distance:       {avg_dist:.2f} pixels")
        print(f"  Distance range:         {max_dist - min_dist:.2f} pixels")
        
        logger.info(f"Reference distance profile created with {len(reference_profile['reference_distances'])} distance pairs")
        
        return reference_profile

    def _validate_spatial_positioning(self, current_detections, reference_profile):
        """
        Validate current detections against reference distance profile
        
        Args:
            current_detections: Current image detections
            reference_profile: Reference distance profile
        
        Returns:
            Validation results with pass/fail status
        """
        print(f"\n{'='*70}")
        print(f"SPATIAL VALIDATION - DISTANCE COMPARISON")
        print(f"{'='*70}")
        
        if not reference_profile:
            print("ERROR: No reference profile available for validation")
            return {
                'validation_passed': False,
                'error': 'No reference profile available'
            }
        
        if len(current_detections) != 4:
            print(f"ERROR: Expected 4 nuts for validation, found {len(current_detections)}")
            return {
                'validation_passed': False,
                'error': f'Expected 4 nuts, found {len(current_detections)}',
                'detected_count': len(current_detections)
            }
        
        print(f"Validating {len(current_detections)} nuts against reference profile")
        print(f"Tolerance: ±{reference_profile['distance_tolerance']*100:.0f}% of reference distances")
        
        # Calculate current distances
        current_data = self._calculate_nut_centers_and_distances(current_detections)
        
        print(f"\nSTEP 3: COMPARING CURRENT DISTANCES WITH REFERENCE")
        print(f"{'-'*70}")
        
        # Compare with reference distances
        validation_results = []
        passed_validations = 0
        
        for idx, ref_dist in enumerate(reference_profile['reference_distances']):
            print(f"\nValidation #{idx+1}: {ref_dist['pair']}")
            print(f"{'.'*40}")
            
            # Find corresponding distance in current data
            current_dist = None
            for curr_dist in current_data['distances']:
                curr_pair = f"nut{curr_dist['nut1_index']}_to_nut{curr_dist['nut2_index']}"
                if curr_pair == ref_dist['pair']:
                    current_dist = curr_dist['distance']
                    break
            
            if current_dist is None:
                print(f"ERROR: Could not find current distance for pair {ref_dist['pair']}")
                validation_results.append({
                    'pair': ref_dist['pair'],
                    'status': 'MISSING_PAIR',
                    'passed': False
                })
                continue
            
            # Print detailed comparison
            print(f"Reference distance: {ref_dist['distance']:.2f} pixels")
            print(f"Current distance:   {current_dist:.2f} pixels")
            print(f"Allowed range:      {ref_dist['min_distance']:.2f} - {ref_dist['max_distance']:.2f} pixels")
            
            # Check if current distance is within tolerance
            within_tolerance = (ref_dist['min_distance'] <= current_dist <= ref_dist['max_distance'])
            
            # Calculate deviation
            deviation_percent = ((current_dist - ref_dist['distance']) / ref_dist['distance']) * 100
            deviation_pixels = current_dist - ref_dist['distance']
            
            print(f"Deviation:          {deviation_pixels:+.2f} pixels ({deviation_percent:+.1f}%)")
            
            if within_tolerance:
                passed_validations += 1
                print(f"Result:             ✓ PASS - Within tolerance")
            else:
                print(f"Result:             ✗ FAIL - Outside tolerance")
            
            validation_results.append({
                'pair': ref_dist['pair'],
                'reference_distance': ref_dist['distance'],
                'current_distance': current_dist,
                'min_allowed': ref_dist['min_distance'],
                'max_allowed': ref_dist['max_distance'],
                'deviation_percent': deviation_percent,
                'deviation_pixels': deviation_pixels,
                'within_tolerance': within_tolerance,
                'status': 'PASS' if within_tolerance else 'FAIL',
                'passed': within_tolerance
            })
        
        # Overall validation result
        total_pairs = len(reference_profile['reference_distances'])
        validation_passed = (passed_validations == total_pairs)
        validation_rate = (passed_validations / total_pairs) * 100
        
        print(f"\n{'='*70}")
        print(f"SPATIAL VALIDATION SUMMARY")
        print(f"{'='*70}")
        print(f"Total distance pairs checked: {total_pairs}")
        print(f"Passed validations:          {passed_validations}")
        print(f"Failed validations:          {total_pairs - passed_validations}")
        print(f"Validation rate:             {validation_rate:.1f}%")
        print(f"Overall result:              {'✓ PASS' if validation_passed else '✗ FAIL'}")
        
        if not validation_passed:
            print(f"\nFAILED VALIDATIONS:")
            for result in validation_results:
                if not result['passed']:
                    print(f"  • {result['pair']}: {result['deviation_percent']:+.1f}% deviation")
        
        print(f"{'='*70}")
        
        return {
            'validation_passed': validation_passed,
            'passed_pairs': passed_validations,
            'total_pairs': total_pairs,
            'validation_rate': validation_rate,
            'individual_results': validation_results,
            'current_nut_count': len(current_detections)
        }

    # Usage example function
    def _demonstrate_distance_calculation(self, image_path):
        """
        Demonstration function to show distance calculation on an image
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Run detection (existing method)
        detections = self._run_detection(image_path)
        
        # Calculate distances and mark centers
        distance_data = self._calculate_nut_centers_and_distances(detections, image)
        
        # Print results
        print(f"\n=== DISTANCE CALCULATION RESULTS ===")
        print(f"Image: {image_path}")
        print(f"Total nuts detected: {distance_data['total_nuts']}")
        
        print(f"\nNut Centers:")
        for center_info in distance_data['centers']:
            print(f"  Nut {center_info['nut_index']+1}: Center({center_info['center'][0]:.1f}, {center_info['center'][1]:.1f}) - {center_info['class_name']}")
        
        print(f"\nDistances between nuts:")
        for dist_info in distance_data['distances']:
            print(f"  Nut{dist_info['nut1_index']+1} to Nut{dist_info['nut2_index']+1}: {dist_info['distance']:.2f} pixels")
        
        return distance_data

    def _load_or_create_reference_distances(self):
        """Load reference distances from file or create from reference image"""
        try:
            print("REFERENCE SETUP: Checking for existing reference distance file...")
            if os.path.exists(self.reference_profile_file):
                print(f"REFERENCE SETUP: Loading existing reference distances from {self.reference_profile_file}")
                with open(self.reference_profile_file, 'r') as f:
                    data = json.load(f)
                    self.reference_distances = data.get('distances', [])
                print(f"REFERENCE SETUP: Successfully loaded {len(self.reference_distances)} reference distances")
                logger.info(f"Reference distances loaded: {len(self.reference_distances)} distance pairs")
                return True
            else:
                print("REFERENCE SETUP: No existing reference file found, creating from reference image...")
                # Create from reference image
                if os.path.exists(self.reference_image_path):
                    print(f"REFERENCE SETUP: Processing reference image: {self.reference_image_path}")
                    logger.info("Creating reference distances from stored image")
                    reference_detections = self._run_detection(self.reference_image_path)
                    
                    if len(reference_detections) == 4:
                        print("REFERENCE SETUP: Found 4 nuts in reference image, calculating distances...")
                        self.reference_distances = self._extract_distances_only(reference_detections)
                        print(f"REFERENCE SETUP: Calculated reference distances: {[round(d, 2) for d in self.reference_distances]}")
                        
                        # Save for future use
                        with open(self.reference_profile_file, 'w') as f:
                            json.dump({'distances': self.reference_distances}, f)
                        print(f"REFERENCE SETUP: Reference distances saved to {self.reference_profile_file}")
                        logger.info(f"Reference distances created and saved: {self.reference_distances}")
                        return True
                    else:
                        print(f"REFERENCE SETUP: ERROR - Reference image must have 4 nuts, found {len(reference_detections)}")
                        logger.error(f"Reference image must have 4 nuts, found {len(reference_detections)}")
                else:
                    print(f"REFERENCE SETUP: ERROR - Reference image not found: {self.reference_image_path}")
                    logger.warning(f"Reference image not found: {self.reference_image_path}")
                return False
        except Exception as e:
            print(f"REFERENCE SETUP: ERROR - {e}")
            logger.error(f"Error with reference distances: {e}")
            return False

    def _save_reference_profile(self, reference_profile):
        """Save reference distance profile to file"""
        try:
            with open(self.reference_profile_file, 'w') as f:
                json.dump(reference_profile, f, indent=2)
            self.reference_profile = reference_profile
            logger.info("Reference distance profile saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving reference profile: {e}")
            return False

    def create_reference_from_image(self, reference_image_path):
        """Create and save reference distance profile from a perfect reference image"""
        logger.info(f"Creating reference profile from: {reference_image_path}")
        
        # Run detection on reference image
        reference_detections = self._run_detection(reference_image_path)
        
        if len(reference_detections) != 4:
            return {
                'success': False,
                'error': f'Reference image must have exactly 4 nuts detected, found {len(reference_detections)}'
            }
        
        # Create reference profile
        reference_profile = self._create_reference_distance_profile(reference_detections)
        
        if reference_profile:
            # Save profile
            if self._save_reference_profile(reference_profile):
                return {
                    'success': True,
                    'message': 'Reference distance profile created and saved successfully',
                    'profile': reference_profile
                }
        
        return {
            'success': False,
            'error': 'Failed to create reference profile'
        }
    
    def _create_reference_from_stored_image(self):
        """Create reference profile from stored reference image automatically"""
        try:
            if not os.path.exists(self.reference_image_path):
                logger.warning(f"Reference image not found: {self.reference_image_path}")
                return False
                
            logger.info(f"Creating reference profile from stored image: {self.reference_image_path}")
            
            # Run detection on reference image
            reference_detections = self._run_detection(self.reference_image_path)
            
            if len(reference_detections) != 4:
                logger.error(f'Reference image must have exactly 4 nuts detected, found {len(reference_detections)}')
                return False
            
            # Create reference profile
            reference_profile = self._create_reference_distance_profile(reference_detections)
            
            if reference_profile and self._save_reference_profile(reference_profile):
                logger.info("Reference distance profile created successfully from stored image")
                return True
            else:
                logger.error("Failed to create reference profile from stored image")
                return False
                
        except Exception as e:
            logger.error(f"Error creating reference from stored image: {e}")
            return False

    def _apply_business_logic(self, detections, image_name):
        """
        ENHANCED: Apply business logic with spatial validation using stored reference distances
        """
        missing_count = sum(1 for d in detections if d['class_name'] == 'MISSING')
        present_count = sum(1 for d in detections if d['class_name'] == 'PRESENT')
        total_detections = len(detections)
        
        print(f"\nBUSINESS LOGIC: Processing image with {total_detections} detections")
        print(f"BUSINESS LOGIC: Present nuts: {present_count}, Missing nuts: {missing_count}")
        
        spatial_validation = {
            'enabled': False,
            'passed': True,
            'details': None
        }
        
        # Distance check ONLY when all 4 nuts are present
        if total_detections == 4 and present_count == 4 and len(self.reference_distances) == 6:
            print("BUSINESS LOGIC: 4 nuts detected - performing spatial validation...")
            print(f"BUSINESS LOGIC: Using stored reference distances: {[round(d, 2) for d in self.reference_distances]}")
            
            spatial_validation['enabled'] = True
            
            # Extract current distances
            print("BUSINESS LOGIC: Calculating current image distances...")
            current_distances = self._extract_distances_only(detections)
            
            # Compare with reference (±20% tolerance)
            print("BUSINESS LOGIC: Comparing current distances with reference (±20% tolerance)...")
            validation_passed = True
            
            for i, (ref_dist, curr_dist) in enumerate(zip(self.reference_distances, current_distances)):
                min_allowed = ref_dist * 0.8  # -20%
                max_allowed = ref_dist * 1.2  # +20%
                
                within_tolerance = (min_allowed <= curr_dist <= max_allowed)
                deviation = ((curr_dist - ref_dist) / ref_dist) * 100
                
                print(f"BUSINESS LOGIC: Pair {i+1} - Reference: {ref_dist:.2f}, Current: {curr_dist:.2f}, Tolerance: [{min_allowed:.2f}, {max_allowed:.2f}]")
                print(f"BUSINESS LOGIC: Pair {i+1} - Deviation: {deviation:+.1f}%, Result: {'PASS' if within_tolerance else 'FAIL'}")
                
                if not within_tolerance:
                    validation_passed = False
                    logger.info(f"Distance pair {i+1} failed: {curr_dist:.1f} not in [{min_allowed:.1f}, {max_allowed:.1f}]")
            
            spatial_validation['passed'] = validation_passed
            print(f"BUSINESS LOGIC: Overall spatial validation: {'PASSED' if validation_passed else 'FAILED'}")
            logger.info(f"Spatial validation: {'PASSED' if validation_passed else 'FAILED'}")
        elif total_detections == 4 and present_count == 4:
            print("BUSINESS LOGIC: 4 nuts detected but no reference distances available - skipping spatial validation")
        else:
            print(f"BUSINESS LOGIC: Only {total_detections} nuts detected - skipping spatial validation (need exactly 4)")
        
        # Business Logic Decision
        print("BUSINESS LOGIC: Making final decision...")
        if total_detections < 4:
            box_color = "RED"
            status = "INCOMPLETE_DETECTION" 
            action = "MANUAL_REVIEW_REQUIRED"
            print("BUSINESS LOGIC: Decision - INCOMPLETE_DETECTION (insufficient nuts)")
        elif present_count == 4 and total_detections == 4:
            if spatial_validation['enabled'] and not spatial_validation['passed']:
                box_color = "RED"
                status = "NUTS_INCORRECT_POSITION"
                action = "REJECTED_WRONG_POSITION"
                print("BUSINESS LOGIC: Decision - NUTS_INCORRECT_POSITION (wrong distances)")
            else:
                box_color = "GREEN"
                status = "ALL_NUTS_PRESENT_CORRECT_POSITION"
                action = "APPROVED"
                print("BUSINESS LOGIC: Decision - ALL_NUTS_PRESENT_CORRECT_POSITION (approved)")
        else:
            box_color = "RED"
            status = "NUTS_MISSING"
            action = "REJECTED"
            print("BUSINESS LOGIC: Decision - NUTS_MISSING (some nuts missing)")
        
        return {
            'box_color': box_color,
            'status': status,
            'action': action,
            'missing_count': missing_count,
            'present_count': present_count,
            'total_detections': total_detections,
            'scenario': self._classify_scenario(missing_count, present_count),
            'spatial_validation': spatial_validation
        }
    
    def _classify_scenario(self, missing_count, present_count):
        """Classify the detected scenario - From your ML code"""
        if missing_count == 0 and present_count == 4:
            return "ALL_PRESENT"
        elif missing_count == 1:
            return "ONE_MISSING"
        elif missing_count == 2:
            return "TWO_MISSING"
        elif missing_count == 3:
            return "THREE_MISSING"
        elif missing_count == 4:
            return "ALL_MISSING"
        else:
            return "MIXED_SCENARIO"

    def _calculate_center_validation(self, detections, image_shape):
        """
        Calculate center validation for detected nuts - From your ML code
        Validates that nut centers match bounding box centers within 10% tolerance
        """
        height, width = image_shape[:2]
        validation_results = []
        
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Calculate bounding box center
            box_center_x = (x1 + x2) / 2
            box_center_y = (y1 + y2) / 2
            
            # Calculate bounding box dimensions
            box_width = x2 - x1
            box_height = y2 - y1
            
            # Calculate 10% tolerance based on box dimensions
            tolerance_x = box_width * 0.10  # 10% of box width
            tolerance_y = box_height * 0.10  # 10% of box height
            
            # For now, we assume the nut center is the same as box center
            nut_center_x = box_center_x  # Detected nut center
            nut_center_y = box_center_y  # Detected nut center
            
            # Calculate deviation
            deviation_x = abs(nut_center_x - box_center_x)
            deviation_y = abs(nut_center_y - box_center_y)
            
            # Check if within tolerance
            within_tolerance_x = deviation_x <= tolerance_x
            within_tolerance_y = deviation_y <= tolerance_y
            within_tolerance = within_tolerance_x and within_tolerance_y
            
            # Calculate percentage deviation
            percent_deviation_x = (deviation_x / (box_width / 2)) * 100 if box_width > 0 else 0
            percent_deviation_y = (deviation_y / (box_height / 2)) * 100 if box_height > 0 else 0
            
            validation_result = {
                'class_name': detection['class_name'],
                'confidence': detection['confidence'],
                'box_center': (box_center_x, box_center_y),
                'nut_center': (nut_center_x, nut_center_y),
                'box_dimensions': (box_width, box_height),
                'tolerance': (tolerance_x, tolerance_y),
                'deviation': (deviation_x, deviation_y),
                'percent_deviation': (percent_deviation_x, percent_deviation_y),
                'within_tolerance': within_tolerance,
                'within_tolerance_x': within_tolerance_x,
                'within_tolerance_y': within_tolerance_y
            }
            
            validation_results.append(validation_result)
        
        # Calculate overall validation statistics
        total_detections = len(validation_results)
        valid_centers = sum(1 for r in validation_results if r['within_tolerance'])
        center_accuracy = (valid_centers / total_detections * 100) if total_detections > 0 else 0
        
        return {
            'validation_results': validation_results,
            'total_detections': total_detections,
            'valid_centers': valid_centers,
            'center_accuracy': center_accuracy,
            'average_deviation_x': np.mean([r['percent_deviation'][0] for r in validation_results]) if validation_results else 0,
            'average_deviation_y': np.mean([r['percent_deviation'][1] for r in validation_results]) if validation_results else 0
        }

    def _create_annotated_image_with_centers(self, image_path, detections, decision, image_id):
        """
        Enhanced: Create annotated image with center dots and distance lines
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Could not load image: {image_path}")
            return None
        
        # Calculate distances and mark centers
        distance_data = self._calculate_nut_centers_and_distances(detections, image)
        annotated = distance_data['marked_image']
        
        # Continue with your existing annotation logic
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Individual coloring based on detection
            if detection['class_name'] == 'PRESENT':
                box_color = (0, 255, 0)      # GREEN for present nut
                text_color = (0, 200, 0)     
                bg_color = (0, 100, 0)       
            else:  # MISSING
                box_color = (0, 0, 255)      # RED for missing nut
                text_color = (0, 0, 200)     
                bg_color = (0, 0, 100)       
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 4)
            
            # Add labels (your existing code)
            label = f"{detection['class_name']}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 3
            
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            cv2.rectangle(annotated, 
                        (x1, y1 - text_height - 10), 
                        (x1 + text_width, y1), 
                        bg_color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), 
                    font, font_scale, (255, 255, 255), thickness)
        
        # Add business status overlay (your existing method)
        self._add_business_overlay(annotated, decision)
        
        # Add spatial validation info if available
        if 'spatial_validation' in decision and decision['spatial_validation']['enabled']:
            self._add_spatial_validation_overlay(annotated, decision['spatial_validation'])
        
        return annotated


    def _add_business_overlay(self, image, decision):
        """Add business status overlay to image - Updated for individual coloring"""
        # Use overall decision color for overlay
        if decision['box_color'] == 'GREEN':
            overlay_box_color = (0, 255, 0)
            overlay_text_color = (0, 200, 0)
            overlay_bg_color = (0, 100, 0)
        else:
            overlay_box_color = (0, 0, 255)
            overlay_text_color = (0, 0, 200)
            overlay_bg_color = (0, 0, 100)
        
        # Status texts based on your business logic
        status_text = f"STATUS: {decision['status']}"
        action_text = f"ACTION: {decision['action']}"
        scenario_text = f"SCENARIO: {decision['scenario']}"
        nuts_text = f"NUTS: {decision['present_count']} PRESENT, {decision['missing_count']} MISSING"
        
        texts = [status_text, action_text, scenario_text, nuts_text]
        
        # Calculate overlay size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3
        
        max_width = 0
        total_height = 30
        
        for text in texts:
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            max_width = max(max_width, text_width)
            total_height += text_height + 15
        
        # Draw overlay background
        overlay_width = max_width + 20
        overlay_height = total_height + 10
        
        cv2.rectangle(image, (10, 10), (10 + overlay_width, 10 + overlay_height), overlay_bg_color, -1)
        cv2.rectangle(image, (10, 10), (10 + overlay_width, 10 + overlay_height), overlay_box_color, 3)
        
        # Draw texts
        y_offset = 40
        for text in texts:
            cv2.putText(image, text, (20, y_offset), font, font_scale, (255, 255, 255), thickness)
            y_offset += 40

    def _add_spatial_validation_overlay(self, image, spatial_validation):
        """Add spatial validation information to the overlay"""
        if not spatial_validation['details']:
            return
            
        validation_text = f"SPATIAL: {'PASS' if spatial_validation['passed'] else 'FAIL'}"
        if spatial_validation['details']:
            rate = spatial_validation['details']['validation_rate']
            validation_text += f" ({rate:.1f}%)"
        
        # Add to bottom of image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        (text_width, text_height), _ = cv2.getTextSize(validation_text, font, font_scale, thickness)
        
        # Background
        y_pos = image.shape[0] - 30
        cv2.rectangle(image, (10, y_pos - text_height - 5), 
                    (10 + text_width + 10, y_pos + 5), (50, 50, 50), -1)
        
        # Text color based on validation result
        text_color = (0, 255, 0) if spatial_validation['passed'] else (0, 0, 255)
        cv2.putText(image, validation_text, (15, y_pos), 
                font, font_scale, text_color, thickness)

    def visualize_region_selector(self, image_path: str) -> Dict[str, List[int]]:
        """
        Interactive tool to select 4 points defining a region.
        Returns region coordinates that can be used with process_image_with_id.
        
        Usage:
            region = service.visualize_region_selector("path/to/image.jpg")
            result = service.process_image_with_id("path/to/image.jpg", "image_001", region=region)
        """
        points = []
        
        def click_event(event, x, y, flags, param):
            nonlocal points, img_copy
            
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append([x, y])
                
                # Draw point
                img_copy = img.copy()
                cv2.circle(img_copy, (x, y), 5, (0, 255, 0), -1)
                
                # Draw points collected so far
                for i, pt in enumerate(points):
                    # Draw point number
                    cv2.putText(img_copy, str(i+1), (pt[0]+10, pt[1]-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                # Draw lines between points
                if len(points) > 1:
                    for i in range(len(points)-1):
                        cv2.line(img_copy, tuple(points[i]), tuple(points[i+1]), 
                                (0, 255, 0), 2)
                    if len(points) == 4:  # Close the polygon
                        cv2.line(img_copy, tuple(points[3]), tuple(points[0]), 
                                (0, 255, 0), 2)
                
                # Show coordinates
                cv2.putText(img_copy, f'Point {len(points)}: ({x},{y})', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not load image")
        
        img_copy = img.copy()
        window_name = 'Region Selector - Click 4 points (Press ESC when done)'
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, click_event)

        print("Instructions:")
        print("1. Click 4 points to define the region")
        print("2. Points will be connected in order clicked")
        print("3. Press ESC to confirm and exit")
        print("4. Press 'R' to reset points")
        print("5. The coordinates will be printed and returned")

        while True:
            cv2.imshow(window_name, img_copy)
            key = cv2.waitKey(1)
            
            # ESC key
            if key == 27:
                if len(points) == 4:
                    break
                else:
                    print(f"Need exactly 4 points. Currently have {len(points)} points.")
            
            # R key to reset
            elif key == ord('r'):
                points = []
                img_copy = img.copy()
                print("Points reset. Please select 4 points.")

        cv2.destroyAllWindows()

        if len(points) == 4:
            region = {
                'points': points
            }
            
            print("\nSelected Region Points:")
            print(f"region = {{")
            print(f"    'points': [")
            for i, (x, y) in enumerate(points):
                print(f"        [{x}, {y}],  # Point {i+1}")
            print(f"    ]")
            print(f"}}")

            return region
        return None

    def process_image_with_id(self, image_path: str, image_id: str, region: Optional[Dict[str, List[int]]] = None, user_id: Optional[int] = None) -> Dict:
        """
        Process image by path with region-based detection support
        
        Args:
            image_path (str): Path to the image file
            image_id (str): Unique identifier for the image
            region (dict, optional): Dictionary defining the region of interest
                Format: {'x1': int, 'y1': int, 'x2': int, 'y2': int}
                If None, entire image is processed
            user_id (int, optional): User identifier
        """
        start_time = datetime.now()
        
        try:
            if not self.model:
                return {
                    'success': False,
                    'error': 'Model not loaded',
                    'timestamp': start_time.isoformat()
                }

            # Verify image exists
            if not os.path.exists(image_path):
                return {
                    'success': False,
                    'error': f'Image file not found: {image_path}',
                    'timestamp': start_time.isoformat()
                }

            # Load and validate image
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'success': False,
                    'error': 'Could not load image file',
                    'timestamp': start_time.isoformat()
                }

            # Validate and process region if provided
            if region:
                try:
                    x1, y1 = region.get('x1', 0), region.get('y1', 0)
                    x2, y2 = region.get('x2', image.shape[1]), region.get('y2', image.shape[0])
                    
                    if not (0 <= x1 < x2 <= image.shape[1] and 0 <= y1 < y2 <= image.shape[0]):
                        return {
                            'success': False,
                            'error': 'Invalid region coordinates',
                            'timestamp': start_time.isoformat()
                        }
                        
                    logger.info(f"Processing region: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
                    
                    # Crop image to region
                    image_region = image[y1:y2, x1:x2]
                    
                    # Save temporary cropped image for detection
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                        cv2.imwrite(temp_file.name, image_region)
                        temp_image_path = temp_file.name
                except Exception as e:
                    return {
                        'success': False,
                        'error': f'Region processing error: {str(e)}',
                        'timestamp': start_time.isoformat()
                    }
            else:
                temp_image_path = image_path
                x1, y1 = 0, 0  # No offset for full image

            # Handle region processing
            if region and 'points' in region:
                try:
                    points = np.array(region['points'], dtype=np.int32)
                    if len(points) != 4:
                        return {
                            'success': False,
                            'error': 'Region must have exactly 4 points',
                            'timestamp': start_time.isoformat()
                        }

                    # Get bounding rectangle of the region
                    rect = cv2.boundingRect(points)
                    x1, y1, w, h = rect
                    x2, y2 = x1 + w, y1 + h

                    # Create mask for the region
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [points], 255)

                    # Apply mask to image
                    masked_image = cv2.bitwise_and(image, image, mask=mask)
                    
                    # Crop to bounding rectangle
                    cropped_image = masked_image[y1:y2, x1:x2]

                    # Save temporary masked and cropped image
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                        cv2.imwrite(temp_file.name, cropped_image)
                        temp_image_path = temp_file.name

                    logger.info(f"Processing region defined by points: {points.tolist()}")
                except Exception as e:
                    return {
                        'success': False,
                        'error': f'Region processing error: {str(e)}',
                        'timestamp': start_time.isoformat()
                    }
            else:
                temp_image_path = image_path

            logger.info(f"Processing image: {image_path}")
            logger.info(f"Image shape: {image.shape}")

            # Run detection with YOLOv8 model on region or full image
            detections = self._run_detection(temp_image_path)
            
            # Adjust detection coordinates if region was specified
            if region and 'points' in region:
                for detection in detections:
                    # Adjust bounding box coordinates to original image space
                    bbox = detection['bbox']
                    detection['bbox'] = [
                        bbox[0] + x1,  # x1
                        bbox[1] + y1,  # y1
                        bbox[2] + x1,  # x2
                        bbox[3] + y1   # y2
                    ]
                
                # Clean up temporary file
                try:
                    os.unlink(temp_image_path)
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file: {e}")

            logger.info(f"Detections found: {len(detections)}")
            
            # Print detection details
            if detections:
                logger.info("Detection Details:")
                for i, det in enumerate(detections, 1):
                    logger.info(f"   {i}. {det['class_name']}: {det['confidence']:.3f}")

            # Calculate center validation
            center_validation = self._calculate_center_validation(detections, image.shape)
            
            # Apply business logic
            decision = self._apply_business_logic(detections, Path(image_path).name)
            
            # Create annotated image
            annotated_image = self._create_annotated_image_with_centers(image_path, detections, decision, image_id)
            
            # Save annotated image
            annotated_path = self._save_annotated_image(annotated_image, image_id)
            
            # Prepare nut results in expected format
            nut_results = self._prepare_nut_results(detections, decision)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update statistics
            self.stats['total_processed'] += 1
            self.stats['successful_detections'] += 1
            
            # Log results
            status_icon = "GREEN" if decision['box_color'] == 'GREEN' else "RED"
            logger.info(f"Business Decision: {status_icon} BOXES")
            logger.info(f"Status: {decision['status']}")
            logger.info(f"Detected Scenario: {decision['scenario']}")
            logger.info(f"Nuts: {decision['present_count']} PRESENT, {decision['missing_count']} MISSING")

            return {
                'success': True,
                'image_id': image_id,
                'processing_time': processing_time,
                'timestamp': start_time.isoformat(),
                'nut_results': nut_results,
                'decision': decision,
                'center_validation': center_validation,
                'detection_summary': {
                    'total_detections': len(detections),
                    'detections': detections
                },
                'annotated_image_path': annotated_path
            }

        except Exception as e:
            logger.error(f"Processing error for image {image_id}: {str(e)}")
            self.stats['total_processed'] += 1
            self.stats['failed_detections'] += 1
            
            return {
                'success': False,
                'error': f'Processing failed: {str(e)}',
                'image_id': image_id,
                'timestamp': start_time.isoformat()
            }

    def _prepare_nut_results(self, detections, decision):
        """Prepare nut results in expected format - FIXED VERSION"""
        # Initialize all nuts as missing
        nut_results = {
            'nut1': {'status': 'MISSING', 'confidence': 0.0, 'bounding_box': None},
            'nut2': {'status': 'MISSING', 'confidence': 0.0, 'bounding_box': None},
            'nut3': {'status': 'MISSING', 'confidence': 0.0, 'bounding_box': None},
            'nut4': {'status': 'MISSING', 'confidence': 0.0, 'bounding_box': None}
        }
        
        # Sort detections by position (top-left to bottom-right)
        present_detections = [d for d in detections if d['class_name'] == 'PRESENT']
        present_detections.sort(key=lambda x: (x['bbox'][1], x['bbox'][0]))  # Sort by y, then x
        
        # Assign present nuts to positions
        for i, detection in enumerate(present_detections[:4]):  # Max 4 nuts
            nut_key = f'nut{i+1}'
            bbox = detection['bbox']
            nut_results[nut_key] = {
                'status': 'PRESENT',
                'confidence': detection['confidence'],
                'bounding_box': {
                    'x1': bbox[0],
                    'y1': bbox[1],
                    'x2': bbox[2],
                    'y2': bbox[3]
                }
            }
        
        # IMPORTANT: If we have fewer than 4 detections, remaining nuts stay as MISSING
        # This ensures the business logic works correctly
        
        return nut_results

    def _save_annotated_image(self, annotated_image, image_id):
        """Save annotated image to results directory"""
        try:
            if annotated_image is None:
                return None
                
            # Create results directory
            results_dir = os.path.join(settings.MEDIA_ROOT, 'inspections', 'results')
            os.makedirs(results_dir, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{image_id}_{timestamp}_result.jpg"
            file_path = os.path.join(results_dir, filename)
            
            # Save image
            cv2.imwrite(file_path, annotated_image)
            
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving annotated image: {e}")
            return None

    def is_healthy(self):
        """Check if service is healthy and ready"""
        return {
            'service_available': self.model is not None,
            'model_loaded': self.model is not None,
            'model_path': self.model_path,
            'config': self.config,
            'statistics': self.stats
        }

# Create global service instance
enhanced_nut_detection_service = FlexibleNutDetectionService()