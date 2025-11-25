# ml_api/storage_service.py - Enhanced Storage with QR-Code-Based Organization

import os
import shutil
from datetime import datetime
from pathlib import Path
from django.conf import settings
from .models import InspectionRecord
import logging

logger = logging.getLogger(__name__)

class EnhancedStorageService:
    """
    Enhanced Storage Service that organizes images into QR_CODE/OK_NG/original_annotated folders
    """
    
    def __init__(self):
        self.base_media_path = Path(settings.CUSTOM_IMAGES_BASE_PATH)
        self.setup_folder_structure()
    
    def setup_folder_structure(self):
        """Create the base folder structure and temp folder"""
        # Create base inspections folder
        self.base_inspections_path = self.base_media_path
        self.base_inspections_path.mkdir(parents=True, exist_ok=True)
        
        # Create temporary processing folder (static)
        self.temp_folder =  Path(settings.MEDIA_ROOT) / 'temp'
        self.temp_folder.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created/verified base inspection folder: {self.base_inspections_path}")
        logger.info(f"Created/verified temp folder: {self.temp_folder}")
    
    def create_qr_folder_structure(self, image_id, test_status):
        # Create status folder first (OK or NG)
        status_folder = self.base_inspections_path / test_status
        status_folder.mkdir(parents=True, exist_ok=True)
        
        # Create QR code folder inside status folder
        qr_folder = status_folder / image_id
        qr_folder.mkdir(parents=True, exist_ok=True)
        
        # Create original and annotated subfolders
        original_folder = qr_folder / 'original'
        annotated_folder = qr_folder / 'annotated'
        
        original_folder.mkdir(parents=True, exist_ok=True)
        annotated_folder.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created QR folder structure: {status_folder}")
        
        return {
            'qr_folder': qr_folder,
            'status_folder': status_folder,
            'original_folder': original_folder,
            'annotated_folder': annotated_folder
        }
    
    def get_status_from_results(self, nuts_present, nuts_absent, total_expected=4):
        """Determine OK/NG status from nut detection results"""
        if nuts_present == total_expected and nuts_absent == 0:
            return 'OK'
        else:
            return 'NG'
    
    def generate_filename(self, image_id, timestamp=None, file_type='original', extension='jpg'):
        """Generate standardized filename"""
        if not timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        return f"{image_id}_{timestamp}_{file_type}.{extension}"
    
    def save_inspection_with_images(self, user, image_id, original_image_path, 
                                   annotated_image_path, nuts_present, nuts_absent, 
                                   confidence_scores=None, processing_time=0.0):
        """
        Save inspection record with QR-code-based image organization
        """
        try:
            # 🔧 FIXED: Explicitly determine test status based on CURRENT ML results
            test_status = self.get_status_from_results(nuts_present, nuts_absent)
            
            # 🔧 DEBUG: Log the decision process for retry scenarios
            logger.info(f"[STORAGE] Processing image_id: {image_id}")
            logger.info(f"[STORAGE] ML Results - Present: {nuts_present}, Missing: {nuts_absent}")
            logger.info(f"[STORAGE] Determined status: {test_status}")
            
            # 🔧 FIXED: Ensure no interference from existing records for same image_id
            existing_records = InspectionRecord.objects.filter(image_id=image_id)
            if existing_records.exists():
                logger.warning(f"[STORAGE] Found {existing_records.count()} existing records for {image_id} - this might be a retry scenario")

            # 🔧 SAFETY CHECK: Verify the status determination is correct
            if nuts_present == 4 and nuts_absent == 0:
                expected_status = 'OK'
            else:
                expected_status = 'NG'

            if test_status != expected_status:
                logger.error(f"[STORAGE] STATUS MISMATCH! Calculated: {test_status}, Expected: {expected_status}")
                logger.error(f"[STORAGE] Using expected status: {expected_status}")
                test_status = expected_status

            logger.info(f"[STORAGE] Final confirmed status: {test_status} for image_id: {image_id}")
            # Generate timestamp for consistent naming
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Generate filenames
            original_filename = self.generate_filename(image_id, timestamp, 'original')
            annotated_filename = self.generate_filename(image_id, timestamp, 'annotated')
            
            # Create QR-code specific folder structure
            folders = self.create_qr_folder_structure(image_id, test_status)
            
            # Full target paths
            target_original_path = folders['original_folder'] / original_filename
            target_annotated_path = folders['annotated_folder'] / annotated_filename
            
            # Copy images to appropriate QR-code folders
            if os.path.exists(original_image_path):
                shutil.copy2(original_image_path, target_original_path)
                logger.info(f"Copied original image to: {target_original_path}")
            else:
                logger.warning(f"⚠️ Original image not found: {original_image_path}")
                return None
            
            if os.path.exists(annotated_image_path):
                shutil.copy2(annotated_image_path, target_annotated_path)
                logger.info(f"Copied annotated image to: {target_annotated_path}")
            else:
                logger.warning(f"Annotated image not found: {annotated_image_path}")
            
            # Create database record
            inspection = InspectionRecord.objects.create(
                user=user,
                image_id=image_id,
                nuts_present=nuts_present,
                nuts_absent=nuts_absent,
                test_status=test_status,
                processing_time=processing_time
            )
            
            # Set appropriate image paths based on status (relative to MEDIA_ROOT)
            relative_original_path = target_original_path.relative_to(self.base_media_path)
            if os.path.exists(target_annotated_path):
                relative_annotated_path = target_annotated_path.relative_to(self.base_media_path)
            else:
                relative_annotated_path = None
            
            if test_status == 'OK':
                inspection.original_image_ok = str(relative_original_path)
                if relative_annotated_path:
                    inspection.annotated_image_ok = str(relative_annotated_path)
            else:  # NG
                inspection.original_image_ng = str(relative_original_path)
                if relative_annotated_path:
                    inspection.annotated_image_ng = str(relative_annotated_path)
            
            # Save confidence scores if provided
            if confidence_scores:
                inspection.set_confidence_scores_list(confidence_scores)
            
            inspection.save()
            
            logger.info(f"Saved inspection record: {inspection.id} with status {test_status}")
            
            return inspection
            
        except Exception as e:
            logger.error(f"Error saving inspection: {e}")
            return None
    
    def get_folder_statistics(self):
        """Get statistics about files in QR-code folders"""
        stats = {}
        
        # Get all QR code folders
        if self.base_inspections_path.exists():
            qr_folders = [f for f in self.base_inspections_path.iterdir() if f.is_dir() and f.name != 'temp']
            
            total_ok_files = 0
            total_ng_files = 0
            total_size = 0
            
            for qr_folder in qr_folders:
                qr_stats = {'folders': {}}
                
                # Check OK and NG folders
                for status in ['OK', 'NG']:
                    status_folder = qr_folder / status
                    if status_folder.exists():
                        # Count files in original and annotated subfolders
                        original_files = len(list((status_folder / 'original').glob('*'))) if (status_folder / 'original').exists() else 0
                        annotated_files = len(list((status_folder / 'annotated').glob('*'))) if (status_folder / 'annotated').exists() else 0
                        
                        folder_files = original_files + annotated_files
                        folder_size = sum(f.stat().st_size for f in status_folder.rglob('*') if f.is_file())
                        
                        qr_stats['folders'][status] = {
                            'original_files': original_files,
                            'annotated_files': annotated_files,
                            'total_files': folder_files,
                            'size_mb': round(folder_size / (1024 * 1024), 2)
                        }
                        
                        if status == 'OK':
                            total_ok_files += folder_files
                        else:
                            total_ng_files += folder_files
                        total_size += folder_size
                
                stats[qr_folder.name] = qr_stats
            
            # Overall summary
            stats['_summary'] = {
                'total_qr_folders': len(qr_folders),
                'total_ok_files': total_ok_files,
                'total_ng_files': total_ng_files,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'base_path': str(self.base_inspections_path)
            }
        
        return stats
    
    def cleanup_temp_files(self, older_than_hours=24):
        """Clean up temporary files older than specified hours"""
        import time
        
        current_time = time.time()
        cutoff_time = current_time - (older_than_hours * 3600)
        
        cleaned_count = 0
        for file_path in self.temp_folder.glob('*'):
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                try:
                    file_path.unlink()
                    cleaned_count += 1
                    logger.info(f"🗑️ Cleaned temp file: {file_path.name}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not clean {file_path}: {e}")
        
        logger.info(f"🧹 Cleaned {cleaned_count} temporary files")
        return cleaned_count
    
    def get_recent_inspections(self, status=None, limit=10):
        """Get recent inspections with proper QR-code-based image paths"""
        queryset = InspectionRecord.objects.all()
        
        if status:
            queryset = queryset.filter(test_status=status)
        
        inspections = queryset[:limit]
        
        results = []
        for inspection in inspections:
            result = {
                'id': str(inspection.id),
                'image_id': inspection.image_id,
                'capture_datetime': inspection.capture_datetime,
                'test_status': inspection.test_status,
                'nuts_present': inspection.nuts_present,
                'nuts_absent': inspection.nuts_absent,
                'original_image_url': f"/media/{inspection.original_image_path}" if inspection.original_image_path else None,
                'annotated_image_url': f"/media/{inspection.annotated_image_path}" if inspection.annotated_image_path else None,
                'confidence_scores': inspection.get_confidence_scores_list(),
                'processing_time': inspection.processing_time
            }
            results.append(result)
        
        return results
    
    def export_to_csv(self, start_date=None, end_date=None, filename=None):
        """Export inspection data to CSV file"""
        import csv
        from datetime import datetime
        
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"inspection_export_{timestamp}.csv"
        
        export_path = self.base_media_path / 'exports'
        export_path.mkdir(exist_ok=True)
        
        full_path = export_path / filename
        
        # Get data
        queryset = InspectionRecord.objects.all()
        if start_date:
            queryset = queryset.filter(capture_datetime__gte=start_date)
        if end_date:
            queryset = queryset.filter(capture_datetime__lte=end_date)
        
        # Write CSV
        with open(full_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'Date_Time', 'Image_ID', 'Test_Status', 'Nuts_Present', 
                'Nuts_Absent', 'Original_Image', 'Annotated_Image', 
                'Processing_Time', 'User', 'Confidence_Scores'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for inspection in queryset:
                writer.writerow({
                    'Date_Time': inspection.capture_datetime.strftime('%Y-%m-%d %H:%M:%S'),
                    'Image_ID': inspection.image_id,
                    'Test_Status': inspection.test_status,
                    'Nuts_Present': inspection.nuts_present,
                    'Nuts_Absent': inspection.nuts_absent,
                    'Original_Image': inspection.original_image_path or '',
                    'Annotated_Image': inspection.annotated_image_path or '',
                    'Processing_Time': inspection.processing_time,
                    'User': inspection.user.username,
                    'Confidence_Scores': inspection.confidence_scores or ''
                })
        
        logger.info(f"📊 Exported {queryset.count()} records to: {full_path}")
        return str(full_path)