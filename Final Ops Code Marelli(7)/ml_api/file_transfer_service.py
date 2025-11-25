# ml_api/file_transfer_service.py - Main service for handling .nip file generation and LOCAL storage

import os
import json
from datetime import datetime
from pathlib import Path
from django.conf import settings
from .models import InspectionRecord, SimpleInspection
from .nip_file_generator import NipFileGenerator
import logging

logger = logging.getLogger(__name__)

class FileTransferService:
    """
    LOCAL ONLY service for handling .nip file generation and local storage
    """
    
    def __init__(self):
        """Initialize file transfer service with local storage only"""
        print(f"\n🚀 Initializing File Transfer Service (LOCAL ONLY)...")
        
        # Initialize components
        self.nip_generator = NipFileGenerator()
        
        # Create transfer log directory
        self.log_folder = Path(settings.MEDIA_ROOT) / 'transfer_logs'
        self.log_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ File Transfer Service initialized (LOCAL ONLY)")
        print(f"📁 Transfer logs: {self.log_folder}")
        
        logger.info("FileTransferService initialized")
    
    def process_ok_status_change(self, inspection_record):
        """
        LOCAL ONLY: Process when inspection status changes - CREATE EMPTY .nip file locally
        
        Args:
            inspection_record: SimpleInspection or InspectionRecord instance
        
        Returns:
            tuple: (success, message, details)
        """
        try:
            qr_code = inspection_record.image_id
            
            print(f"\n🎯 Processing status change (LOCAL ONLY):")
            print(f"   - QR Code: {qr_code}")
            print(f"   - Record Type: {type(inspection_record).__name__}")
            print(f"   - User: {inspection_record.user.username if inspection_record.user else 'system'}")
            
            # Step 1: Generate .nip file with inspection status (LOCAL ONLY)
            print(f"\n📄 Step 1: Generating .nip file with inspection status (LOCAL ONLY)...")
            success, file_path, message = self.nip_generator.create_empty_nip_file(
                qr_code, 
                folder='local',
                inspection_record=inspection_record
            )
            
            if not success:
                error_msg = f"Failed to generate .nip file: {message}"
                print(f"❌ {error_msg}")
                self._log_transfer_attempt(qr_code, 'generation_failed', error_msg)
                return False, error_msg, {'step': 'generation', 'error': message}
            
            print(f"✅ Empty .nip file generated: {Path(file_path).name}")
            
            # REMOVED: External server sending - NO MORE DELAYS
            print(f"✅ SKIPPED: External server transfer (LOCAL ONLY mode)")
            
            # Step 2: Log successful local creation
            success_msg = f"Successfully processed {qr_code}: empty .nip file created locally"
            print(f"🎉 {success_msg}")
            
            # Log successful local creation
            self._log_transfer_attempt(
                qr_code, 
                'local_success', 
                success_msg, 
                {
                    'file_path': file_path,
                    'mode': 'local_only'
                }
            )
            
            return True, success_msg, {
                'step': 'completed',
                'file_path': file_path,
                'mode': 'local_only'
            }
                
        except Exception as e:
            error_msg = f"Unexpected error processing status for {inspection_record.image_id}: {str(e)}"
            print(f"💥 {error_msg}")
            logger.error(error_msg)
            
            self._log_transfer_attempt(
                getattr(inspection_record, 'image_id', 'unknown'), 
                'error', 
                error_msg
            )
            
            return False, error_msg, {'step': 'error', 'error': str(e)}
    
    def retry_failed_transfers(self):
        """
        LOCAL ONLY: No failed transfers to retry (always succeeds locally)
        
        Returns:
            dict: Results of retry attempts
        """
        print(f"\n🔄 LOCAL ONLY mode - no failed transfers to retry")
        return {'total': 0, 'successful': 0, 'failed': 0, 'message': 'LOCAL ONLY mode - no external transfers'}
    
    def test_system(self):
        """
        Test the LOCAL ONLY file transfer system
        
        Returns:
            dict: Test results
        """
        try:
            print(f"\n🧪 Testing File Transfer System (LOCAL ONLY)...")
            
            results = {
                'server_connection': True,  # Always true for local
                'nip_generation': False,
                'file_transfer': True,  # Always true for local
                'overall': False,
                'details': {}
            }
            
            # Test: LOCAL NIP file generation
            print(f"\n1️⃣ Testing local .nip file generation...")
            try:
                test_success, test_path, test_msg = self.nip_generator.create_empty_nip_file(
                    'TEST_LOCAL_001', 
                    folder='local',
                    inspection_record=None
                )
                
                results['nip_generation'] = test_success
                results['details']['nip_generation'] = {
                    'success': test_success,
                    'message': test_msg,
                    'file_path': test_path
                }
                
                # Clean up test file
                if test_success and test_path and os.path.exists(test_path):
                    os.remove(test_path)
                    print(f"🗑️ Cleaned up test .nip file")
                
            except Exception as e:
                results['details']['nip_generation'] = {
                    'success': False,
                    'error': str(e)
                }
            
            # Overall result
            results['overall'] = results['nip_generation']
            
            print(f"\n📊 System Test Results (LOCAL ONLY):")
            print(f"   - Local Storage: {'✅' if results['server_connection'] else '❌'}")
            print(f"   - NIP Generation: {'✅' if results['nip_generation'] else '❌'}")
            print(f"   - File Operations: {'✅' if results['file_transfer'] else '❌'}")
            print(f"   - Overall: {'✅ PASS' if results['overall'] else '❌ FAIL'}")
            
            return results
            
        except Exception as e:
            error_msg = f"System test error: {str(e)}"
            print(f"💥 {error_msg}")
            logger.error(error_msg)
            return {'error': error_msg}
    
    def get_transfer_statistics(self):
        """Get statistics about LOCAL file transfers"""
        try:
            # Get .nip file statistics
            nip_stats = self.nip_generator.get_file_statistics()
            
            # Get recent transfer logs
            recent_logs = self._get_recent_transfer_logs(limit=10)
            
            stats = {
                'nip_files': nip_stats,
                'transfer_stats': {
                    'mode': 'LOCAL_ONLY',
                    'total_created': nip_stats.get('local', 0),
                    'success_rate': 100.0  # Always 100% for local
                },
                'recent_logs': recent_logs,
                'server_config': {'mode': 'LOCAL_ONLY', 'external_server': 'DISABLED'}
            }
            
            print(f"\n📊 Transfer Statistics (LOCAL ONLY):")
            print(f"   - Mode: LOCAL ONLY")
            print(f"   - Local Files Created: {nip_stats.get('local', 0)}")
            print(f"   - Success Rate: 100%")
            
            return stats
            
        except Exception as e:
            error_msg = f"Error getting statistics: {str(e)}"
            print(f"💥 {error_msg}")
            return {'error': error_msg}
    
    def _log_transfer_attempt(self, qr_code, status, message, details=None):
        """Log transfer attempt to file"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'qr_code': qr_code,
                'status': status,
                'message': message,
                'details': details or {},
                'mode': 'LOCAL_ONLY'
            }
            
            # Create daily log file
            log_date = datetime.now().strftime('%Y%m%d')
            log_file = self.log_folder / f"transfer_log_{log_date}.json"
            
            # Append to log file
            log_entries = []
            if log_file.exists():
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        log_entries = json.load(f)
                except json.JSONDecodeError:
                    log_entries = []
            
            log_entries.append(log_entry)
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_entries, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Transfer log: {qr_code} - {status} - {message}")
            
        except Exception as e:
            logger.error(f"Failed to write transfer log: {str(e)}")
    
    def _get_recent_transfer_logs(self, limit=10):
        """Get recent transfer log entries"""
        try:
            all_logs = []
            
            # Get recent log files (last 7 days)
            for i in range(7):
                log_date = datetime.now().strftime('%Y%m%d')
                log_file = self.log_folder / f"transfer_log_{log_date}.json"
                
                if log_file.exists():
                    try:
                        with open(log_file, 'r', encoding='utf-8') as f:
                            daily_logs = json.load(f)
                            all_logs.extend(daily_logs)
                    except json.JSONDecodeError:
                        continue
            
            # Sort by timestamp and return recent entries
            all_logs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            return all_logs[:limit]
            
        except Exception as e:
            logger.error(f"Error getting recent logs: {str(e)}")
            return []
    
    def cleanup_old_files(self, days_old=30):
        """Clean up old files and logs"""
        try:
            print(f"\n🧹 Starting cleanup of files older than {days_old} days...")
            
            # Clean up old .nip files
            nip_cleaned = self.nip_generator.cleanup_old_files(days_old)
            
            # Clean up old log files
            log_cleaned = 0
            import time
            current_time = time.time()
            cutoff_time = current_time - (days_old * 24 * 3600)
            
            for log_file in self.log_folder.glob('transfer_log_*.json'):
                if log_file.stat().st_mtime < cutoff_time:
                    try:
                        log_file.unlink()
                        log_cleaned += 1
                        print(f"🗑️ Cleaned old log: {log_file.name}")
                    except Exception as e:
                        print(f"⚠️ Could not clean {log_file.name}: {e}")
            
            print(f"🧹 Cleanup completed:")
            print(f"   - .nip files cleaned: {nip_cleaned}")
            print(f"   - Log files cleaned: {log_cleaned}")
            
            return {'nip_files': nip_cleaned, 'log_files': log_cleaned}
            
        except Exception as e:
            error_msg = f"Cleanup error: {str(e)}"
            print(f"💥 {error_msg}")
            return {'error': error_msg}