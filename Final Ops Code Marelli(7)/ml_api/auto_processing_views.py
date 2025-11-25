# ml_api/auto_processing_views.py - NEW FILE

import json
import logging
import os
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse, HttpResponseRedirect
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from django.views import View
from django.utils import timezone
from django.contrib import messages
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
import threading
import time
import uuid

# Import your models
from .models import (
    InspectionRecord, NutResult, QRScannerLog, 
    ProcessingQueue, SystemConfiguration
)

# Import your enhanced service
from .services import enhanced_nut_detection_service

logger = logging.getLogger(__name__)

class AutoProcessingWorkflowView(View):
    """
    Main view for the auto-processing workflow
    Image ID → Auto Process → Auto Display → Auto Save → Next
    """
    
    @method_decorator(login_required)
    def get(self, request):
        """Render the main auto-processing interface"""
        # Get system configuration
        config = SystemConfiguration.get_config()
        
        # Get recent inspections
        recent_inspections = InspectionRecord.objects.filter(
            user=request.user
        ).order_by('-created_at')[:10]
        
        # Get processing statistics
        stats = enhanced_nut_detection_service.get_statistics()
        
        # Get pending queue count
        pending_count = ProcessingQueue.objects.filter(status='queued').count()
        
        context = {
            'config': config,
            'recent_inspections': recent_inspections,
            'stats': stats,
            'pending_count': pending_count,
        }
        
        return render(request, 'ml_api/auto_processing_workflow.html', context)


@require_http_methods(["POST"])
@login_required
@csrf_exempt
def start_inspection_with_id(request):
    """
    Start new inspection with Image ID
    AUTO PROCESS → AUTO DISPLAY → AUTO SAVE → NEXT
    """
    try:
        data = json.loads(request.body)
        image_id = data.get('image_id', '').strip()
        source_type = data.get('source_type', 'manual_upload')
        auto_process = data.get('auto_process', True)
        
        # Validate image ID
        if not image_id:
            return JsonResponse({
                'success': False,
                'error': 'Image ID is required'
            }, status=400)
        
        # Check if image ID already exists
        if InspectionRecord.objects.filter(image_id=image_id).exists():
            return JsonResponse({
                'success': False,
                'error': f'Image ID "{image_id}" already exists'
            }, status=400)
        
        # Validate image ID format (basic validation)
        config = SystemConfiguration.get_config()
        if len(image_id) > config.image_id_max_length:
            return JsonResponse({
                'success': False,
                'error': f'Image ID too long (max {config.image_id_max_length} characters)'
            }, status=400)
        
        # Create inspection record (pending state)
        inspection = InspectionRecord.objects.create(
            image_id=image_id,
            user=request.user,
            source_type=source_type,
            status='pending'
        )
        
        logger.info(f"✅ New inspection created: {image_id} by {request.user.username}")
        
        return JsonResponse({
            'success': True,
            'message': f'Inspection "{image_id}" created. Ready for image upload.',
            'inspection_id': str(inspection.id),
            'image_id': image_id,
            'next_step': 'upload_image',
            'auto_process': auto_process
        })
        
    except Exception as e:
        logger.error(f"Error starting inspection: {e}")
        return JsonResponse({
            'success': False,
            'error': f'Error starting inspection: {str(e)}'
        }, status=500)


@require_http_methods(["POST"])
@login_required
@csrf_exempt
def upload_and_auto_process(request):
    """
    Upload image and trigger auto-processing
    """
    try:
        inspection_id = request.POST.get('inspection_id')
        auto_process = request.POST.get('auto_process', 'true').lower() == 'true'
        
        # Get inspection record
        inspection = get_object_or_404(InspectionRecord, id=inspection_id, user=request.user)
        
        # Check if image file is provided
        if 'image_file' not in request.FILES:
            return JsonResponse({
                'success': False,
                'error': 'No image file provided'
            }, status=400)
        
        image_file = request.FILES['image_file']
        
        # Validate file
        if not image_file.content_type.startswith('image/'):
            return JsonResponse({
                'success': False,
                'error': 'Invalid file type. Please upload an image.'
            }, status=400)
        
        # Save image file
        inspection.image_file = image_file
        inspection.save()
        
        logger.info(f"📁 Image uploaded for {inspection.image_id}")
        
        # If auto-process is enabled, trigger processing
        if auto_process:
            # Add to processing queue
            queue_entry = ProcessingQueue.objects.create(
                inspection=inspection,
                priority=1  # Normal priority
            )
            
            # Start processing in background
            threading.Thread(
                target=process_inspection_background,
                args=(inspection.id,),
                daemon=True
            ).start()
            
            return JsonResponse({
                'success': True,
                'message': f'Image uploaded and processing started for "{inspection.image_id}"',
                'inspection_id': str(inspection.id),
                'image_id': inspection.image_id,
                'auto_processing': True,
                'next_step': 'wait_for_results'
            })
        else:
            return JsonResponse({
                'success': True,
                'message': f'Image uploaded for "{inspection.image_id}". Ready for manual processing.',
                'inspection_id': str(inspection.id),
                'image_id': inspection.image_id,
                'auto_processing': False,
                'next_step': 'manual_process'
            })
        
    except Exception as e:
        logger.error(f"Error in upload and auto-process: {e}")
        return JsonResponse({
            'success': False,
            'error': f'Upload error: {str(e)}'
        }, status=500)


@require_http_methods(["POST"])
@login_required
@csrf_exempt
def capture_and_auto_process(request):
    """
    Capture from camera and trigger auto-processing
    """
    try:
        data = json.loads(request.body)
        inspection_id = data.get('inspection_id')
        
        # Get inspection record
        inspection = get_object_or_404(InspectionRecord, id=inspection_id, user=request.user)
        
        # Import camera service
        from camera_integration.services.hikrobot_service import camera_service
        
        # Check if camera is connected
        if not camera_service.is_connected:
            return JsonResponse({
                'success': False,
                'error': 'Camera not connected. Please connect camera first.'
            }, status=400)
        
        # Capture image from camera
        frame = camera_service.get_frame()
        if frame is None:
            return JsonResponse({
                'success': False,
                'error': 'Failed to capture image from camera'
            }, status=500)
        
        # Save captured frame
        import cv2
        import tempfile
        
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        cv2.imwrite(temp_file.name, frame)
        temp_file.close()
        
        # Save to inspection record
        with open(temp_file.name, 'rb') as f:
            file_content = ContentFile(f.read())
            inspection.image_file.save(
                f'{inspection.image_id}_captured.jpg',
                file_content,
                save=True
            )
        
        # Clean up temp file
        os.unlink(temp_file.name)
        
        # Update source type
        inspection.source_type = 'camera_capture'
        inspection.save()
        
        logger.info(f"📷 Image captured from camera for {inspection.image_id}")
        
        # Auto-process the captured image
        queue_entry = ProcessingQueue.objects.create(
            inspection=inspection,
            priority=2  # Higher priority for camera captures
        )
        
        # Start processing in background
        threading.Thread(
            target=process_inspection_background,
            args=(inspection.id,),
            daemon=True
        ).start()
        
        return JsonResponse({
            'success': True,
            'message': f'Image captured and processing started for "{inspection.image_id}"',
            'inspection_id': str(inspection.id),
            'image_id': inspection.image_id,
            'next_step': 'wait_for_results'
        })
        
    except Exception as e:
        logger.error(f"Error in capture and auto-process: {e}")
        return JsonResponse({
            'success': False,
            'error': f'Capture error: {str(e)}'
        }, status=500)


def process_inspection_background(inspection_id):
    """
    Background processing function
    AUTO PROCESS → AUTO DISPLAY → AUTO SAVE
    """
    try:
        # Get inspection and update status
        inspection = InspectionRecord.objects.get(id=inspection_id)
        inspection.start_processing()
        
        # Update queue status
        queue_entry = ProcessingQueue.objects.get(inspection=inspection)
        queue_entry.status = 'processing'
        queue_entry.started_at = timezone.now()
        queue_entry.save()
        
        logger.info(f"🔄 Starting background processing for {inspection.image_id}")
        
        # Get image file path
        image_path = inspection.image_file.path
        
        # Process with enhanced ML service
        results = enhanced_nut_detection_service.process_image_with_id(
            image_path=image_path,
            image_id=inspection.image_id,
            user_id=inspection.user.id
        )
        
        # Complete inspection with results
        inspection.complete_processing(results)
        
        # Save individual nut results
        if results.get('success') and results.get('nut_results'):
            for nut_position, nut_data in results['nut_results'].items():
                NutResult.objects.create(
                    inspection=inspection,
                    nut_position=nut_position,
                    status=nut_data['status'],
                    confidence=nut_data['confidence'],
                    bounding_box=nut_data.get('coordinates', {}).get('bbox'),
                    center_coordinates=nut_data.get('coordinates', {}).get('center')
                )
        
        # Update queue status
        queue_entry.status = 'completed'
        queue_entry.completed_at = timezone.now()
        queue_entry.save()
        
        logger.info(f"✅ Background processing completed for {inspection.image_id}: {inspection.overall_result}")
        
    except Exception as e:
        logger.error(f"❌ Background processing error for {inspection_id}: {e}")
        
        try:
            # Mark inspection as failed
            inspection = InspectionRecord.objects.get(id=inspection_id)
            inspection.fail_processing(str(e))
            
            # Update queue status
            queue_entry = ProcessingQueue.objects.get(inspection=inspection)
            queue_entry.status = 'failed'
            queue_entry.completed_at = timezone.now()
            queue_entry.save()
            
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {cleanup_error}")


@require_http_methods(["GET"])
@login_required
def get_inspection_status(request, inspection_id):
    """
    Get real-time status of inspection processing
    """
    try:
        inspection = get_object_or_404(InspectionRecord, id=inspection_id, user=request.user)
        
        # Get queue status if exists
        queue_status = None
        try:
            queue_entry = ProcessingQueue.objects.get(inspection=inspection)
            queue_status = {
                'status': queue_entry.status,
                'queued_at': queue_entry.queued_at.isoformat(),
                'started_at': queue_entry.started_at.isoformat() if queue_entry.started_at else None,
                'completed_at': queue_entry.completed_at.isoformat() if queue_entry.completed_at else None,
            }
        except ProcessingQueue.DoesNotExist:
            pass
        
        # Get nut results if completed
        nut_results = {}
        if inspection.status == 'completed':
            for nut_result in inspection.nut_results.all():
                nut_results[nut_result.nut_position] = {
                    'status': nut_result.status,
                    'confidence': nut_result.confidence,
                    'bounding_box': nut_result.bounding_box,
                    'center_coordinates': nut_result.center_coordinates
                }
        
        return JsonResponse({
            'success': True,
            'inspection': {
                'id': str(inspection.id),
                'image_id': inspection.image_id,
                'status': inspection.status,
                'overall_result': inspection.overall_result,
                'quality_score': inspection.quality_score,
                'processing_time_seconds': inspection.processing_time_seconds,
                'created_at': inspection.created_at.isoformat(),
                'processing_started_at': inspection.processing_started_at.isoformat() if inspection.processing_started_at else None,
                'processing_completed_at': inspection.processing_completed_at.isoformat() if inspection.processing_completed_at else None,
                'has_result_image': bool(inspection.result_image_file),
                'error_message': inspection.error_message
            },
            'queue_status': queue_status,
            'nut_results': nut_results,
            'ready_for_next': inspection.status in ['completed', 'failed']
        })
        
    except Exception as e:
        logger.error(f"Error getting inspection status: {e}")
        return JsonResponse({
            'success': False,
            'error': f'Status error: {str(e)}'
        }, status=500)


@require_http_methods(["GET"])
@login_required
def view_inspection_results(request, inspection_id):
    """
    View detailed inspection results
    """
    try:
        inspection = get_object_or_404(InspectionRecord, id=inspection_id, user=request.user)
        
        # Get nut results
        nut_results = inspection.nut_results.all().order_by('nut_position')
        
        context = {
            'inspection': inspection,
            'nut_results': nut_results,
            'business_logic': inspection.business_logic_results or {},
        }
        
        return render(request, 'ml_api/inspection_results.html', context)
        
    except Exception as e:
        logger.error(f"Error viewing inspection results: {e}")
        return JsonResponse({
            'success': False,
            'error': f'Error viewing results: {str(e)}'
        }, status=500)


@require_http_methods(["POST"])
@login_required
@csrf_exempt
def confirm_and_next(request):
    """
    Confirm current inspection and move to next
    """
    try:
        data = json.loads(request.body)
        inspection_id = data.get('inspection_id')
        
        inspection = get_object_or_404(InspectionRecord, id=inspection_id, user=request.user)
        
        # Log confirmation
        logger.info(f"✅ Inspection confirmed by user: {inspection.image_id} - {inspection.overall_result}")
        
        return JsonResponse({
            'success': True,
            'message': f'Inspection "{inspection.image_id}" confirmed. Ready for next inspection.',
            'next_step': 'new_inspection',
            'last_result': inspection.overall_result
        })
        
    except Exception as e:
        logger.error(f"Error confirming inspection: {e}")
        return JsonResponse({
            'success': False,
            'error': f'Confirmation error: {str(e)}'
        }, status=500)


@require_http_methods(["GET"])
@login_required
def get_processing_statistics(request):
    """
    Get real-time processing statistics
    """
    try:
        # Get ML service statistics
        ml_stats = enhanced_nut_detection_service.get_statistics()
        
        # Get database statistics
        total_inspections = InspectionRecord.objects.filter(user=request.user).count()
        completed_inspections = InspectionRecord.objects.filter(user=request.user, status='completed').count()
        failed_inspections = InspectionRecord.objects.filter(user=request.user, status='failed').count()
        
        # Get results breakdown
        pass_count = InspectionRecord.objects.filter(user=request.user, overall_result='PASS').count()
        fail_count = InspectionRecord.objects.filter(user=request.user, overall_result='FAIL').count()
        
        # Calculate rates
        completion_rate = (completed_inspections / total_inspections * 100) if total_inspections > 0 else 0
        pass_rate = (pass_count / completed_inspections * 100) if completed_inspections > 0 else 0
        
        # Get queue status
        pending_queue = ProcessingQueue.objects.filter(status='queued').count()
        processing_queue = ProcessingQueue.objects.filter(status='processing').count()
        
        return JsonResponse({
            'success': True,
            'statistics': {
                'ml_service': ml_stats,
                'inspections': {
                    'total': total_inspections,
                    'completed': completed_inspections,
                    'failed': failed_inspections,
                    'completion_rate': completion_rate
                },
                'results': {
                    'pass_count': pass_count,
                    'fail_count': fail_count,
                    'pass_rate': pass_rate
                },
                'queue': {
                    'pending': pending_queue,
                    'processing': processing_queue
                }
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting statistics: {e}")
        return JsonResponse({
            'success': False,
            'error': f'Statistics error: {str(e)}'
        }, status=500)