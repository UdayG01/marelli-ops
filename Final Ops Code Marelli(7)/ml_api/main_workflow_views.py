# ml_api/main_workflow_views.py - Main Single Image Workflow

from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator
from django.views import View
from django.core.files.storage import default_storage
from django.conf import settings
import json
import os
import tempfile
import logging
from datetime import datetime
from PIL import Image
import uuid

from .models import InspectionRecord, NutResult, CameraCaptureSession, ProcessingJob
from .services import enhanced_nut_detection_service

logger = logging.getLogger(__name__)

@login_required
def main_workflow_page(request):
    """
    Main single image workflow interface - Entry point
    """
    return render(request, 'ml_api/main_workflow.html')

@method_decorator(login_required, name='dispatch')
class SingleImageProcessingView(View):
    """
    Process single image with complete auto-workflow
    Your requirements: Image ID → Upload/Camera → Auto Process → Results → Save
    """
    
    def post(self, request):
        try:
            # Step 1: Get Image ID (manual entry or QR scan)
            image_id = request.POST.get('image_id', '').strip()
            if not image_id:
                return JsonResponse({
                    'success': False,
                    'error': 'Image ID is required'
                })
            
            # Step 2: Get Image Source (upload or camera)
            image_source = request.POST.get('image_source', 'upload')  # 'upload' or 'camera'
            
            if image_source == 'camera':
                # Camera capture workflow
                return self._process_camera_image(request, image_id)
            else:
                # Upload workflow
                return self._process_uploaded_image(request, image_id)
                
        except Exception as e:
            logger.error(f"Single image processing error: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': f'Processing error: {str(e)}'
            })
    
    def _process_uploaded_image(self, request, image_id):
        """
        Process uploaded image
        """
        uploaded_file = request.FILES.get('image')
        if not uploaded_file:
            return JsonResponse({
                'success': False,
                'error': 'No image file uploaded'
            })
        
        # Validate image file
        try:
            pil_image = Image.open(uploaded_file)
            pil_image.verify()
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': f'Invalid image file: {str(e)}'
            })
        
        # Save to temporary file for processing
        temp_file = tempfile.NamedTemporaryFile(
            suffix='.jpg', 
            delete=False,
            dir=os.path.join(settings.MEDIA_ROOT, 'temp')
        )
        
        # Ensure temp directory exists
        os.makedirs(os.path.dirname(temp_file.name), exist_ok=True)
        
        # Save uploaded file
        uploaded_file.seek(0)  # Reset file pointer
        for chunk in uploaded_file.chunks():
            temp_file.write(chunk)
        temp_file.close()
        
        # Step 3: Auto Process with YOLOv8
        result = self._auto_process_image(temp_file.name, image_id, uploaded_file.name)
        
        # Step 4: Auto Save Results
        if result['success']:
            saved_inspection = self._auto_save_results(
                request.user, 
                image_id, 
                uploaded_file.name, 
                temp_file.name, 
                result
            )
            result['inspection_id'] = str(saved_inspection.id)
            result['auto_saved'] = True
        
        # Clean up temp file
        try:
            os.unlink(temp_file.name)
        except:
            pass
        
        return JsonResponse(result)
    
    def _process_camera_image(self, request, image_id):
        """
        Process camera captured image
        """
        # This integrates with your Hikrobot camera
        camera_image_path = request.POST.get('camera_image_path')
        
        if not camera_image_path or not os.path.exists(camera_image_path):
            return JsonResponse({
                'success': False,
                'error': 'Camera image not found. Please capture image first.'
            })
        
        # Step 3: Auto Process with YOLOv8
        result = self._auto_process_image(camera_image_path, image_id, f"{image_id}_camera.jpg")
        
        # Step 4: Auto Save Results
        if result['success']:
            saved_inspection = self._auto_save_results(
                request.user, 
                image_id, 
                f"{image_id}_camera.jpg", 
                camera_image_path, 
                result
            )
            result['inspection_id'] = str(saved_inspection.id)
            result['auto_saved'] = True
        
        return JsonResponse(result)
    
    def _auto_process_image(self, image_path, image_id, filename):
        """
        Step 3: Auto Process with YOLOv8 Model
        """
        logger.info(f"Auto-processing image: {image_id}")
        
        # Use your enhanced detection service
        result = enhanced_nut_detection_service.process_image_with_id(
            image_path=image_path,
            image_id=image_id,
            user_id=None
        )
        
        if not result['success']:
            return {
                'success': False,
                'error': result.get('error', 'Processing failed'),
                'image_id': image_id
            }
        
        # Return enhanced result for display
        return {
            'success': True,
            'image_id': image_id,
            'processing_time_seconds': result['processing_time_seconds'],
            'nut_results': result['nut_results'],
            'business_logic': result['business_logic'],
            'detection_summary': result['detection_summary'],
            'timestamp': datetime.now().isoformat()
        }
    
    def _auto_save_results(self, user, image_id, filename, image_path, result):
        """
        Step 4: Auto Save Results (Image ID + Image + Status of all 4 nuts)
        """
        try:
            # Create inspection record
            inspection = InspectionRecord.objects.create(
                user=user,
                image_id=image_id,
                original_filename=filename,
                processing_time_seconds=result['processing_time_seconds'],
                overall_result=result['business_logic']['production_decision'],
                confidence_score=result['detection_summary']['average_confidence'],
                quality_score=result['business_logic']['quality_score'],
                detection_results=result['detection_summary'],
                business_logic_results=result['business_logic']
            )
            
            # Save individual nut results (nut1, nut2, nut3, nut4)
            for nut_position, nut_data in result['nut_results'].items():
                NutResult.objects.create(
                    inspection=inspection,
                    nut_position=nut_position,
                    status=nut_data['status'],  # PRESENT or MISSING
                    confidence=nut_data['confidence'],
                    bounding_box=nut_data.get('bounding_box', {}),
                    detection_metadata=nut_data.get('metadata', {})
                )
            
            # Save processed image to permanent location
            try:
                permanent_path = os.path.join(
                    settings.MEDIA_ROOT,
                    'inspections',
                    'original',
                    f"{image_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                )
                
                os.makedirs(os.path.dirname(permanent_path), exist_ok=True)
                
                if image_path != permanent_path:
                    import shutil
                    shutil.copy2(image_path, permanent_path)
                
                inspection.original_image_path = os.path.relpath(permanent_path, settings.MEDIA_ROOT)
                inspection.save()
                
            except Exception as e:
                logger.error(f"Failed to save image permanently: {str(e)}")
            
            logger.info(f"Auto-saved inspection: {inspection.id} for image: {image_id}")
            return inspection
            
        except Exception as e:
            logger.error(f"Failed to auto-save results: {str(e)}")
            raise

@login_required
def get_inspection_results(request, inspection_id):
    """
    Get detailed inspection results for display
    """
    try:
        if request.user.role == 'admin':
            inspection = InspectionRecord.objects.get(id=inspection_id)
        else:
            inspection = InspectionRecord.objects.get(id=inspection_id, user=request.user)
        
        nut_results = NutResult.objects.filter(inspection=inspection).order_by('nut_position')
        
        # Format results for display
        formatted_nut_results = {}
        for nut in nut_results:
            formatted_nut_results[nut.nut_position] = {
                'status': nut.status,
                'confidence': nut.confidence,
                'bounding_box': nut.bounding_box
            }
        
        return JsonResponse({
            'success': True,
            'inspection': {
                'id': str(inspection.id),
                'image_id': inspection.image_id,
                'overall_result': inspection.overall_result,
                'quality_score': inspection.quality_score,
                'processing_time': inspection.processing_time_seconds,
                'created_at': inspection.created_at.isoformat(),
                'nut_results': formatted_nut_results,
                'business_logic': inspection.business_logic_results
            }
        })
        
    except InspectionRecord.DoesNotExist:
        return JsonResponse({
            'success': False,
            'error': 'Inspection not found'
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })

@csrf_exempt
@login_required
def qr_scanner_endpoint(request):
    """
    QR Scanner endpoint (for future RS-232 integration)
    Currently simulates QR scan data
    """
    if request.method == 'POST':
        try:
            # In future, this will receive data from RS-232 USB scanner
            scanned_data = request.POST.get('scanned_data', '')
            
            if not scanned_data:
                return JsonResponse({
                    'success': False,
                    'error': 'No QR data received'
                })
            
            # Validate QR data format
            if len(scanned_data) < 3 or len(scanned_data) > 50:
                return JsonResponse({
                    'success': False,
                    'error': 'Invalid QR code format'
                })
            
            # TODO: Add RS-232 communication here
            # For now, just return the scanned data
            return JsonResponse({
                'success': True,
                'image_id': scanned_data,
                'source': 'QR_Scanner',
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({
        'success': False,
        'error': 'Method not allowed'
    })

@login_required
def workflow_statistics(request):
    """
    Get workflow statistics for dashboard
    """
    try:
        if request.user.role == 'admin':
            inspections = InspectionRecord.objects.all()
        else:
            inspections = InspectionRecord.objects.filter(user=request.user)
        
        total_count = inspections.count()
        pass_count = inspections.filter(overall_result='PASS').count()
        fail_count = inspections.filter(overall_result='FAIL').count()
        
        success_rate = 0
        if total_count > 0:
            success_rate = (pass_count / total_count) * 100
        
        # Recent activity (last 24 hours)
        from django.utils import timezone
        yesterday = timezone.now() - timezone.timedelta(days=1)
        recent_count = inspections.filter(created_at__gte=yesterday).count()
        
        return JsonResponse({
            'success': True,
            'statistics': {
                'total_inspections': total_count,
                'pass_count': pass_count,
                'fail_count': fail_count,
                'success_rate': round(success_rate, 2),
                'recent_24h': recent_count
            }
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })