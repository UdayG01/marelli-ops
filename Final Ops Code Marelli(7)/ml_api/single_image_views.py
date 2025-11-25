# ml_api/single_image_views.py - Single Image Workflow Views

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
def single_image_workflow(request):
    """
    Main single image workflow interface
    """
    return render(request, 'ml_api/single_image_workflow.html')

@method_decorator(login_required, name='dispatch')
class SingleImageProcessingView(View):
    """
    Process single image with auto-workflow
    """
    
    def post(self, request):
        try:
            image_id = request.POST.get('image_id', '').strip()
            auto_process = request.POST.get('auto_process', 'false') == 'true'
            image_path = request.POST.get('image_path')  # For camera captures
            
            if not image_id:
                return JsonResponse({
                    'success': False,
                    'error': 'Image ID is required'
                })
            
            # Handle image source
            if image_path:
                # Camera captured image
                if not os.path.exists(image_path):
                    return JsonResponse({
                        'success': False,
                        'error': 'Captured image not found'
                    })
                processing_image_path = image_path
                original_filename = f"{image_id}_camera_capture.jpg"
            else:
                # Uploaded image
                uploaded_file = request.FILES.get('image')
                if not uploaded_file:
                    return JsonResponse({
                        'success': False,
                        'error': 'No image provided'
                    })
                
                # Save uploaded image temporarily
                temp_file = tempfile.NamedTemporaryFile(
                    suffix='.jpg', 
                    delete=False,
                    dir=getattr(settings, 'MEDIA_ROOT', '/tmp')
                )
                
                # Save uploaded file
                for chunk in uploaded_file.chunks():
                    temp_file.write(chunk)
                temp_file.close()
                
                processing_image_path = temp_file.name
                original_filename = uploaded_file.name
            
            # Validate image
            try:
                with Image.open(processing_image_path) as img:
                    img.verify()
            except Exception as e:
                return JsonResponse({
                    'success': False,
                    'error': f'Invalid image file: {str(e)}'
                })
            
            # Process with enhanced service
            logger.info(f"Processing image: {image_id}")
            
            result = enhanced_nut_detection_service.process_image_with_id(
                image_path=processing_image_path,
                image_id=image_id,
                user_id=request.user.id
            )
            
            if not result['success']:
                return JsonResponse({
                    'success': False,
                    'error': result.get('error', 'Processing failed')
                })
            
            # Auto-save to database if requested
            auto_saved = False
            if auto_process:
                try:
                    inspection_record = self._save_inspection_result(
                        request.user,
                        image_id,
                        original_filename,
                        processing_image_path,
                        result
                    )
                    auto_saved = True
                    logger.info(f"Auto-saved inspection: {inspection_record.id}")
                except Exception as e:
                    logger.error(f"Failed to auto-save: {str(e)}")
            
            # Clean up temp files
            if not image_path and os.path.exists(processing_image_path):
                try:
                    os.unlink(processing_image_path)
                except:
                    pass
            
            # Return enhanced result
            return JsonResponse({
                'success': True,
                'image_id': image_id,
                'processing_time_seconds': result['processing_time_seconds'],
                'nut_results': result['nut_results'],
                'business_logic': result['business_logic'],
                'detection_summary': result['detection_summary'],
                'auto_saved': auto_saved,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Single image processing error: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': f'Processing error: {str(e)}'
            })
    
    def _save_inspection_result(self, user, image_id, filename, image_path, result):
        """
        Save inspection result to database
        """
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
        
        # Save individual nut results
        for nut_position, nut_data in result['nut_results'].items():
            NutResult.objects.create(
                inspection=inspection,
                nut_position=nut_position,
                status=nut_data['status'],
                confidence=nut_data['confidence'],
                bounding_box=nut_data.get('bounding_box', {}),
                detection_metadata=nut_data.get('metadata', {})
            )
        
        # Save processed image to media directory
        try:
            media_path = os.path.join(
                settings.MEDIA_ROOT,
                'inspections',
                'original',
                f"{image_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            )
            
            os.makedirs(os.path.dirname(media_path), exist_ok=True)
            
            if image_path != media_path:
                import shutil
                shutil.copy2(image_path, media_path)
            
            inspection.original_image_path = os.path.relpath(media_path, settings.MEDIA_ROOT)
            inspection.save()
            
        except Exception as e:
            logger.error(f"Failed to save image: {str(e)}")
        
        return inspection

@login_required
def get_inspection_status(request, inspection_id):
    """
    Get status of a specific inspection
    """
    try:
        if request.user.role == 'admin':
            inspection = InspectionRecord.objects.get(id=inspection_id)
        else:
            inspection = InspectionRecord.objects.get(id=inspection_id, user=request.user)
        
        nut_results = NutResult.objects.filter(inspection=inspection).order_by('nut_position')
        
        return JsonResponse({
            'success': True,
            'inspection': {
                'id': str(inspection.id),
                'image_id': inspection.image_id,
                'overall_result': inspection.overall_result,
                'quality_score': inspection.quality_score,
                'processing_time': inspection.processing_time_seconds,
                'created_at': inspection.created_at.isoformat(),
                'nut_results': [
                    {
                        'position': nut.nut_position,
                        'status': nut.status,
                        'confidence': nut.confidence
                    }
                    for nut in nut_results
                ]
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

@login_required
def recent_inspections_api(request):
    """
    Get recent inspections for current user
    """
    try:
        if request.user.role == 'admin':
            inspections = InspectionRecord.objects.all()
        else:
            inspections = InspectionRecord.objects.filter(user=request.user)
        
        inspections = inspections.order_by('-created_at')[:20]
        
        inspection_data = []
        for inspection in inspections:
            inspection_data.append({
                'id': str(inspection.id),
                'image_id': inspection.image_id,
                'overall_result': inspection.overall_result,
                'quality_score': inspection.quality_score,
                'created_at': inspection.created_at.isoformat(),
                'processing_time': inspection.processing_time_seconds
            })
        
        return JsonResponse({
            'success': True,
            'inspections': inspection_data
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })

@login_required
def inspection_statistics(request):
    """
    Get inspection statistics for dashboard
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

@csrf_exempt
@login_required
def delete_inspection(request, inspection_id):
    """
    Delete inspection (admin only)
    """
    if request.method != 'DELETE':
        return JsonResponse({'success': False, 'error': 'Method not allowed'})
    
    if request.user.role != 'admin':
        return JsonResponse({'success': False, 'error': 'Access denied'})
    
    try:
        inspection = InspectionRecord.objects.get(id=inspection_id)
        
        # Delete associated files
        if inspection.original_image_path:
            file_path = os.path.join(settings.MEDIA_ROOT, inspection.original_image_path)
            if os.path.exists(file_path):
                os.unlink(file_path)
        
        if inspection.result_image_path:
            file_path = os.path.join(settings.MEDIA_ROOT, inspection.result_image_path)
            if os.path.exists(file_path):
                os.unlink(file_path)
        
        inspection.delete()
        
        return JsonResponse({
            'success': True,
            'message': 'Inspection deleted successfully'
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

class BulkInspectionView(View):
    """
    Handle bulk inspection operations (admin only)
    """
    
    @method_decorator(login_required)
    def post(self, request):
        if request.user.role != 'admin':
            return JsonResponse({'success': False, 'error': 'Access denied'})
        
        try:
            data = json.loads(request.body)
            action = data.get('action')
            inspection_ids = data.get('inspection_ids', [])
            
            if action == 'delete':
                deleted_count = 0
                for inspection_id in inspection_ids:
                    try:
                        inspection = InspectionRecord.objects.get(id=inspection_id)
                        inspection.delete()
                        deleted_count += 1
                    except InspectionRecord.DoesNotExist:
                        continue
                
                return JsonResponse({
                    'success': True,
                    'message': f'Deleted {deleted_count} inspections'
                })
            
            elif action == 'export':
                # TODO: Implement export functionality
                return JsonResponse({
                    'success': False,
                    'error': 'Export functionality not implemented yet'
                })
            
            else:
                return JsonResponse({
                    'success': False,
                    'error': 'Invalid action'
                })
                
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })