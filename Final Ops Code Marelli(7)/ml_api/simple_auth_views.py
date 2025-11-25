# ml_api/simple_auth_views.py - CAMERA INTEGRATION ADDED

import os
import glob
import logging
import cv2
import traceback
from django.conf import settings

from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import json
import re
from datetime import datetime
import os  # For os.path.basename()
from .models import CustomUser, SimpleInspection

# QR Processing Lock System
import threading
QR_PROCESSING_LOCK = threading.Lock()
CURRENT_PROCESSING_QR = None
PROCESSING_START_TIME = None

from django.contrib.auth import get_user_model
from django.db import transaction
import json
from django.http import JsonResponse
from .file_transfer_service import FileTransferService
from .models import CustomUser, SimpleInspection, InspectionRecord  # 🆕 Add InspectionRecord



# Set up logging
logger = logging.getLogger(__name__)


# 🎯 PREDEFINED ADMIN ACCOUNTS - YOU CAN EDIT THIS SECTION
PREDEFINED_ADMINS = {
    'admin': 'admin123',           # username: password
    'supervisor': 'super123',      # username: password
    'manager': 'manager123',       # username: password
    # ADD MORE ADMIN ACCOUNTS HERE AS NEEDED
}

def simple_login_view(request):
    """
    Login view handling only login form - SIGNUP REMOVED
    """
    if request.method == 'POST':
        form_type = request.POST.get('form_type', 'login')
        
        # Only handle login form submission - SIGNUP REMOVED
        if form_type == 'login':
            username = request.POST.get('username')
            password = request.POST.get('password')
            
            # 🆕 NEW: Check if this is a predefined admin
            is_admin = False
            if username in PREDEFINED_ADMINS and password == PREDEFINED_ADMINS[username]:
                is_admin = True
                print(f"🔑 Admin login detected: {username}")
            
            # Authenticate user
            user = authenticate(request, username=username, password=password)
            
            if user is not None:
                # 🆕 NEW: Auto-promote predefined admins
                if is_admin and user.role != 'admin':
                    user.role = 'admin'
                    user.save()
                    print(f"👑 User {username} promoted to admin")
                
                login(request, user)
                
                # Redirect based on role (existing logic)
                if user.role == 'admin':
                    return redirect('ml_api:simple_admin_dashboard')
                else:
                    return redirect('ml_api:image_id_entry')
            else:
                # 🆕 NEW: Special handling for predefined admins not in database
                if is_admin:
                    try:
                        # Create admin user if doesn't exist
                        user = CustomUser.objects.create_user(
                            username=username,
                            password=password,
                            email=f"{username}@company.com",  # Default email
                            role='admin'
                        )
                        login(request, user)
                        messages.success(request, f'Admin account created and logged in: {username}')
                        return redirect('ml_api:simple_admin_dashboard')
                    except Exception as e:
                        messages.error(request, f'Error creating admin account: {str(e)}')
                else:
                    messages.error(request, 'Invalid username or password.')
        
        # SIGNUP HANDLING COMPLETELY REMOVED
    
    return render(request, 'ml_api/simple_login.html')

@login_required
def update_inspection_status_view(request, inspection_id):
    """
    Update inspection status (OK/NG) with role-based permissions
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            new_status = data.get('status')  # 'OK' or 'NG'
            
            if new_status not in ['OK', 'NG']:
                return JsonResponse({'success': False, 'error': 'Invalid status'})
            
            # Get the inspection record
            try:
                inspection = InspectionRecord.objects.get(id=inspection_id)
            except InspectionRecord.DoesNotExist:
                return JsonResponse({'success': False, 'error': 'Inspection not found'})
            
            # Permission check
            if request.user.role == 'user':
                # Users can only change status of their own current images
                if inspection.user != request.user:
                    return JsonResponse({
                        'success': False, 
                        'error': 'Permission denied: You can only modify your own inspections'
                    })
                
                # Check if this is the latest inspection (current image)
                latest_inspection = InspectionRecord.objects.filter(
                    user=request.user
                ).order_by('-capture_datetime').first()
                
                if latest_inspection and inspection.id != latest_inspection.id:
                    return JsonResponse({
                        'success': False, 
                        'error': 'Permission denied: You can only modify your current/latest inspection'
                    })
            
            elif request.user.role == 'admin':
                # Admins can change any inspection status
                pass
            else:
                return JsonResponse({'success': False, 'error': 'Insufficient permissions'})
            
            # Update the status
            old_status = inspection.test_status
            inspection.test_status = new_status
            
            # Update nuts count based on status
            if new_status == 'OK':
                inspection.nuts_present = 4
                inspection.nuts_absent = 0
            else:  # NG
                # Keep existing counts or set default
                if inspection.nuts_present + inspection.nuts_absent != 4:
                    inspection.nuts_present = 3  # Example default
                    inspection.nuts_absent = 1
            
            inspection.save()
            
            # MODIFIED: FILE TRANSFER INTEGRATION FOR BOTH OK AND NG
            transfer_result = None
            print(f"\n🎯 Status changed to {new_status} - Initiating file transfer...")
            print(f"   - QR Code: {inspection.image_id}")
            print(f"   - Changed by: {request.user.username} ({request.user.role})")
            
            try:
                # Initialize file transfer service
                transfer_service = FileTransferService()
                
                # MODIFIED: Process file transfer for BOTH statuses
                transfer_success, transfer_message, transfer_details = transfer_service.process_ok_status_change(inspection)
                
                if transfer_success:
                    print(f"✅ File transfer successful: {transfer_message}")
                    transfer_result = {
                        'success': True,
                        'message': transfer_message,
                        'details': transfer_details
                    }
                else:
                    print(f"❌ File transfer failed: {transfer_message}")
                    transfer_result = {
                        'success': False,
                        'message': transfer_message,
                        'details': transfer_details
                    }
                    
            except Exception as e:
                error_msg = f"File transfer error: {str(e)}"
                print(f"💥 {error_msg}")
                transfer_result = {
                    'success': False,
                    'message': error_msg,
                    'details': {'error': str(e)}
                }
            
            # ENHANCED RESPONSE WITH TRANSFER INFO
            response_data = {
                'success': True, 
                'message': f'Status updated from {old_status} to {new_status}',
                'new_status': new_status,
                'updated_by': request.user.username,
                'user_role': request.user.role
            }
            
            # Add transfer result if applicable
            if transfer_result:
                response_data['file_transfer'] = transfer_result
                
                # Add user-friendly message
                if transfer_result['success']:
                    response_data['message'] += f" and .nip file sent to external server"
                else:
                    response_data['message'] += f" but .nip file transfer failed"
            
            return JsonResponse(response_data)
            
        except json.JSONDecodeError:
            return JsonResponse({'success': False, 'error': 'Invalid JSON data'})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Method not allowed'})

@login_required
def inspection_list_view(request):
    """
    View inspection list with status toggle capability
    """
    if request.user.role == 'admin':
        # Admins see all inspections
        inspections = InspectionRecord.objects.all().order_by('-capture_datetime')[:50]
    else:
        # Users see only their inspections
        inspections = InspectionRecord.objects.filter(
            user=request.user
        ).order_by('-capture_datetime')[:20]
    
    context = {
        'inspections': inspections,
        'user_role': request.user.role,
    }
    
    return render(request, 'ml_api/inspection_list.html', context)


# REPLACE the existing simple_admin_dashboard function in simple_auth_views.py with this:

@login_required
def simple_admin_dashboard(request):
    """
    Simple Admin Dashboard with Failed Inspection Percentage
    """
    if request.user.role != 'admin':
        messages.error(request, 'Access denied. Admin privileges required.')
        return redirect('ml_api:simple_user_dashboard')
    
    # Get basic statistics
    total_users = CustomUser.objects.count()
    total_inspections = SimpleInspection.objects.count()
    recent_inspections = SimpleInspection.objects.order_by('-created_at')[:10]
    failed_inspections = SimpleInspection.objects.filter(overall_result='FAIL').count()
    
    # Calculate failed inspection percentage
    if total_inspections > 0:
        failed_percentage = round((failed_inspections / total_inspections) * 100, 1)
    else:
        failed_percentage = 0.0
    
    # Recent activity (last 24 hours)
    from django.utils import timezone
    yesterday = timezone.now() - timezone.timedelta(days=1)
    recent_count = SimpleInspection.objects.filter(created_at__gte=yesterday).count()
    
    context = {
        'total_users': total_users,
        'total_inspections': total_inspections,
        'failed_inspections': failed_inspections,
        'failed_percentage': failed_percentage,  # NEW: Added percentage calculation
        'recent_inspections': recent_inspections,
        'recent_count': recent_count,
    }
    
    return render(request, 'ml_api/simple_admin_dashboard.html', context)

@login_required
def simple_user_dashboard(request):
    """
    Simple User Dashboard
    """
    # Get user's inspections
    user_inspections = SimpleInspection.objects.filter(
        user=request.user
    ).order_by('-created_at')[:20]
    
    # Get user's failed inspections
    failed_inspections = SimpleInspection.objects.filter(
        user=request.user,
        overall_result='FAIL'
    ).order_by('-created_at')[:10]
    
    # 🆕 ADD: Get today's inspections count
    from django.utils import timezone
    today = timezone.now().date()
    today_inspections = SimpleInspection.objects.filter(
        user=request.user,
        created_at__date=today
    ).count()
    
    context = {
        'user_inspections': user_inspections,
        'failed_inspections': failed_inspections,
        'today_inspections': today_inspections,  # 🆕 NEW: Add this line
    }
    
    return render(request, 'ml_api/simple_user_dashboard.html', context)

@login_required
def image_id_entry_view(request):
    """
    Step 1: Image ID Entry - Manual input or QR scan
    """

    import requests

    url = "http://127.0.0.1:8000/api/ml/refresh-server/"  # change if needed

    response = requests.post(url)

    print(response)

    return render(request, 'ml_api/image_id_entry.html')

@login_required
def validate_image_id(request):
    """
    AJAX endpoint to validate Image ID format
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_id = data.get('image_id', '').strip()
            
            # Validate Image ID format
            validation_result = _validate_image_id_format(image_id)
            
            return JsonResponse(validation_result)
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({
        'success': False,
        'error': 'Method not allowed'
    })
    

@csrf_exempt
@login_required
def qr_scanner_endpoint(request):
    """
    QR Scanner endpoint for RS-232 integration - WITH PROCESSING LOCK
    """
    global CURRENT_PROCESSING_QR, PROCESSING_START_TIME
    
    if request.method == 'POST':
        try:
            # Get scanned data from RS-232
            scanned_data = request.POST.get('scanned_data', '').strip()
            
            if not scanned_data:
                return JsonResponse({
                    'success': False,
                    'error': 'No QR data received'
                })
            
            # CHECK PROCESSING LOCK - NEW FUNCTIONALITY
            with QR_PROCESSING_LOCK:
                if CURRENT_PROCESSING_QR is not None:
                    # Calculate processing time
                    processing_duration = (datetime.now() - PROCESSING_START_TIME).total_seconds()
                    
                    logger.warning(f"[QR_BLOCKED] QR scan blocked: {scanned_data}")
                    logger.warning(f"[QR_BLOCKED] Currently processing: {CURRENT_PROCESSING_QR}")
                    logger.warning(f"[QR_BLOCKED] Processing duration: {processing_duration:.1f}s")
                    
                    return JsonResponse({
                        'success': False,
                        'error': f'Processing in progress. Please wait until current QR code "{CURRENT_PROCESSING_QR}" completes processing.',
                        'blocked': True,
                        'current_processing_qr': CURRENT_PROCESSING_QR,
                        'processing_duration_seconds': round(processing_duration, 1),
                        'message': 'Scanner temporarily blocked during processing'
                    })
            
            # Validate scanned QR data
            validation_result = _validate_image_id_format(scanned_data)
            
            if validation_result['valid']:
                # 🆕 NEW: Send QR code to Node-RED immediately after validation
                try:
                    qr_success, qr_message = send_qr_to_node_red(scanned_data, 'scanner')
                    logger.info(f"[QR->NODE-RED] Scanner QR sent: {qr_success} - {qr_message}")
                except Exception as e:
                    logger.error(f"[QR->NODE-RED] Scanner QR send error: {e}")
                
                # SET PROCESSING LOCK - NEW FUNCTIONALITY
                with QR_PROCESSING_LOCK:
                    CURRENT_PROCESSING_QR = scanned_data
                    PROCESSING_START_TIME = datetime.now()
                
                logger.info(f"[QR_ACCEPTED] QR scan accepted: {scanned_data}")
                logger.info(f"[QR_LOCK] Processing lock set for: {scanned_data}")
                
                return JsonResponse({
                    'success': True,
                    'image_id': scanned_data,
                    'source': 'QR_Scanner',
                    'timestamp': datetime.now().isoformat(),
                    'message': f'Successfully scanned: {scanned_data}',
                    'processing_locked': True
                })
            else:
                return JsonResponse({
                    'success': False,
                    'error': f'Invalid QR code format: {validation_result["message"]}'
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

# Step 1: Update simple_auth_views.py - Modify image_source_selection_view to redirect directly to camera

@login_required
def image_source_selection_view(request):
    """
    Step 2: DIRECT REDIRECT TO CAMERA (Modified)
    Skip source selection - go directly to camera capture
    """
    # Get Image ID from query parameter
    image_id = request.GET.get('image_id', '')
    
    if not image_id:
        messages.error(request, 'No Image ID provided. Please start from Image ID entry.')
        return redirect('ml_api:image_id_entry')
    
    # Validate Image ID format
    validation_result = _validate_image_id_format(image_id)
    if not validation_result['valid']:
        messages.error(request, f'Invalid Image ID: {validation_result["message"]}')
        return redirect('ml_api:image_id_entry')
    
    # DIRECT REDIRECT TO CAMERA CAPTURE (NO SOURCE SELECTION)
    return redirect(f'/api/ml/camera-capture/?image_id={image_id}')

# Step 2: Simplified Camera Capture Template - Remove unnecessary elements

# 📷 NEW: CAMERA CAPTURE VIEW FOR WORKFLOW INTEGRATION
@login_required
def camera_capture_view(request):
    """
    CAMERA CAPTURE PAGE - Enhanced with session management
    """
    # Get Image ID from query parameter
    image_id = request.GET.get('image_id', '')
    
    if not image_id:
        messages.error(request, 'No Image ID provided. Please start from Image ID entry.')
        return redirect('ml_api:image_id_entry')
    
    # Validate Image ID format
    validation_result = _validate_image_id_format(image_id)
    if not validation_result['valid']:
        messages.error(request, f'Invalid Image ID: {validation_result["message"]}')
        return redirect('ml_api:image_id_entry')
    
    # Check for duplicate Image ID
    # if SimpleInspection.objects.filter(image_id=image_id).exists():
    #     messages.error(request, f'Image ID "{image_id}" already exists. Please use a different ID.')
    #     return redirect('ml_api:image_id_entry')
    
    # **NEW: Reset trigger system for new session**
    from .views import camera_manager
    camera_manager.reset_trigger_system()
    
    # **NEW: Clear old image_id from cache and set new one**
    from django.core.cache import cache
    cache.delete('current_image_id')  # Clear old
    cache.set('current_image_id', image_id, timeout=3600)  # Set new
    
    # **NEW: Store session trigger baseline**
    request.session['session_start_trigger_count'] = camera_manager.trigger_count
    request.session['current_image_id'] = image_id
    
    logger.info(f"[SESSION] New camera capture session started:")
    logger.info(f"[SESSION]   Image ID: {image_id}")
    logger.info(f"[SESSION]   Baseline trigger count: {camera_manager.trigger_count}")
    
    context = {
        'image_id': image_id,
        'session_trigger_baseline': camera_manager.trigger_count,  # Pass to frontend
    }
    
    return render(request, 'ml_api/camera_capture.html', context)

# ml_api/simple_auth_views.py - ENHANCED camera_capture_and_process function
# Replace your existing function with this enhanced version
@csrf_exempt
@login_required
def camera_capture_and_process(request):
    """
    ENHANCED CAMERA CAPTURE + ML PROCESSING - Complete workflow with OK/NG storage
    1. Capture image from camera
    2. Save to original directory
    3. Process with ML model
    4. Save results to database
    5. ENHANCED: Organize images into OK/NG folders
    6. ENHANCED: Save to InspectionRecord database
    7. Return results
    """
    
    # MOVE GLOBAL TO THE VERY TOP - BEFORE ANY OTHER CODE
    global CURRENT_PROCESSING_QR, PROCESSING_START_TIME
    
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_id = data.get('image_id', '').strip()
            
            # ============================================================================
            # QR PROCESSING LOCK CHECK - IMMEDIATE BLOCKING
            # ============================================================================
            with QR_PROCESSING_LOCK:
                # Check if ANY processing is currently active
                if CURRENT_PROCESSING_QR is not None:
                    processing_duration = (datetime.now() - PROCESSING_START_TIME).total_seconds()
                    logger.warning(f"[QR_BLOCKED] Processing blocked for: {image_id}")
                    logger.warning(f"[QR_BLOCKED] Currently processing: {CURRENT_PROCESSING_QR}")
                    logger.warning(f"[QR_BLOCKED] Processing duration: {processing_duration:.1f}s")
                    
                    return JsonResponse({
                        'success': False,
                        'error': f'Processing in progress. Please wait until current QR code "{CURRENT_PROCESSING_QR}" completes processing.',
                        'blocked': True,
                        'current_processing_qr': CURRENT_PROCESSING_QR,
                        'processing_duration_seconds': round(processing_duration, 1),
                        'message': 'Processing temporarily blocked during another operation'
                    })
                
                # Check if this SAME image_id is already being processed
                if CURRENT_PROCESSING_QR == image_id:
                    logger.warning(f"[QR_DUPLICATE] Duplicate processing attempt for: {image_id}")
                    return JsonResponse({
                        'success': False,
                        'error': f'Image ID "{image_id}" is already being processed.',
                        'duplicate': True,
                        'message': 'Duplicate processing attempt blocked'
                    })
                
                # SET PROCESSING LOCK IMMEDIATELY
                CURRENT_PROCESSING_QR = image_id
                PROCESSING_START_TIME = datetime.now()
                logger.info(f"[QR_LOCK] Processing lock set for: {image_id}")
            
            import time
            
            # ============================================================================
            # TIMING ANALYSIS - START
            # ============================================================================
            start_time = time.time()
            print(f"\n{'='*60}")
            print(f"[TIMING] Starting camera capture processing...")
            print(f"[TIMING] Start time: {time.strftime('%H:%M:%S', time.localtime(start_time))}")
            print(f"{'='*60}")
            
            # Validate Image ID
            if not image_id:
                # CLEAR LOCK ON ERROR
                with QR_PROCESSING_LOCK:
                    CURRENT_PROCESSING_QR = None
                    PROCESSING_START_TIME = None
                return JsonResponse({
                    'success': False,
                    'error': 'Image ID is required'
                })
            
            validation_result = _validate_image_id_format(image_id)
            if not validation_result['valid']:
                # CLEAR LOCK ON ERROR
                with QR_PROCESSING_LOCK:
                    CURRENT_PROCESSING_QR = None
                    PROCESSING_START_TIME = None
                return JsonResponse({
                    'success': False,
                    'error': f'Invalid Image ID: {validation_result["message"]}'
                })
            
            # Check for duplicate Image ID
            # count = SimpleInspection.objects.filter(image_id=image_id).count()
            # if count > 0:
            #     image_id = f"{image_id}_{count}"
            #     # CLEAR LOCK ON ERROR
            #     with QR_PROCESSING_LOCK:
            #         CURRENT_PROCESSING_QR = None
            #         PROCESSING_START_TIME = None
            #     return JsonResponse({
            #         'success': True,
            #         'error': f'Image ID "{image_id}" already exists. Please use a different ID.'
            #     })
            
            # Import camera manager
            from .views import camera_manager
            
            # Check if camera is connected
            if not camera_manager.is_connected:
                # CLEAR LOCK ON ERROR
                with QR_PROCESSING_LOCK:
                    CURRENT_PROCESSING_QR = None
                    PROCESSING_START_TIME = None
                return JsonResponse({
                    'success': False,
                    'error': 'Camera not connected. Please connect camera first.'
                })
            
            validation_time = time.time()
            print(f"[TIMING] Validation completed: {validation_time - start_time:.3f} seconds")
            
            # ============================================================================
            # CAMERA CAPTURE PHASE
            # ============================================================================
            print(f"[TIMING] Starting camera capture phase...")
            
            # Capture image from camera (EXISTING CODE - UNCHANGED)
            original_dir = os.path.join(settings.CUSTOM_IMAGES_BASE_PATH, 'temp')
            os.makedirs(original_dir, exist_ok=True)
            
            success, capture_result = camera_manager.capture_image(original_dir)
            
            if not success:
                # CLEAR LOCK ON ERROR
                with QR_PROCESSING_LOCK:
                    CURRENT_PROCESSING_QR = None
                    PROCESSING_START_TIME = None
                return JsonResponse({
                    'success': False,
                    'error': f'Camera capture failed: {capture_result}'
                })
            
            # Get captured image details (EXISTING CODE - UNCHANGED)
            captured_filepath = capture_result['filepath']
            captured_filename = capture_result['filename']
            
            # Rename file to include image_id (EXISTING CODE - UNCHANGED)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            new_filename = f"{image_id}_{timestamp}_camera.jpg"
            new_filepath = os.path.join(original_dir, new_filename)
            
            # Move/rename the captured file (EXISTING CODE - UNCHANGED)
            import shutil
            shutil.move(captured_filepath, new_filepath)
            
            print(f"Camera captured: {new_filepath}")
            
            capture_time = time.time()
            print(f"[TIMING] ✅ Camera capture completed: {capture_time - validation_time:.3f} seconds")
            
            # ============================================================================
            # ML PROCESSING PHASE
            # ============================================================================
            print(f"[TIMING] Starting ML processing phase...")
            
            # Process with ML model (EXISTING CODE - UNCHANGED)
            processing_result = _process_with_yolov8_model(new_filepath, image_id)
            
            if not processing_result['success']:
                # CLEAR LOCK ON ERROR
                with QR_PROCESSING_LOCK:
                    CURRENT_PROCESSING_QR = None
                    PROCESSING_START_TIME = None
                return JsonResponse({
                    'success': False,
                    'error': processing_result['error']
                })
            
            # Extract results from YOLOv8 processing (EXISTING CODE - UNCHANGED)
            detection_data = processing_result['data']
            nut_results = detection_data['nut_results']
            decision = detection_data['decision']
            
            # Use ML decision for counts (EXISTING CODE - UNCHANGED)
            present_count = decision['present_count']
            missing_count = decision['missing_count']
            
            print(f"ML Results - Present: {present_count}, Missing: {missing_count}")
            
            # Determine individual nut statuses based on ML results (EXISTING CODE - UNCHANGED)
            nut_statuses = ['MISSING', 'MISSING', 'MISSING', 'MISSING']
            
            # Assign PRESENT status based on ML detection (EXISTING CODE - UNCHANGED)
            for nut_key in ['nut1', 'nut2', 'nut3', 'nut4']:
                if nut_key in nut_results and nut_results[nut_key]['status'] == 'PRESENT':
                    nut_index = int(nut_key.replace('nut', '')) - 1
                    nut_statuses[nut_index] = 'PRESENT'
            
            # Overall result based on ML decision (EXISTING CODE - UNCHANGED)
            # 🆕 NEW: Check total nuts detected, if ≠ 4 then always FAIL
            total_nuts_detected = present_count + missing_count
            if present_count == 4 and missing_count == 0:
                overall_result = 'PASS'  # Only PASS if all 4 nuts are present
            else:
                overall_result = 'FAIL'   # FAIL if less than 4 present or any missing'

            ml_time = time.time()
            print(f"[TIMING] ✅ ML processing completed: {ml_time - capture_time:.3f} seconds")
            
            # ============================================================================
            # NODE-RED COMMUNICATION PHASE
            # ============================================================================
            print(f"[TIMING] Starting Node-RED communication...")
            
            # 🆕 NEW: Send result to Node-RED
            # 🆕 MODIFIED: Send to Node-RED only for PASS results
            # 🔧 FIXED: Do NOT send ANY results to Node-RED during processing
            # Node-RED signals will be sent only when "Next Inspection" is clicked
            
            print(f"[NODE-RED] 🔕 Skipping Node-RED signal during processing: {image_id} = {overall_result}")
            print(f"[NODE-RED] 📋 Signal will be sent when user clicks 'Next Inspection'")
            node_red_success, node_red_message = True, f"{overall_result} result - Node-RED signal deferred to Next Inspection click"                
            
            node_red_time = time.time()
            print(f"[TIMING] ✅ Node-RED communication completed: {node_red_time - ml_time:.3f} seconds")
            
            # ============================================================================
            # DATABASE SAVE PHASE (SIMPLE INSPECTION)
            # ============================================================================
            print(f"[TIMING] Starting database save (SimpleInspection)...")
            
            # Save to database (EXISTING CODE - UNCHANGED)
            inspection = SimpleInspection.objects.create(
                user=request.user,
                image_id=image_id,
                filename=new_filename,
                overall_result=overall_result,
                nut1_status=nut_statuses[0],
                nut2_status=nut_statuses[1],
                nut3_status=nut_statuses[2],
                nut4_status=nut_statuses[3],
                processing_time=detection_data.get('processing_time', 0.0)
            )
            
            print(f"Saved to database: {inspection.id}")
            
            # Get annotated image path (EXISTING CODE - UNCHANGED)
            annotated_image_path = detection_data.get('annotated_image_path', '')
            
            if annotated_image_path and os.path.exists(annotated_image_path):
                annotated_filename = os.path.basename(annotated_image_path)
            else:
                # Check for existing result files
                results_dir = os.path.join(settings.MEDIA_ROOT, 'inspections', 'results')
                pattern = f"{image_id}_*_result.jpg"
                matching_files = glob.glob(os.path.join(results_dir, pattern))
                
                if matching_files:
                    latest_file = max(matching_files, key=os.path.getctime)
                    annotated_filename = os.path.basename(latest_file)
                    annotated_image_path = latest_file
                else:
                    timestamp_new = datetime.now().strftime('%Y%m%d_%H%M%S')
                    annotated_filename = f"{image_id}_{timestamp_new}_result.jpg"
                    annotated_image_path = os.path.join(results_dir, annotated_filename)
            
            db_save_time = time.time()
            print(f"[TIMING] ✅ Database save completed: {db_save_time - node_red_time:.3f} seconds")
            
            # ============================================================================
            # 🆕 ENHANCED FUNCTIONALITY - OK/NG STORAGE (NEW CODE ADDED)
            # ============================================================================
            print(f"[TIMING] Starting enhanced storage phase...")
            
            try:
                # Import enhanced storage service
                from .storage_service import EnhancedStorageService
                
                # Initialize storage service
                storage_service = EnhancedStorageService()
                
                # Extract confidence scores for enhanced storage
                confidence_scores = []
                for nut_key in ['nut1', 'nut2', 'nut3', 'nut4']:
                    if nut_key in nut_results:
                        confidence_scores.append(nut_results[nut_key].get('confidence', 0.0))
                
                storage_init_time = time.time()
                print(f"[TIMING] Enhanced storage initialized: {storage_init_time - db_save_time:.3f} seconds")
                
                # Save to enhanced database with OK/NG organization
                enhanced_inspection = storage_service.save_inspection_with_images(
                    user=request.user,
                    image_id=image_id,
                    original_image_path=new_filepath,
                    annotated_image_path=annotated_image_path,
                    nuts_present=present_count,
                    nuts_absent=missing_count,
                    confidence_scores=confidence_scores,
                    processing_time=detection_data.get('processing_time', 0.0)
                )

# ADD this right after the enhanced_inspection = storage_service.save_inspection_with_images(...) call:

                # **NEW: CLEANUP TEMP FILE AFTER ENHANCED STORAGE COPIES IT**
                if enhanced_inspection:
                    try:
                        if os.path.exists(new_filepath):
                            os.remove(new_filepath)
                            print(f"🗑️ Temp file cleaned up after enhanced storage: {os.path.basename(new_filepath)}")
                    except Exception as e:
                        print(f"⚠️ Could not clean up temp file: {e}")
                
                # Rest of your existing code continues...

                storage_time = time.time()
                print(f"[TIMING] ✅ Enhanced storage save completed: {storage_time - storage_init_time:.3f} seconds")

                # ============================================================================
                # FILE TRANSFER PHASE (.nip files)
                # ============================================================================
                print(f"[TIMING] Starting file transfer phase...")

                # MODIFIED: Create .nip file for BOTH OK and NG statuses
                if enhanced_inspection:
                    print(f"\n🎯 Status is {enhanced_inspection.test_status} - Initiating file transfer...")
                    print(f"   - QR Code: {enhanced_inspection.image_id}")
                    print(f"   - Inspection ID: {enhanced_inspection.id}")
                    
                    try:
                        from .file_transfer_service import FileTransferService
                        transfer_service = FileTransferService()
                        
                        file_transfer_start = time.time()
                        
                        # MODIFIED: Process file transfer for both OK and NG
                        success, message, details = transfer_service.process_ok_status_change(enhanced_inspection)
                        
                        file_transfer_time = time.time()
                        print(f"[TIMING] ✅ File transfer service completed: {file_transfer_time - file_transfer_start:.3f} seconds")
                        
                        if success:
                            print(f"✅ File transfer successful: {message}")
                        else:
                            print(f"❌ File transfer failed: {message}")
                            
                    except Exception as e:
                        file_transfer_time = time.time()
                        print(f"[TIMING] ❌ File transfer error: {file_transfer_time - storage_time:.3f} seconds")
                        print(f"💥 File transfer error: {str(e)}")
                else:
                    file_transfer_time = storage_time
                    print(f"[TIMING] ⚠️ Enhanced inspection not created - skipping file transfer")
                
                if enhanced_inspection:
                    print(f"🎯 Enhanced storage: {enhanced_inspection.test_status} folder - {enhanced_inspection.id}")
                    enhanced_storage_success = True
                    enhanced_folder = enhanced_inspection.test_status
                else:
                    print("Enhanced storage failed, continuing with existing workflow")
                    enhanced_storage_success = False
                    enhanced_folder = 'OK' if missing_count == 0 else 'NG'
                    
            except ImportError:
                storage_time = db_save_time
                file_transfer_time = db_save_time
                print("Enhanced storage service not available, continuing with existing workflow")
                enhanced_storage_success = False
                enhanced_folder = 'OK' if missing_count == 0 else 'NG'
            except Exception as e:
                storage_time = db_save_time
                file_transfer_time = db_save_time
                print(f"Enhanced storage error: {e}, continuing with existing workflow")
                enhanced_storage_success = False
                enhanced_folder = 'OK' if missing_count == 0 else 'NG'
            
            # ============================================================================
            # RESPONSE PREPARATION PHASE
            # ============================================================================
            print(f"[TIMING] Preparing response...")
            
            # Return complete results (EXISTING CODE + ENHANCED DATA)
            response_data = {
                'success': True,
                'image_id': image_id,
                'overall_result': overall_result,
                'nut_results': {
                    'nut1': {'status': nut_statuses[0], 'confidence': nut_results.get('nut1', {}).get('confidence', 0.0)},
                    'nut2': {'status': nut_statuses[1], 'confidence': nut_results.get('nut2', {}).get('confidence', 0.0)},
                    'nut3': {'status': nut_statuses[2], 'confidence': nut_results.get('nut3', {}).get('confidence', 0.0)},
                    'nut4': {'status': nut_statuses[3], 'confidence': nut_results.get('nut4', {}).get('confidence', 0.0)},
                },
                'summary': {
                    'present_count': present_count,
                    'missing_count': missing_count,
                    'quality_score': (present_count / 4) * 100
                },
                'processing_info': {
                    'processing_time': detection_data.get('processing_time', 0.0),
                    'method': 'camera',
                    'filename': new_filename,
                    'total_detections': detection_data.get('total_detections', 0),
                    'confidence_threshold': detection_data.get('confidence_threshold', 0.5)
                },
                'detection_details': {
                    'detections': detection_data.get('detections', []),
                    'center_validation': detection_data.get('center_validation', {}),
                    'business_decision': decision
                },
                'image_paths': {
                    'original': f"/media/inspections/original/{new_filename}",
                    'annotated': f"/media/inspections/results/{annotated_filename}",
                },
                'inspection_id': str(inspection.id),
                'timestamp': datetime.now().isoformat(),
                
                # 🆕 ENHANCED DATA (NEW FIELDS ADDED)
                'enhanced_storage': {
                    'enabled': enhanced_storage_success,
                    'folder': enhanced_folder,
                    'test_status': 'OK' if missing_count == 0 else 'NG',
                    'nuts_present': present_count,
                    'nuts_absent': missing_count
                },
                # 🆕 NEW: Node-RED integration status
                'node_red': {
                    'sent': node_red_success,
                    'message': node_red_message
                }
            }
            
            # ============================================================================
            # CLEAR PROCESSING LOCK - NEW FUNCTIONALITY (CORRECTLY PLACED)
            # ============================================================================
            with QR_PROCESSING_LOCK:
                if CURRENT_PROCESSING_QR == image_id:
                    processing_duration = (datetime.now() - PROCESSING_START_TIME).total_seconds()
                    logger.info(f"[QR_UNLOCK] Processing completed for: {image_id}")
                    logger.info(f"[QR_UNLOCK] Total processing time: {processing_duration:.1f}s")
                    CURRENT_PROCESSING_QR = None
                    PROCESSING_START_TIME = None
            
            # ADD TO RESPONSE DATA
            response_data['qr_processing'] = {
                'lock_cleared': True,
                'ready_for_next_scan': True
            }
            
            # ============================================================================
            # TIMING ANALYSIS - FINAL SUMMARY
            # ============================================================================
            end_time = time.time()
            total_time = end_time - start_time
            
            print(f"\n{'='*60}")
            print(f"[TIMING] ⏱️  PROCESSING COMPLETE - DETAILED BREAKDOWN:")
            print(f"{'='*60}")
            print(f"[TIMING] 1. Validation:           {validation_time - start_time:.3f}s")
            print(f"[TIMING] 2. Camera Capture:       {capture_time - validation_time:.3f}s")
            print(f"[TIMING] 3. ML Processing:        {ml_time - capture_time:.3f}s")
            print(f"[TIMING] 4. Node-RED Comm:        {node_red_time - ml_time:.3f}s")
            print(f"[TIMING] 5. Database Save:        {db_save_time - node_red_time:.3f}s")
            print(f"[TIMING] 6. Enhanced Storage:     {storage_time - db_save_time:.3f}s")
            print(f"[TIMING] 7. File Transfer (.nip): {file_transfer_time - storage_time:.3f}s")
            print(f"[TIMING] 8. Response Prep:        {end_time - file_transfer_time:.3f}s")
            print(f"{'='*60}")
            print(f"[TIMING] 🎯 TOTAL PROCESSING TIME: {total_time:.3f} seconds")
            print(f"[TIMING] 📊 Image ID: {image_id}")
            print(f"[TIMING] 🏆 Result: {overall_result}")
            print(f"{'='*60}")
            
            return JsonResponse(response_data)
            
        except Exception as e:
            # ============================================================================
            # CLEAR PROCESSING LOCK ON ERROR - NEW FUNCTIONALITY
            # ============================================================================
            with QR_PROCESSING_LOCK:
                if CURRENT_PROCESSING_QR:
                    logger.info(f"[QR_UNLOCK] Clearing lock due to error: {CURRENT_PROCESSING_QR}")
                    CURRENT_PROCESSING_QR = None
                    PROCESSING_START_TIME = None
            
            import traceback
            traceback.print_exc()
            return JsonResponse({
                'success': False,
                'error': f'Camera processing error: {str(e)}'
            })
    
    return JsonResponse({
        'success': False,
        'error': 'Method not allowed'
    })
    
    
@login_required
def simple_workflow_page(request):
    """
    Simple single image workflow page - redirects to Image ID entry
    """
    return redirect('ml_api:image_id_entry')

@login_required
def simple_process_image(request):
    """
    Step 3 & 4: Image processing and results (UPLOAD METHOD)
    Enhanced with actual YOLOv8 model integration - FIXED VERSION
    """
    if request.method == 'POST':
        try:
            image_id = request.POST.get('image_id', '').strip()
            uploaded_file = request.FILES.get('image')
            image_source = request.POST.get('image_source', 'upload')  # 'upload' or 'camera'
            
            # Validate Image ID
            if not image_id:
                return JsonResponse({
                    'success': False,
                    'error': 'Image ID is required'
                })
            
            validation_result = _validate_image_id_format(image_id)
            if not validation_result['valid']:
                return JsonResponse({
                    'success': False,
                    'error': f'Invalid Image ID: {validation_result["message"]}'
                })
            
            # Check for duplicate Image ID
            # count = SimpleInspection.objects.filter(image_id=image_id).count()
            # if count > 0:
            #     image_id = f"{image_id}_{count}"
            #     return JsonResponse({
            #         'success': True,
            #         'error': f'Image ID "{image_id}" already exists. Changed to new ID.'
            #     })
            
            # Handle image file and save to processing directory
            if image_source == 'upload':
                if not uploaded_file:
                    return JsonResponse({
                        'success': False,
                        'error': 'Image file is required for upload method'
                    })
                
                # Save uploaded image to processing directory
                image_path, filename = _save_uploaded_image(uploaded_file, image_id)
                
            else:
                # Camera capture - filename would be generated
                filename = f"{image_id}_camera_capture.jpg"
                # For camera, you would get the image path from camera service
                image_path = None  # TODO: Get from camera service
                
                return JsonResponse({
                    'success': False,
                    'error': 'Camera capture not implemented yet. Please use upload method.'
                })
            
            # Process image with actual YOLOv8 model
            processing_result = _process_with_yolov8_model(image_path, image_id)
            
            if not processing_result['success']:
                return JsonResponse({
                    'success': False,
                    'error': processing_result['error']
                })
            
            # Extract results from YOLOv8 processing
            detection_data = processing_result['data']
            nut_results = detection_data['nut_results']
            decision = detection_data['decision']
            
            # FIXED: Use the correct counts from your ML business logic
            present_count = decision['present_count']
            missing_count = decision['missing_count']
            
            print(f"DEBUG: ML detected - Present: {present_count}, Missing: {missing_count}")
            print(f"DEBUG: Nut results from ML: {nut_results}")
            
            # FIXED: Determine individual nut statuses based on ACTUAL ML results
            nut_statuses = ['MISSING', 'MISSING', 'MISSING', 'MISSING']  # Initialize all as missing
            
            # Assign PRESENT status to the correct positions based on ML detection
            present_count_assigned = 0
            for nut_key in ['nut1', 'nut2', 'nut3', 'nut4']:
                if nut_key in nut_results and nut_results[nut_key]['status'] == 'PRESENT':
                    nut_index = int(nut_key.replace('nut', '')) - 1  # Convert nut1->0, nut2->1, etc.
                    nut_statuses[nut_index] = 'PRESENT'
                    present_count_assigned += 1
            
            # Verify our assignment matches ML results
            actual_present = nut_statuses.count('PRESENT')
            actual_missing = nut_statuses.count('MISSING')
            
            print(f"DEBUG: Final nut statuses: {nut_statuses}")
            print(f"DEBUG: Assigned - Present: {actual_present}, Missing: {actual_missing}")
            
            # Use ML decision for overall result (this is correct)
            # 🔧 FIXED: Overall result - PASS only if exactly 4 nuts present
            if present_count == 4 and missing_count == 0:
                overall_result = 'PASS'  # Only PASS if all 4 nuts are present
            else:
                overall_result = 'FAIL'   # FAIL if less than 4 present or any missin
            
            # 🆕 NEW: Send result to Node-RED
            try:
                from .views import send_to_node_red
                node_red_success, node_red_message = send_to_node_red(
                    image_id=image_id,
                    overall_result=overall_result,
                    nut_results=nut_results
                )
                
                if node_red_success:
                    print(f"[NODE-RED] ✅ Sent manual processing result to Node-RED: {image_id} = {overall_result}")
                else:
                    print(f"[NODE-RED] ❌ Failed to send to Node-RED: {node_red_message}")
            except Exception as e:
                print(f"[NODE-RED] Error sending to Node-RED: {e}")
                node_red_success, node_red_message = False, str(e)
            
            # Save to database with CORRECT nut statuses
            inspection = SimpleInspection.objects.create(
                user=request.user,
                image_id=image_id,
                filename=filename,
                overall_result=overall_result,
                nut1_status=nut_statuses[0],  # These should now be correct
                nut2_status=nut_statuses[1],
                nut3_status=nut_statuses[2],
                nut4_status=nut_statuses[3],
                processing_time=detection_data.get('processing_time', 0.0)
            )
            
            print(f"DEBUG: Saved to database - Nut1: {nut_statuses[0]}, Nut2: {nut_statuses[1]}, Nut3: {nut_statuses[2]}, Nut4: {nut_statuses[3]}")
            
            # FIXED: Get the actual annotated image path from processing (MOVED OUTSIDE return statement)
            # FIXED: Get the actual annotated image path from processing
            annotated_image_path = detection_data.get('annotated_image_path', '')
            print(f"DEBUG: Annotated image path from services: {annotated_image_path}")

            if annotated_image_path and os.path.exists(annotated_image_path):
                # Extract just the filename from the full path
                annotated_filename = os.path.basename(annotated_image_path)
                print(f"DEBUG: Using actual annotated filename: {annotated_filename}")
            else:
                # Check what files actually exist in results directory
                import glob
                results_dir = os.path.join(settings.MEDIA_ROOT, 'inspections', 'results')
                pattern = f"{image_id}_*_result.jpg"
                matching_files = glob.glob(os.path.join(results_dir, pattern))
                
                if matching_files:
                    # Use the most recent matching file
                    latest_file = max(matching_files, key=os.path.getctime)
                    annotated_filename = os.path.basename(latest_file)
                    print(f"DEBUG: Found existing result file: {annotated_filename}")
                else:
                    # Fallback filename pattern
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    annotated_filename = f"{image_id}_{timestamp}_result.jpg"
                    print(f"DEBUG: Using fallback filename: {annotated_filename}")
            
            # Return results using the CORRECT counts from ML
            # ADD THIS LINE RIGHT HERE:
            print(f"DEBUG: Returning annotated path: /media/inspections/results/{annotated_filename}")

            # Return results using the CORRECT counts from ML
            return JsonResponse({
                'success': True,
                'image_id': image_id,
                'overall_result': overall_result,
                'nut_results': {
                    'nut1': {'status': nut_statuses[0], 'confidence': nut_results.get('nut1', {}).get('confidence', 0.0)},
                    'nut2': {'status': nut_statuses[1], 'confidence': nut_results.get('nut2', {}).get('confidence', 0.0)},
                    'nut3': {'status': nut_statuses[2], 'confidence': nut_results.get('nut3', {}).get('confidence', 0.0)},
                    'nut4': {'status': nut_statuses[3], 'confidence': nut_results.get('nut4', {}).get('confidence', 0.0)},
                },
                'summary': {
                    'present_count': present_count,
                    'missing_count': missing_count,
                    'quality_score': (present_count / 4) * 100
                },
                'processing_info': {
                    'processing_time': detection_data.get('processing_time', 0.0),
                    'method': image_source,
                    'filename': filename,
                    'total_detections': detection_data.get('total_detections', 0),
                    'confidence_threshold': detection_data.get('confidence_threshold', 0.5)
                },
                'detection_details': {
                    'detections': detection_data.get('detections', []),
                    'center_validation': detection_data.get('center_validation', {}),
                    'business_decision': decision
                },
                'image_paths': {
                    'original': f"/media/inspections/original/{filename}",
                    'annotated': f"/media/inspections/results/{annotated_filename}",
                },
                'inspection_id': str(inspection.id),
                'timestamp': datetime.now().isoformat(),
                # 🆕 NEW: Node-RED integration status
                'node_red': {
                    'sent': node_red_success,
                    'message': node_red_message
                }
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JsonResponse({
                'success': False,
                'error': f'Processing error: {str(e)}'
            })
    
    return JsonResponse({
        'success': False,
        'error': 'Method not allowed'
    })

@login_required
def results_display_view(request):
    """
    Display inspection results with enhanced image handling - FIXED VERSION
    """
    image_id = request.GET.get('image_id')
    
    if not image_id:
        messages.error(request, 'No Image ID provided.')
        return redirect('ml_api:simple_user_dashboard')
    
    try:
        # Get the latest inspection for this image_id
        inspection = SimpleInspection.objects.filter(image_id=image_id).order_by('-created_at').first()
        
        if not inspection:
            messages.error(request, f'No inspection found for Image ID: {image_id}')
            return redirect('ml_api:simple_user_dashboard')
        
        # **FIXED: Determine the correct folder based on inspection result**
        # Convert PASS/FAIL to OK/NG for folder structure
# REPLACE with this ENHANCED version that checks BOTH folders:

        # **FIXED: Determine the correct folder based on inspection result**
        # Convert PASS/FAIL to OK/NG for folder structure
        result_folder = 'OK' if inspection.overall_result == 'PASS' else 'NG'
        
        # **NEW: Create search paths for BOTH OK and NG folders to handle logic mismatches**
        custom_original_paths = [
            f'{result_folder}/{image_id}/original/',
            f'{"NG" if result_folder == "OK" else "OK"}/{image_id}/original/',
            f'{image_id}/{result_folder}/original/',
            f'{image_id}/original/',
            f'temp/',
        ]

        custom_annotated_paths = [
            f'{result_folder}/{image_id}/annotated/',
            f'{"NG" if result_folder == "OK" else "OK"}/{image_id}/annotated/',
            f'{image_id}/{result_folder}/annotated/',
            f'{image_id}/annotated/',
        ]      
        # **MINIMAL FIX: Add custom path search locations**
        # Custom path locations (NEW - PRIORITY SEARCH)
        # **FIXED: Add custom path search locations that check BOTH folders**
        # Custom path locations (NEW - PRIORITY SEARCH)
        # Custom path search order: status first, then image_id
        
        # **EXISTING: Original media path locations (PRESERVED)**
        original_possible_paths = [
            f'inspections/{result_folder}/{image_id}/original/',
            f'inspections/{image_id}/{result_folder}/original/',
            f'inspections/{image_id}/original/',
            f'inspections/original/',
        ]

        annotated_possible_paths = [
            f'inspections/{result_folder}/{image_id}/annotated/',
            f'inspections/{image_id}/{result_folder}/annotated/',
            f'inspections/{image_id}/annotated/',
            f'inspections/results/',
            f'inspections/triggered_inspections/{image_id}/annotated/',
            f'inspections/triggered_image/{result_folder}/{image_id}/annotated/',
        ]
        
        # **NEW: Search custom path FIRST for original image**
        original_image_path = None  # ENSURE THIS LINE EXISTS AT THE TOP
        for path in custom_original_paths:
            full_path = os.path.join(settings.CUSTOM_IMAGES_BASE_PATH, path)
            logger.info(f"[DEBUG] Checking custom path: {full_path}")
            
            if os.path.exists(full_path):
                logger.info(f"[DEBUG] Path exists, listing files...")
                try:
                    files_in_path = os.listdir(full_path)
                    logger.info(f"[DEBUG] Files found: {files_in_path}")
                    
                    for file in files_in_path:
                        logger.info(f"[DEBUG] Checking file: {file}")
                        # FIXED: More flexible matching
                        file_matches = (
                            image_id in file or 
                            inspection.filename == file or
                            file.startswith(image_id) or
                            (inspection.filename and os.path.splitext(inspection.filename)[0] in file)
                        )
                        
                        if file_matches and file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            # Create a custom media URL that we'll handle
                            original_image_path = f'/api/ml/custom_media/{path}{file}'
                            logger.info(f"[RESULTS] ✅ Found original in custom path: {original_image_path}")
                            break
                except Exception as e:
                    logger.error(f"[DEBUG] Error listing files in {full_path}: {e}")
            else:
                logger.info(f"[DEBUG] Path does not exist: {full_path}")
                
            if original_image_path:
                break
        
        # **EXISTING: Search for original image in media path (PRESERVED)**
        if not original_image_path:  # ENSURE original_image_path is initialized above
            for path in original_possible_paths:
                full_path = os.path.join(settings.MEDIA_ROOT, path)
                if os.path.exists(full_path):
                    for file in os.listdir(full_path):
                        if (image_id in file or inspection.filename == file) and file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            original_image_path = f'/media/{path}{file}'
                            break
                    if original_image_path:
                        break
        
        # **EXISTING: Search for original image in media path (PRESERVED)**
        if not original_image_path:
            for path in original_possible_paths:
                full_path = os.path.join(settings.MEDIA_ROOT, path)
                if os.path.exists(full_path):
                    for file in os.listdir(full_path):
                        if (image_id in file or inspection.filename == file) and file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            original_image_path = f'/media/{path}{file}'
                            break
                    if original_image_path:
                        break
        
        # **NEW: Search custom path FIRST for annotated image**
        annotated_image_path = None
        for path in custom_annotated_paths:
            full_path = os.path.join(settings.CUSTOM_IMAGES_BASE_PATH, path)
            if os.path.exists(full_path):
                for file in os.listdir(full_path):
                    if (image_id in file and any(keyword in file.lower() for keyword in ['annotated', 'result', 'processed', 'detection'])) and file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        annotated_image_path = f'/api/ml/custom_media/{path}{file}'
                        logger.info(f"[RESULTS] Found annotated in custom path: {annotated_image_path}")
                        break
                if annotated_image_path:
                    break
        
        # **EXISTING: Search for annotated image in media path (PRESERVED)**
        if not annotated_image_path:
            for path in annotated_possible_paths:
                full_path = os.path.join(settings.MEDIA_ROOT, path)
                if os.path.exists(full_path):
                    for file in os.listdir(full_path):
                        if (image_id in file and any(keyword in file.lower() for keyword in ['annotated', 'result', 'processed', 'detection'])) and file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            annotated_image_path = f'/media/{path}{file}'
                            logger.info(f"[RESULTS] Found annotated image: {annotated_image_path}")
                            break
                    if annotated_image_path:
                        break
        
        # **EXISTING: Enhanced storage database check (PRESERVED)**
        if not annotated_image_path:
            try:
                from .models import InspectionRecord
                enhanced_inspection = InspectionRecord.objects.filter(image_id=image_id).order_by('-capture_datetime').first()
                
                if enhanced_inspection:
                    if enhanced_inspection.annotated_image_path and os.path.exists(enhanced_inspection.annotated_image_path):
                        annotated_image_path = enhanced_inspection.annotated_image_path.replace(settings.MEDIA_ROOT, '/media').replace('\\', '/')
                        logger.info(f"[RESULTS] Found enhanced storage annotated image: {annotated_image_path}")
                    
                    if not original_image_path and enhanced_inspection.original_image_path and os.path.exists(enhanced_inspection.original_image_path):
                        original_image_path = enhanced_inspection.original_image_path.replace(settings.MEDIA_ROOT, '/media').replace('\\', '/')
                
            except ImportError:
                logger.info("[RESULTS] Enhanced storage not available")
            except Exception as e:
                logger.warning(f"[RESULTS] Error checking enhanced storage: {e}")
        
        # **EXISTING: Debug logging (PRESERVED)**
        logger.info(f"[RESULTS] Image ID: {image_id}")
        logger.info(f"[RESULTS] Result folder: {result_folder}")
        logger.info(f"[RESULTS] Original image path: {original_image_path}")
        logger.info(f"[RESULTS] Annotated image path: {annotated_image_path}")
        
        # **EXISTING: Context and return (PRESERVED)**
        context = {
            'inspection': inspection,
            'image_id': image_id,
            'result_folder': result_folder,  # Pass the folder name
            'original_image_path': original_image_path,
            'annotated_image_path': annotated_image_path,
            'annotated_filename': os.path.basename(annotated_image_path) if annotated_image_path else None,
        }
        
        return render(request, 'ml_api/results_display.html', context)
        
    except Exception as e:
        logger.error(f"[RESULTS] Error in results display: {e}")
        messages.error(request, f'Error loading results: {str(e)}')
        return redirect('ml_api:simple_user_dashboard')

def _save_uploaded_image(uploaded_file, image_id):
    """
    Save uploaded image to processing directory
    """
    import os
    from django.conf import settings
    from django.core.files.storage import default_storage
    
    # Create directories if they don't exist - CHANGED TO USE CUSTOM PATH
    original_dir = os.path.join(settings.CUSTOM_IMAGES_BASE_PATH, 'temp')
    os.makedirs(original_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_extension = os.path.splitext(uploaded_file.name)[1]
    filename = f"{image_id}_{timestamp}{file_extension}"
    
    # Save file
    file_path = os.path.join(original_dir, filename)
    
    with open(file_path, 'wb+') as destination:
        for chunk in uploaded_file.chunks():
            destination.write(chunk)
    
    return file_path, filename

def _process_with_yolov8_model(image_path, image_id):
    """
    Process image with actual YOLOv8 model (using your exact ML logic)
    """
    try:
        import cv2
        import numpy as np
        from pathlib import Path
        from django.conf import settings
        import os
        
        # Import YOLOv8 model from your services
        from .services import enhanced_nut_detection_service
        
        # Verify image exists
        if not os.path.exists(image_path):
            return {
                'success': False,
                'error': f'Image file not found: {image_path}'
            }
        
        # Load and validate image
        image = cv2.imread(image_path)
        if image is None:
            return {
                'success': False,
                'error': 'Could not load image file'
            }
        
        print(f"Processing image: {image_path}")
        print(f"Image shape: {image.shape}")
        
        # Use your enhanced nut detection service with your ML logic
        start_time = datetime.now()
        
        # Process with your YOLOv8 model using your exact business logic
        result = enhanced_nut_detection_service.process_image_with_id(
            image_path=image_path,
            image_id=image_id,
            user_id=None
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        if not result['success']:
            return {
                'success': False,
                'error': result.get('error', 'YOLOv8 processing failed')
            }
        
        # Extract results using your business logic
        nut_results = result['nut_results']
        decision = result['decision']
        center_validation = result.get('center_validation', {})
        detections = result.get('detection_summary', {}).get('detections', [])
        
        return {
            'success': True,
            'data': {
                'nut_results': nut_results,
                'decision': decision,
                'detections': detections,
                'center_validation': center_validation,
                'processing_time': processing_time,
                'total_detections': len(detections),
                'confidence_threshold': 0.5,
                'annotated_image_path': result.get('annotated_image_path')
            }
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': f'YOLOv8 processing error: {str(e)}'
        }

def simple_logout_view(request):
    """
    Simple logout
    """
    logout(request)
    messages.success(request, 'You have been logged out successfully.')
    return redirect('ml_api:simple_login')

# 🆕 NEW: FILE TRANSFER MANAGEMENT VIEWS - ADD THESE AT THE END OF THE FILE

@login_required
def file_transfer_dashboard(request):
    """
    Dashboard for monitoring file transfer system
    """
    if request.user.role != 'admin':
        messages.error(request, 'Access denied. Admin privileges required.')
        return redirect('ml_api:simple_user_dashboard')
    
    try:
        # Initialize transfer service
        transfer_service = FileTransferService()
        
        # Get statistics
        stats = transfer_service.get_transfer_statistics()
        
        context = {
            'stats': stats,
            'page_title': 'File Transfer Dashboard'
        }
        
        return render(request, 'ml_api/file_transfer_dashboard.html', context)
        
    except Exception as e:
        messages.error(request, f'Error loading file transfer dashboard: {str(e)}')
        return redirect('ml_api:simple_admin_dashboard')

@login_required
def retry_failed_transfers(request):
    """
    AJAX endpoint to retry failed file transfers
    """
    if request.user.role != 'admin':
        return JsonResponse({'success': False, 'error': 'Admin access required'})
    
    if request.method == 'POST':
        try:
            print(f"\n🔄 Manual retry initiated by: {request.user.username}")
            
            # Initialize transfer service
            transfer_service = FileTransferService()
            
            # Retry failed transfers
            results = transfer_service.retry_failed_transfers()
            
            return JsonResponse({
                'success': True,
                'message': f"Retry completed: {results.get('successful', 0)} successful, {results.get('failed', 0)} failed",
                'results': results
            })
            
        except Exception as e:
            error_msg = f"Retry failed: {str(e)}"
            print(f"💥 {error_msg}")
            return JsonResponse({'success': False, 'error': error_msg})
    
    return JsonResponse({'success': False, 'error': 'Method not allowed'})

@login_required
def test_file_transfer_system(request):
    """
    AJAX endpoint to test file transfer system
    """
    if request.user.role != 'admin':
        return JsonResponse({'success': False, 'error': 'Admin access required'})
    
    if request.method == 'POST':
        try:
            print(f"\n🧪 System test initiated by: {request.user.username}")
            
            # Initialize transfer service
            transfer_service = FileTransferService()
            
            # Run system test
            test_results = transfer_service.test_system()
            
            return JsonResponse({
                'success': test_results.get('overall', False),
                'message': 'System test completed',
                'results': test_results
            })
            
        except Exception as e:
            error_msg = f"System test failed: {str(e)}"
            print(f"💥 {error_msg}")
            return JsonResponse({'success': False, 'error': error_msg})
    
    return JsonResponse({'success': False, 'error': 'Method not allowed'})

@login_required
def override_authentication_view(request):
    """
    Override authentication for status changes
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            username = data.get('username')
            password = data.get('password')
            image_id = data.get('image_id')
            
            # Check if this is a predefined admin
            is_admin = False
            if username in PREDEFINED_ADMINS and password == PREDEFINED_ADMINS[username]:
                is_admin = True
            
            # Authenticate user
            user = authenticate(request, username=username, password=password)
            
            if user is not None or is_admin:
                # Find the inspection record
                try:
                    inspection = InspectionRecord.objects.filter(image_id=image_id).latest('capture_datetime')
                except InspectionRecord.DoesNotExist:
                    # Try SimpleInspection as fallback
                    try:
                        simple_inspection = SimpleInspection.objects.filter(image_id=image_id).latest('created_at')
                        return JsonResponse({
                            'success': False,
                            'error': 'Legacy inspection format not supported for override'
                        })
                    except SimpleInspection.DoesNotExist:
                        return JsonResponse({
                            'success': False,
                            'error': 'Inspection not found'
                        })
                
                # Determine user role
                if is_admin:
                    user_role = 'admin'
                elif user:
                    user_role = user.role
                else:
                    user_role = 'user'
                
                # Check permissions
                if user_role == 'user':
                    if user and inspection.user != user:
                        return JsonResponse({
                            'success': False,
                            'error': 'Users can only modify their own inspections'
                        })
                    
                    # Check if this is the latest inspection
                    if user:
                        latest_inspection = InspectionRecord.objects.filter(
                            user=user
                        ).order_by('-capture_datetime').first()
                        
                        if latest_inspection and inspection.id != latest_inspection.id:
                            return JsonResponse({
                                'success': False,
                                'error': 'Users can only modify their current/latest inspection'
                            })
                
                return JsonResponse({
                    'success': True,
                    'user_role': user_role,
                    'inspection_id': str(inspection.id),
                    'current_status': inspection.test_status,
                    'message': f'Authentication successful for {username}'
                })
            else:
                return JsonResponse({
                    'success': False,
                    'error': 'Invalid username or password'
                })
                
        except json.JSONDecodeError:
            return JsonResponse({'success': False, 'error': 'Invalid JSON data'})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Method not allowed'})


@login_required
def retry_failed_image(request, image_id):
    """
    MODIFIED: Retry functionality that preserves all existing data and generates incremental image_id
    1. Generate new image_id with suffix (X_1, X_2, etc.)
    2. Keep all existing files and database records
    3. Redirect to camera capture with new image_id
    """
    if request.method == 'POST':
        try:
            print(f"\n🔄 RETRY REQUEST INITIATED (PRESERVE DATA)")
            print(f"   - Original Image ID: {image_id}")
            print(f"   - User: {request.user.username}")
            print(f"   - Timestamp: {datetime.now()}")
            
            # ============================================================================
            # STEP 1: GENERATE NEW IMAGE_ID WITH INCREMENTAL SUFFIX
            # ============================================================================
            
            def find_next_available_image_id(base_image_id):
                """Find the next available image_id with incremental suffix"""
                # Check if base image_id contains existing suffix
                if '_' in base_image_id and base_image_id.split('_')[-1].isdigit():
                    # Extract base without suffix (e.g., "ABC1234567_2" -> "ABC1234567")
                    base_without_suffix = '_'.join(base_image_id.split('_')[:-1])
                else:
                    # No existing suffix
                    base_without_suffix = base_image_id
                
                print(f"📋 Base image_id (without suffix): {base_without_suffix}")
                
                # Find existing suffixes in database
                existing_suffixes = set()
                
                # Check SimpleInspection
                simple_inspections = SimpleInspection.objects.filter(
                    image_id__startswith=base_without_suffix
                )
                for inspection in simple_inspections:
                    if inspection.image_id == base_without_suffix:
                        existing_suffixes.add(0)  # Original has suffix 0
                    elif inspection.image_id.startswith(base_without_suffix + '_'):
                        suffix_part = inspection.image_id[len(base_without_suffix + '_'):]
                        if suffix_part.isdigit():
                            existing_suffixes.add(int(suffix_part))
                
                # Check InspectionRecord (enhanced storage)
                try:
                    inspection_records = InspectionRecord.objects.filter(
                        image_id__startswith=base_without_suffix
                    )
                    for record in inspection_records:
                        if record.image_id == base_without_suffix:
                            existing_suffixes.add(0)  # Original has suffix 0
                        elif record.image_id.startswith(base_without_suffix + '_'):
                            suffix_part = record.image_id[len(base_without_suffix + '_'):]
                            if suffix_part.isdigit():
                                existing_suffixes.add(int(suffix_part))
                except Exception as e:
                    print(f"⚠️ Could not check InspectionRecord: {e}")
                
                print(f"📊 Found existing suffixes: {sorted(existing_suffixes)}")
                
                # Find next available suffix
                next_suffix = 1
                while next_suffix in existing_suffixes:
                    next_suffix += 1
                
                new_image_id = f"{base_without_suffix}_{next_suffix}"
                print(f"🆔 Generated new image_id: {new_image_id}")
                
                return new_image_id, base_without_suffix, next_suffix
            
            # Generate the new image_id
            new_image_id, base_id, suffix_number = find_next_available_image_id(image_id)
            
            # ============================================================================
            # STEP 2: VERIFY NEW IMAGE_ID IS UNIQUE
            # ============================================================================
            
            # Double-check that new image_id doesn't exist
            if SimpleInspection.objects.filter(image_id=new_image_id).exists():
                return JsonResponse({
                    'success': False,
                    'error': f'Generated image_id {new_image_id} already exists. Please try again.'
                })
            
            try:
                if InspectionRecord.objects.filter(image_id=new_image_id).exists():
                    return JsonResponse({
                        'success': False,
                        'error': f'Generated image_id {new_image_id} already exists in enhanced records. Please try again.'
                    })
            except:
                pass  # Enhanced storage may not be available
            
            print(f"✅ Verified new image_id is unique: {new_image_id}")
            
            # ============================================================================
            # STEP 3: LOG PRESERVATION (NO DELETION)
            # ============================================================================
            
            print(f"\n📋 DATA PRESERVATION SUMMARY:")
            print(f"   - Original image_id: {image_id} (PRESERVED)")
            print(f"   - New image_id: {new_image_id}")
            print(f"   - Base: {base_id}")
            print(f"   - Suffix: _{suffix_number}")
            print(f"   - All existing files: PRESERVED")
            print(f"   - All database records: PRESERVED")
            print(f"   - Action: Redirect to camera capture with new image_id")
            
            # ============================================================================ 
            # STEP 4: RETURN SUCCESS WITH REDIRECT URL
            # ============================================================================
            
            # IMPORTANT: Use the exact same URL format as your current working system
            redirect_url = f'/api/ml/camera-capture/?image_id={new_image_id}'
            
            print(f"📷 Prepared redirect to: {redirect_url}")
            print(f"🔄 RETRY PREPARATION COMPLETED SUCCESSFULLY (NO DATA DELETED)")
            
            return JsonResponse({
                'success': True,
                'message': f'All data for {image_id} preserved. New attempt: {new_image_id}',
                'original_image_id': image_id,
                'new_image_id': new_image_id,
                'suffix_number': suffix_number,
                'files_deleted': 0,  # Keep this field for compatibility with frontend
                'redirect_url': redirect_url,  # This is the key field the frontend expects
                'preservation_note': 'All existing files and records preserved'
            })
            
        except Exception as e:
            print(f"💥 RETRY ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            return JsonResponse({
                'success': False,
                'error': f'Retry failed: {str(e)}'
            })
    
    return JsonResponse({
        'success': False,
        'error': 'Method not allowed'
    })
                                                         

@login_required
def camera_check_view(request):
    """
    Camera Check page - Test camera functionality and capture images
    Available to all users (both admin and regular users)
    """
    return render(request, 'ml_api/camera_check.html')


@login_required
def get_qr_processing_status(request):
    """
    Get current QR processing status
    """
    global CURRENT_PROCESSING_QR, PROCESSING_START_TIME
    
    with QR_PROCESSING_LOCK:
        if CURRENT_PROCESSING_QR is not None:
            processing_duration = (datetime.now() - PROCESSING_START_TIME).total_seconds()
            return JsonResponse({
                'processing': True,
                'current_qr': CURRENT_PROCESSING_QR,
                'processing_duration_seconds': round(processing_duration, 1),
                'estimated_remaining_seconds': max(0, 6 - processing_duration)  # Based on your 5-6 sec estimate
            })
        else:
            return JsonResponse({
                'processing': False,
                'ready_for_scan': True
            })
            
@csrf_exempt
@login_required
def send_qr_to_node_red_endpoint(request):
    """
    API endpoint to send QR code to Node-RED (for manual entry)
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_id = data.get('image_id')
            source_type = data.get('source_type', 'manual')
            
            if not image_id:
                return JsonResponse({
                    'success': False,
                    'error': 'No image_id provided'
                })
            
            # 🆕 NEW: Check for duplicate Image ID before sending to Node-RED
            # count = SimpleInspection.objects.filter(image_id=image_id).count()
            # auto_modified = False
            # if count >= 1:
            #     new_image_id = f"{image_id}_{count}"
            #     auto_modified = True
            #     success, message = send_qr_to_node_red(new_image_id, source_type)
            #     return JsonResponse({
            #         'success': True,
            #         'auto_modified' : auto_modified,
            #         'error': f'Image ID "{image_id}" already exists. Cannot send duplicate to Node-RED.',
            #         'duplicate': True,
            #         'occurences' : count,
            #         'image_id': new_image_id,
            #         'source_type': source_type
            #     })
            # else:
            #     auto_modified = False
            
            # Send QR code to Node-RED
            success, message = send_qr_to_node_red(image_id, source_type)
            
            return JsonResponse({
                'success': success,
                'message': message,
                'image_id': image_id,
                'source_type': source_type,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({'success': False, 'error': 'Method not allowed'})
# Helper Functions

def _validate_image_id_format(image_id):
    """
    Validate Image ID format according to requirements (UPDATED for retry suffixes)
    """
    if not image_id:
        return {'valid': False, 'message': 'Image ID cannot be empty'}
    
    # NEW: Check if this is a retry image_id (has suffix like _1, _2, etc.)
    if '_' in image_id and image_id.split('_')[-1].isdigit():
        # This is a retry image_id - validate the base part only

        parts = image_id.split('_')
        base_part = '_'.join(parts[:-1])  # Everything except the last part
        suffix_part = parts[-1]          # The numeric suffix
        
        # Validate base part (should be 10 characters)
        if len(base_part) > 15:
            return {'valid': False, 'message': f'Base Image ID must be exactly 10 characters (got {len(base_part)})'}
        
        # Validate base part format
        if not re.match(r'^[A-Za-z0-9_-]+$', base_part):
            return {'valid': False, 'message': 'Base Image ID can only contain letters, numbers, underscore, and hyphen'}
        
        # Validate suffix is numeric
        if not suffix_part.isdigit():
            return {'valid': False, 'message': 'Retry suffix must be numeric'}
        
        return {'valid': True, 'message': f'Valid retry Image ID: {image_id} (base: {base_part}, retry: {suffix_part})'}
    
    else:
        # Original image_id - must be exactly 10 characters
        if len(image_id) != 10:
            return {'valid': False, 'message': 'Image ID must be exactly 10 characters long'}
        
        # Allow letters, numbers, underscore, and hyphen
        if not re.match(r'^[A-Za-z0-9_-]+$', image_id):
            return {'valid': False, 'message': 'Image ID can only contain letters, numbers, underscore, and hyphen'}
        
        return {'valid': True, 'message': f'Valid Image ID: {image_id}'}


# ADD THIS FUNCTION TO simple_auth_views.py (after the camera_check_view function)
@csrf_exempt
@login_required
def simple_camera_capture(request):
    """
    Simple camera capture for testing - just capture and return image path
    Used by Camera Check page
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_id = data.get('image_id', f'test_{int(datetime.now().timestamp())}')
            
            # Import camera manager
            from .views import camera_manager
            
            # Check if camera is connected
            if not camera_manager.is_connected:
                return JsonResponse({
                    'success': False,
                    'error': 'Camera not connected. Please connect camera first.'
                })
            
            # Get frame directly from camera manager (since capture works for you)
            frame = camera_manager.get_frame()
            if frame is None:
                return JsonResponse({
                    'success': False,
                    'error': 'No frame available from camera'
                })
            
            # Save the frame to file
            test_dir = os.path.join(settings.MEDIA_ROOT, 'captures')
            os.makedirs(test_dir, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"camera_test_{timestamp}.jpg"
            filepath = os.path.join(test_dir, filename)
            
            # Save image using cv2 (since your camera capture works)
            import cv2
            success = cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            if not success:
                return JsonResponse({
                    'success': False,
                    'error': 'Failed to save captured image'
                })
            
            # Get file size
            file_size = os.path.getsize(filepath)
            
            # Return success with image path
            return JsonResponse({
                'success': True,
                'message': 'Test image captured successfully!',
                'image_path': f'/media/captures/{filename}',
                'filename': filename,
                'image_id': image_id,
                'file_size': file_size,
                'timestamp': timestamp
            })
            
        except Exception as e:
            logger.error(f"Camera check capture error: {e}")
            return JsonResponse({
                'success': False,
                'error': f'Capture error: {str(e)}'
            })
    
    return JsonResponse({'success': False, 'error': 'Method not allowed'})
# ADD THESE FUNCTIONS TO THE END OF simple_auth_views.py

import csv
from django.http import HttpResponse
from django.utils import timezone
from datetime import timedelta

@login_required
def reports_filter_view(request):
    """
    View Reports filter page
    """
    if request.user.role != 'admin':
        messages.error(request, 'Access denied. Admin privileges required.')
        return redirect('ml_api:simple_user_dashboard')
    
    return render(request, 'ml_api/reports_filter.html')

# REPLACE the existing generate_report_view function in simple_auth_views.py with this updated version:

# REPLACE the existing generate_report_view function in simple_auth_views.py with this updated version:

@login_required
def generate_report_view(request):
    """
    Generate CSV report based on date and time range filters (UPDATED WITH DUPLICATE REMOVAL)
    """
    if request.user.role != 'admin':
        return JsonResponse({'success': False, 'error': 'Admin access required'})
    
    if request.method == 'POST':
        try:
            # Get filter parameters (CHANGED TO HANDLE DATETIME)
            start_datetime_str = request.POST.get('start_datetime')  # Format: YYYY-MM-DD HH:MM
            end_datetime_str = request.POST.get('end_datetime')      # Format: YYYY-MM-DD HH:MM
            parts_filter = request.POST.get('parts_filter')         # 'all', 'ok', or 'ng'
            
            # Validate datetime inputs
            if not start_datetime_str or not end_datetime_str:
                return JsonResponse({'success': False, 'error': 'Start date/time and end date/time are required'})
            
            try:
                # Parse datetime strings
                from datetime import datetime
                start_datetime = datetime.strptime(start_datetime_str, '%Y-%m-%d %H:%M')
                end_datetime = datetime.strptime(end_datetime_str, '%Y-%m-%d %H:%M')
                
                # Make timezone aware
                from django.utils import timezone
                start_datetime = timezone.make_aware(start_datetime)
                end_datetime = timezone.make_aware(end_datetime)
                
            except ValueError:
                return JsonResponse({'success': False, 'error': 'Invalid datetime format. Use YYYY-MM-DD HH:MM'})
            
            # Validate datetime range
            if start_datetime >= end_datetime:
                return JsonResponse({'success': False, 'error': 'Start date/time must be before end date/time'})
            
            # Get data from both models using datetime range filter
            simple_inspections = SimpleInspection.objects.filter(
                created_at__gte=start_datetime,
                created_at__lte=end_datetime
            ).order_by('-created_at')
            
            inspection_records = InspectionRecord.objects.filter(
                capture_datetime__gte=start_datetime,
                capture_datetime__lte=end_datetime
            ).order_by('-capture_datetime')
            
            # NEW: Use a dictionary to track unique DMC numbers and keep only the latest entry
            unique_inspections = {}
            
            # Process SimpleInspection data
            for inspection in simple_inspections:
                # Apply parts filter
                if parts_filter == 'ok' and inspection.overall_result != 'PASS':
                    continue
                elif parts_filter == 'ng' and inspection.overall_result != 'FAIL':
                    continue
                
                # Determine status (convert PASS/FAIL to OK/NG)
                status = 'OK' if inspection.overall_result == 'PASS' else 'NG'
                
                inspection_data = {
                    'dmc_number': inspection.image_id,
                    'status': status,
                    'timestamp': inspection.created_at,
                    'timestamp_str': inspection.created_at.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # NEW: Only keep the latest entry for each DMC number
                dmc_key = inspection.image_id
                if dmc_key not in unique_inspections or inspection.created_at > unique_inspections[dmc_key]['timestamp']:
                    unique_inspections[dmc_key] = inspection_data
            
            # Process InspectionRecord data
            for inspection in inspection_records:
                # Apply parts filter
                if parts_filter == 'ok' and inspection.test_status != 'OK':
                    continue
                elif parts_filter == 'ng' and inspection.test_status != 'NG':
                    continue
                
                inspection_data = {
                    'dmc_number': inspection.image_id,
                    'status': inspection.test_status,
                    'timestamp': inspection.capture_datetime,
                    'timestamp_str': inspection.capture_datetime.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # NEW: Only keep the latest entry for each DMC number
                dmc_key = inspection.image_id
                if dmc_key not in unique_inspections or inspection.capture_datetime > unique_inspections[dmc_key]['timestamp']:
                    unique_inspections[dmc_key] = inspection_data
            
            # NEW: Convert dictionary back to list and sort by timestamp
            csv_data = list(unique_inspections.values())
            csv_data.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # Add serial numbers
            for i, row in enumerate(csv_data, 1):
                row['serial_number'] = i
            
            # Create CSV response with datetime in filename
            response = HttpResponse(content_type='text/csv')
            
            # Create filename with datetime range
            start_file_str = start_datetime.strftime('%Y%m%d_%H%M')
            end_file_str = end_datetime.strftime('%Y%m%d_%H%M')
            filename = f"inspection_report_{start_file_str}_to_{end_file_str}.csv"
            response['Content-Disposition'] = f'attachment; filename="{filename}"'
            
            # Write CSV
            writer = csv.writer(response)
            
            # Write header in correct order - Serial Number, Timestamp, DMC Number, Status
            writer.writerow(['Serial Number', 'Timestamp', 'DMC Number', 'Status'])
            
            # Write data in correct column order (no duplicates)
            for row in csv_data:
                writer.writerow([
                    row['serial_number'],
                    row['timestamp_str'],
                    row['dmc_number'],
                    row['status']
                ])
            
            return response
            
        except Exception as e:
            logger.error(f"Report generation error: {e}")
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Method not allowed'})

# ADD THESE FUNCTIONS TO THE END OF simple_auth_views.py

# ============================================================================
# USER MANAGEMENT API ENDPOINTS
# ============================================================================

@login_required
def user_management_list(request):
    """Get list of all users for admin dashboard"""
    if request.user.role != 'admin':
        return JsonResponse({'success': False, 'error': 'Admin access required'})
    
    try:
        users = CustomUser.objects.all().order_by('username')
        users_data = []
        
        for user in users:
            users_data.append({
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'role': user.role,
                'employee_id': user.employee_id or '',
                'date_joined': user.date_joined.isoformat(),
                'is_active': user.is_active,
                'last_login': user.last_login.isoformat() if user.last_login else None
            })
        
        return JsonResponse({
            'success': True,
            'users': users_data,
            'total_count': len(users_data)
        })
        
    except Exception as e:
        logger.error(f"Error loading user list: {e}")
        return JsonResponse({'success': False, 'error': str(e)})

@login_required
def user_management_add(request):
    """Add new user"""
    if request.user.role != 'admin':
        return JsonResponse({'success': False, 'error': 'Admin access required'})
    
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            username = data.get('username', '').strip()
            email = data.get('email', '').strip()
            password = data.get('password', '')
            role = data.get('role', 'user')
            employee_id = data.get('employee_id', '').strip()
            
            # Validation
            if not username:
                return JsonResponse({'success': False, 'error': 'Username is required'})
            
            if not email:
                return JsonResponse({'success': False, 'error': 'Email is required'})
            
            if not password:
                return JsonResponse({'success': False, 'error': 'Password is required'})
            
            if len(password) < 8:
                return JsonResponse({'success': False, 'error': 'Password must be at least 8 characters long'})
            
            if role not in ['user', 'admin']:
                return JsonResponse({'success': False, 'error': 'Invalid role'})
            
            # Check if username already exists
            if CustomUser.objects.filter(username=username).exists():
                return JsonResponse({'success': False, 'error': 'Username already exists'})
            
            # Check if email already exists
            if CustomUser.objects.filter(email=email).exists():
                return JsonResponse({'success': False, 'error': 'Email already registered'})
            
            # Prevent creating predefined admin usernames
            if username in PREDEFINED_ADMINS:
                return JsonResponse({'success': False, 'error': 'This username is reserved'})
            
            # Create user
            user = CustomUser.objects.create_user(
                username=username,
                email=email,
                password=password,
                role=role,
                employee_id=employee_id
            )
            
            logger.info(f"Admin {request.user.username} created new user: {username} ({role})")
            
            return JsonResponse({
                'success': True,
                'message': f'User "{username}" created successfully',
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,
                    'role': user.role,
                    'employee_id': user.employee_id
                }
            })
            
        except json.JSONDecodeError:
            return JsonResponse({'success': False, 'error': 'Invalid JSON data'})
        except Exception as e:
            logger.error(f"Error adding user: {e}")
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Method not allowed'})

@login_required
def user_management_delete(request, user_id):
    """Delete user"""
    if request.user.role != 'admin':
        return JsonResponse({'success': False, 'error': 'Admin access required'})
    
    if request.method == 'DELETE':
        try:
            # Get user to delete
            try:
                user_to_delete = CustomUser.objects.get(id=user_id)
            except CustomUser.DoesNotExist:
                return JsonResponse({'success': False, 'error': 'User not found'})
            
            # Prevent admin from deleting themselves
            if user_to_delete.id == request.user.id:
                return JsonResponse({'success': False, 'error': 'Cannot delete yourself'})
            
            # Prevent deleting predefined admin accounts
            if user_to_delete.username in PREDEFINED_ADMINS:
                return JsonResponse({'success': False, 'error': 'Cannot delete predefined admin account'})
            
            username = user_to_delete.username
            user_to_delete.delete()
            
            logger.info(f"Admin {request.user.username} deleted user: {username}")
            
            return JsonResponse({
                'success': True,
                'message': f'User "{username}" deleted successfully'
            })
            
        except Exception as e:
            logger.error(f"Error deleting user: {e}")
            return JsonResponse({'success': False, 'error': str(e)})
    
    return JsonResponse({'success': False, 'error': 'Method not allowed'})

@login_required
def user_management_count(request):
    """Get current user count"""
    if request.user.role != 'admin':
        return JsonResponse({'success': False, 'error': 'Admin access required'})
    
    try:
        count = CustomUser.objects.count()
        return JsonResponse({
            'success': True,
            'count': count
        })
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})
    
def send_qr_to_node_red(image_id, source_type):
    """
    Send QR code (image_id) to Node-RED immediately when scanned/entered
    """
    try:
        from django.conf import settings
        import requests
        
        # Use existing Node-RED endpoint or create new one for QR codes
        node_red_url = getattr(settings, 'NODE_RED_QR_ENDPOINT', 'http://localhost:1880/qr-received')
        timeout = getattr(settings, 'NODE_RED_TIMEOUT', 5)
        
        payload = {
            'qr_code': image_id,
            'image_id': image_id,
            'source_type': source_type,  # 'manual' or 'scanner'
            'timestamp': datetime.now().isoformat(),
            'source': 'marelli_ml_api'
        }
        
        response = requests.post(
            node_red_url,
            json=payload,
            timeout=timeout,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            logger.info(f"[NODE-RED] QR code sent successfully: {image_id} ({source_type})")
            return True, "QR code sent to Node-RED successfully"
        else:
            logger.warning(f"[NODE-RED] HTTP {response.status_code}: {response.text}")
            return False, f"Node-RED responded with status {response.status_code}"
            
    except Exception as e:
        logger.error(f"[NODE-RED] Error sending QR code: {str(e)}")
        return False, f"Node-RED QR error: {str(e)}"
    
    
# ADD this temporary debug function to your simple_auth_views.py
@login_required
def debug_custom_path_files(request, image_id):
    """Debug endpoint to check what files exist in custom path"""
    try:
        from django.conf import settings
        import os
        
        debug_info = {
            'image_id': image_id,
            'custom_base_path': settings.CUSTOM_IMAGES_BASE_PATH,
            'paths_checked': {}
        }
        
        # Check all possible paths
        paths_to_check = [
            'temp/',
            f'{image_id}/',
            f'{image_id}/OK/',
            f'{image_id}/NG/',
            f'{image_id}/OK/original/',
            f'{image_id}/NG/original/',
        ]
        
        for path in paths_to_check:
            full_path = os.path.join(settings.CUSTOM_IMAGES_BASE_PATH, path)
            
            if os.path.exists(full_path):
                try:
                    files = os.listdir(full_path)
                    debug_info['paths_checked'][path] = {
                        'exists': True,
                        'files': files
                    }
                except Exception as e:
                    debug_info['paths_checked'][path] = {
                        'exists': True,
                        'error': str(e)
                    }
            else:
                debug_info['paths_checked'][path] = {
                    'exists': False
                }
        
        return JsonResponse({
            'success': True,
            'debug_info': debug_info
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })