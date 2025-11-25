# ml_api/urls.py - COMPLETE FIXED FILE

from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from . import views
# Add this import at the top of your ml_api/urls.py if not already present
from django.views.static import serve
from django.conf import settings
# Updated simple authentication views with Image ID workflow
# ADD this import at the TOP of your ml_api/urls.py:
from .expiry_views import system_expiry_status, update_expiry_time, expiry_management_page, emergency_extend



from .simple_auth_views import (
    debug_custom_path_files,
    get_qr_processing_status,
    retry_failed_image,
    simple_camera_capture,
    simple_login_view,
    simple_logout_view,
    simple_admin_dashboard,
    simple_user_dashboard,
    simple_workflow_page,
    simple_process_image,
    # New Image ID workflow views
    image_id_entry_view,
    validate_image_id,
    qr_scanner_endpoint,
    image_source_selection_view,
    results_display_view,
    # Camera integration views (keep the original camera_capture_view)
    camera_capture_view,
    # Enhanced Authentication views
    inspection_list_view,
    update_inspection_status_view,
    # File transfer imports
    file_transfer_dashboard,
    retry_failed_transfers,
    test_file_transfer_system,
    # Override authentication
    override_authentication_view,
    camera_capture_and_process,
    # FIXED: Import reports functions from simple_auth_views
    reports_filter_view,
    generate_report_view,
    
    retry_failed_image,  # 🆕 ADD THIS LINE
    
    user_management_list,
    user_management_add,
    user_management_delete,
    user_management_count,
    
    camera_check_view,
    get_qr_processing_status,  # ADD THIS LINE
    send_qr_to_node_red_endpoint,  # 🆕 ADD THIS LINE
)

# FIXED: Import only functions that actually exist in views.py
from .views import (
    # Core detection views
    NutDetectionView, 
    detection_page,
    # Camera functionality imports - VERIFIED TO EXIST
    connect_camera,
    disconnect_camera,
    camera_status,
    send_fail_result_to_node_red,
    video_stream,
    get_current_frame_base64,
    camera_control_page,
    # Trigger mode imports
    enable_trigger_mode,
    disable_trigger_mode,
    get_trigger_status,
    get_trigger_result_status,
    # NEW: API endpoint for overall_result by image_id
    get_overall_result_by_image_id,
    # NEW: Node-RED test endpoint
    test_node_red_connection,
    # NEW: Node-RED receive endpoint
    receive_from_node_red,
    # NEW: Node-RED get message endpoint
    get_node_red_message,
    # NEW: Store image_id in cache endpoint
    store_image_id,
    # NEW: Finalize triggered inspection endpoint
    finalize_triggered_inspection,
    # NEW: Reset trigger session endpoint  
    reset_trigger_session,
    # NEW: Debug inspection files endpoint
    debug_inspection_files,
)

app_name = 'ml_api'

urlpatterns = [
    # 🏠 AUTHENTICATION SYSTEM
    path('', simple_login_view, name='simple_login'),
    path('login/', simple_login_view, name='simple_login'),
    path('logout/', simple_logout_view, name='simple_logout'),
    
    # 📊 DASHBOARDS (Role-based)
    path('admin-dashboard/', simple_admin_dashboard, name='simple_admin_dashboard'),
    path('user-dashboard/', simple_user_dashboard, name='simple_user_dashboard'),
    
    # 🆕 INSPECTION MANAGEMENT
    path('inspections/', inspection_list_view, name='inspection_list'),
    path('inspection/<uuid:inspection_id>/update-status/', update_inspection_status_view, name='update_inspection_status'),
    
    # 📊 REPORTS FUNCTIONALITY - FIXED: Import from simple_auth_views
    path('reports/', reports_filter_view, name='reports_filter'),
    path('generate-report/', generate_report_view, name='generate_report'),

    # 🆕 FILE TRANSFER MANAGEMENT
    path('file-transfer-dashboard/', file_transfer_dashboard, name='file_transfer_dashboard'),
    path('retry-failed-transfers/', retry_failed_transfers, name='retry_failed_transfers'),
    path('test-file-transfer/', test_file_transfer_system, name='test_file_transfer'),
    path('override-auth/', override_authentication_view, name='override_auth'),
     
    # 🏭 COMPLETE IMAGE WORKFLOW
    # Step 1: Image ID Entry (Manual/QR Scanner)
    path('image-id/', image_id_entry_view, name='image_id_entry'),
    path('validate-image-id/', validate_image_id, name='validate_image_id'),
    path('qr-scan/', qr_scanner_endpoint, name='qr_scanner'),
    
    # Step 2: Image Source Selection (Upload/Camera)
    path('image-source/', image_source_selection_view, name='image_source_selection'),
    
    # 📷 Step 2B: CAMERA CAPTURE (Integrated into workflow)
    path('camera-capture/', camera_capture_view, name='camera_capture'),
    
    # 🔧 MAIN FIX: Manual capture with same processing as trigger
    path('camera/capture-and-process/', camera_capture_and_process, name='capture_and_process'),
    
    # Step 3 & 4: Processing and Results
    path('process-image/', simple_process_image, name='simple_process_image'),
    path('results/', results_display_view, name='results_display'),
    
    # 🎯 WORKFLOW ENTRY POINTS
    path('workflow/', simple_workflow_page, name='simple_workflow'),
    path('start/', image_id_entry_view, name='start_workflow'),
    
    # 🔧 LEGACY/WORKING ENDPOINTS (Keep for compatibility)
    path('health/', NutDetectionView.as_view(), name='health_check'),
    path('detect-nuts/', NutDetectionView.as_view(), name='detect_nuts'),
    path('detection/', detection_page, name='detection_page'),
    
    # 📷 CAMERA CONTROL ENDPOINTS
    path('camera/connect/', connect_camera, name='connect_camera'),
    path('camera/disconnect/', disconnect_camera, name='disconnect_camera'),
    path('camera/status/', camera_status, name='camera_status'),
    path('camera/frame/', get_current_frame_base64, name='get_frame'),
    path('camera/stream/', video_stream, name='video_stream'),
    path('camera/capture/', simple_camera_capture, name='simple_camera_capture'),  # <- This line already exists
    path('camera/', camera_control_page, name='camera_control'),
    
    # 🎯 TRIGGER MODE ENDPOINTS
    path('camera/trigger/enable/', enable_trigger_mode, name='enable_trigger_mode'),
    path('camera/trigger/disable/', disable_trigger_mode, name='disable_trigger_mode'),
    path('camera/trigger/status/', get_trigger_status, name='get_trigger_status'),
    path('camera/trigger-result-status/<int:trigger_count>/', get_trigger_result_status, name='trigger_result_status'),
    path('camera/reset-session/', reset_trigger_session, name='reset_trigger_session'),
    
    # 🔧 API ENDPOINTS
    # NEW: API endpoint to get overall_result by image_id
    path('overall_result/<str:image_id>/', get_overall_result_by_image_id, name='get_overall_result_by_image_id'),
    
    # 🆕 NEW: Node-RED integration test endpoint
    path('test-node-red/', test_node_red_connection, name='test_node_red'),
    
    # 🆕 NEW: Node-RED receive endpoint (bidirectional communication)
    path('from-node-red/', receive_from_node_red, name='receive_from_node_red'),
    
    # 🆕 NEW: Node-RED get message endpoint
    path('get-node-red-message/', get_node_red_message, name='get_node_red_message'),
    
    # 🆕 NEW: Store image_id in cache for triggered captures
    path('store-image-id/', store_image_id, name='store_image_id'),
    
    # 🆕 NEW: Finalize triggered inspection (move to final location)
    path('finalize-triggered-inspection/', finalize_triggered_inspection, name='finalize_triggered_inspection'),
    
    # 🆕 NEW: Debug inspection files
    path('debug/inspection-files/<str:image_id>/', debug_inspection_files, name='debug_inspection_files'),
    
    # 🆕 NEW: Retry failed image functionality
    path('retry-failed-image/<str:image_id>/', retry_failed_image, name='retry_failed_image'),
    
    # 👥 USER MANAGEMENT ENDPOINTS (ADD THESE)
    path('user-management/list/', user_management_list, name='user_management_list'),
    path('user-management/add/', user_management_add, name='user_management_add'),
    path('user-management/delete/<int:user_id>/', user_management_delete, name='user_management_delete'),
    path('user-management/count/', user_management_count, name='user_management_count'),
    
    path('camera-check/', camera_check_view, name='camera_check'),
    
    # QR Processing Status
    path('qr-processing-status/', get_qr_processing_status, name='qr_processing_status'),
    path('send-fail-to-node-red/', send_fail_result_to_node_red, name='send_fail_to_node_red'),
    
    # 🆕 NEW: Send QR code to Node-RED endpoint
    path('send-qr-to-node-red/', send_qr_to_node_red_endpoint, name='send_qr_to_node_red'),
    
    path('custom_media/<path:path>', serve, {
        'document_root': settings.CUSTOM_IMAGES_BASE_PATH,
    }),
    
    # Add this line to your urlpatterns in urls.py:
    path('debug/custom-path-files/<str:image_id>/', debug_custom_path_files, name='debug_custom_path_files'),


    # 🆕SYSTEM EXPIRY MANAGEMENT ENDPOINTS
    path('system-expiry-status/', system_expiry_status, name='system_expiry_status'),
    path('update-expiry-time/', update_expiry_time, name='update_expiry_time'),
    path('expiry-management/', expiry_management_page, name='expiry_management'),
    path('emergency-extend/', emergency_extend, name='emergency_extend'),

]