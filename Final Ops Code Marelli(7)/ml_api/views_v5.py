# ml_api/views.py - COMPLETE UPDATED VERSION WITH ALL FIXES

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, JSONParser
from PIL import Image
from datetime import datetime
import io
import base64
import json
import logging
import requests
import os
import sys
import threading
import time
import ctypes
import numpy as np
import cv2
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings
from django.shortcuts import render

# Set up logging
logger = logging.getLogger(__name__)

def send_to_node_red(image_id, overall_result, nut_results=None, trigger_count=None):
    """
    Send inspection results to Node-RED system
    """
    try:
        # Node-RED endpoint configuration
        from django.conf import settings
        node_red_url = getattr(settings, 'NODE_RED_ENDPOINT', 'http://localhost:1880/inspection-result')
        timeout = getattr(settings, 'NODE_RED_TIMEOUT', 5)  # 5 seconds timeout
        
        # Prepare payload
        payload = {
            'image_id': image_id,
            'overall_result': overall_result,
            'timestamp': datetime.now().isoformat(),
            'source': 'marelli_ml_api'
        }
        
        # Add optional data
        if nut_results:
            payload['nut_results'] = nut_results
        
        if trigger_count:
            payload['trigger_count'] = trigger_count
            payload['capture_type'] = 'trigger'
        else:
            payload['capture_type'] = 'manual'
        
        # Send HTTP POST request to Node-RED
        response = requests.post(
            node_red_url,
            json=payload,
            timeout=timeout,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code == 200:
            logger.info(f"[NODE-RED] Successfully sent result for {image_id}: {overall_result}")
            return True, "Successfully sent to Node-RED"
        else:
            logger.warning(f"[NODE-RED] HTTP {response.status_code}: {response.text}")
            return False, f"Node-RED responded with status {response.status_code}"
            
    except requests.exceptions.ConnectionError:
        logger.warning(f"[NODE-RED] Connection refused - Node-RED may not be running on {node_red_url}")
        return False, "Node-RED connection refused"
    except requests.exceptions.Timeout:
        logger.warning(f"[NODE-RED] Request timeout after {timeout}s")
        return False, "Node-RED request timeout"
    except Exception as e:
        logger.error(f"[NODE-RED] Error sending to Node-RED: {str(e)}")
        return False, f"Node-RED error: {str(e)}"

# Import the nut detection service
try:
    from .services import enhanced_nut_detection_service
    SERVICE_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import nut detection service: {e}")
    SERVICE_AVAILABLE = False


class NutDetectionView(APIView):
    """
    Industrial Nut Detection API View
    
    Endpoints:
    - POST: Process uploaded image for nut detection
    - GET: Health check
    """
    parser_classes = [MultiPartParser, JSONParser]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize detection service once
        if SERVICE_AVAILABLE:
            try:
                self.detection_service = enhanced_nut_detection_service  # Fixed: removed ()
                logger.info("Nut detection service initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize nut detection service: {e}")
                self.detection_service = None
        else:
            self.detection_service = None
    
    def post(self, request):
        """
        Process uploaded image for nut detection
        
        Expected form data:
        - image: Image file (required)
        - text: Additional text input (optional)
        
        Returns:
        JSON response with nut detection results
        """
        try:
            # Check if service is available
            if self.detection_service is None:
                return Response(
                    {'error': 'Nut detection service not available. Check model file and dependencies.'}, 
                    status=status.HTTP_503_SERVICE_UNAVAILABLE
                )
            
            # Get image and text from request
            image_file = request.FILES.get('image')
            text_input = request.data.get('text', '')
            
            if not image_file:
                return Response(
                    {'error': 'No image provided'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Process the image using PIL then convert for nut detection
            try:
                pil_image = Image.open(image_file)
                filename = image_file.name
                
                # Convert PIL image to file path temporarily for processing
                import tempfile
                import os
                
                # Save PIL image to temporary file
                temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                pil_image.save(temp_file.name, 'JPEG')
                temp_file.close()
                
                try:
                    # Process with the enhanced service
                    result = self.detection_service.process_image_with_id(
                        image_path=temp_file.name,
                        image_id=f"upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        user_id=getattr(request.user, 'id', None) if hasattr(request, 'user') else None
                    )
                    
                    # Add text analysis if provided
                    if text_input:
                        result['text_analysis'] = {
                            'text_input': text_input,
                            'text_length': len(text_input),
                            'text_preview': text_input[:100] + '...' if len(text_input) > 100 else text_input
                        }
                    
                    # Add image metadata
                    result['image_metadata'] = {
                        'filename': filename,
                        'format': pil_image.format,
                        'size': pil_image.size,
                        'mode': pil_image.mode
                    }
                    
                finally:
                    # Clean up temp file
                    try:
                        os.unlink(temp_file.name)
                    except:
                        pass
                
                # Return appropriate HTTP status
                response_status = status.HTTP_200_OK if result.get('success', False) else status.HTTP_400_BAD_REQUEST
                
                return Response(result, status=response_status)
                
            except Exception as e:
                logger.error(f"Image processing error: {e}")
                return Response(
                    {
                        'success': False,
                        'error': f'Image processing failed: {str(e)}',
                        'timestamp': datetime.now().isoformat()
                    }, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
        except Exception as e:
            logger.error(f"API error: {e}")
            return Response(
                {
                    'success': False,
                    'error': 'Internal server error',
                    'timestamp': datetime.now().isoformat()
                }, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    
    def get(self, request):
        """
        Health check endpoint
        
        Returns:
        JSON response with service status
        """
        try:
            # Check if service is available
            service_available = self.detection_service is not None
            model_loaded = False
            
            if service_available:
                # Check if model is actually loaded
                model_loaded = (
                    hasattr(self.detection_service, 'model') and 
                    self.detection_service.model is not None
                )
            
            health_status = {
                'success': True,
                'service': 'Industrial Nut Detection API',
                'status': 'healthy' if model_loaded else 'unhealthy',
                'service_available': service_available,
                'model_loaded': model_loaded,
                'timestamp': datetime.now().isoformat()
            }
            
            if service_available and hasattr(self.detection_service, 'model_path'):
                health_status['model_path'] = self.detection_service.model_path
                
            if service_available and hasattr(self.detection_service, 'config'):
                health_status['config'] = self.detection_service.config
                
            if service_available and hasattr(self.detection_service, 'get_statistics'):
                try:
                    health_status['statistics'] = self.detection_service.get_statistics()
                except Exception as e:
                    logger.warning(f"Could not get statistics: {e}")
            
            if not model_loaded:
                if not service_available:
                    health_status['error'] = 'Service not available - check imports and dependencies'
                else:
                    health_status['error'] = 'Model not loaded - check model file path and format'
            
            return Response(health_status, status=status.HTTP_200_OK)
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return Response({
                'success': False,
                'service': 'Industrial Nut Detection API',
                'status': 'unhealthy',
                'service_available': False,
                'model_loaded': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def options(self, request, *args, **kwargs):
        """
        Handle OPTIONS requests for CORS
        """
        return Response(
            {
                'name': 'Nut Detection',
                'description': 'Industrial Nut Detection API View\n\nEndpoints:\n- POST: Process uploaded image for nut detection\n- GET: Health check',
                'renders': ['application/json', 'text/html'],
                'parses': ['multipart/form-data', 'application/json']
            },
            status=status.HTTP_200_OK
        )


# Alternative Base64 View for backward compatibility
class NutDetectionBase64View(APIView):
    """Base64 image endpoint for nut detection"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if SERVICE_AVAILABLE:
            try:
                self.detection_service = enhanced_nut_detection_service()
            except Exception as e:
                logger.error(f"Failed to initialize nut detection service: {e}")
                self.detection_service = None
        else:
            self.detection_service = None
    
    def post(self, request):
        try:
            if self.detection_service is None:
                return Response(
                    {'error': 'Nut detection service not available'}, 
                    status=status.HTTP_503_SERVICE_UNAVAILABLE
                )
            
            # Get base64 image and text
            image_base64 = request.data.get('image_base64')
            text_input = request.data.get('text', '')
            
            if not image_base64:
                return Response(
                    {'error': 'No image_base64 provided'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            try:
                # Decode base64 image
                image_data = base64.b64decode(image_base64)
                image = Image.open(io.BytesIO(image_data))
                
                # Process with nut detection
                result = self.detection_service.process_image_from_pil(image, "base64_image")
                
                # Add text analysis if provided
                if text_input:
                    result['text_analysis'] = {
                        'text_input': text_input,
                        'text_length': len(text_input)
                    }
                
                response_status = status.HTTP_200_OK if result['success'] else status.HTTP_400_BAD_REQUEST
                return Response(result, status=response_status)
                
            except Exception as e:
                logger.error(f"Base64 processing error: {e}")
                return Response(
                    {'error': f'Base64 processing failed: {str(e)}'}, 
                    status=status.HTTP_400_BAD_REQUEST
                )
            
        except Exception as e:
            logger.error(f"Base64 API error: {e}")
            return Response(
                {'error': 'Internal server error'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


# Keep the old views for backward compatibility (renamed)
class MLPredictionView(APIView):
    """Legacy ML prediction view (redirects to nut detection)"""
    parser_classes = [MultiPartParser, JSONParser]
    
    def post(self, request):
        # Redirect to new nut detection view
        nut_view = NutDetectionView()
        nut_view.setup(request)
        return nut_view.post(request)


class MLPredictionBase64View(APIView):
    """Legacy base64 prediction view (redirects to nut detection)"""
    
    def post(self, request):
        # Redirect to new nut detection base64 view
        nut_view = NutDetectionBase64View()
        nut_view.setup(request)
        return nut_view.post(request)


# HTML page view
from django.shortcuts import render

def detection_page(request):
    """
    Render the HTML page for nut detection
    """
    return render(request, 'ml_api/detection.html')


# ========== CAMERA FUNCTIONALITY FIXED BELOW ==========
# Camera control imports
import sys
import os
import cv2
import numpy as np
import threading
import time
import json
import ctypes
import tempfile
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings

# Add camera drivers to path
camera_driver_path = os.path.join(settings.BASE_DIR, 'MvImport')

print(f"Camera driver path: {camera_driver_path}")
print(f"Path exists: {os.path.exists(camera_driver_path)}")

# Add to Python path
if camera_driver_path not in sys.path:
    sys.path.insert(0, camera_driver_path)
    print(f"Added to sys.path: {camera_driver_path}")

# Simple import exactly like your working code
try:
    from MvImport import *
    from MvImport.MvCameraControl_class import *
    HIKROBOT_AVAILABLE = True
    print("Hikrobot SDK imported successfully")
    
except ImportError as e:
    HIKROBOT_AVAILABLE = False
    print(f"Failed to import Hikrobot SDK: {e}")
    logger.warning(f"Hikrobot SDK not available: {e}")

class HikrobotCameraManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            # Camera objects
            self.camera = None
            self.device_list = None
            self.is_streaming = False
            self.is_connected = False
            self.current_frame = None
            self.frame_lock = threading.Lock()
            
            # Trigger mode functionality
            self.is_trigger_mode = False
            self.is_monitoring_trigger = False
            self.trigger_count = 0
            self.last_trigger_time = None
            self.monitoring_thread = None
            self.stop_monitoring = threading.Event()
            
            # Collection limits
            self.MAX_PROCESSING_STATUSES = 100
            self.MAX_TRIGGERED_FRAMES = 10
            
            # Cleanup tracking
            self.last_cleanup_time = datetime.now()
            self.cleanup_interval = 300  # 5 minutes
            
            # Data structures with locks
            self.trigger_processing_status = {}
            self.processing_lock = threading.Lock()
            self.triggered_frames = []
            self.triggered_frames_lock = threading.Lock()
            self.final_move_queue = {}
            
            self.initialized = True
    
    def connect(self):
        """Connect to the first available Hikrobot camera"""
        try:
            logger.info("[CAMERA] Starting camera connection process...")
            
            if not HIKROBOT_AVAILABLE:
                logger.error("[CAMERA] Hikrobot SDK not available")
                return False, "Hikrobot SDK not available"
            
            if self.is_connected:
                logger.info("[CAMERA] Camera already connected")
                return True, "Already connected"
            
            logger.info("[CAMERA] Creating camera object...")
            # Create camera object
            self.camera = MvCamera()
            
            # Get device list
            self.device_list = MV_CC_DEVICE_INFO_LIST()
            tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE
            
            logger.info("[CAMERA] Enumerating devices...")
            # Enumerate devices
            ret = self.camera.MV_CC_EnumDevices(tlayerType, self.device_list)
            if ret != 0:
                logger.error(f"[CAMERA] Failed to enumerate devices. Error: {ret}")
                return False, f"Failed to enumerate devices. Error: {ret}"
            
            logger.info(f"[CAMERA] Found {self.device_list.nDeviceNum} device(s)")
            if self.device_list.nDeviceNum == 0:
                logger.error("[CAMERA] No Hikrobot cameras found")
                return False, "No Hikrobot cameras found"
            
            # Connect to first camera
            stDeviceList = cast(self.device_list.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
            
            logger.info("[CAMERA] Creating camera handle...")
            # Create handle
            ret = self.camera.MV_CC_CreateHandle(stDeviceList)
            if ret != 0:
                logger.error(f"[CAMERA] Failed to create handle. Error: {ret}")
                return False, f"Failed to create handle. Error: {ret}"
            
            logger.info("[CAMERA] Opening device...")
            # Open device
            ret = self.camera.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
            if ret != 0:
                logger.error(f"[CAMERA] Failed to open device. Error: {ret}")
                return False, f"Failed to open device. Error: {ret}"
            
            # FIXED: Don't start in trigger mode by default, start in free-running mode first
            logger.info("[CAMERA] Starting grabbing in free-running mode...")
            # Ensure trigger mode is OFF first
            ret = self.camera.MV_CC_SetEnumValue("TriggerMode", 0)
            if ret != 0:
                logger.warning(f"Failed to disable trigger mode initially. Error: {ret}")
            
            # Start grabbing in free-running mode
            ret = self.camera.MV_CC_StartGrabbing()
            if ret != 0:
                logger.error(f"[CAMERA] Failed to start grabbing. Error: {ret}")
                return False, f"Failed to start grabbing. Error: {ret}"
            
            self.is_streaming = True
            self.is_connected = True
            self.is_trigger_mode = False  # Start in free-running mode
            logger.info("[CAMERA] Camera connected successfully in free-running mode")
            
            return True, "Camera connected successfully"
            
        except Exception as e:
            logger.error(f"[CAMERA] Connection exception: {str(e)}")
            return False, f"Connection error: {str(e)}"
    
    def disconnect(self):
        """Disconnect from camera"""
        try:
            # Stop trigger monitoring if active
            if self.is_monitoring_trigger:
                self.disable_trigger_mode()
            
            if self.camera and self.is_streaming:
                self.camera.MV_CC_StopGrabbing()
                self.camera.MV_CC_CloseDevice()
                self.camera.MV_CC_DestroyHandle()
            
            self.is_streaming = False
            self.is_connected = False
            self.is_trigger_mode = False
            return True, "Camera disconnected"
            
        except Exception as e:
            return False, f"Disconnect error: {str(e)}"
    
    def enable_trigger_mode(self):
        """Enable trigger mode for Line 0 detection - FIXED VERSION"""
        if not self.is_connected:
            return False, "Camera not connected"
        
        try:
            logger.info("[TRIGGER] Enabling trigger mode...")
            
            # Stop current grabbing
            if self.is_streaming:
                ret = self.camera.MV_CC_StopGrabbing()
                if ret != 0:
                    logger.warning(f"Failed to stop grabbing before trigger mode. Error: {ret}")
            
            # FIXED: Set trigger mode parameters step by step using numeric values
            
            # 1. Set trigger mode ON
            ret = self.camera.MV_CC_SetEnumValue("TriggerMode", 1)
            if ret != 0:
                logger.error(f"[TRIGGER] Failed to set trigger mode ON. Error: {ret}")
                return False, f"Failed to set trigger mode ON. Error: {ret}"
            
            # 2. Set trigger source to Line0 (hardware trigger)
            # Line0 = 0, Line1 = 1, Line2 = 2, Software = 7
            ret = self.camera.MV_CC_SetEnumValue("TriggerSource", 0)  # Line0
            if ret != 0:
                logger.error(f"[TRIGGER] Failed to set trigger source to Line0. Error: {ret}")
                return False, f"Failed to set trigger source to Line0. Error: {ret}"
            
            # 3. Set trigger activation to Rising Edge
            # RisingEdge = 0, FallingEdge = 1, LevelHigh = 2, LevelLow = 3
            ret = self.camera.MV_CC_SetEnumValue("TriggerActivation", 0)  # Rising Edge
            if ret != 0:
                logger.warning(f"Failed to set trigger activation to rising edge. Error: {ret}")
                # Try Level High as alternative
                ret = self.camera.MV_CC_SetEnumValue("TriggerActivation", 2)  # Level High
                if ret != 0:
                    logger.warning(f"Failed to set trigger activation to level high. Error: {ret}")
            
            # 4. Set acquisition mode to continuous
            # SingleFrame = 0, MultiFrame = 1, Continuous = 2
            ret = self.camera.MV_CC_SetEnumValue("AcquisitionMode", 2)  # Continuous
            if ret != 0:
                logger.warning(f"Failed to set acquisition mode to continuous. Error: {ret}")
            
            # 5. OPTIONAL: Set trigger delay (microseconds) - set to 0 for immediate
            ret = self.camera.MV_CC_SetFloatValue("TriggerDelay", 0.0)
            if ret != 0:
                logger.warning(f"Failed to set trigger delay. Error: {ret}")
            
            # 6. OPTIONAL: Enable trigger cache mode for better performance
            ret = self.camera.MV_CC_SetBoolValue("TriggerCacheEnable", True)
            if ret != 0:
                logger.warning(f"Failed to enable trigger cache. Error: {ret}")
            
            # 5. Start grabbing in trigger mode
            ret = self.camera.MV_CC_StartGrabbing()
            if ret != 0:
                logger.error(f"[TRIGGER] Failed to start grabbing in trigger mode. Error: {ret}")
                return False, f"Failed to start grabbing in trigger mode. Error: {ret}"
            
            self.is_trigger_mode = True
            self.is_streaming = True
            
            # Clear any previous triggered frames
            with self.triggered_frames_lock:
                self.triggered_frames.clear()
            
            # Start trigger monitoring thread - FIXED VERSION
            self.stop_monitoring.clear()
            self.monitoring_thread = threading.Thread(target=self._monitor_trigger_signal_fixed, daemon=True)
            self.monitoring_thread.start()
            self.is_monitoring_trigger = True
            
            logger.info("[TRIGGER] Trigger mode enabled successfully")
            
            # ADDED: Verify trigger settings
            self._log_trigger_settings()
            
            return True, "Trigger mode enabled successfully. Camera will capture on Line 0 signal."
            
        except Exception as e:
            logger.error(f"[TRIGGER] Failed to enable trigger mode: {str(e)}")
            return False, f"Failed to enable trigger mode: {str(e)}"
    
    def disable_trigger_mode(self):
        """Disable trigger mode and return to free-running mode"""
        if not self.is_connected:
            return False, "Camera not connected"
        
        try:
            logger.info("[TRIGGER] Disabling trigger mode...")
            
            # Stop trigger monitoring
            if self.is_monitoring_trigger:
                self.stop_monitoring.set()
                if self.monitoring_thread and self.monitoring_thread.is_alive():
                    self.monitoring_thread.join(timeout=2.0)
                self.is_monitoring_trigger = False
            
            # Stop grabbing
            if self.is_streaming:
                ret = self.camera.MV_CC_StopGrabbing()
                if ret != 0:
                    logger.warning(f"Failed to stop grabbing. Error: {ret}")
            
            # Set trigger mode OFF (free-running)
            ret = self.camera.MV_CC_SetEnumValue("TriggerMode", 0)
            if ret != 0:
                return False, f"Failed to set trigger mode OFF. Error: {ret}"
            
            # Start grabbing in free-running mode
            ret = self.camera.MV_CC_StartGrabbing()
            if ret != 0:
                return False, f"Failed to start grabbing in free-running mode. Error: {ret}"
            
            self.is_trigger_mode = False
            self.is_streaming = True
            
            logger.info("[TRIGGER] Trigger mode disabled successfully")
            return True, "Trigger mode disabled. Camera in free-running mode."
            
        except Exception as e:
            logger.error(f"[TRIGGER] Failed to disable trigger mode: {str(e)}")
            return False, f"Failed to disable trigger mode: {str(e)}"
    
    def _monitor_trigger_signal_fixed(self):
        """FIXED: Monitor trigger signal with memory management and performance optimizations"""
        logger.info("[TRIGGER] Started monitoring trigger signal on Line 0 (OPTIMIZED VERSION)")
        
        # Add cleanup interval tracking
        last_cleanup = datetime.now()
        CLEANUP_INTERVAL = 300  # 5 minutes
        MAX_TRIGGERED_FRAMES = 10  # Keep only last 10 frames
        
        while not self.stop_monitoring.is_set() and self.is_monitoring_trigger:
            try:
                if not self.is_connected or not self.is_trigger_mode:
                    break
                    
                # Capture frame
                frame = self._capture_triggered_frame_fixed()
                
                if frame is not None:
                    self.trigger_count += 1
                    self.last_trigger_time = datetime.now()
                    
                    # Keep triggered frames list bounded
                    with self.triggered_frames_lock:
                        self.triggered_frames.append({
                            'frame': frame.copy(),
                            'trigger_count': self.trigger_count,
                            'timestamp': self.last_trigger_time,
                            'processed': False
                        })
                        # Keep only last N frames
                        if len(self.triggered_frames) > MAX_TRIGGERED_FRAMES:
                            self.triggered_frames = self.triggered_frames[-MAX_TRIGGERED_FRAMES:]
                    
                    logger.info(f"[TRIGGER] TRIGGER #{self.trigger_count} DETECTED! Captured at {self.last_trigger_time.strftime('%H:%M:%S')}")
                    
                    # Process triggered image
                    self._save_and_process_triggered_image_enhanced(frame, self.trigger_count, self.last_trigger_time)
                    
                    # Periodic cleanup of old processing statuses
                    current_time = datetime.now()
                    if (current_time - last_cleanup).total_seconds() > CLEANUP_INTERVAL:
                        with self.processing_lock:
                            # Remove completed statuses older than 1 hour
                            old_triggers = []
                            for count, status in self.trigger_processing_status.items():
                                if status.get('complete'):
                                    completed_at = datetime.fromisoformat(status.get('completed_at', current_time.isoformat()))
                                    if (current_time - completed_at).total_seconds() > 3600:  # 1 hour
                                        old_triggers.append(count)
                            
                            # Remove old triggers
                            for count in old_triggers:
                                del self.trigger_processing_status[count]
                        
                        # Update cleanup timestamp
                        last_cleanup = current_time
                        if old_triggers:
                            logger.info(f"[CLEANUP] Removed {len(old_triggers)} old trigger statuses")
                
                # Small delay between checks
                time.sleep(0.001)  # 1ms delay
                    
            except Exception as e:
                logger.error(f"Trigger monitoring error: {e}")
                time.sleep(0.01)  # Longer delay on error
        
        logger.info("[TRIGGER] Stopped monitoring trigger signal")
        
        # Final cleanup
        with self.triggered_frames_lock:
            self.triggered_frames.clear()
    
    def _capture_triggered_frame_fixed(self):
        """FIXED: Capture frame when trigger signal is received"""
        try:
            if not self.is_trigger_mode or not self.is_streaming:
                return None
            
            # Get payload size
            stParam = MVCC_INTVALUE()
            ret = self.camera.MV_CC_GetIntValue("PayloadSize", stParam)
            if ret != 0:
                return None
            
            nPayloadSize = stParam.nCurValue
            
            # Create buffer
            pData = (ctypes.c_ubyte * nPayloadSize)()
            stFrameInfo = MV_FRAME_OUT_INFO_EX()
            
            # FIXED: Get frame with appropriate timeout for trigger mode
            # In trigger mode, this will wait for the trigger signal
            ret = self.camera.MV_CC_GetOneFrameTimeout(pData, nPayloadSize, stFrameInfo, 50)  # 50ms timeout
            if ret != 0:
                return None  # No trigger received within timeout
            
            # FIXED: Proper frame conversion
            frame_width = stFrameInfo.nWidth
            frame_height = stFrameInfo.nHeight
            
            # Convert buffer to numpy array
            np_array = np.frombuffer(pData, dtype=np.uint8)
            
            # Calculate bytes per pixel
            total_bytes = len(np_array)
            expected_size = frame_width * frame_height
            bytes_per_pixel = total_bytes // expected_size if expected_size > 0 else 1
            
            if bytes_per_pixel == 1:
                # Grayscale image
                image = np_array[:expected_size].reshape((frame_height, frame_width))
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif bytes_per_pixel == 3:
                # RGB image
                rgb_size = frame_width * frame_height * 3
                if total_bytes >= rgb_size:
                    image = np_array[:rgb_size].reshape((frame_height, frame_width, 3))
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                else:
                    # Fallback to grayscale
                    image = np_array[:expected_size].reshape((frame_height, frame_width))
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                # Unknown format, fallback to grayscale
                if total_bytes >= expected_size:
                    image = np_array[:expected_size].reshape((frame_height, frame_width))
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                else:
                    logger.error(f"Invalid frame format: {bytes_per_pixel} bytes per pixel")
                    return None
            
            logger.info(f"[CAPTURE] Successfully captured triggered frame: {frame_width}x{frame_height}, {bytes_per_pixel} bpp")
            return image
            
        except Exception as e:
            logger.error(f"Trigger capture error: {e}")
            return None
    
    def _save_and_process_triggered_image_enhanced(self, frame, trigger_count, timestamp):
        """FIXED: Save triggered image using SAME workflow as manual capture"""
        try:
            # Generate image_id using trigger timestamp
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
            temp_image_id = f"triggered_capture_{timestamp_str}"
            
            # FIXED: Mark as captured (not processing) immediately
            with self.processing_lock:
                self.trigger_processing_status[trigger_count] = {
                    'captured': True,        # NEW: Image captured flag
                    'processing': True,      # Processing started
                    'complete': False,       # Processing not complete yet
                    'image_id': temp_image_id,  # Temporary ID
                    'started_at': timestamp.isoformat(),
                    'trigger_timestamp': timestamp_str,
                    'trigger_count': trigger_count
                }
            
            # FIXED: Use CUSTOM directory path
            original_dir = os.path.join(settings.CUSTOM_IMAGES_BASE_PATH, 'temp')
            os.makedirs(original_dir, exist_ok=True)
            
            # FIXED: Use SAME filename format as manual capture
            filename = f"{temp_image_id}_{timestamp_str}_camera.jpg"
            filepath = os.path.join(original_dir, filename)
            
            # Save image with high quality
            success = cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            if success:
                logger.info(f"[CAPTURED] Triggered image saved: {filename} with temp_image_id: {temp_image_id}")
                
                # Update status with file path
                with self.processing_lock:
                    self.trigger_processing_status[trigger_count].update({
                        'filepath': filepath,
                        'filename': filename,
                        'captured_at': timestamp.isoformat()
                    })
                
                # FIXED: Start processing using SAME workflow as manual capture
                processing_thread = threading.Thread(
                    target=self._process_triggered_image_same_as_manual,
                    args=(filepath, temp_image_id, trigger_count, timestamp),
                    daemon=True
                )
                processing_thread.start()
                
                return filepath
            else:
                logger.error(f"[ERROR] Failed to save triggered image: {filename}")
                with self.processing_lock:
                    self.trigger_processing_status[trigger_count].update({
                        'captured': False,
                        'complete': True,
                        'processing': False,
                        'error': 'Failed to save image'
                    })
                return None
                
        except Exception as e:
            logger.error(f"Error saving triggered image: {e}")
            with self.processing_lock:
                if trigger_count in self.trigger_processing_status:
                    self.trigger_processing_status[trigger_count].update({
                        'captured': False,
                        'complete': True,
                        'processing': False,
                        'error': str(e)
                    })
            return None
    
    def _process_triggered_image_same_as_manual(self, image_path, temp_image_id, trigger_count, timestamp):
        """FIXED: Process triggered image using SAME workflow as manual capture"""
        try:
            logger.info(f"[PROCESSING] Starting SAME processing as manual for triggered image: {temp_image_id}")
            
            # FIXED: Import and use the SAME processing function from simple_auth_views
            from .simple_auth_views import _process_with_yolov8_model
            
            # FIXED: Use SAME YOLOv8 processing function as manual capture
            processing_result = _process_with_yolov8_model(image_path, temp_image_id)
            
            if not processing_result['success']:
                logger.error(f"[ERROR] YOLOv8 processing failed for triggered image: {temp_image_id}")
                # 🆕 NEW: Send failure to Node-RED
                send_to_node_red(temp_image_id, 'PROCESSING_FAILED', trigger_count=trigger_count)
                
                with self.processing_lock:
                    self.trigger_processing_status[trigger_count].update({
                        'complete': True,
                        'processing': False,
                        'error': processing_result.get('error', 'YOLOv8 processing failed')
                    })
                return
            
            # Extract results (SAME as manual capture)
            detection_data = processing_result['data']
            nut_results = detection_data['nut_results']
            decision = detection_data['decision']
            
            # Use ML decision for counts (SAME as manual capture)
            present_count = decision['present_count']
            missing_count = decision['missing_count']
            
            logger.info(f"[TRIGGER] ML Results - Present: {present_count}, Missing: {missing_count}")
            
            # FIXED: Create database records (SAME as manual capture)
            from .models import SimpleInspection, CustomUser
            
            # Determine individual nut statuses (SAME logic as manual capture)
            nut_statuses = ['MISSING', 'MISSING', 'MISSING', 'MISSING']
            for nut_key in ['nut1', 'nut2', 'nut3', 'nut4']:
                if nut_key in nut_results and nut_results[nut_key]['status'] == 'PRESENT':
                    nut_index = int(nut_key.replace('nut', '')) - 1
                    nut_statuses[nut_index] = 'PRESENT'
            
            # Get system user for triggered captures
            try:
                system_user = CustomUser.objects.get(username='system')
            except CustomUser.DoesNotExist:
                system_user = CustomUser.objects.first()  # Fallback to first user
            
            # Overall result based on ML decision
            # 🔧 FIXED: Overall result - PASS only if exactly 4 nuts present
            if present_count == 4 and missing_count == 0:
                overall_result = 'PASS'  # Only PASS if all 4 nuts are present
            else:
                overall_result = 'FAIL'   # FAIL if less than 4 present or any missing

            # NEW: Get the user-entered image_id from cache/session
            user_entered_image_id = self._get_user_entered_image_id()
            final_image_id = user_entered_image_id if user_entered_image_id else temp_image_id
            
            logger.info(f"[IMAGE_ID] Using final image_id: {final_image_id} (user entered: {user_entered_image_id}, temp: {temp_image_id})")
            
            # 🆕 NEW: Send result to Node-RED immediately after processing
            # 🔧 FIXED: Only send PASS results during processing, defer FAIL results
            if overall_result == 'PASS':
                # Send PASS results immediately during processing
                node_red_success, node_red_message = send_to_node_red(
                    image_id=final_image_id,
                    overall_result=overall_result,
                    nut_results=nut_results,
                    trigger_count=trigger_count
                )
                
                if node_red_success:
                    logger.info(f"[NODE-RED] ✅ Sent trigger PASS result to Node-RED: {final_image_id} = {overall_result}")
                else:
                    logger.warning(f"[NODE-RED] ❌ Failed to send trigger PASS result to Node-RED: {node_red_message}")
            else:
                # Defer FAIL results - will be sent when "Next Inspection" is clicked
                logger.info(f"[NODE-RED] 🔕 Trigger FAIL result deferred: {final_image_id} = {overall_result}")
                node_red_success, node_red_message = True, "Trigger FAIL result deferred to Next Inspection click"
            
            
            # FIXED: Create SimpleInspection record (SAME as manual capture)
            inspection = SimpleInspection.objects.create(
                user=system_user,
                image_id=final_image_id,  # Use final image_id
                filename=os.path.basename(image_path),
                overall_result=overall_result,
                nut1_status=nut_statuses[0],
                nut2_status=nut_statuses[1],
                nut3_status=nut_statuses[2],
                nut4_status=nut_statuses[3],
                processing_time=detection_data.get('processing_time', 0.0)
            )
            
            # FIXED: Enhanced storage (SAME as manual capture)
            try:
                from .storage_service import EnhancedStorageService
                storage_service = EnhancedStorageService()
                
                # Extract confidence scores for enhanced storage
                confidence_scores = []
                for nut_key in ['nut1', 'nut2', 'nut3', 'nut4']:
                    if nut_key in nut_results:
                        confidence_scores.append(nut_results[nut_key].get('confidence', 0.0))
                
                enhanced_inspection = storage_service.save_inspection_with_images(
                    user=system_user,
                    image_id=final_image_id,  # Use final image_id
                    original_image_path=image_path,
                    annotated_image_path=detection_data.get('annotated_image_path'),
                    nuts_present=present_count,
                    nuts_absent=missing_count,
                    confidence_scores=confidence_scores,
                    processing_time=detection_data.get('processing_time', 0.0)
                )
                
                # NEW: Move inspection directory to triggered_inspections with user-entered image_id
                if enhanced_inspection and user_entered_image_id:
                    logger.info(f"[MOVE] About to move directory - temp_image_id: {temp_image_id}, user_entered_image_id: {user_entered_image_id}")
                    move_success = self._move_inspection_directory(temp_image_id, user_entered_image_id, overall_result)
                    logger.info(f"[MOVE] Directory move result: {move_success}")
                else:
                    if not enhanced_inspection:
                        logger.warning("[MOVE] No enhanced_inspection created - skipping directory move")
                    if not user_entered_image_id:
                        logger.warning("[MOVE] No user_entered_image_id found - skipping directory move")
                        logger.info(f"[MOVE] Will use temporary directory structure with temp_image_id: {temp_image_id}")
                
                # MODIFIED: Process file transfer for BOTH OK and NG
                if enhanced_inspection:
                    logger.info(f"\n🎯 Trigger status is {enhanced_inspection.test_status} - Initiating file transfer...")
                    try:
                        from .file_transfer_service import FileTransferService
                        transfer_service = FileTransferService()
                        success, message, details = transfer_service.process_ok_status_change(enhanced_inspection)
                        if success:
                            logger.info(f"✅ File transfer successful: {message}")
                        else:
                            logger.info(f"❌ File transfer failed: {message}")
                    except Exception as e:
                        logger.info(f"💥 File transfer error: {str(e)}")
                        
            except ImportError:
                logger.info("Enhanced storage service not available, continuing with existing workflow")
            except Exception as e:
                logger.info(f"Enhanced storage error: {e}, continuing with existing workflow")
            
            logger.info(f"[SUCCESS] Triggered image processing completed: {final_image_id}")
            
            # FIXED: Mark processing as complete with proper data
            with self.processing_lock:
                self.trigger_processing_status[trigger_count].update({
                    'processing': False,
                    'complete': True,
                    'completed_at': datetime.now().isoformat(),
                    'detection_result': processing_result,
                    'success': True,
                    'image_id': final_image_id,  # Update with final image_id
                    'inspection_id': str(inspection.id),
                    'nuts_detected': present_count,
                    'overall_result': overall_result,
                    'node_red_sent': node_red_success,  # 🆕 NEW: Track Node-RED status
                    'node_red_message': node_red_message  # 🆕 NEW: Track Node-RED message
                })
            
            logger.info(f"[TRIGGER] Trigger #{trigger_count} processing completed successfully! Image ID: {final_image_id}")
            
        except Exception as e:
            logger.error(f"[ERROR] Error processing triggered image: {e}")
            # 🆕 NEW: Send error to Node-RED
            send_to_node_red(temp_image_id, 'ERROR', trigger_count=trigger_count)
            
            with self.processing_lock:
                self.trigger_processing_status[trigger_count].update({
                    'processing': False,
                    'complete': True,
                    'error': str(e),
                    'completed_at': datetime.now().isoformat(),
                    'image_id': temp_image_id
                })
    
    def _get_user_entered_image_id(self):
        """Get the user-entered image_id from cache or session"""
        try:
            from django.core.cache import cache
            
            # Try to get from cache first (for external requests)
            cache_key = 'current_image_id'
            user_image_id = cache.get(cache_key)
            
            if user_image_id:
                logger.info(f"[IMAGE_ID] Retrieved from cache: {user_image_id}")
                return user_image_id
            
            # Try to get from session storage (fallback)
            # Note: This won't work for triggered captures as they don't have session context
            # The cache approach is better for this use case
            
            logger.warning("[IMAGE_ID] No user-entered image_id found in cache")
            return None
            
        except Exception as e:
            logger.error(f"[IMAGE_ID] Error retrieving user image_id: {e}")
            return None

    def _move_inspection_directory(self, temp_image_id, final_image_id, overall_result):
        """Move inspection directory from temp location to triggered_inspections with final image_id"""
        try:
            import shutil
            
            # Source directory path (current location)
            source_base = os.path.join(settings.MEDIA_ROOT, 'inspections')
            source_dir = os.path.join(source_base, temp_image_id)
            
            # Target directory path (new location) - FIXED: Under inspections/triggered_inspections
            target_base = os.path.join(settings.MEDIA_ROOT, 'inspections', 'triggered_inspections')
            target_dir = os.path.join(target_base, final_image_id)
            
            logger.info(f"[MOVE] Attempting to move directory:")
            logger.info(f"[MOVE]   Source: {source_dir}")
            logger.info(f"[MOVE]   Target: {target_dir}")
            
            # Check if source directory exists
            if not os.path.exists(source_dir):
                logger.warning(f"[MOVE] Source directory does not exist: {source_dir}")
                
                # List what directories DO exist in inspections
                inspections_dir = os.path.join(settings.MEDIA_ROOT, 'inspections')
                if os.path.exists(inspections_dir):
                    existing_dirs = [d for d in os.listdir(inspections_dir) if os.path.isdir(os.path.join(inspections_dir, d))]
                    logger.info(f"[MOVE] Existing directories in inspections: {existing_dirs}")
                    
                    # Try to find a directory that contains the temp_image_id
                    for existing_dir in existing_dirs:
                        if temp_image_id in existing_dir:
                            logger.info(f"[MOVE] Found matching directory: {existing_dir}")
                            source_dir = os.path.join(inspections_dir, existing_dir)
                            break
                
                if not os.path.exists(source_dir):
                    logger.error(f"[MOVE] Still cannot find source directory: {source_dir}")
                    return False
            
            # Create target base directory if it doesn't exist
            os.makedirs(target_base, exist_ok=True)
            logger.info(f"[MOVE] Created target base directory: {target_base}")
            
            # Remove target directory if it already exists
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
                logger.info(f"[MOVE] Removed existing target directory: {target_dir}")
            
            # Move the directory
            shutil.move(source_dir, target_dir)
            
            logger.info(f"[MOVE] ✅ Successfully moved inspection directory:")
            logger.info(f"[MOVE]    FROM: {source_dir}")
            logger.info(f"[MOVE]    TO:   {target_dir}")
            
            # Verify the move by checking source doesn't exist and target does
            source_exists = os.path.exists(source_dir)
            target_exists = os.path.exists(target_dir)
            
            logger.info(f"[MOVE] Post-move verification:")
            logger.info(f"[MOVE]   Source exists: {source_exists}")
            logger.info(f"[MOVE]   Target exists: {target_exists}")
            
            if target_exists and not source_exists:
                # Store the current location for final move
                with self.processing_lock:
                    if hasattr(self, 'final_move_queue'):
                        self.final_move_queue[final_image_id] = {
                            'current_path': target_dir,
                            'overall_result': overall_result,
                            'timestamp': datetime.now().isoformat()
                        }
                    else:
                        self.final_move_queue = {final_image_id: {
                            'current_path': target_dir,
                            'overall_result': overall_result,
                            'timestamp': datetime.now().isoformat()
                        }}
                
                # List contents for verification
                contents = []
                for root, dirs, files in os.walk(target_dir):
                    level = root.replace(target_dir, '').count(os.sep)
                    indent = ' ' * 2 * level
                    contents.append(f"{indent}{os.path.basename(root)}/")
                    subindent = ' ' * 2 * (level + 1)
                    for file in files:
                        contents.append(f"{subindent}{file}")
                
                logger.info(f"[MOVE] Directory structure after move:")
                for item in contents:
                    logger.info(f"[MOVE] {item}")
                
                return True
            else:
                logger.error(f"[MOVE] Move verification failed - source still exists: {source_exists}, target exists: {target_exists}")
                return False
            
        except Exception as e:
            logger.error(f"[MOVE] Error moving inspection directory: {e}")
            logger.error(f"[MOVE] Failed to move from {temp_image_id} to {final_image_id}")
            import traceback
            logger.error(f"[MOVE] Traceback: {traceback.format_exc()}")
            return False

    def move_to_final_triggered_location(self, image_id, overall_result):
        """NEW: Move directory to final triggered_image location with OK/NG structure"""
        try:
            import shutil
            
            # Source directory (current location in triggered_inspections)
            source_dir = os.path.join(settings.MEDIA_ROOT, 'inspections', 'triggered_inspections', image_id)
            
            # Determine OK or NG based on overall_result
            ok_ng_folder = 'OK' if overall_result == 'PASS' else 'NG'
            
            # Target directory (final location)
            target_base = os.path.join(settings.MEDIA_ROOT, 'inspections', 'triggered_image', image_id, ok_ng_folder)
            
            logger.info(f"[FINAL_MOVE] Moving to final triggered location:")
            logger.info(f"[FINAL_MOVE]   Source: {source_dir}")
            logger.info(f"[FINAL_MOVE]   Target: {target_base}")
            logger.info(f"[FINAL_MOVE]   Result: {overall_result} -> {ok_ng_folder}")
            
            if not os.path.exists(source_dir):
                logger.error(f"[FINAL_MOVE] Source directory not found: {source_dir}")
                return False
            
            # Create target directory structure
            os.makedirs(target_base, exist_ok=True)
            
            # Create annotated and original subdirectories
            annotated_dir = os.path.join(target_base, 'annotated')
            original_dir = os.path.join(target_base, 'original')
            os.makedirs(annotated_dir, exist_ok=True)
            os.makedirs(original_dir, exist_ok=True)
            
            # Move files to appropriate directories
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    source_file = os.path.join(root, file)
                    
                    # Determine if file should go to annotated or original
                    if 'annotated' in file.lower() or 'processed' in file.lower() or 'result' in file.lower():
                        target_file = os.path.join(annotated_dir, file)
                    else:
                        target_file = os.path.join(original_dir, file)
                    
                    # Copy file to target location
                    shutil.copy2(source_file, target_file)
                    logger.info(f"[FINAL_MOVE] Copied: {file} -> {'annotated' if 'annotated' in target_file else 'original'}")
            
            # Remove source directory after successful copy
            shutil.rmtree(source_dir)
            logger.info(f"[FINAL_MOVE] ✅ Successfully moved to final location: {target_base}")
            
            # Clean up from queue
            with self.processing_lock:
                if hasattr(self, 'final_move_queue') and image_id in self.final_move_queue:
                    del self.final_move_queue[image_id]
            
            return True
            
        except Exception as e:
            logger.error(f"[FINAL_MOVE] Error moving to final triggered location: {e}")
            import traceback
            logger.error(f"[FINAL_MOVE] Traceback: {traceback.format_exc()}")
            return False

    def get_trigger_processing_status(self, trigger_count):
        """Get processing status for specific trigger"""
        with self.processing_lock:
            return self.trigger_processing_status.get(trigger_count, {
                'processing': False,
                'complete': False,
                'error': 'Trigger not found'
            })
    
    def cleanup_old_data(self):
        """Cleanup old processing data to prevent memory bloat"""
        try:
            current_time = datetime.now()
            
            # Clean triggered frames
            with self.triggered_frames_lock:
                if len(self.triggered_frames) > self.MAX_TRIGGERED_FRAMES:
                    self.triggered_frames = self.triggered_frames[-self.MAX_TRIGGERED_FRAMES:]
            
            # Clean processing statuses
            with self.processing_lock:
                if len(self.trigger_processing_status) > self.MAX_PROCESSING_STATUSES:
                    # Keep only most recent entries
                    sorted_counts = sorted(self.trigger_processing_status.keys())
                    to_remove = sorted_counts[:-self.MAX_PROCESSING_STATUSES]
                    for count in to_remove:
                        del self.trigger_processing_status[count]
            
            # Clean final move queue
            if hasattr(self, 'final_move_queue'):
                old_moves = []
                for image_id, data in self.final_move_queue.items():
                    timestamp = datetime.fromisoformat(data['timestamp'])
                    if (current_time - timestamp).total_seconds() > 3600:  # 1 hour
                        old_moves.append(image_id)
                
                for image_id in old_moves:
                    del self.final_move_queue[image_id]
            
            self.last_cleanup_time = current_time
            
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
            
    def get_trigger_status(self):
        """Get current trigger mode status with recent triggers"""
        # Check if cleanup is needed
        current_time = datetime.now()
        if (current_time - self.last_cleanup_time).total_seconds() > self.cleanup_interval:
            self.cleanup_old_data()
        
        # Get recent triggers (last 10)
        recent_triggers = []
        with self.processing_lock:
            for count in sorted(self.trigger_processing_status.keys())[-10:]:
                status = self.trigger_processing_status[count]
                recent_triggers.append({
                    'trigger_count': count,
                    'processing': status.get('processing', False),
                    'complete': status.get('complete', False),
                    'success': status.get('success', False),
                    'started_at': status.get('started_at'),
                    'completed_at': status.get('completed_at'),
                    'error': status.get('error')
                })
        
        return {
            'is_trigger_mode': self.is_trigger_mode,
            'is_monitoring': self.is_monitoring_trigger,
            'trigger_count': self.trigger_count,
            'last_trigger_time': self.last_trigger_time.isoformat() if self.last_trigger_time else None,
            'recent_triggers': recent_triggers
        }
    
    def get_frame(self):
        """Get current frame from camera - FIXED for both modes"""
        try:
            if not self.is_streaming:
                return None
            
            # FIXED: Handle both trigger and free-running modes
            if self.is_trigger_mode:
                # In trigger mode, return the last triggered frame if available
                with self.triggered_frames_lock:
                    if self.triggered_frames:
                        latest_triggered = self.triggered_frames[-1]
                        return latest_triggered['frame']
                    else:
                        # No triggered frames available, return placeholder
                        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(placeholder, f'Waiting For Trigger... (Count: {self.trigger_count})', 
                                   (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)
                        return placeholder
            
            # Free-running mode - get live frame
            # Get payload size
            stParam = MVCC_INTVALUE()
            ret = self.camera.MV_CC_GetIntValue("PayloadSize", stParam)
            if ret != 0:
                return None
            
            nPayloadSize = stParam.nCurValue
            
            # Create buffer
            pData = (ctypes.c_ubyte * nPayloadSize)()
            stFrameInfo = MV_FRAME_OUT_INFO_EX()
            
            # Get frame with timeout
            ret = self.camera.MV_CC_GetOneFrameTimeout(pData, nPayloadSize, stFrameInfo, 1000)
            if ret != 0:
                return None
            
            # Convert to numpy array
            frame_width = stFrameInfo.nWidth
            frame_height = stFrameInfo.nHeight
            np_array = np.frombuffer(pData, dtype=np.uint8)
            
            bytes_per_pixel = len(np_array) // (frame_width * frame_height)
            
            if bytes_per_pixel == 1:
                # Grayscale
                image = np_array[:frame_width * frame_height].reshape((frame_height, frame_width))
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif bytes_per_pixel == 3:
                # RGB
                image = np_array[:frame_width * frame_height * 3].reshape((frame_height, frame_width, 3))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                # Unknown format, try grayscale
                size = frame_width * frame_height
                if len(np_array) >= size:
                    image = np_array[:size].reshape((frame_height, frame_width))
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                else:
                    return None
            
            with self.frame_lock:
                self.current_frame = image.copy()
            
            return image
            
        except Exception as e:
            logger.error(f"Frame capture error: {e}")
            return None
    
    def capture_image(self, save_path=None):
        """Capture and save a single image - FIXED for manual capture"""
        try:
            # FIXED: In trigger mode, don't try to capture manually
            if self.is_trigger_mode:
                return False, "Cannot manually capture in trigger mode. Disable trigger mode first or wait for trigger signal."
            
            frame = self.get_frame()
            if frame is None:
                return False, "Failed to capture frame"
            
            if save_path is None:
                save_path = os.path.join(settings.CUSTOM_IMAGES_BASE_PATH, 'temp')
                # Create save directory if it doesn't exist
                os.makedirs(save_path, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
            filename = f"manual_capture_{timestamp}.jpg"
            filepath = os.path.join(save_path, filename)
            
            # Save image with high quality
            success = cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            if success:
                file_size = os.path.getsize(filepath)
                return True, {
                    'filename': filename,
                    'filepath': filepath,
                    'size': file_size,
                    'timestamp': timestamp,
                    'capture_type': 'manual'
                }
            else:
                return False, "Failed to save image"
                
        except Exception as e:
            return False, f"Capture error: {str(e)}"
    
    def manual_trigger_test(self):
        """FIXED: Manual trigger test for debugging"""
        if not self.is_trigger_mode:
            return False, "Not in trigger mode"
        
        try:
            logger.info("[TEST] Manual trigger test initiated...")
            
            # Simulate a software trigger
            ret = self.camera.MV_CC_SetCommandValue("TriggerSoftware")
            if ret != 0:
                logger.error(f"Failed to send software trigger. Error: {ret}")
                return False, f"Software trigger failed. Error: {ret}"
            
            logger.info("[TEST] Software trigger sent successfully")
            return True, "Software trigger sent. Check for captured frame."
            
        except Exception as e:
            logger.error(f"Manual trigger test error: {e}")
            return False, f"Manual trigger test error: {str(e)}"
    
    def _log_trigger_settings(self):
        """Log current trigger settings for debugging"""
        try:
            if not self.is_connected:
                return
            
            logger.info("[DEBUG] Current Trigger Settings:")
            
            # Get trigger mode
            trigger_mode = MVCC_ENUMVALUE()
            ret = self.camera.MV_CC_GetEnumValue("TriggerMode", trigger_mode)
            if ret == 0:
                logger.info(f"  - Trigger Mode: {trigger_mode.nCurValue} (0=Off, 1=On)")
            
            # Get trigger source
            trigger_source = MVCC_ENUMVALUE()
            ret = self.camera.MV_CC_GetEnumValue("TriggerSource", trigger_source)
            if ret == 0:
                source_names = {0: "Line0", 1: "Line1", 2: "Line2", 7: "Software"}
                source_name = source_names.get(trigger_source.nCurValue, f"Unknown({trigger_source.nCurValue})")
                logger.info(f"  - Trigger Source: {trigger_source.nCurValue} ({source_name})")
            
            # Get trigger activation
            trigger_activation = MVCC_ENUMVALUE()
            ret = self.camera.MV_CC_GetEnumValue("TriggerActivation", trigger_activation)
            if ret == 0:
                activation_names = {0: "RisingEdge", 1: "FallingEdge", 2: "LevelHigh", 3: "LevelLow"}
                activation_name = activation_names.get(trigger_activation.nCurValue, f"Unknown({trigger_activation.nCurValue})")
                logger.info(f"  - Trigger Activation: {trigger_activation.nCurValue} ({activation_name})")
            
            # Get acquisition mode
            acq_mode = MVCC_ENUMVALUE()
            ret = self.camera.MV_CC_GetEnumValue("AcquisitionMode", acq_mode)
            if ret == 0:
                mode_names = {0: "SingleFrame", 1: "MultiFrame", 2: "Continuous"}
                mode_name = mode_names.get(acq_mode.nCurValue, f"Unknown({acq_mode.nCurValue})")
                logger.info(f"  - Acquisition Mode: {acq_mode.nCurValue} ({mode_name})")
            
            # Get trigger delay
            trigger_delay = MVCC_FLOATVALUE()
            ret = self.camera.MV_CC_GetFloatValue("TriggerDelay", trigger_delay)
            if ret == 0:
                logger.info(f"  - Trigger Delay: {trigger_delay.fCurValue} microseconds")
            
        except Exception as e:
            logger.warning(f"Failed to log trigger settings: {e}")
    
    def get_detailed_camera_info(self):
        """Get detailed camera information for debugging"""
        if not self.is_connected:
            return {"error": "Camera not connected"}
        
        try:
            info = {}
            
            # Device info
            if self.device_list and self.device_list.nDeviceNum > 0:
                device_info = cast(self.device_list.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)).contents
                info['device_type'] = "GigE" if device_info.nTLayerType == MV_GIGE_DEVICE else "USB"
                
                if device_info.nTLayerType == MV_GIGE_DEVICE:
                    gige_info = cast(byref(device_info.SpecialInfo.stGigEInfo), POINTER(MV_GIGE_DEVICE_INFO)).contents
                    info['device_ip'] = f"{(gige_info.nCurrentIp & 0xff000000) >> 24}.{(gige_info.nCurrentIp & 0x00ff0000) >> 16}.{(gige_info.nCurrentIp & 0x0000ff00) >> 8}.{gige_info.nCurrentIp & 0x000000ff}"
                    info['device_model'] = gige_info.chModelName.decode('utf-8') if gige_info.chModelName else "Unknown"
            
            # Current settings
            self._log_trigger_settings()
            
            return info
            
        except Exception as e:
            return {"error": f"Failed to get camera info: {str(e)}"}

    def reset_trigger_system(self):
        """Reset trigger system for new session"""
        try:
            logger.info("[RESET] Resetting trigger system for new session...")
            
            with self.processing_lock:
                # Clear old processing statuses (keep only recent ones)
                current_time = datetime.now()
                old_triggers = []
                
                for trigger_count, status in self.trigger_processing_status.items():
                    if status.get('completed_at'):
                        try:
                            completed_time = datetime.fromisoformat(status['completed_at'])
                            if (current_time - completed_time).total_seconds() > 300:  # 5 minutes old
                                old_triggers.append(trigger_count)
                        except:
                            old_triggers.append(trigger_count)
                
                # Remove old triggers
                for trigger_count in old_triggers:
                    del self.trigger_processing_status[trigger_count]
                    logger.info(f"[RESET] Removed old trigger status: {trigger_count}")
            
            # Clear triggered frames
            with self.triggered_frames_lock:
                self.triggered_frames.clear()
            
            # Clear final move queue
            if hasattr(self, 'final_move_queue'):
                self.final_move_queue.clear()
            
            logger.info(f"[RESET] Trigger system reset completed. Current count: {self.trigger_count}")
            return True
            
        except Exception as e:
            logger.error(f"[RESET] Error resetting trigger system: {e}")
            return False

    def get_new_triggers_since(self, last_known_count):
        """Get only NEW triggers since the last known count"""
        try:
            with self.processing_lock:
                new_triggers = []
                for trigger_count in range(last_known_count + 1, self.trigger_count + 1):
                    if trigger_count in self.trigger_processing_status:
                        status = self.trigger_processing_status[trigger_count]
                        new_triggers.append({
                            'trigger_count': trigger_count,
                            'captured': status.get('captured', False),
                            'processing': status.get('processing', False),
                            'complete': status.get('complete', False),
                            'image_id': status.get('image_id'),
                            'started_at': status.get('started_at')
                        })
                
                return new_triggers
                
        except Exception as e:
            logger.error(f"[TRIGGERS] Error getting new triggers: {e}")
            return []


# Global camera manager instance
camera_manager = HikrobotCameraManager()

from pathlib import Path

# ========== TRIGGER STATUS MONITORING VIEWS ==========

@require_http_methods(["GET"])
def get_trigger_status_view(request):
    """
    Django view wrapper for the existing get_trigger_status method.
    This calls your existing method from the camera_manager instance.
    """
    try:
        # Use the existing global camera_manager instance that's already in your views.py
        status = camera_manager.get_trigger_status()
        
        # Return the status as JSON without any modifications
        return JsonResponse(status)
        
    except Exception as e:
        logger.error(f"Error in get_trigger_status_view: {e}")
        return JsonResponse({
            'error': f'Error getting trigger status: {str(e)}',
            'is_trigger_mode': False,
            'is_monitoring': False,
            'trigger_count': 0,
            'last_trigger_time': None,
            'recent_triggers': []
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def refresh_server(request):
    """
    Endpoint to trigger server refresh when trigger count exceeds limit.
    This will trigger Django's auto-reload by touching manage.py.
    """
    try:
        # Get the path to manage.py
        project_root = Path("C:/Users/manis/Downloads/Final Ops Code/Final Ops Code")
        manage_py_path = project_root / "manage.py"
        
        if not manage_py_path.exists():
            # Try to find manage.py relative to current file
            current_dir = Path(__file__).parent.parent
            manage_py_path = current_dir / "manage.py"
        
        if manage_py_path.exists():
            # Reset the trigger count first
            try:
                # Use the existing global camera_manager
                old_count = camera_manager.trigger_count
                camera_manager.trigger_count = 0
                camera_manager.trigger_processing_status.clear()
                camera_manager.last_trigger_time = None
                
                logger.info(f"[REFRESH] Trigger count reset from {old_count} to 0 before server refresh")
                
            except Exception as e:
                logger.warning(f"[REFRESH] Could not reset trigger count: {e}")
            
            # Trigger Django auto-reload by touching manage.py
            # This mimics what your batch file does: copy /b "manage.py" +,,
            manage_py_path.touch()
            
            # Log the refresh
            logger.info(f"[REFRESH] Django auto-reload triggered via manage.py touch")
            
            # Alternative method - modify the file timestamp
            os.utime(str(manage_py_path), None)
            
            return JsonResponse({
                'success': True,
                'message': 'Django auto-reload triggered successfully',
                'method': 'manage.py touch'
            })
        else:
            return JsonResponse({
                'success': False,
                'error': f'manage.py not found at {manage_py_path}'
            }, status=404)
            
    except Exception as e:
        logger.error(f"[REFRESH] Error in refresh_server: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def trigger_batch_refresh(request):
    """
    Alternative method that sends a signal to the running batch file.
    This creates a trigger file that could be monitored by the batch.
    """
    try:
        # Create a trigger file that could be monitored by the batch
        trigger_file = Path("C:/Users/manis/Downloads/Final Ops Code/Final Ops Code/refresh_trigger.txt")
        
        # Write current timestamp to trigger file
        with open(trigger_file, 'w') as f:
            f.write(f"Refresh triggered at {datetime.now().isoformat()}\n")
            f.write("trigger_count_exceeded\n")
        
        # Also try the manage.py touch method as backup
        project_root = Path("C:/Users/manis/Downloads/Final Ops Code/Final Ops Code")
        manage_py_path = project_root / "manage.py"
        
        if manage_py_path.exists():
            # Touch manage.py to trigger Django auto-reload
            manage_py_path.touch()
            
            # Clean up trigger file after a delay
            def cleanup():
                time.sleep(5)
                if trigger_file.exists():
                    trigger_file.unlink()
            
            cleanup_thread = threading.Thread(target=cleanup)
            cleanup_thread.daemon = True
            cleanup_thread.start()
            
            return JsonResponse({
                'success': True,
                'message': 'Batch refresh triggered via file signal'
            })
        else:
            return JsonResponse({
                'success': False,
                'error': 'Could not locate manage.py for refresh'
            }, status=404)
        
    except Exception as e:
        logger.error(f"[REFRESH] Error in trigger_batch_refresh: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@require_http_methods(["GET"])
def health_check(request):
    """
    Simple health check endpoint to verify server is running.
    Used by frontend to detect when server is back online after refresh.
    Also returns trigger count status.
    """
    try:
        # Use the existing global camera_manager
        trigger_count = camera_manager.trigger_count if camera_manager.is_connected else 0
        server_refreshed = trigger_count == 0  # If count is 0, likely just refreshed
        
    except Exception as e:
        logger.error(f"Error getting trigger count in health check: {e}")
        trigger_count = 0
        server_refreshed = False
    
    return JsonResponse({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'trigger_count': trigger_count,
        'server_refreshed': server_refreshed,
        'camera_connected': camera_manager.is_connected if 'camera_manager' in globals() else False
    })


@csrf_exempt
@require_http_methods(["POST"])
def reset_trigger_count(request):
    """
    Manually reset the trigger count without restarting server.
    Useful for testing and maintenance.
    """
    try:
        # Use the existing global camera_manager
        if not camera_manager.is_connected:
            return JsonResponse({
                'success': False,
                'error': 'Camera not connected'
            }, status=400)
        
        # Store old count for logging
        old_count = camera_manager.trigger_count
        
        # Reset trigger count and related data
        camera_manager.trigger_count = 0
        camera_manager.trigger_processing_status.clear()
        camera_manager.last_trigger_time = None
        
        # If you have additional reset logic in your class, add it here
        if hasattr(camera_manager, 'triggered_frames'):
            with camera_manager.triggered_frames_lock:
                camera_manager.triggered_frames.clear()
        
        logger.info(f"[RESET] Trigger count manually reset from {old_count} to 0")
        
        return JsonResponse({
            'success': True,
            'message': f'Trigger count reset from {old_count} to 0',
            'old_count': old_count,
            'new_count': 0,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"[RESET] Error in reset_trigger_count: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


# FIXED: Add the missing capture_and_process function
@csrf_exempt
@require_http_methods(["POST"])
def capture_and_process(request):
    """Capture image and process it immediately - SAME AS TRIGGER PROCESSING"""
    if not camera_manager.is_connected:
        return JsonResponse({
            'success': False,
            'error': 'Camera not connected'
        })
    
    if camera_manager.is_trigger_mode:
        return JsonResponse({
            'success': False,
            'error': 'Cannot manually capture in trigger mode. Disable trigger mode first.'
        })
    
    try:
        # Get image_id from request
        data = json.loads(request.body) if request.body else {}
        image_id = data.get('image_id', f"manual_capture_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}")
        
        logger.info(f"[PROCESSING] Starting manual capture and process for image_id: {image_id}")
        
        # Capture frame
        frame = camera_manager.get_frame()
        if frame is None:
            return JsonResponse({
                'success': False,
                'error': 'Failed to capture frame from camera'
            })
        
        # Save image - CHANGED TO USE CUSTOM PATH  
        save_path = os.path.join(settings.CUSTOM_IMAGES_BASE_PATH, 'temp')
        os.makedirs(save_path, exist_ok=True)
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"manual_capture_{timestamp_str}.jpg"
        filepath = os.path.join(save_path, filename)
        
        # Save with high quality
        success = cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        if not success:
            return JsonResponse({
                'success': False,
                'error': 'Failed to save captured image'
            })
        
        logger.info(f"[SAVED] Manual image saved: {filename}")
        
        # Process using the SAME service as trigger mode
        if not SERVICE_AVAILABLE:
            return JsonResponse({
                'success': False,
                'error': 'Detection service not available'
            })
        
        # SAME PROCESSING AS TRIGGER MODE
        result = enhanced_nut_detection_service.process_image_with_id(
            image_path=filepath,
            image_id=image_id,
            user_id=getattr(request.user, 'id', None) if hasattr(request, 'user') else None
        )
        
        # Save processing results - SAME AS TRIGGER MODE
        results_path = os.path.join(settings.MEDIA_ROOT, 'camera_captures', 'processed')
        os.makedirs(results_path, exist_ok=True)
        
        result_filename = f"result_manual_{timestamp_str}.json"
        result_filepath = os.path.join(results_path, result_filename)
        
        # Add manual capture metadata (similar to trigger metadata)
        result['capture_metadata'] = {
            'capture_type': 'manual',
            'capture_time': datetime.now().isoformat(),
            'image_path': filepath,
            'image_id': image_id,
            'processing_time': datetime.now().isoformat()
        }
        
        with open(result_filepath, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        logger.info(f"[SUCCESS] Manual capture processed successfully: {result_filename}")
        
        return JsonResponse({
            'success': True,
            'message': 'Image captured and processed successfully',
            'image_id': image_id,
            'result': result,
            'result_path': result_filepath
        })
        
    except Exception as e:
        logger.error(f"[ERROR] Manual capture and process error: {e}")
        return JsonResponse({
            'success': False,
            'error': f'Processing failed: {str(e)}'
        })


# Camera API Views - FIXED VERSIONS
@csrf_exempt
@require_http_methods(["POST"])
def connect_camera(request):
    """Connect to Hikrobot camera"""
    success, message = camera_manager.connect()
    return JsonResponse({
        'success': success,
        'message': message,
        'connected': camera_manager.is_connected,
        'trigger_mode': camera_manager.is_trigger_mode
    })


@csrf_exempt
@require_http_methods(["POST"])
def disconnect_camera(request):
    """Disconnect from camera"""
    success, message = camera_manager.disconnect()
    return JsonResponse({
        'success': success,
        'message': message,
        'connected': camera_manager.is_connected
    })


@csrf_exempt
@require_http_methods(["GET"])
def camera_status(request):
    """Get camera connection status including trigger mode"""
    status_data = {
        'connected': camera_manager.is_connected,
        'streaming': camera_manager.is_streaming,
        'sdk_available': HIKROBOT_AVAILABLE
    }
    
    # Add trigger mode status if camera is connected
    if camera_manager.is_connected:
        trigger_status = camera_manager.get_trigger_status()
        status_data.update(trigger_status)
    
    return JsonResponse(status_data)


def generate_frames():
    """Generate frames for video streaming - FIXED"""
    while True:
        if camera_manager.is_connected and camera_manager.is_streaming:
            frame = camera_manager.get_frame()
            if frame is not None:
                # FIXED: Add overlay information
                if camera_manager.is_trigger_mode:
                    # Add trigger mode overlay
                    cv2.putText(frame, f'TRIGGER MODE - Count: {camera_manager.trigger_count}', 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    if camera_manager.last_trigger_time:
                        time_str = camera_manager.last_trigger_time.strftime('%H:%M:%S')
                        cv2.putText(frame, f'Last Trigger: {time_str}', 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                else:
                    # Add free-running mode overlay
                    cv2.putText(frame, 'FREE-RUNNING MODE', 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                # Send placeholder frame if no camera frame
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                if camera_manager.is_trigger_mode:
                    cv2.putText(placeholder, f'Waiting For Trigger... (Count: {camera_manager.trigger_count})', 
                               (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)
                else:
                    cv2.putText(placeholder, 'No Frame Available', (200, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                _, buffer = cv2.imencode('.jpg', placeholder)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # Send "disconnected" frame
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, 'Camera Disconnected', (180, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            _, buffer = cv2.imencode('.jpg', placeholder)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.033)  # ~30 FPS


def video_stream(request):
    """Stream video from camera"""
    return StreamingHttpResponse(
        generate_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


@csrf_exempt
@require_http_methods(["GET"])
def get_current_frame_base64(request):
    """Get current frame as base64 encoded image"""
    if not camera_manager.is_connected:
        return JsonResponse({
            'success': False,
            'message': 'Camera not connected'
        })
    
    try:
        frame = camera_manager.get_frame()
        if frame is not None:
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return JsonResponse({
                'success': True,
                'image': f'data:image/jpeg;base64,{img_base64}',
                'timestamp': datetime.now().isoformat(),
                'trigger_mode': camera_manager.is_trigger_mode,
                'trigger_count': camera_manager.trigger_count
            })
        else:
            return JsonResponse({
                'success': False,
                'message': 'No frame available'
            })
            
    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': f'Error: {str(e)}'
        })


# FIXED: Trigger Mode API Endpoints
@csrf_exempt
@require_http_methods(["POST"])
def enable_trigger_mode(request):
    """Enable trigger mode for Line 0 detection - FIXED"""
    if not camera_manager.is_connected:
        return JsonResponse({
            'success': False,
            'message': 'Camera not connected'
        })
    
    success, message = camera_manager.enable_trigger_mode()
    status_data = camera_manager.get_trigger_status()
    
    return JsonResponse({
        'success': success,
        'message': message,
        'trigger_status': status_data
    })


@csrf_exempt
@require_http_methods(["POST"])
def disable_trigger_mode(request):
    """Disable trigger mode and return to free-running"""
    if not camera_manager.is_connected:
        return JsonResponse({
            'success': False,
            'message': 'Camera not connected'
        })
    
    success, message = camera_manager.disable_trigger_mode()
    status_data = camera_manager.get_trigger_status()
    
    return JsonResponse({
        'success': success,
        'message': message,
        'trigger_status': status_data
    })


@csrf_exempt
@require_http_methods(["GET"])
def get_trigger_status(request):
    """Get trigger mode status with recent triggers"""
    if not camera_manager.is_connected:
        return JsonResponse({
            'success': False,
            'message': 'Camera not connected'
        })
    
    status_data = camera_manager.get_trigger_status()
    return JsonResponse({
        'success': True,
        'trigger_status': status_data
    })


# CORRECTED: Fixed trigger result status endpoint with proper image_id handling
@csrf_exempt
@require_http_methods(["GET"])
def get_trigger_result_status(request, trigger_count):
    """ENHANCED: Get processing status for specific trigger including capture status"""
    if not camera_manager.is_connected:
        return JsonResponse({
            'success': False,
            'message': 'Camera not connected'
        })
    
    try:
        trigger_count = int(trigger_count)
        
        with camera_manager.processing_lock:
            # Check if this trigger exists
            if trigger_count not in camera_manager.trigger_processing_status:
                return JsonResponse({
                    'success': True,
                    'trigger_count': trigger_count,
                    'captured': False,
                    'processing': False,
                    'processing_complete': False,
                    'image_id': None
                })
            
            status = camera_manager.trigger_processing_status[trigger_count]
            
            return JsonResponse({
                'success': True,
                'trigger_count': trigger_count,
                'captured': status.get('captured', False),          # NEW: Image captured
                'processing': status.get('processing', False),      # Processing in progress
                'processing_complete': status.get('complete', False), # Processing complete
                'image_id': status.get('image_id'),
                'error': status.get('error'),
                'started_at': status.get('started_at'),
                'completed_at': status.get('completed_at'),
                'detection_result': status.get('detection_result'),
                'detection_success': status.get('success', False),
                'nuts_detected': status.get('nuts_detected'),
                'overall_result': status.get('overall_result')
            })
            
    except ValueError:
        return JsonResponse({
            'success': False,
            'message': 'Invalid trigger count'
        })
    except Exception as e:
        logger.error(f"[ERROR] Error getting trigger status: {e}")
        return JsonResponse({
            'success': False,
            'message': f'Error: {str(e)}'
        })


# NEW: Manual trigger test endpoint for debugging
@csrf_exempt
@require_http_methods(["POST"])
def manual_trigger_test(request):
    """Manual trigger test for debugging"""
    if not camera_manager.is_connected:
        return JsonResponse({
            'success': False,
            'message': 'Camera not connected'
        })
    
    success, message = camera_manager.manual_trigger_test()
    return JsonResponse({
        'success': success,
        'message': message,
        'trigger_count': camera_manager.trigger_count
    })


# NEW: Camera info endpoint for debugging
@csrf_exempt
@require_http_methods(["GET"])
def get_camera_info(request):
    """Get detailed camera information for debugging"""
    if not camera_manager.is_connected:
        return JsonResponse({
            'success': False,
            'message': 'Camera not connected'
        })
    
    camera_info = camera_manager.get_detailed_camera_info()
    return JsonResponse({
        'success': True,
        'camera_info': camera_info
    })


# NEW: API endpoint to get overall_result by image_id
@csrf_exempt
@require_http_methods(["GET"])
def get_overall_result_by_image_id(request, image_id):
    """
    API endpoint to get overall_result and nut statuses for a given image_id.
    Example: GET /api/overall_result/<image_id>/
    """
    try:
        from .models import SimpleInspection
        inspection = SimpleInspection.objects.filter(image_id=image_id).order_by('-created_at').first()
        if not inspection:
            return JsonResponse({
                'success': False,
                'message': f'No inspection found for image_id: {image_id}'
            }, status=404)
        
        return JsonResponse({
            'success': True,
            'image_id': inspection.image_id,
            'overall_result': inspection.overall_result,
            'nut1_status': inspection.nut1_status,
            'nut2_status': inspection.nut2_status,
            'nut3_status': inspection.nut3_status,
            'nut4_status': inspection.nut4_status,
            'timestamp': inspection.created_at.isoformat() if hasattr(inspection, 'created_at') else None,
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': f'Error: {str(e)}'
        }, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def test_node_red_connection(request):
    """Test Node-RED connection"""
    try:
        success, message = send_to_node_red(
            image_id="TEST_CONNECTION",
            overall_result="TEST",
            nut_results={"test": "connection"}
        )
        
        return JsonResponse({
            'success': success,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })


@csrf_exempt
@require_http_methods(["POST"])
def receive_from_node_red(request):
    """Enhanced endpoint to receive messages from Node-RED and trigger form submission"""
    try:
        # Handle both JSON and form data
        if request.content_type == 'application/json':
            data = json.loads(request.body)
        else:
            data = request.POST.dict()
            
        message_id = data.get('id', 'unknown')
        message_content = data.get('message', '')
        
        logger.info(f"[NODE-RED->DJANGO] Received: ID={message_id}, Message={message_content}")
        
        # For external requests (like Node-RED), use cache instead of session
        from django.core.cache import cache
        
        # Store the message in cache with a short TTL (30 seconds)
        cache_key = f'node_red_message'
        cache.set(cache_key, {
            'content': message_content,
            'id': message_id,
            'timestamp': datetime.now().isoformat()
        }, timeout=30)  # 30 seconds timeout
        
        # Also try to store in session if available
        if hasattr(request, 'session'):
            request.session['node_red_message'] = {
                'content': message_content,
                'id': message_id,
                'timestamp': datetime.now().isoformat()
            }
        
        # Simple response
        return JsonResponse({
            'success': True,
            'received_id': message_id,
            'received_message': message_content,
            'reply': f'Message received and stored: {message_content}',
            'timestamp': datetime.now().isoformat(),
            'stored_in': 'cache_and_session'
        })
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}")
        return JsonResponse({
            'success': False,
            'error': f'Invalid JSON: {str(e)}'
        }, status=400)
    except Exception as e:
        logger.error(f"Error in receive_from_node_red: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def get_node_red_message(request):
    """Get stored Node-RED message for frontend"""
    try:
        from django.core.cache import cache
        
        # Try to get from cache first (for external requests)
        cache_key = f'node_red_message'
        message_data = cache.get(cache_key)
        
        if message_data:
            # Clear the message after retrieving it
            cache.delete(cache_key)
            
            return JsonResponse({
                'success': True,
                'has_message': True,
                'message_content': message_data['content'],
                'message_id': message_data['id'],
                'timestamp': message_data['timestamp'],
                'source': 'cache'
            })
        
        # Fallback to session (for same-domain requests)
        if hasattr(request, 'session') and 'node_red_message' in request.session:
            message_data = request.session['node_red_message']
            # Clear the message after retrieving it
            del request.session['node_red_message']
            
            return JsonResponse({
                'success': True,
                'has_message': True,
                'message_content': message_data['content'],
                'message_id': message_data['id'],
                'timestamp': message_data['timestamp'],
                'source': 'session'
            })
        
        return JsonResponse({
            'success': True,
            'has_message': False,
            'message_content': None,
            'source': 'none'
        })
            
    except Exception as e:
        logger.error(f"Error in get_node_red_message: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        })


@csrf_exempt
@require_http_methods(["POST"])
def store_image_id(request):
    """Store user-entered image_id in cache for triggered captures"""
    try:
        import json
        data = json.loads(request.body)
        image_id = data.get('image_id')
        
        if not image_id:
            return JsonResponse({
                'success': False,
                'error': 'No image_id provided'
            })
        
        from django.core.cache import cache
        
        # Store in cache with 1 hour timeout
        cache_key = 'current_image_id'
        cache.set(cache_key, image_id, timeout=3600)  # 1 hour
        
        logger.info(f"[CACHE] Stored image_id in cache: {image_id}")
        
        return JsonResponse({
            'success': True,
            'message': f'Image ID {image_id} stored in cache',
            'image_id': image_id
        })
        
    except Exception as e:
        logger.error(f"Error storing image_id in cache: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        })


@csrf_exempt
@require_http_methods(["POST"])
def finalize_triggered_inspection(request):
    """NEW: Finalize triggered inspection by moving to final location"""
    try:
        import json
        data = json.loads(request.body)
        image_id = data.get('image_id')
        overall_result = data.get('overall_result')
        
        if not image_id:
            return JsonResponse({
                'success': False,
                'error': 'No image_id provided'
            })
        
        if not overall_result:
            # Try to get result from database
            from .models import SimpleInspection
            try:
                inspection = SimpleInspection.objects.filter(image_id=image_id).order_by('-created_at').first()
                if inspection:
                    overall_result = inspection.overall_result
                    # Convert to PASS/FAIL format if needed
                    if overall_result == 'OK':
                        overall_result = 'PASS'
                    elif overall_result == 'NG':
                        overall_result = 'FAIL'
                else:
                    return JsonResponse({
                        'success': False,
                        'error': 'No inspection found for image_id and no overall_result provided'
                    })
            except Exception as e:
                return JsonResponse({
                    'success': False,
                    'error': f'Error retrieving inspection: {str(e)}'
                })
        
        # Perform the final move
        success = camera_manager.move_to_final_triggered_location(image_id, overall_result)
        
        if success:
            logger.info(f"[FINALIZE] Successfully finalized triggered inspection: {image_id} -> {overall_result}")
            return JsonResponse({
                'success': True,
                'message': f'Inspection {image_id} moved to final location',
                'image_id': image_id,
                'overall_result': overall_result,
                'final_location': f'triggered_image/{image_id}/{"OK" if overall_result == "PASS" else "NG"}/'
            })
        else:
            return JsonResponse({
                'success': False,
                'error': 'Failed to move inspection to final location'
            })
        
    except Exception as e:
        logger.error(f"[FINALIZE] Error finalizing triggered inspection: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        })


@csrf_exempt
@require_http_methods(["POST"])
def reset_trigger_session(request):
    """Reset trigger system for new capture session"""
    if not camera_manager.is_connected:
        return JsonResponse({
            'success': False,
            'message': 'Camera not connected'
        })
    
    try:
        # Reset trigger system
        reset_success = camera_manager.reset_trigger_system()
        
        # Get current status
        status_data = camera_manager.get_trigger_status()
        
        # Clear cache
        from django.core.cache import cache
        cache.delete('current_image_id')
        
        logger.info(f"[API] Trigger session reset - Success: {reset_success}")
        
        return JsonResponse({
            'success': reset_success,
            'message': 'Trigger session reset successfully' if reset_success else 'Failed to reset trigger session',
            'current_trigger_count': camera_manager.trigger_count,
            'trigger_status': status_data
        })
        
    except Exception as e:
        logger.error(f"[API] Error resetting trigger session: {e}")
        return JsonResponse({
            'success': False,
            'message': f'Reset failed: {str(e)}'
        })


@csrf_exempt
@require_http_methods(["GET"])
def debug_inspection_files(request, image_id):
    """Debug endpoint to check what files exist for an image_id"""
    try:
        import os
        from django.conf import settings
        
        files_found = {}
        
        # Check all possible locations
        locations_to_check = [
            ('inspections/original', 'Original Images'),
            ('inspections/results', 'Results'),
            ('inspections/annotated', 'Annotated'),
            (f'inspections/{image_id}', f'Image ID Directory ({image_id})'),
            (f'inspections/{image_id}/original', f'{image_id}/Original'),
            (f'inspections/{image_id}/annotated', f'{image_id}/Annotated'),
            (f'inspections/{image_id}/results', f'{image_id}/Results'),
            (f'inspections/triggered_inspections/{image_id}', f'Triggered/{image_id}'),
            (f'inspections/triggered_image/{image_id}', f'Final Triggered/{image_id}'),
        ]
        
        for location, description in locations_to_check:
            full_path = os.path.join(settings.MEDIA_ROOT, location)
            if os.path.exists(full_path):
                files_in_location = []
                for root, dirs, files in os.walk(full_path):
                    for file in files:
                        rel_path = os.path.relpath(os.path.join(root, file), full_path)
                        files_in_location.append(rel_path)
                
                if files_in_location:
                    files_found[description] = files_in_location
        
        # Check database records
        try:
            from .models import SimpleInspection
            inspection = SimpleInspection.objects.filter(image_id=image_id).first()
            db_info = {
                'found': bool(inspection),
                'filename': inspection.filename if inspection else None,
                'overall_result': inspection.overall_result if inspection else None,
                'created_at': inspection.created_at.isoformat() if inspection else None
            }
        except Exception as e:
            db_info = {'error': str(e)}
        
        return JsonResponse({
            'success': True,
            'image_id': image_id,
            'files_found': files_found,
            'database_info': db_info,
            'media_root': settings.MEDIA_ROOT
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })


def camera_control_page(request):
    """
    Render the HTML page for camera control
    """
    return render(request, 'ml_api/camera_control.html')


@csrf_exempt
@require_http_methods(["POST"])
def send_fail_result_to_node_red(request):
    """
    Send FAIL result to Node-RED when Next Inspection is clicked
    """
    try:
        data = json.loads(request.body)
        image_id = data.get('image_id')
        overall_result = data.get('overall_result')
        
        if not image_id:
            return JsonResponse({
                'success': False,
                'error': 'Invalid request - image_id required'
            })

        # Accept both FAIL and NG (since your system uses both)
        if overall_result not in ['FAIL', 'NG']:
            return JsonResponse({
                'success': False,
                'error': 'Invalid request - expected FAIL or NG result'
            })
        
        # Send FAIL result to Node-RED
        # Send NG/FAIL result to Node-RED
        logger.info(f"[NODE-RED] Sending NG result via Next Inspection click: {image_id} = {overall_result}")

        success, message = send_to_node_red(
            image_id=image_id,
            overall_result=overall_result,
            nut_results=None
        )

        if success:
            logger.info(f"[NODE-RED] ✅ Successfully sent NG result to Node-RED: {image_id}")
        else:
            logger.warning(f"[NODE-RED] ❌ Failed to send NG result to Node-RED: {message}")
        
        # 🆕 NEW: Print statements to monitor Node-RED message sending
        print(f"[NODE-RED MONITOR] ========================================")
        print(f"[NODE-RED MONITOR] Next Inspection Click - Image ID: {image_id}")
        print(f"[NODE-RED MONITOR] Overall Result: {overall_result}")
        print(f"[NODE-RED MONITOR] Node-RED Success: {success}")
        print(f"[NODE-RED MONITOR] Node-RED Message: {message}")
        if success:
            print(f"[NODE-RED MONITOR] ✅ Next Inspection Node-RED message sent successfully!")
        else:
            print(f"[NODE-RED MONITOR] ❌ Next Inspection Node-RED message failed to send!")
        print(f"[NODE-RED MONITOR] ========================================")

        return JsonResponse({
            'success': success,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })
    



