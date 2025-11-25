# ml_api/external_server_client.py - HTTP POST client for sending .nip files to external server

import requests
import json
import time
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ExternalServerClient:
    """
    HTTP POST client for sending .nip files to external server
    """
    
    # 🔧 CONFIGURABLE SERVER SETTINGS - Change these as needed
    DEFAULT_SERVER_IP = "192.168.0.104"
    DEFAULT_SERVER_PORT = 8080
    DEFAULT_ENDPOINT = "/receive_nip"
    DEFAULT_TIMEOUT = 5
    
    def __init__(self, server_ip=None, server_port=None, endpoint=None):
        """
        Initialize external server client
        
        Args:
            server_ip: IP address of external server (default: 192.168.0.104)
            server_port: Port of external server (default: 8080)
            endpoint: API endpoint (default: /receive_nip)
        """
        self.server_ip = "192.168.0.109"  
        self.server_port = 8888
        self.endpoint = endpoint or self.DEFAULT_ENDPOINT
        
        # Build server URL
        self.server_url = f"http://{self.server_ip}:{self.server_port}{self.endpoint}"
        
        # Retry configuration
        self.max_retries = 1
        self.retry_delays = [0]  # seconds between retries
        
        print(f"🌐 External Server Client Initialized:")
        print(f"   - Server URL: {self.server_url}")
        print(f"   - Max Retries: {self.max_retries}")
        print(f"   - Retry Delays: {self.retry_delays} seconds")
        
        logger.info(f"ExternalServerClient initialized for {self.server_url}")
    
    def send_nip_file(self, file_path, qr_code):
        """
        Send .nip file to external server via HTTP POST
        
        Args:
            file_path: Path to .nip file
            qr_code: QR code identifier
        
        Returns:
            tuple: (success, response_data, message)
        """
        try:
            print(f"\n🚀 Attempting to send .nip file:")
            print(f"   - File: {Path(file_path).name}")
            print(f"   - QR Code: {qr_code}")
            print(f"   - Target: {self.server_url}")
            
            # Read .nip file content
            with open(file_path, 'r', encoding='utf-8') as f:
                nip_content = json.load(f)
            
            # Prepare request payload
            payload = {
                'qr_code': qr_code,
                'nip_data': nip_content,
                'timestamp': datetime.now().isoformat(),
                'source': 'Industrial_Nut_Detection_System'
            }
            
            # Prepare headers
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'NutDetection-Client/1.0',
                'X-QR-Code': qr_code
            }
            
            print(f"📦 Payload prepared:")
            print(f"   - QR Code: {payload['qr_code']}")
            print(f"   - Overall Status: {nip_content['inspection_results']['overall_status']}")
            print(f"   - Timestamp: {payload['timestamp']}")
            
            # Send HTTP POST request
            start_time = time.time()
            
            response = requests.post(
                self.server_url,
                json=payload,
                headers=headers,
                timeout=self.DEFAULT_TIMEOUT
            )
            
            request_time = time.time() - start_time
            
            print(f"📡 HTTP Request completed:")
            print(f"   - Status Code: {response.status_code}")
            print(f"   - Response Time: {request_time:.2f} seconds")
            
            # Handle response
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    print(f"✅ Server Response: {response_data}")
                    
                    # Check if response matches expected format
                    if (response_data.get('status') == 'received' and 
                        response_data.get('code') == 200):
                        
                        success_msg = f"Successfully sent {qr_code}.nip to external server"
                        print(f"🎯 {success_msg}")
                        logger.info(f"Successfully sent .nip file: {qr_code}")
                        
                        return True, response_data, success_msg
                    else:
                        # Unexpected response format
                        error_msg = f"Unexpected response format: {response_data}"
                        print(f"⚠️ {error_msg}")
                        logger.warning(error_msg)
                        
                        return False, response_data, error_msg
                        
                except json.JSONDecodeError:
                    # Response is not JSON
                    error_msg = f"Invalid JSON response: {response.text}"
                    print(f"❌ {error_msg}")
                    logger.error(error_msg)
                    
                    return False, {'raw_response': response.text}, error_msg
            else:
                # HTTP error status
                error_msg = f"HTTP {response.status_code}: {response.text}"
                print(f"❌ {error_msg}")
                logger.error(error_msg)
                
                return False, {'status_code': response.status_code, 'error': response.text}, error_msg
                
        except requests.exceptions.Timeout:
            error_msg = f"Request timeout after {self.DEFAULT_TIMEOUT} seconds"
            print(f"⏰ {error_msg}")
            logger.error(error_msg)
            return False, None, error_msg
            
        except requests.exceptions.ConnectionError:
            error_msg = f"Connection error - server unreachable at {self.server_url}"
            print(f"🔌 {error_msg}")
            logger.error(error_msg)
            return False, None, error_msg
            
        except Exception as e:
            error_msg = f"Unexpected error sending .nip file: {str(e)}"
            print(f"💥 {error_msg}")
            logger.error(error_msg)
            return False, None, error_msg
    
    def send_with_retry(self, file_path, qr_code):
        """
        Send .nip file with automatic retry logic
        
        Args:
            file_path: Path to .nip file
            qr_code: QR code identifier
        
        Returns:
            tuple: (success, response_data, message, attempt_count)
        """
        print(f"\n🔄 Starting send with retry for QR: {qr_code}")
        
        for attempt in range(1, self.max_retries + 1):
            print(f"\n📋 Attempt {attempt}/{self.max_retries}")
            
            success, response_data, message = self.send_nip_file(file_path, qr_code)
            
            if success:
                print(f"✅ Success on attempt {attempt}")
                return True, response_data, message, attempt
            else:
                print(f"❌ Attempt {attempt} failed: {message}")
                
                # If not the last attempt, wait before retrying
                if attempt < self.max_retries:
                    delay = self.retry_delays[attempt - 1]
                    print(f"⏳ Waiting {delay} seconds before retry...")
                    time.sleep(delay)
                else:
                    final_msg = f"All {self.max_retries} attempts failed. Final error: {message}"
                    print(f"💀 {final_msg}")
                    logger.error(f"Failed to send {qr_code} after {self.max_retries} attempts")
                    return False, response_data, final_msg, attempt
    
    def test_connection(self):
        """
        Test connection to external server
        
        Returns:
            tuple: (success, message)
        """
        try:
            print(f"\n🔍 Testing connection to external server:")
            print(f"   - URL: {self.server_url}")
            
            # Simple GET request to test connectivity
            test_url = f"http://{self.server_ip}:{self.server_port}/"
            
            response = requests.get(test_url, timeout=10)
            
            print(f"✅ Connection test successful:")
            print(f"   - Status Code: {response.status_code}")
            print(f"   - Server appears to be reachable")
            
            return True, f"Server reachable at {self.server_ip}:{self.server_port}"
            
        except requests.exceptions.ConnectionError:
            error_msg = f"Connection failed - server unreachable at {self.server_ip}:{self.server_port}"
            print(f"🔌 {error_msg}")
            return False, error_msg
            
        except requests.exceptions.Timeout:
            error_msg = f"Connection timeout - server may be slow at {self.server_ip}:{self.server_port}"
            print(f"⏰ {error_msg}")
            return False, error_msg
            
        except Exception as e:
            error_msg = f"Connection test failed: {str(e)}"
            print(f"💥 {error_msg}")
            return False, error_msg
    
    def get_server_info(self):
        """Get current server configuration"""
        info = {
            'server_ip': self.server_ip,
            'server_port': self.server_port,
            'endpoint': self.endpoint,
            'full_url': self.server_url,
            'max_retries': self.max_retries,
            'retry_delays': self.retry_delays,
            'timeout': self.DEFAULT_TIMEOUT
        }
        
        print(f"ℹ️ Server Configuration:")
        for key, value in info.items():
            print(f"   - {key}: {value}")
        
        return info
    
    def update_server_settings(self, server_ip=None, server_port=None, endpoint=None):
        """
        Update server settings
        
        Args:
            server_ip: New server IP
            server_port: New server port
            endpoint: New endpoint path
        """
        if server_ip:
            self.server_ip = server_ip
            print(f"🔧 Updated server IP: {server_ip}")
        
        if server_port:
            self.server_port = server_port
            print(f"🔧 Updated server port: {server_port}")
        
        if endpoint:
            self.endpoint = endpoint
            print(f"🔧 Updated endpoint: {endpoint}")
        
        # Rebuild URL
        self.server_url = f"http://{self.server_ip}:{self.server_port}{self.endpoint}"
        print(f"🔧 New server URL: {self.server_url}")
        
        logger.info(f"Server settings updated: {self.server_url}")
    
    def send_test_payload(self):
        """
        Send a test payload to verify server communication
        
        Returns:
            tuple: (success, response, message)
        """
        try:
            print(f"\n🧪 Sending test payload to server...")
            
            test_payload = {
                'qr_code': 'TEST_123',
                'nip_data': {
                    'metadata': {
                        'qr_code': 'TEST_123',
                        'timestamp': datetime.now().isoformat(),
                        'generated_by': 'Test_Client',
                        'version': '1.0'
                    },
                    'inspection_results': {
                        'overall_status': 'OK',
                        'individual_nuts': {
                            'nut1': 'OK',
                            'nut2': 'OK',
                            'nut3': 'OK',
                            'nut4': 'OK'
                        }
                    }
                },
                'timestamp': datetime.now().isoformat(),
                'source': 'Test_Client'
            }
            
            headers = {
                'Content-Type': 'application/json',
                'User-Agent': 'NutDetection-TestClient/1.0',
                'X-QR-Code': 'TEST_123'
            }
            
            response = requests.post(
                self.server_url,
                json=test_payload,
                headers=headers,
                timeout=self.DEFAULT_TIMEOUT
            )
            
            print(f"📡 Test request sent:")
            print(f"   - Status Code: {response.status_code}")
            print(f"   - Response: {response.text}")
            
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    if (response_data.get('status') == 'received' and 
                        response_data.get('code') == 200):
                        
                        print(f"✅ Test successful - server responding correctly")
                        return True, response_data, "Test successful"
                    else:
                        print(f"⚠️ Test response unexpected: {response_data}")
                        return False, response_data, "Unexpected response format"
                except json.JSONDecodeError:
                    print(f"❌ Test failed - invalid JSON response")
                    return False, response.text, "Invalid JSON response"
            else:
                print(f"❌ Test failed - HTTP {response.status_code}")
                return False, response.text, f"HTTP error {response.status_code}"
                
        except Exception as e:
            error_msg = f"Test failed: {str(e)}"
            print(f"💥 {error_msg}")
            return False, None, error_msg