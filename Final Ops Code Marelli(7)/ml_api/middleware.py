import pytz
from datetime import datetime, timedelta
from django.http import HttpResponse, JsonResponse
from django.utils.deprecation import MiddlewareMixin
from django.conf import settings
import logging

logger = logging.getLogger(__name__)

class SystemExpiryMiddleware(MiddlewareMixin):
    """
    Middleware to check system expiry and crash application when time is reached
    """
    
    def __init__(self, get_response):
        self.get_response = get_response
        super().__init__(get_response)
    
    def process_request(self, request):
        """Check system expiry on every request"""
        
        # Skip if expiry check is disabled
        if not getattr(settings, 'SYSTEM_EXPIRY', {}).get('ENABLED', False):
            return None
        
        expiry_config = settings.SYSTEM_EXPIRY
        
        # Parse expiry datetime
        try:
            expiry_datetime_str = expiry_config['EXPIRY_DATETIME']
            timezone = pytz.timezone(expiry_config['TIMEZONE'])
            
            # Parse the datetime string (assume naive, then localize to India timezone)
            expiry_datetime_naive = datetime.strptime(expiry_datetime_str, '%Y-%m-%d %H:%M:%S')
            expiry_datetime = timezone.localize(expiry_datetime_naive)
            
        except Exception as e:
            logger.error(f"Invalid expiry datetime configuration: {e}")
            return None
        
        # Get current time in India timezone
        current_datetime = datetime.now(timezone)
        
        # Calculate time difference
        time_diff = expiry_datetime - current_datetime
        
        # CHECK: SYSTEM EXPIRED - PERMANENT CRASH
        if time_diff.total_seconds() <= 0:
            logger.critical(f"SYSTEM EXPIRED: {current_datetime} >= {expiry_datetime}")
            return self._system_expired_response(request)
        
        return None
    
    def _system_expired_response(self, request):
        """Return system expired response - PERMANENT CRASH"""
        crash_message = settings.SYSTEM_EXPIRY.get('CRASH_MESSAGE', 'System has expired and is permanently disabled.')
        
        # Log the permanent crash
        logger.critical("SYSTEM PERMANENTLY DISABLED - EXPIRY TIME REACHED")
        
        # Return HTML crash page
        return HttpResponse(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>System Expired</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 50px; text-align: center; background: #f5f5f5; }}
                    .container {{ max-width: 600px; margin: 0 auto; padding: 40px; background: white; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
                    .error {{ color: #d32f2f; font-size: 24px; font-weight: bold; margin-bottom: 20px; }}
                    .message {{ color: #333; font-size: 18px; line-height: 1.6; }}
                    .timestamp {{ color: #666; font-size: 14px; margin-top: 30px; }}
                    .skull {{ font-size: 48px; margin-bottom: 20px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="skull">💀</div>
                    <div class="error">SYSTEM EXPIRED</div>
                    <div class="message">{crash_message}</div>
                    <div class="timestamp">System permanently disabled at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
                </div>
            </body>
            </html>
        """, status=503)
