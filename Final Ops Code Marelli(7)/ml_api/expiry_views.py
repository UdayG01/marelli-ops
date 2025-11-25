import pytz
from datetime import datetime, timedelta
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from django.conf import settings
import json
import logging

logger = logging.getLogger(__name__)

@require_http_methods(["GET"])
def system_expiry_status(request):
    """Get current system expiry status"""
    try:
        if not getattr(settings, 'SYSTEM_EXPIRY', {}).get('ENABLED', False):
            return JsonResponse({
                'success': True,
                'expiry_enabled': False,
                'message': 'System expiry checking is disabled'
            })
        
        expiry_config = settings.SYSTEM_EXPIRY
        
        # Parse expiry datetime
        expiry_datetime_str = expiry_config['EXPIRY_DATETIME']
        timezone = pytz.timezone(expiry_config['TIMEZONE'])
        
        expiry_datetime_naive = datetime.strptime(expiry_datetime_str, '%Y-%m-%d %H:%M:%S')
        expiry_datetime = timezone.localize(expiry_datetime_naive)
        current_datetime = datetime.now(timezone)
        
        # Calculate time difference
        time_diff = expiry_datetime - current_datetime
        is_expired = time_diff.total_seconds() <= 0
        
        if is_expired:
            return JsonResponse({
                'success': False,
                'expiry_enabled': True,
                'expired': True,
                'message': 'System has expired',
                'expiry_datetime': expiry_datetime.strftime('%Y-%m-%d %H:%M:%S %Z')
            })
        
        # Format time remaining
        total_seconds = int(time_diff.total_seconds())
        days = total_seconds // 86400
        hours = (total_seconds % 86400) // 3600
        minutes = (total_seconds % 3600) // 60
        
        return JsonResponse({
            'success': True,
            'expiry_enabled': True,
            'expired': False,
            'expiry_datetime': expiry_datetime.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'current_datetime': current_datetime.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'time_remaining': {
                'days': days,
                'hours': hours,
                'minutes': minutes,
                'total_seconds': total_seconds,
                'formatted': f"{days} days, {hours} hours, {minutes} minutes"
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting system expiry status: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        })

@csrf_exempt
@login_required
@require_http_methods(["GET", "POST"])
def update_expiry_time(request):
    """Update system expiry time (admin only)"""
    
    # Check if user is admin
    if not hasattr(request, 'user') or request.user.role != 'admin':
        return JsonResponse({
            'success': False,
            'error': 'Admin access required'
        })
    
    if request.method == 'GET':
        # Show current expiry time and update form
        try:
            expiry_config = getattr(settings, 'SYSTEM_EXPIRY', {})
            
            if not expiry_config.get('ENABLED', False):
                return JsonResponse({
                    'success': True,
                    'expiry_enabled': False,
                    'current_expiry': None,
                    'message': 'System expiry is disabled'
                })
            
            return JsonResponse({
                'success': True,
                'expiry_enabled': True,
                'current_expiry': expiry_config.get('EXPIRY_DATETIME'),
                'timezone': expiry_config.get('TIMEZONE'),
                'warning_days': expiry_config.get('WARNING_DAYS'),
                'warning_hours': expiry_config.get('WARNING_HOURS')
            })
            
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    elif request.method == 'POST':
        # Update expiry time
        try:
            data = json.loads(request.body)
            new_expiry_datetime = data.get('expiry_datetime')
            
            if not new_expiry_datetime:
                return JsonResponse({
                    'success': False,
                    'error': 'expiry_datetime is required'
                })
            
            # Validate datetime format
            try:
                datetime.strptime(new_expiry_datetime, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                return JsonResponse({
                    'success': False,
                    'error': 'Invalid datetime format. Use YYYY-MM-DD HH:MM:SS'
                })
            
            # Update settings (this will only work for the current process)
            settings.SYSTEM_EXPIRY['EXPIRY_DATETIME'] = new_expiry_datetime
            
            logger.info(f"System expiry time updated by admin {request.user.username}: {new_expiry_datetime}")
            
            return JsonResponse({
                'success': True,
                'message': f'Expiry time updated to {new_expiry_datetime}',
                'new_expiry': new_expiry_datetime,
                'updated_by': request.user.username
            })
            
        except json.JSONDecodeError:
            return JsonResponse({
                'success': False,
                'error': 'Invalid JSON data'
            })
        except Exception as e:
            logger.error(f"Error updating expiry time: {e}")
            return JsonResponse({
                'success': False,
                'error': str(e)
            })

@csrf_exempt
@login_required
@require_http_methods(["POST"])
def emergency_extend(request):
    """Emergency extend system by 24 hours (admin only)"""
    
    if not hasattr(request, 'user') or request.user.role != 'admin':
        return JsonResponse({
            'success': False,
            'error': 'Admin access required'
        })
    
    try:
        # Get current expiry
        expiry_config = settings.SYSTEM_EXPIRY
        current_expiry_str = expiry_config['EXPIRY_DATETIME']
        timezone = pytz.timezone(expiry_config['TIMEZONE'])
        
        # Parse current expiry
        current_expiry_naive = datetime.strptime(current_expiry_str, '%Y-%m-%d %H:%M:%S')
        current_expiry = timezone.localize(current_expiry_naive)
        
        # Extend by 24 hours
        new_expiry = current_expiry + timedelta(hours=24)
        new_expiry_str = new_expiry.strftime('%Y-%m-%d %H:%M:%S')
        
        # Update settings
        settings.SYSTEM_EXPIRY['EXPIRY_DATETIME'] = new_expiry_str
        
        logger.warning(f"EMERGENCY EXTENSION: System extended by 24 hours by admin {request.user.username}")
        logger.warning(f"EMERGENCY EXTENSION: New expiry time: {new_expiry_str}")
        
        return JsonResponse({
            'success': True,
            'message': 'System extended by 24 hours',
            'old_expiry': current_expiry_str,
            'new_expiry': new_expiry_str,
            'extended_by': request.user.username,
            'extension_time': '24 hours'
        })
        
    except Exception as e:
        logger.error(f"Emergency extension failed: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        })

@login_required
def expiry_management_page(request):
    """Admin page for managing system expiry"""
    if not hasattr(request, 'user') or request.user.role != 'admin':
        from django.http import HttpResponseForbidden
        return HttpResponseForbidden("Admin access required")
    
    return render(request, 'ml_api/expiry_management.html')
