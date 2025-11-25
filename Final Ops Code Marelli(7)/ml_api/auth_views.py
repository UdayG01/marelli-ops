# ml_api/simple_auth_views.py - Simple Login System

from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from datetime import datetime

from .models import CustomUser, SimpleInspection

def simple_login_view(request):
    """
    Simple dual login page for Admin and User
    """
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        login_type = request.POST.get('login_type', 'user')
        
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            # Check if user role matches selected login type
            if login_type == 'admin' and user.role != 'admin':
                messages.error(request, 'You do not have admin privileges.')
                return render(request, 'ml_api/simple_login.html')
            
            login(request, user)
            
            # Redirect based on role
            if user.role == 'admin':
                return redirect('ml_api:simple_admin_dashboard')
            else:
                return redirect('ml_api:simple_user_dashboard')
        else:
            messages.error(request, 'Invalid username or password.')
    
    return render(request, 'ml_api/simple_login.html')

@login_required
def simple_admin_dashboard(request):
    """
    Simple Admin Dashboard
    """
    if request.user.role != 'admin':
        messages.error(request, 'Access denied. Admin privileges required.')
        return redirect('ml_api:simple_user_dashboard')
    
    # Get basic statistics
    total_users = CustomUser.objects.count()
    total_inspections = SimpleInspection.objects.count()
    recent_inspections = SimpleInspection.objects.order_by('-created_at')[:10]
    failed_inspections = SimpleInspection.objects.filter(overall_result='FAIL').count()
    
    context = {
        'total_users': total_users,
        'total_inspections': total_inspections,
        'failed_inspections': failed_inspections,
        'recent_inspections': recent_inspections,
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
    
    context = {
        'user_inspections': user_inspections,
        'failed_inspections': failed_inspections,
    }
    
    return render(request, 'ml_api/simple_user_dashboard.html', context)

@login_required
def simple_workflow_page(request):
    """
    Simple single image workflow page
    """
    return render(request, 'ml_api/simple_workflow.html')

@login_required
def simple_process_image(request):
    """
    Simple image processing endpoint
    """
    if request.method == 'POST':
        try:
            image_id = request.POST.get('image_id', '').strip()
            uploaded_file = request.FILES.get('image')
            
            if not image_id:
                return JsonResponse({
                    'success': False,
                    'error': 'Image ID is required'
                })
            
            if not uploaded_file:
                return JsonResponse({
                    'success': False,
                    'error': 'Image file is required'
                })
            
            # Simple processing simulation
            # In real implementation, this would call your YOLOv8 model
            import random
            
            # Simulate nut detection results
            nut_statuses = []
            for i in range(4):
                status = 'PRESENT' if random.random() > 0.2 else 'MISSING'
                nut_statuses.append(status)
            
            # Determine overall result
            missing_count = nut_statuses.count('MISSING')
            overall_result = 'PASS' if missing_count == 0 else 'FAIL'
            
            # Save to database
            inspection = SimpleInspection.objects.create(
                user=request.user,
                image_id=image_id,
                filename=uploaded_file.name,
                overall_result=overall_result,
                nut1_status=nut_statuses[0],
                nut2_status=nut_statuses[1],
                nut3_status=nut_statuses[2],
                nut4_status=nut_statuses[3],
                processing_time=1.2
            )
            
            return JsonResponse({
                'success': True,
                'image_id': image_id,
                'overall_result': overall_result,
                'nut_results': {
                    'nut1': nut_statuses[0],
                    'nut2': nut_statuses[1],
                    'nut3': nut_statuses[2],
                    'nut4': nut_statuses[3],
                },
                'missing_count': missing_count,
                'processing_time': 1.2,
                'inspection_id': str(inspection.id)
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

def simple_logout_view(request):
    """
    Simple logout
    """
    logout(request)
    messages.success(request, 'You have been logged out successfully.')
    return redirect('ml_api:simple_login')