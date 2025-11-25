# ml_backend/urls.py - Update your main URLs file

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from django.http import HttpResponseRedirect

def redirect_to_ml_api(request):
    """Redirect root URL to ML API interface"""
    return HttpResponseRedirect('/api/ml/')

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/ml/', include('ml_api.urls')),
    path('camera/', include('camera_integration.urls')),  # 🆕 NEW: Camera integration URLs
    path('', redirect_to_ml_api, name='home'),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)