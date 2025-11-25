# ml_api/admin.py
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser, SimpleInspection, InspectionRecord

# Register CustomUser with enhanced admin interface
@admin.register(CustomUser)
class CustomUserAdmin(UserAdmin):
    """
    Custom User Admin with role-based fields
    """
    list_display = ('username', 'email', 'role', 'employee_id', 'date_joined', 'is_active')
    list_filter = ('role', 'is_active', 'date_joined')
    search_fields = ('username', 'email', 'employee_id')
    ordering = ('-date_joined',)
    
    # Add custom fields to the user form
    fieldsets = UserAdmin.fieldsets + (
        ('Additional Info', {
            'fields': ('role', 'employee_id', 'department', 'phone_number')
        }),
    )
    
    # Add custom fields to the add user form
    add_fieldsets = UserAdmin.add_fieldsets + (
        ('Additional Info', {
            'fields': ('role', 'employee_id', 'department', 'phone_number')
        }),
    )

# Register SimpleInspection
@admin.register(SimpleInspection)
class SimpleInspectionAdmin(admin.ModelAdmin):
    """
    Simple Inspection Admin
    """
    list_display = ('image_id', 'user', 'overall_result', 'created_at')
    list_filter = ('overall_result', 'created_at')
    search_fields = ('image_id', 'user__username')
    ordering = ('-created_at',)
    readonly_fields = ('id', 'created_at')
    
    fieldsets = (
        ('Basic Info', {
            'fields': ('user', 'image_id', 'filename', 'overall_result')
        }),
        ('Nut Results', {
            'fields': ('nut1_status', 'nut2_status', 'nut3_status', 'nut4_status')
        }),
        ('Metadata', {
            'fields': ('processing_time', 'created_at')
        }),
    )

# Register InspectionRecord
@admin.register(InspectionRecord)
class InspectionRecordAdmin(admin.ModelAdmin):
    """
    Enhanced Inspection Record Admin
    """
    list_display = ('image_id', 'user', 'test_status', 'nuts_present', 'nuts_absent', 'capture_datetime')
    list_filter = ('test_status', 'capture_datetime', 'user')
    search_fields = ('image_id', 'user__username')
    ordering = ('-capture_datetime',)
    readonly_fields = ('id', 'created_at', 'updated_at')
    
    fieldsets = (
        ('Basic Info', {
            'fields': ('user', 'image_id', 'test_status', 'capture_datetime')
        }),
        ('Nut Detection Results', {
            'fields': ('nuts_present', 'nuts_absent', 'total_nuts_expected', 'confidence_scores')
        }),
        ('Image Paths', {
            'fields': ('original_image_ok', 'original_image_ng', 'annotated_image_ok', 'annotated_image_ng')
        }),
        ('Metadata', {
            'fields': ('processing_time', 'created_at', 'updated_at')
        }),
    )
    
    def get_queryset(self, request):
        """Optimize queryset with select_related"""
        return super().get_queryset(request).select_related('user')