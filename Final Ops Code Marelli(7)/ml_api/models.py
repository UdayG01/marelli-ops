# ml_api/models.py - Enhanced with OK/NG Storage (Keeping All Existing Code)

import uuid
from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils import timezone
import json

class CustomUser(AbstractUser):
    """
    Custom User model with basic fields for login system
    """
    ROLE_CHOICES = [
        ('admin', 'Administrator'),
        ('user', 'User'),
    ]
    
    # Basic additional fields
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='user')
    employee_id = models.CharField(max_length=50, blank=True, help_text="Employee ID")
    department = models.CharField(max_length=100, blank=True, help_text="Department")
    phone_number = models.CharField(max_length=20, blank=True, help_text="Phone number")
    
    # Avoid conflicts with related names
    groups = models.ManyToManyField(
        'auth.Group',
        verbose_name='groups',
        blank=True,
        related_name='custom_user_set',
        related_query_name='custom_user',
    )
    user_permissions = models.ManyToManyField(
        'auth.Permission',
        verbose_name='user permissions',
        blank=True,
        related_name='custom_user_set',
        related_query_name='custom_user',
    )
    
    class Meta:
        verbose_name = "User"
        verbose_name_plural = "Users"
    
    def __str__(self):
        return f"{self.username} ({self.role})"

class SimpleInspection(models.Model):
    """
    Simple inspection model for testing workflow
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='simple_inspections')
    
    # Basic fields
    image_id = models.CharField(max_length=100, help_text="Image identifier")
    filename = models.CharField(max_length=255, help_text="Image filename")
    overall_result = models.CharField(
        max_length=10,
        choices=[('PASS', 'Pass'), ('FAIL', 'Fail')],
        default='PASS'
    )
    
    # Nut results (simple approach)
    nut1_status = models.CharField(max_length=10, choices=[('PRESENT', 'Present'), ('MISSING', 'Missing')], default='PRESENT')
    nut2_status = models.CharField(max_length=10, choices=[('PRESENT', 'Present'), ('MISSING', 'Missing')], default='PRESENT')
    nut3_status = models.CharField(max_length=10, choices=[('PRESENT', 'Present'), ('MISSING', 'Missing')], default='PRESENT')
    nut4_status = models.CharField(max_length=10, choices=[('PRESENT', 'Present'), ('MISSING', 'Missing')], default='PRESENT')
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    processing_time = models.FloatField(default=0.0)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "Simple Inspection"
        verbose_name_plural = "Simple Inspections"
    
    def __str__(self):
        return f"{self.image_id} - {self.overall_result}"


# ============================================================================
# NEW ENHANCED MODELS - ADDED FOR OK/NG FOLDER ORGANIZATION
# ============================================================================

class InspectionRecord(models.Model):
    """
    Enhanced Inspection Record with OK/NG folder organization
    Stores: date_time, image_id (QR code), nuts_present, nuts_absent
    Organizes images into OK/NG folders based on test results
    """
    # Primary key
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # User who performed the inspection
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='inspection_records')
    
    # Core inspection data - YOUR REQUIRED FIELDS
    image_id = models.CharField(max_length=100, help_text="QR Code / Image ID", db_index=True)
    capture_datetime = models.DateTimeField(auto_now_add=True, help_text="Date and time of image capture")
    
    # Image storage paths (organized by OK/NG status)
    original_image_ok = models.CharField(max_length=500, blank=True, 
                                        help_text="Path to original image in OK folder")
    original_image_ng = models.CharField(max_length=500, blank=True, 
                                        help_text="Path to original image in NG folder")
    annotated_image_ok = models.CharField(max_length=500, blank=True, 
                                         help_text="Path to annotated image in OK folder")
    annotated_image_ng = models.CharField(max_length=500, blank=True, 
                                         help_text="Path to annotated image in NG folder")
    
    # Nut detection results - YOUR REQUIRED FIELDS
    nuts_present = models.IntegerField(default=0, help_text="Number of nuts detected as present")
    nuts_absent = models.IntegerField(default=0, help_text="Number of nuts detected as absent")
    total_nuts_expected = models.IntegerField(default=4, help_text="Total nuts expected")
    
    # Test result status - OK/NG ORGANIZATION
    TEST_STATUS_CHOICES = [
        ('OK', 'OK - All nuts present (PASS)'),
        ('NG', 'NG - Some nuts missing (FAIL)'),
    ]
    test_status = models.CharField(max_length=2, choices=TEST_STATUS_CHOICES, 
                                  help_text="OK = Pass (all 4 nuts), NG = Fail (missing nuts)")
    
    # Additional metadata
    processing_time = models.FloatField(default=0.0, help_text="Processing time in seconds")
    confidence_scores = models.TextField(blank=True, help_text="JSON array of confidence scores")
    
    # Audit fields
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-capture_datetime']
        verbose_name = "Inspection Record"
        verbose_name_plural = "Inspection Records"
        indexes = [
            models.Index(fields=['image_id']),
            models.Index(fields=['capture_datetime']),
            models.Index(fields=['test_status']),
            models.Index(fields=['user', 'capture_datetime']),
        ]

    def __str__(self):
        return f"{self.image_id} - {self.test_status} ({self.capture_datetime.strftime('%Y-%m-%d %H:%M:%S')})"
    
    @property
    def original_image_path(self):
        """Get the appropriate original image path based on status"""
        if self.test_status == 'OK':
            return self.original_image_ok
        else:
            return self.original_image_ng
    
    @property
    def annotated_image_path(self):
        """Get the appropriate annotated image path based on status"""
        if self.test_status == 'OK':
            return self.annotated_image_ok
        else:
            return self.annotated_image_ng
    
    def get_confidence_scores_list(self):
        """Get confidence scores as a Python list"""
        if self.confidence_scores:
            try:
                return json.loads(self.confidence_scores)
            except json.JSONDecodeError:
                return []
        return []
    
    def set_confidence_scores_list(self, scores_list):
        """Set confidence scores from a Python list"""
        self.confidence_scores = json.dumps(scores_list)

    @classmethod
    def get_ok_count(cls, start_date=None, end_date=None):
        """Get count of OK inspections"""
        queryset = cls.objects.filter(test_status='OK')
        if start_date:
            queryset = queryset.filter(capture_datetime__gte=start_date)
        if end_date:
            queryset = queryset.filter(capture_datetime__lte=end_date)
        return queryset.count()
    
    @classmethod
    def get_ng_count(cls, start_date=None, end_date=None):
        """Get count of NG inspections"""
        queryset = cls.objects.filter(test_status='NG')
        if start_date:
            queryset = queryset.filter(capture_datetime__gte=start_date)
        if end_date:
            queryset = queryset.filter(capture_datetime__lte=end_date)
        return queryset.count()
    
    @classmethod
    def get_daily_summary(cls, date=None):
        """Get daily summary of inspections"""
        if not date:
            date = timezone.now().date()
        
        daily_inspections = cls.objects.filter(
            capture_datetime__date=date
        )
        
        total_count = daily_inspections.count()
        ok_count = daily_inspections.filter(test_status='OK').count()
        ng_count = daily_inspections.filter(test_status='NG').count()
        
        return {
            'date': date,
            'total': total_count,
            'ok_count': ok_count,
            'ng_count': ng_count,
            'pass_rate': round((ok_count / max(total_count, 1)) * 100, 2)
        }