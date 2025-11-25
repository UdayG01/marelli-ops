# ml_backend/settings.py

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Add this line here - BEFORE INSTALLED_APPS
AUTH_USER_MODEL = 'ml_api.CustomUser'

SECRET_KEY = 'django-insecure-your-secret-key-here-change-in-production'
DEBUG = True
ALLOWED_HOSTS = ['localhost', '127.0.0.1', '0.0.0.0']

# Updated INSTALLED_APPS with camera integration
INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'ml_api',
    'camera_integration',  # 🆕 NEW: Camera integration app
]

# Updated MIDDLEWARE with CORS
MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',  # 🆕 NEW: Add CORS middleware
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'ml_backend.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'ml_backend.wsgi.application'

# Database
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# Password validation
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Internationalization
LANGUAGE_CODE = 'en-us'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# Static files (CSS, JavaScript, Images)
STATIC_URL = 'static/'

# Default primary key field type
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Cache configuration for Node-RED message storage
CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.locmem.LocMemCache',
        'LOCATION': 'unique-snowflake',
    }
}

# Add REST Framework configuration
REST_FRAMEWORK = {
    'DEFAULT_PARSER_CLASSES': [
        'rest_framework.parsers.JSONParser',
        'rest_framework.parsers.MultiPartParser',
        'rest_framework.parsers.FormParser',
    ],
}

# Media files configuration (for image uploads)
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# Increase file upload size limit if needed
FILE_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024  # 10MB
DATA_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024   # 10MB

# Industrial Nut Detection Settings
NUT_DETECTION_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'industrial_nut_detection.pt')

# 🆕 NEW: CORS settings for frontend integration
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

CORS_ALLOW_CREDENTIALS = True

CORS_ALLOW_ALL_ORIGINS = True  # Only for development

# 🆕 NEW: Camera Integration Settings
CAMERA_SETTINGS = {
    'CAPTURE_PATH': os.path.join(MEDIA_ROOT, 'camera_captures'),
    'RAW_IMAGES_PATH': os.path.join(MEDIA_ROOT, 'camera_captures', 'raw'),
    'PROCESSED_IMAGES_PATH': os.path.join(MEDIA_ROOT, 'camera_captures', 'processed'),
    'FAILED_IMAGES_PATH': os.path.join(MEDIA_ROOT, 'camera_captures', 'failed'),
    'MAX_STORAGE_GB': 10,  # Maximum storage for camera captures
    'CLEANUP_DAYS': 30,    # Days to keep old captures
    'AUTO_PROCESS_CAPTURES': True,  # Automatically process captured images
}

# 🆕 NEW: Hikrobot SDK settings
HIKROBOT_SDK = {
    'SDK_PATH': os.path.join(BASE_DIR, 'hikrobot_sdk'),
    'CONFIG_PATH': os.path.join(BASE_DIR, 'hikrobot_sdk', 'config'),
    'DEFAULT_CONFIG': 'camera_settings.json',
}

# 🆕 NEW: Background task settings (for image processing)
BACKGROUND_TASKS = {
    'PROCESSING_WORKERS': 2,  # Number of background processing workers
    'QUEUE_SIZE': 100,        # Maximum queue size for processing jobs
    'TIMEOUT_SECONDS': 300,   # Timeout for processing jobs
}

# 🆕 NEW: Authentication settings
LOGIN_URL = '/api/ml/login/'
LOGIN_REDIRECT_URL = '/api/ml/dashboard/'
LOGOUT_REDIRECT_URL = '/api/ml/login/'

# 🆕 UPDATED: Enhanced Logging configuration
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': os.path.join(BASE_DIR, 'nut_detection.log'),
            'formatter': 'verbose',
        },
        'camera_file': {  # 🆕 NEW: Camera-specific log file
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': os.path.join(BASE_DIR, 'camera_integration.log'),
            'formatter': 'verbose',
        },
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['file', 'console'],
            'level': 'INFO',
            'propagate': True,
        },
        'ml_api': {
            'handlers': ['file', 'console'],
            'level': 'INFO',
            'propagate': False,
        },
        'camera_integration': {  # 🆕 NEW: Camera integration logger
            'handlers': ['camera_file', 'console'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}

# 🆕 NEW: Node-RED Integration Settings
NODE_RED_ENDPOINT = 'http://127.0.0.1:1880/inspection-result'
NODE_RED_TIMEOUT = 10  # seconds - increased timeout
NODE_RED_ENABLED = True  # Set to False to disable Node-RED integration

# 🆕 NEW: Node-RED QR Code endpoint
NODE_RED_QR_ENDPOINT = 'http://127.0.0.1:1880/qr-received'