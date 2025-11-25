# Image ID Flow Implementation for Triggered Captures

## Overview

This implementation adds functionality to move inspection directories from temporary locations to permanent locations with user-entered image IDs after triggered capture processing is complete.

## Changes Made

### 1. Modified `views.py`

#### Updated `_save_and_process_triggered_image_enhanced` method:
- Changed `image_id` to `temp_image_id` for temporary storage
- Uses format: `triggered_capture_{timestamp_str}`

#### Updated `_process_triggered_image_same_as_manual` method:
- Added logic to retrieve user-entered image_id from cache
- Implemented directory moving after successful processing
- Uses `final_image_id` (user-entered or temp as fallback)

#### Added new helper methods:
- `_get_user_entered_image_id()`: Retrieves image_id from Django cache
- `_move_inspection_directory()`: Moves inspection folder to final location

#### Added new API endpoint:
- `store_image_id()`: Stores user-entered image_id in cache for triggered captures

### 2. Modified `urls.py`

#### Added new URL pattern:
```python
path('store-image-id/', store_image_id, name='store_image_id'),
```

### 3. Modified `image_id_entry.html`

#### Updated JavaScript:
- Added `getCookie()` function for CSRF token handling
- Modified `proceedToImageSource()` to store image_id in cache via API call

## Flow Explanation

### Current Workflow:

1. **User enters Image ID**: User fills form in `api/ml/image-id` (form id = `qr-form`)
2. **Image ID stored in cache**: JavaScript calls `/api/ml/store-image-id/` to store in Django cache
3. **Trigger detected**: Camera detects Line 0 trigger signal
4. **Temporary capture**: Image saved as `triggered_capture_{timestamp_str}`
5. **Processing**: ML processing runs using temporary image_id
6. **Directory created**: Results stored in `inspections/triggered_capture_{timestamp}/OK_or_NG/annotated_and_original`
7. **Retrieve user ID**: System gets user-entered image_id from cache
8. **Move directory**: After processing complete, directory moved to `triggered_inspections/{user_entered_image_id}`
9. **Final structure**: `triggered_inspections/{user_entered_image_id}/OK_or_NG/annotated_and_original`

### Directory Structure Changes:

**Before (temporary):**
```
media/inspections/triggered_capture_20250127_143059_123/
├── OK/
│   ├── annotated/
│   │   └── annotated_image.jpg
│   └── original/
│       └── original_image.jpg
└── results.json
```

**After (final):**
```
media/triggered_inspections/{user_entered_image_id}/
├── OK/
│   ├── annotated/
│   │   └── annotated_image.jpg
│   └── original/
│       └── original_image.jpg
└── results.json
```

## Key Features

### Cache Management:
- User-entered image_id stored with 1-hour timeout
- Fallback to temporary ID if cache miss
- Thread-safe retrieval

### Directory Moving:
- Preserves complete directory structure
- Handles both OK and NG results
- Removes existing target directories if present
- Detailed logging for debugging

### Error Handling:
- Graceful fallback to temporary ID
- Comprehensive error logging
- Non-blocking operation (processing continues even if move fails)

## API Endpoints

### Store Image ID
```
POST /api/ml/store-image-id/
Content-Type: application/json

{
    "image_id": "USER_ENTERED_ID_123"
}
```

**Response:**
```json
{
    "success": true,
    "message": "Image ID USER_ENTERED_ID_123 stored in cache",
    "image_id": "USER_ENTERED_ID_123"
}
```

## Testing

A test script `test_image_id_flow.py` has been created to verify:
1. Cache storage and retrieval
2. Directory moving functionality
3. Camera manager helper methods

**Run tests:**
```bash
python test_image_id_flow.py
```

## Configuration

### Cache Settings
The implementation uses Django's default cache. For production, consider configuring Redis or Memcached:

```python
# settings.py
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        }
    }
}
```

### Media Directory Structure
Ensure the following directories exist:
- `media/inspections/` (temporary storage)
- `media/triggered_inspections/` (final storage)

## Security Considerations

- Image ID validation prevents directory traversal
- CSRF protection on API endpoints
- Cache timeout prevents indefinite storage
- Directory permissions should be restricted appropriately

## Logging

The implementation provides detailed logging:
- `[IMAGE_ID]` prefix for image ID operations
- `[MOVE]` prefix for directory operations
- `[CACHE]` prefix for cache operations

**Example log output:**
```
[IMAGE_ID] Retrieved from cache: USER_ID_123
[IMAGE_ID] Using final image_id: USER_ID_123 (user entered: USER_ID_123, temp: triggered_capture_20250127_143059_123)
[MOVE] ✅ Successfully moved inspection directory:
[MOVE]    FROM: /media/inspections/triggered_capture_20250127_143059_123
[MOVE]    TO:   /media/triggered_inspections/USER_ID_123
```

## Troubleshooting

### Common Issues:

1. **Cache not working**: Check Django cache configuration
2. **Directory move fails**: Verify permissions on media directories
3. **Image ID not found**: Check cache timeout and form submission
4. **JavaScript errors**: Verify CSRF token configuration

### Debug Commands:

```python
# Check cache contents
from django.core.cache import cache
print(cache.get('current_image_id'))

# List cache keys (Redis only)
from django.core.cache import cache
print(cache.keys('*'))

# Check directory permissions
import os
import stat
path = '/path/to/media/triggered_inspections'
print(oct(stat.S_IMODE(os.lstat(path).st_mode)))
```

This implementation ensures that triggered captures are properly organized with user-meaningful identifiers while maintaining the existing workflow and error handling capabilities.
