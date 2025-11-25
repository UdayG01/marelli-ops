# Triggered Capture Flow with Directory Management

## Complete Flow Explanation

### Overview
This document explains the complete flow of triggered captures, from user input to final directory organization and results display.

## Step-by-Step Flow

### 1. User Setup Phase
```
User Action: Navigate to /api/ml/image-id/
↓
Enter Image ID in form (id="qr-form")
↓
JavaScript stores Image ID in Django cache (1 hour timeout)
↓
User proceeds to camera capture page
```

**Technical Details:**
- Form submits to `/api/ml/store-image-id/`
- Image ID stored with cache key: `'current_image_id'`
- Cache timeout: 3600 seconds (1 hour)

### 2. Camera Trigger Detection
```
Hardware trigger signal detected on Line 0
↓
Camera captures frame automatically
↓
Temporary image ID generated: triggered_capture_{timestamp}
↓
Image saved to: media/inspections/original/
```

**Technical Details:**
- Trigger monitoring runs in background thread
- Frame captured with 50ms timeout
- High quality JPEG saved (95% quality)
- Processing starts immediately in separate thread

### 3. ML Processing Phase
```
YOLOv8 model processes the image
↓
Results generated:
- Nut detection results (PRESENT/MISSING)
- Confidence scores
- Overall result (PASS/FAIL)
↓
Database records created
```

**Technical Details:**
- Uses same `_process_with_yolov8_model` function as manual processing
- Creates `SimpleInspection` record
- Enhanced storage service saves detailed results
- Processing time tracked

### 4. Directory Structure Creation
```
Initial directory structure created:
media/inspections/triggered_capture_{timestamp}/
├── OK/ (or NG/)
│   ├── annotated/
│   │   └── annotated_image.jpg
│   └── original/
│       └── original_image.jpg
└── results.json
```

**Technical Details:**
- Directory named with temporary ID
- Result directory (OK/NG) based on ML decision
- Annotated image shows detection boxes
- Original image preserved for reference

### 5. Image ID Retrieval and Directory Move
```
System retrieves user-entered Image ID from cache
↓
If Image ID found:
  - Use user Image ID as final_image_id
  - Move directory to new location
If Image ID not found:
  - Use temporary ID as final_image_id
  - Keep directory in original location
```

**Technical Details:**
- Cache lookup with key `'current_image_id'`
- Fallback mechanism prevents processing failure
- Directory move uses `shutil.move()` for atomic operation

### 6. Final Directory Organization
```
Directory moved to final location:
media/inspections/triggered_inspections/{user_entered_image_id}/
├── OK/ (or NG/)
│   ├── annotated/
│   │   └── annotated_image.jpg
│   └── original/
│       └── original_image.jpg
└── results.json
```

**Key Points:**
- ✅ **Final path**: `media/inspections/triggered_inspections/{user_image_id}/`
- ✅ **Source removed**: Original temporary directory deleted
- ✅ **Structure preserved**: Complete directory structure maintained
- ✅ **Verification**: Move success verified before completion

### 7. Results and Notifications
```
Processing completion triggers:
├── Database update with final Image ID
├── Node-RED notification sent
├── File transfer processing (if configured)
└── Status tracking updated
```

**Technical Details:**
- `SimpleInspection` record updated with final Image ID
- Node-RED receives inspection results
- File transfer service processes OK/NG results
- Trigger processing status marked as complete

## Directory Structure Comparison

### Before Processing (Temporary)
```
media/
└── inspections/
    ├── original/
    │   └── triggered_capture_20250127_143059_123_camera.jpg
    └── triggered_capture_20250127_143059_123/
        └── OK/
            ├── annotated/
            │   └── annotated_image.jpg
            └── original/
                └── original_image.jpg
```

### After Processing (Final)
```
media/
└── inspections/
    ├── original/
    │   └── triggered_capture_20250127_143059_123_camera.jpg
    └── triggered_inspections/
        └── USER_ENTERED_ID_123/
            └── OK/
                ├── annotated/
                │   └── annotated_image.jpg
                └── original/
                    └── original_image.jpg
```

## Results Page Integration

### Current Results Page Access
The results page can be accessed via:
```
/api/ml/results/?image_id={user_entered_image_id}
```

### Results Page Data Sources
1. **Database Query**: `SimpleInspection` objects filtered by `image_id`
2. **File System**: Images loaded from final directory location
3. **Enhanced Storage**: Detailed inspection data from enhanced storage service

### Results Page Display
```
Results Page Shows:
├── Inspection Summary
│   ├── Image ID: {user_entered_image_id}
│   ├── Overall Result: PASS/FAIL
│   ├── Processing Time: X.XX seconds
│   └── Timestamp: YYYY-MM-DD HH:MM:SS
├── Nut Detection Details
│   ├── Nut 1: PRESENT/MISSING (confidence: XX%)
│   ├── Nut 2: PRESENT/MISSING (confidence: XX%)
│   ├── Nut 3: PRESENT/MISSING (confidence: XX%)
│   └── Nut 4: PRESENT/MISSING (confidence: XX%)
└── Images
    ├── Original Image
    └── Annotated Image (with detection boxes)
```

## Error Handling and Fallbacks

### Cache Miss Scenario
```
If user Image ID not found in cache:
├── Warning logged: "No user-entered image_id found in cache"
├── Use temporary ID as final_image_id
├── Directory remains in original location
└── Processing continues normally
```

### Directory Move Failure
```
If directory move fails:
├── Error logged with full traceback
├── Original directory structure preserved
├── Processing marked as complete
└── User can still access results via temporary ID
```

### Processing Failure
```
If ML processing fails:
├── Error status recorded in database
├── Node-RED notified of failure
├── Processing marked as complete with error
└── Directory cleanup may be required
```

## Performance Considerations

### Timing Breakdown
1. **Trigger Detection**: < 1ms
2. **Image Capture**: 50ms timeout
3. **ML Processing**: 2-5 seconds (typical)
4. **Directory Move**: < 100ms
5. **Total Processing**: 3-6 seconds

### Concurrency Handling
- Multiple triggers processed in separate threads
- Thread-safe status tracking with locks
- Atomic directory operations
- Database transactions for consistency

## Monitoring and Debugging

### Log Levels and Prefixes
```
[TRIGGER] - Trigger detection and capture
[PROCESSING] - ML model processing
[IMAGE_ID] - Image ID retrieval and management
[MOVE] - Directory move operations
[NODE-RED] - External system notifications
[CACHE] - Cache operations
```

### Status Tracking
Each trigger maintains status in `trigger_processing_status`:
```python
{
    'captured': True/False,
    'processing': True/False,
    'complete': True/False,
    'image_id': 'final_image_id',
    'success': True/False,
    'error': 'error_message',
    'started_at': 'timestamp',
    'completed_at': 'timestamp',
    'trigger_count': 123
}
```

## Configuration Requirements

### Cache Configuration
```python
# Recommended for production
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        },
        'TIMEOUT': 3600,  # 1 hour
    }
}
```

### Directory Permissions
```bash
# Ensure write permissions
chmod 755 media/inspections/
chmod 755 media/inspections/triggered_inspections/
```

### Required Services
1. **Django Application**: Main processing logic
2. **Cache Backend**: Redis or Memcached (recommended)
3. **Database**: PostgreSQL or MySQL (recommended)
4. **Node-RED**: External system integration (optional)
5. **File Transfer Service**: Result distribution (optional)

## Testing the Flow

### Manual Test Sequence
1. **Setup**: Enter Image ID via web form
2. **Trigger**: Send hardware trigger signal or use software trigger
3. **Monitor**: Watch logs for processing completion
4. **Verify**: Check final directory structure
5. **Results**: Access results page with user Image ID

### Automated Testing
```bash
# Run the test script
python test_image_id_flow.py
```

This comprehensive flow ensures that triggered captures are properly organized with user-meaningful identifiers while maintaining system reliability and performance.
