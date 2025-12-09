# Architecture Document: Separated Service and Worker Models

## Overview
The data masking service has been refactored to separate model initialization between the API service and RQ workers. This eliminates redundant model loading and ensures each process only loads what it needs.

## Architecture

### 1. Service Process (`python main.py service`)
- **Purpose**: Handles text masking requests in real-time and submits table masking tasks to the queue
- **Models loaded**:
  - `text_masking_worker`: For real-time text masking requests
  - `crypto_manager`: For cryptographic operations
- **Models NOT loaded**:
  - `table_masking_worker`: Not needed since table processing is handled by workers

### 2. Worker Process (`python main.py worker`)
- **Purpose**: Processes table masking tasks from the queue
- **Models loaded**:
  - `table_masking_worker`: For batch table processing
  - `crypto_manager`: For cryptographic operations (worker's own instance)
- **Models NOT loaded**:
  - `text_masking_worker`: Not needed since workers only process table tasks

## File Structure

```
queue-service-rq/
├── service.py          # API service with text_masking_worker only
├── worker.py           # RQ worker that pre-initializes models
├── worker_models.py    # Worker-specific model management
├── task_processor.py   # Table processing logic for workers
└── main.py            # Entry point for both service and worker
```

## Key Benefits

1. **No Redundant Loading**:
   - Service loads only text masking models
   - Workers load only table masking models
   - Each process loads exactly what it needs

2. **Clear Separation of Concerns**:
   - Service handles real-time requests
   - Workers handle batch processing
   - No shared state between processes

3. **Efficient Resource Usage**:
   - Models are loaded once per process
   - Workers pre-initialize models at startup (configurable)
   - No circular import issues

## Startup Sequence

```bash
# Start worker first (will load its own models)
python main.py worker

# Then start service (will load only text models)
python main.py service
```

## Expected GPU Usage

- **Service process**: 1 model instance (text_masking_worker)
- **Each worker process**: 1 model instance (table_masking_worker)
- **Total**: 1 + N models where N is the number of workers

## Module Details

### worker_models.py
- Manages model initialization for worker processes
- Provides `get_table_masking_worker()` and `get_crypto_manager()`
- Models are initialized once per worker process

### task_processor.py
- Contains the `process_table_masking_task` function
- Imports models from `worker_models` (not `service`)
- Executes table masking tasks independently

### service.py
- Only initializes `text_masking_worker` for text requests
- Submits table tasks to RQ queue
- No longer has `table_masking_worker`

## Configuration

The worker can be configured to pre-initialize models at startup (default behavior):
```python
# In worker.py
initialize_worker_models()  # Loads models immediately
```

If you prefer lazy loading (models loaded on first task), comment out the pre-initialization code.

## Migration Notes

The previous attempt at sharing models between processes failed because:
- Processes don't share memory
- Each process needs its own model instances
- Circular imports caused initialization issues

This architecture accepts the reality of process separation and optimizes for it.