# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based facial recognition service using FastAPI. The system performs face detection, embedding generation, and identity verification through cosine similarity comparison.

## Core Architecture

- **Main Service**: `face_service.py` - FastAPI application that handles face recognition requests
- **Model**: Uses `facenet_pytorch` with InceptionResnetV1 and MTCNN for face detection and embedding generation
- **Storage**: Redis for temporary embedding storage, local `.npy` files for registered embeddings
- **Pre-trained Model**: VGGFace2 model weights in `models/20180402-114759-vggface2.pt`

## Development Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
uvicorn face_service:app --host 0.0.0.0 --port 8000

# Run test files directly
python tests/test1.py
python tests/redis_test.py
python tests/save_embed.py
```

## Key Components

### Face Recognition Pipeline
1. Image upload via `/recognize` endpoint
2. Face detection using MTCNN (Multi-task CNN)
3. Feature extraction using InceptionResnetV1
4. L2 normalization of embeddings
5. Cosine similarity comparison with threshold (default: 0.8)

### Storage Strategy
- **Redis**: Temporary user embeddings with 7-day TTL (key format: `guard:{employeeId}`)
- **File System**: Permanent registered embeddings in `tests/embeddings/` as `.npy` files

### Environment Configuration
Required environment variables:
- `REDIS_HOST` (default: localhost)
- `REDIS_PORT` (default: 6379)
- `REDIS_DB` (default: 0)
- `MODEL_PATH` (path to model weights)
- `EMBEDDING_THRESHOLD` (default: 0.8)

## Test Data Structure

- `tests/images/` - Test face images for various individuals
- `tests/embeddings/` - Pre-computed embeddings as NumPy arrays
- Test utilities for embedding generation and Redis operations

## Key Technical Details

- Device auto-detection (CUDA/CPU) for model inference
- Async request handling with thread pool execution
- EXIF orientation handling for uploaded images
- First-time registration flow vs. existing user verification