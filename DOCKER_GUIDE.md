# Docker Build and Deployment Guide for SDG Classifier

## Quick Start - Building the Docker Image

### 1. Basic Build Command

```bash
# Build the image with a tag
docker build -t sdg-classifier:latest .
```

### 2. Build with Specific Version Tag

```bash
# Build with version tag
docker build -t sdg-classifier:v1.0.0 .
```

### 3. Build with Multiple Tags

```bash
# Build with multiple tags
docker build -t sdg-classifier:latest -t sdg-classifier:v1.0.0 .
```

## Running the Docker Container

### 1. Run with Default Settings (Streamlit App)

```bash
# Run the container and map port 8501
docker run -p 8501:8501 sdg-classifier:latest
```

Then open your browser and go to: `http://localhost:8501`

### 2. Run in Detached Mode (Background)

```bash
# Run in background
docker run -d -p 8501:8501 --name sdg-classifier-app sdg-classifier:latest
```

### 3. Run with Volume Mounts (For Development)

```bash
# Mount local directories to persist data and models
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/metrics:/app/metrics \
  sdg-classifier:latest
```

### 4. Run with Environment Variables

```bash
# Run with custom environment variables
docker run -p 8501:8501 \
  -e STREAMLIT_SERVER_PORT=8080 \
  -e PYTHONUNBUFFERED=1 \
  sdg-classifier:latest
```

## Managing Docker Containers

### View Running Containers

```bash
docker ps
```

### View All Containers (including stopped)

```bash
docker ps -a
```

### Stop a Running Container

```bash
docker stop sdg-classifier-app
```

### Start a Stopped Container

```bash
docker start sdg-classifier-app
```

### Remove a Container

```bash
docker rm sdg-classifier-app
```

### View Container Logs

```bash
# Follow logs in real-time
docker logs -f sdg-classifier-app

# View last 100 lines
docker logs --tail 100 sdg-classifier-app
```

## Docker Image Management

### List Docker Images

```bash
docker images
```

### Remove an Image

```bash
docker rmi sdg-classifier:latest
```

### Save Image to File (For Transfer)

```bash
# Save image to tar file
docker save sdg-classifier:latest -o sdg-classifier.tar

# Load image from tar file (on another machine)
docker load -i sdg-classifier.tar
```

## Advanced Docker Commands

### 1. Build with No Cache (Clean Build)

```bash
docker build --no-cache -t sdg-classifier:latest .
```

### 2. Build with Build Arguments

```bash
docker build --build-arg PYTHON_VERSION=3.10 -t sdg-classifier:latest .
```

### 3. Interactive Shell Access

```bash
# Access container shell (while running)
docker exec -it sdg-classifier-app /bin/bash

# Run container with interactive shell
docker run -it --rm sdg-classifier:latest /bin/bash
```

### 4. Inspect Container

```bash
# View container details
docker inspect sdg-classifier-app

# View specific information (e.g., IP address)
docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' sdg-classifier-app
```

## Docker Compose (Optional)

If you want to use Docker Compose, create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  sdg-classifier:
    build: .
    image: sdg-classifier:latest
    container_name: sdg-classifier-app
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./metrics:/app/metrics
      - ./reports:/app/reports
    environment:
      - PYTHONUNBUFFERED=1
      - STREAMLIT_SERVER_HEADLESS=true
    restart: unless-stopped
```

Then use:

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Rebuild and restart
docker-compose up -d --build
```

## Pushing to Docker Registry

### 1. Push to Docker Hub

```bash
# Login to Docker Hub
docker login

# Tag the image with your username
docker tag sdg-classifier:latest yourusername/sdg-classifier:latest

# Push to Docker Hub
docker push yourusername/sdg-classifier:latest
```

### 2. Push to Private Registry

```bash
# Tag for private registry
docker tag sdg-classifier:latest registry.example.com/sdg-classifier:latest

# Push to private registry
docker push registry.example.com/sdg-classifier:latest
```

## Troubleshooting

### Build Fails Due to Dependencies

```bash
# Try building with more verbose output
docker build --progress=plain -t sdg-classifier:latest .
```

### Container Exits Immediately

```bash
# Check logs
docker logs sdg-classifier-app

# Run with interactive mode to see errors
docker run -it --rm sdg-classifier:latest
```

### Port Already in Use

```bash
# Use a different port
docker run -p 8080:8501 sdg-classifier:latest
```

### Out of Disk Space

```bash
# Clean up unused images and containers
docker system prune -a

# Remove specific dangling images
docker image prune
```

## Best Practices

1. **Always use specific tags** instead of `latest` in production
2. **Use .dockerignore** to reduce build context size (already created)
3. **Multi-stage builds** for smaller production images (if needed)
4. **Health checks** to ensure container is running properly
5. **Use volumes** for persistent data
6. **Set resource limits** in production:

```bash
docker run -p 8501:8501 \
  --memory="2g" \
  --cpus="1.5" \
  sdg-classifier:latest
```

## Current Dockerfile Features

Your Dockerfile includes:
- ✅ Python 3.10 slim base image (lightweight)
- ✅ Proper environment variables for Python and Streamlit
- ✅ Build tools for scikit-learn compilation
- ✅ Requirements installation with pip cache disabled
- ✅ Port 8501 exposed for Streamlit
- ✅ Streamlit app as entry point

## Next Steps

1. Build your image: `docker build -t sdg-classifier:latest .`
2. Run your container: `docker run -p 8501:8501 sdg-classifier:latest`
3. Access the app at: `http://localhost:8501`
4. (Optional) Push to Docker Hub for sharing
