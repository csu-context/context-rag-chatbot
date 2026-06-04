#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/home/ubuntu/app"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting EC2 Boot Deployment script..."

if [ -f "$APP_DIR/.env" ]; then
    echo "Loading configurations from .env..."
    # Export ECR configurations safely while filtering
    while IFS= read -r line || [ -n "$line" ]; do
        if [[ ! "$line" =~ ^# ]] && [[ -n "$line" ]]; then
            if [[ "$line" =~ ^(AWS_ACCOUNT_ID|AWS_REGION|ECR_REGISTRY|ECR_REPOSITORY|IMAGE_TAG)= ]]; then
                export "$line"
            fi
        fi
    done < "$APP_DIR/.env"
fi

# Set default values if not defined
AWS_REGION="${AWS_REGION:-ap-northeast-2}"
ECR_REGISTRY="${ECR_REGISTRY:-}"

if [ -n "$ECR_REGISTRY" ]; then
    echo "Logging in to Amazon ECR registry: $ECR_REGISTRY..."
    # Authenticate to ECR using IAM role assigned to the EC2 Instance Profile
    aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$ECR_REGISTRY"
else
    echo "ECR_REGISTRY is not configured. Skipping ECR login."
fi

cd "$APP_DIR"

echo "Pre-cleaning dangling/unused docker resources to free up space..."
docker image prune -f

echo "Pulling latest Docker images..."
docker compose -f docker-compose.yml -f docker-compose.gpu.yml pull

echo "Starting containers..."
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d

echo "Cleaning up old/dangling docker images to optimize disk space..."
docker image prune -f

echo "Deployment completed successfully!"
