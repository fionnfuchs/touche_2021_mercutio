#!/bin/sh
# Note: Using sh instead of bash is needed to run inside the docker:* image in gitlab-ci

set -eux

cd "$(dirname "$0")/.." || exit

IMAGE_NAME="touche2-ul-mercutio"
COMMIT_ID="$(git rev-parse HEAD)"

docker build -t "$IMAGE_NAME:$COMMIT_ID" -t "$IMAGE_NAME:latest" -f docker/Dockerfile .
