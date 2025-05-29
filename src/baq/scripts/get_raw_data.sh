#!/bin/bash

# Script to download latest.csv from AWS S3
# Source: s3://soccer-storage/webapp-storage/data/raw/latest_data/latest.csv
# Destination: data/raw_data/latest.csv

set -e  # Exit on any error

# Define paths
S3_SOURCE="s3://soccer-storage/webapp-storage/data/raw/latest_data/latest.csv"
LOCAL_DEST="data/raw_data/latest.csv"
DEST_DIR="data/raw_data"

echo "Starting download of latest.csv from AWS S3..."

# Create destination directory if it doesn't exist
if [ ! -d "$DEST_DIR" ]; then
    echo "Creating directory: $DEST_DIR"
    mkdir -p "$DEST_DIR"
fi

# Download file from S3
echo "Downloading from: $S3_SOURCE"
echo "Saving to: $LOCAL_DEST"

if aws s3 cp "$S3_SOURCE" "$LOCAL_DEST"; then
    echo "✅ Successfully downloaded latest.csv"
    echo "File saved to: $LOCAL_DEST"
    
    # Show file info
    if [ -f "$LOCAL_DEST" ]; then
        FILE_SIZE=$(du -h "$LOCAL_DEST" | cut -f1)
        echo "File size: $FILE_SIZE"
    fi
else
    echo "❌ Failed to download latest.csv"
    exit 1
fi

echo "Download completed!"
