#!/bin/bash

# Define URLs and filenames
URL_302e="NIST 302E URL goes here"
URL_302h="NIST 302H URL goes here"
FILE_302e="302e.zip"
FILE_302h="302h.zip"

# Download function
download_file() {
  local url="$1"
  local file="$2"

  if [ -f "$file" ]; then
    echo "File $file already exists. Skipping download."
  else
    echo "Downloading $file..."
    wget -O "$file" "$url"
    if [ $? -eq 0 ]; then
      echo "$file downloaded successfully."
    else
      echo "Failed to download $file."
      exit 1
    fi
  fi
}

# Create directories and download files
echo "Starting downloads in current working directory: $(pwd)"
download_file "$URL_302e" "$FILE_302e"
download_file "$URL_302h" "$FILE_302h"

echo "All downloads are complete. Files are located in: $(pwd)"
