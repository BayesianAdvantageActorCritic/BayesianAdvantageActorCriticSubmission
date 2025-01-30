#!/usr/bin/env bash

# Source and destination directories
SRC_DIR="./figures"
DEST_DIR="./compressed_figures"

# Choose a Ghostscript PDFSETTINGS level:
# Possible options: /screen /ebook /printer /prepress /default
COMPRESSION_LEVEL="/screen"

echo "Starting PDF compression..."
echo "Source:      $SRC_DIR"
echo "Destination: $DEST_DIR"
echo "Compression: $COMPRESSION_LEVEL"
echo

# Create destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Loop through all PDF files in the source directory
for pdf_file in "$SRC_DIR"/*.pdf; do
  if [ ! -f "$pdf_file" ]; then
    echo "No PDF files found in $SRC_DIR."
    exit 0
  fi
  
  # Extract the base name (without path and extension)
  filename="$(basename "$pdf_file" .pdf)"
    
  echo "Converting PDF to JPEG(s): $filename"

  # Convert each PDF page to JPEG
  gs \
    -dNOPAUSE \
    -dBATCH \
    -sDEVICE=jpeg \
    -dJPEGQ=20 \
    -r200 \
    -sOutputFile="$DEST_DIR/${filename}.jpg" \
    "$pdf_file"
done

echo
echo "PDF conversion complete."
