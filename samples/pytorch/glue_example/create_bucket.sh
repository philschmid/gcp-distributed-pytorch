#!/bin/bash
echo "Creating Bucket, upload trainer, etc..."

BUCKET_NAME=<your bucket>

REGION=us-central1


gsutil mb -l ${REGION} gs://${BUCKET_NAME}


