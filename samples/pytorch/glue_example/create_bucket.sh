#!/bin/bash
echo "Creating Bucket, upload trainer, etc..."

BUCKET_NAME=ddp-pytorch-test

REGION=us-central1


gsutil mb -l ${REGION} gs://${BUCKET_NAME}


