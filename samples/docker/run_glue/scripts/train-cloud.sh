echo "Submitting AI Platform PyTorch job"

# Variables needed
PROJECT_ID=huggingface-ml
REGION=us-central1 # REGION: select a region from https://cloud.google.com/ml-engine/docs/regions
JOB_NAME=pytorch_job_$(date +%Y%m%d_%H%M%S) # JOB_NAME: the name of your job running on AI Platform.
BUCKET_NAME=ddp-pytorch-test # need to be adjsuted
IMAGE_REPO_NAME=ddp-pytorch-test # IMAGE_REPO_NAME: the image will be stored on Cloud Container Registry

MACHINE_TYPE=complex_model_m_p100 # Machines used for training


IMAGE_TAG=run_glue  # IMAGE_TAG: an easily identifiable tag for your docker image
IMAGE_URI=gcr.io/${PROJECT_ID}/${IMAGE_REPO_NAME}:${IMAGE_TAG} # IMAGE_URI: the complete URI location for Cloud Container Registry

# Build the docker image
docker build -f Dockerfile -t ${IMAGE_URI} ./

# Deploy the docker image to Cloud Container Registry
docker push ${IMAGE_URI}

# Submit your training job
echo "Submitting the training job"

# These variables are passed to the docker image
JOB_DIR=gs://${BUCKET_NAME}/${JOB_NAME}/models


gcloud ai-platform jobs submit training ${JOB_NAME} \
    --region ${REGION} \
    --master-image-uri ${IMAGE_URI} \
    --job-dir ${JOB_DIR} \
    --scale-tier custom \
    --master-machine-type ${MACHINE_TYPE} \
    --worker-machine-type ${MACHINE_TYPE} \
    --worker-count 1 \
    -- \
    --model_name_or_path bert-base-cased \
    --task_name mrpc \
    --do_train True \
    --do_eval True \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --output_dir /tmp/mrpc/


# Stream the logs from the job
gcloud ai-platform jobs stream-logs ${JOB_NAME}

# Verify the model was exported
echo "Verify the model was exported:"
gsutil ls ${JOB_DIR}/model_*
