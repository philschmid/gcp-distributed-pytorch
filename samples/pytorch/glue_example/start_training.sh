#!/bin/bash
echo "Submitting AI Platform PyTorch job"

# BUCKET_NAME: Change to your bucket name.
BUCKET_NAME=ddp-pytorch-test # need to be adjsuted

# The PyTorch image provided by AI Platform Training.
IMAGE_URI=gcr.io/cloud-ml-public/training/pytorch-gpu.1-6

# Machines used for training
MACHINE_TYPE=complex_model_m_p100


# JOB_NAME: the name of your job running on AI Platform.
JOB_NAME=pytorch_job_$(date +%Y%m%d_%H%M%S)


# REGION: select a region from https://cloud.google.com/ml-engine/docs/regions
# or use the default '`us-central1`'. The region is where the job will be run.
REGION=us-central1

# JOB_DIR: Where to store prepared package and upload output model.
JOB_DIR=gs://${BUCKET_NAME}/${JOB_NAME}/models

# JOB_DIR: Where the store datasets are located. 
# comment if you provide train_data
# TRAIN_FILES=${GCS_TAXI_TRAIN_SMALL}
# EVAL_FILES=${GCS_TAXI_EVAL_SMALL}

# create trainer package
echo "creating Trainer package"

python setup.py sdist --formats=gztar

MODULE_NAME=trainer.task
PACKAGE_PATH=dist/trainer-0.1.tar.gz
PATH_TO_PACKAGED_TRAINER=gs://${BUCKET_NAME}/code/trainer-0.1.tar.gz

gsutil cp ${PACKAGE_PATH} gs://${BUCKET_NAME}/code/

# start training
# https://cloud.google.com/ai-platform/training/docs/reference/rest/v1/projects.jobs#TrainingInput
gcloud ai-platform jobs submit training ${JOB_NAME} \
    --region ${REGION} \
    --master-image-uri ${IMAGE_URI} \
    --job-dir ${JOB_DIR} \
    --module-name ${MODULE_NAME} \
    --packages ${PATH_TO_PACKAGED_TRAINER} \
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
    # --train-files ${TRAIN_FILES} \
    # --eval-files ${EVAL_FILES} \

# Stream the logs from the job
gcloud ai-platform jobs stream-logs ${JOB_NAME}

# Verify the model was exported
echo "Verify the model was exported:"
gsutil ls ${JOB_DIR}/model_*


python -u -c 'import sys, setuptools, tokenize; sys.argv[0] = '\"'\"'/tmp/pip-req-build-e0dr_6jo/setup.py'\"'\"'; __file__='\"'\"'/tmp/pip-req-build-e0dr_6jo/setup.py'\"'\"';f=getattr(tokenize, '\"'\"'open'\"'\"', open)(__file__);code=f.read().replace('\"'\"'\\r\\n'\"'\"', '\"'\"'\\n'\"'\"');f.close();exec(compile(code, __file__, '\"'\"'exec'\"'\"'))' bdist_wheel -d /tmp/pip-wheel-q37ipymm\n",
