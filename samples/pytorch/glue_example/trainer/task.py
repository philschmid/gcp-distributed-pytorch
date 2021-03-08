from argparse import ArgumentParser
from google.cloud import storage
import os
import json
import datetime
import subprocess
import logging
import glob

from trainer.run_glue import main

logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser()
    parsed, unknown = parser.parse_known_args()
    for arg in unknown:
        if arg.startswith(("-", "--")):
            parser.add_argument(arg.split("=")[0])

    return parser.parse_args()


def upload_local_directory_to_gcs(local_path, bucket, gcs_path):
    assert os.path.isdir(local_path)
    for local_file in glob.glob(local_path + "/**"):
        if not os.path.isfile(local_file):
            upload_local_directory_to_gcs(local_file, bucket, gcs_path + "/" + os.path.basename(local_file))
    else:
        remote_path = os.path.join(gcs_path, local_file[1 + len(local_path) :])
        blob = bucket.blob(remote_path)
        blob.upload_from_filename(local_file)


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()

    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print("Blob {} downloaded to {}.".format(source_blob_name, destination_file_name))


def save_model(args):
    """Saves the model to Google Cloud Storage
    Args:
      args: contains name for saved model.
    """
    scheme = "gs://"
    bucket_name = args.job_dir[len(scheme) :].split("/")[0]

    prefix = "{}{}/".format(scheme, bucket_name)
    bucket_path = args.job_dir[len(prefix) :].rstrip("/")

    datetime_ = datetime.datetime.now().strftime("model_%Y%m%d_%H%M%S")
    gcs_path = "{}/{}/{}".format(bucket_path, datetime_, args.model_name_or_path)

    bucket = storage.Client().bucket(bucket_name)
    upload_local_directory_to_gcs(args.output_dir, bucket, gcs_path)


def main():
    args = parse_args()

    print(os.environ)
    print(args.job_dir)
    print(args.train_script)
    scheme = "gs://"
    bucket_name = args.train_script[len(scheme) :].split("/")[0]

    prefix = "{}{}/".format(scheme, bucket_name)
    bucket_path = args.train_script[len(prefix) :].rstrip("/")

    download_blob(bucket_name, bucket_path, "train.py")

    num_gpus = 4  # TODO: find a better way to get the gpu number
    num_nodes = os.environ["WORLD_SIZE"]
    rank = os.environ["RANK"]
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]
    os.environ["NCCL_DEBUG"] = "INFO"

    if int(num_nodes) > 1:
        cmd = f"""python -m torch.distributed.launch \
            --nnodes={num_nodes}  \
            --node_rank={rank}  \
            --nproc_per_node={num_gpus}  \
            --master_addr={master_addr}  \
            --master_port={master_port} \
            ./train.py \
            {"".join([f" --{parameter} {value}" for parameter,value in args.__dict__.items()])}"""
    else:
        cmd = f"""python -m torch.distributed.launch \
        --nproc_per_node={num_gpus}  \
        ./train.py \
        {"".join([f" --{parameter} {value}" for parameter,value in args.__dict__.items()])}"""
    try:
        subprocess.run(cmd, shell=True)
    except Exception as e:
        logger.info(e)

    save_model(args)


if __name__ == "__main__":
    main()