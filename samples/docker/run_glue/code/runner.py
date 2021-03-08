from argparse import ArgumentParser
from google.cloud import storage
import os
import datetime
import subprocess
import logging
import glob

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
            ./run_glue.py \
            {"".join([f" --{parameter} {value}" for parameter,value in args.__dict__.items()])}"""
    else:
        cmd = f"""python -m torch.distributed.launch \
        --nproc_per_node={num_gpus}  \
        ./run_glue.py \
        {"".join([f" --{parameter} {value}" for parameter,value in args.__dict__.items()])}"""
    try:
        subprocess.run(cmd, shell=True)
    except Exception as e:
        logger.info(e)

    save_model(args)


if __name__ == "__main__":
    main()