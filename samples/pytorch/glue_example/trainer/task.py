from argparse import ArgumentParser
import os
import json
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
    port = 8888
    num_gpus = int(os.environ["SM_NUM_GPUS"])
    hosts = json.loads(os.environ["SM_HOSTS"])
    num_nodes = len(hosts)
    current_host = os.environ["SM_CURRENT_HOST"]
    rank = hosts.index(current_host)
    os.environ["NCCL_DEBUG"] = "INFO"

    if num_nodes > 1:
        cmd = f"""python -m torch.distributed.launch \
            --nnodes={num_nodes}  \
            --node_rank={rank}  \
            --nproc_per_node={num_gpus}  \
            --master_addr={hosts[0]}  \
            --master_port={port} \
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