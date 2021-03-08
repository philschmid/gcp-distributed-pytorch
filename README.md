# Example for a distributed Training with Hugging Face and GCP

Example for distributed training on multiple-gpus

# TODO: Add Script to setup and enable install

# Getting started

1. Install `gcloud` https://cloud.google.com/sdk/docs/install

2. login into `gcloud`

```bash
gcloud auth login
```

3. Setup Project

a. create new gcloud project

```bash
gcloud projects create <PROJECT_ID>
```

After you created a project you have to [enable the API](https://console.cloud.google.com/flows/enableapi?apiid=ml.googleapis.com&authuser=3&_ga=2.51847977.1880909477.1613492124-671625421.1584534077&_gac=1.250041588.1613492125.Cj0KCQiA962BBhCzARIsAIpWEL2YIQ_6F49jxU4fLshNuBzidmLM671wecZTzyG7z_aCxrOcCz4lB5caAkL7EALw_wcB)

b. Init existing project

```bash
gcloud config set project  <PROJECT_ID>
```

4. Start training with

before you can start training adjust the variables in `start_training.sh`

```bash
./start_training.sh
```

### Extra

1. Zip you Trainer manually https://cloud.google.com/ai-platform/prediction/docs/custom-prediction-routines

```bash
python setup.py sdist --formats=gztar
```

## Ressources

- https://cloud.google.com/ai-platform/training/docs/distributed-pytorch

- https://cloud.google.com/ai-platform/training/docs/getting-started-pytorch

- pytorch documentation https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py

## Image_uris

| Container image URI                               | PyTorch version | Supported processors |
| ------------------------------------------------- | --------------- | -------------------- |
| `gcr.io/cloud-ml-public/training/pytorch-xla.1-6` | 1.6             | CPU, TPU             |
| `gcr.io/cloud-ml-public/training/pytorch-gpu.1-6` | 1.6             | GPU                  |
| `gcr.io/cloud-ml-public/training/pytorch-cpu.1-4` | 1.4             | CPU                  |
| `gcr.io/cloud-ml-public/training/pytorch-gpu.1-4` | 1.4             | GPU                  |
