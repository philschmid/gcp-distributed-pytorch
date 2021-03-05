from functools import reduce
from googleapiclient import discovery
from googleapiclient import errors

# Store your full project ID in a variable in the format the API needs.

project_id = 'huggingface-ml'
job_name = "ddp-distributed-test"
region = 'us-central1'
image_uri = "gcr.io/cloud-ml-public/training/pytorch-gpu.1-6"
# Build a representation of the Cloud ML API.
ml = discovery.build('ml', 'v1')

# hyperparameter
hyperparameter = {}

# Create a dictionary with the fields from the request body. https://cloud.google.com/ai-platform/training/docs/reference/rest/v1/projects.jobs#TrainingInput
training_inputs = {
    'scaleTier': 'CUSTOM',
    'masterType': 'complex_model_m_p100', # https://cloud.google.com/ai-platform/training/docs/machine-types
    'masterConfig': {
      'image_uri':image_uri
    },
    'workerCount': 1,
    'packageUris': ['gs://my/trainer/path/package-0.0.0.tar.gz'],
    'pythonModule': 'trainer.task',
    'args':  reduce(lambda x,y: x+y, [[f"--{parameter}",value] for parameter,value in hyperparameter.items()]),
    'region': region,
    'jobDir': 'gs://my/training/job/directory',
    'runtimeVersion': '2.4',
    'pythonVersion': '3.7',
    'scheduling': {'maxWaitTime': '3600s', 'maxRunningTime': '7200s'},
}


job_spec = {'jobId': job_name, 'trainingInput': training_inputs}



# Create a request to call projects.models.create.
request = ml.projects().models().create(
              parent=f'projects/{project_id}', body=job_spec)

# Make the call.
try:
    response = request.execute()
    print(response)
except errors.HttpError as err:
    # Something went wrong, print out some information.
    print('There was an error creating the model. Check the details:')
    print(err._get_reason())