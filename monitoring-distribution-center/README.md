# Amazon Monitoring Distribution Center - Capstone Project

This repo you will find the following use-case, predict the number of items in a bin. The business problem that we are going train to address is to get the number of items in a bin and helping to manage inventory and orders

## Project Set Up and Installation

Environments:

* Development: Amazon SageMaker studio
* Data storage: Amazon S3
* Deployment: Amazon SageMaker training jobs and endpoints

the workflow is:

* imports and parameters
* data preparation
* hyper parameter optimization
* model training
* deployment
* testing

model develpement framework

* pytorch

## Dataset

The data is provided by Amazon and can be found in the following link: https://registry.opendata.aws/amazon-bin-imagery/. the data is provided in a .zip file that has to be unzip. The unzip file have 5 files with names 1,2,3,4 and 5 that correspond the labels or number of items by bin. Each file have different number of images:

- 1 item 1K images
- 2 items 1.8K images
- 3 items 2.3K images
- 4 items 2K images
- 5 items 1.8 images

### Overview

a sample image:

![alt text](https://github.com/miguelmayhem92/Udacity_projects/blob/main/monitoring-distribution-center/screenshots/sample_images.jpg) 

the images are jpg and some characteristics are:

* images are somewhat dark
* some images seems to be difficult to classify (even using image inspection)
* some images seems to have incorrect labels

### Access

first it is processed in the local environement (Sagemaker studio) and then uploaded to S3:

![alt text](https://github.com/miguelmayhem92/Udacity_projects/blob/main/monitoring-distribution-center/screenshots/s3bucket.jpg) 

## Model Training

Now that data is split and available in S3, it is time to train some models starting for hyper parameter tunning. The base model architecture is a Resnet base model that is finetuned for 5 classes. Here the set up of the hyperparameter tunning job:


* larning rate tuning between 0.001 and 0.1
* batch size options: 16, 32 and 64
* epochs between 2 and 6

and the estimator parameters are:

```
estimator = PyTorch(
    entry_point = 'hpo.py',
    base_job_name = 'amazon-monitoring-model-hpo',
    estimator_name  = 'amazon-monitoring-hpo',
    role = role,
    instance_count = 4, # some more instances to see if it improves tuning training execution time
    instance_type = 'ml.m5.2xlarge', 
    py_version = 'py36',
    framework_version = '1.8'
)
```
note that I set number instances equal to 4 to help with the training and actually it did. the resuts are:

![alt text](https://github.com/miguelmayhem92/Udacity_projects/blob/main/monitoring-distribution-center/screenshots/hpo-models.jpg) 


th best hyperparameters are:

```
best_hyperparameters ={
    'batch_size': 64,
    'epochs':2,
    'lr': 0.06524108473213124
}
```
and now a single training job with the hyper parameter can be trigger:

![alt text](https://github.com/miguelmayhem92/Udacity_projects/blob/main/monitoring-distribution-center/screenshots/training-job.jpg) 

the results of the previous model are:

![alt text](https://github.com/miguelmayhem92/Udacity_projects/blob/main/monitoring-distribution-center/screenshots/training_eval.jpg) 

### Deployment

to deploy the model:
 
 I used the compressed model that is available in S3 (in that way I do not depend on the model in active session). To deploy the model I followed the instructions in this official documentation https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#deploy-endpoints-from-model-data. In short a PytorchModel Object is going to upload the model in S3. Additional features are:
 
 * a unit test for the input data (raise one error if the input data is not a valid image)
 * Reshape step (because images do not have the same size and the model has an input shape!) 

![alt text](https://github.com/miguelmayhem92/Udacity_projects/blob/main/monitoring-distribution-center/screenshots/endpoint.jpg) 

### Testing the end point

![alt text](https://github.com/miguelmayhem92/Udacity_projects/blob/main/monitoring-distribution-center/screenshots/test_image.jpg) 


## Machine Learning Pipeline

Here I share a diagram of the pipeline:

![alt text](https://github.com/miguelmayhem92/Udacity_projects/blob/main/monitoring-distribution-center/screenshots/pipeline.jpg) 

1. treat the data in local
2. export the data to s3 such that execute training jobs
3. search for best hyper parameters
4. train the final model
5. deploy model

## Standout Suggestions

* Model performance can improve if more training data is provided and further arechitectures are tested
* Model training time can be improved if GPU resources are used
* Maybe vanishing gradient hooks are making the task to take longer, (because the machine is assessing if the model weight optimization is doing well) I would suggest to remove this hook and see if the model trains faster
