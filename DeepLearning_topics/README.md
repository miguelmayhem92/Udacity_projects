# Image Classification using AWS SageMaker

Use AWS Sagemaker to train a pretrained model that can perform image classification by using the Sagemaker profiling, debugger, hyperparameter tuning and other good ML engineering practices. This can be done on either the provided dog breed classication data set or one of your choice.

## Project Set Up and Installation
This project is done in AWS SageMaker using s3 for data storage and SageMaker studio for training, validation and deployment

## Dataset
The provided dataset is the dogbreed classification dataset which can be found in the classroom.

this dataset contains data for 133 breeds or classes. The following plot shows the breed distribution in the training set:

![alt text](https://github.com/miguelmayhem92/udacity_projects/blob/main/DeepLearning_topics/screenshots/train_breed_dist.jpg)

### Access

the data was quickly assessed and then uploaded to S3:

![alt text](https://github.com/miguelmayhem92/udacity_projects/blob/main/DeepLearning_topics/screenshots/s3_data_split.jpg)

the dataset contains:

* 6680 images for traning
* 835 images for validation
* 836 images for test

## Hyperparameter Tuning

The choosen model for this task  was ResNet50 (pretrained model) plus a custom layer for finetuning

![alt text](https://github.com/miguelmayhem92/udacity_projects/blob/main/DeepLearning_topics/screenshots/resnet.png)

RestNet is a good model because it is powerful model for computer vision. In short the archetecture relies on a very long chains of convolutional cells, each cell contains convolutional layers from the previous cell that is used as a residual layer that helps to add noise to the CNN and to address overfit

for hyperparameter tunning I used:

* larning rate tuning between 0.001 and 0.1
* batch size options: 16, 32 and 64
* epochs between 2 and 6

the training jobs in AWS SageMakers:

![alt text](https://github.com/miguelmayhem92/udacity_projects/blob/main/DeepLearning_topics/screenshots/training_jobs.jpg)

and the best hyperparameters are:

```
{
    'batch_size': 64,
    'epochs': 3,
    'lr': 0.014088958140267716
}
```

## Debugging and Profiling

After getting the best hyperparameters, the actual model was trained using profiling and debuging rules
the rules are:

* Vanishing gradient (whether weights tends to 0 or are very near to 0)
* poor_weight_initialization (whether initial weights are not good enought for validation inside the training loop)
* ProfilerReport (resource utilization for the training job)

### Results

The training phase did not trigger alarms concerning the Vanishing gradient or weight initialization, but the profiler gave some insights:

* the training job took 56 minutes (big part of the time was the training loop)
* the training job spend 34% of the time in a OTHERS tasks (no traning no evaluation) so it suggest to inspect why a significant amount of time is spent in Others jobs and not traning and evaluation
* CPU resources was spent in MkldnnConvolutionBackward (14%) and Convolutions Forward (8.64%) as main CPU's resources expenditure
* the profiler rule that was mainly activated was 'StepOutlier' meaning that there are bottlenecks during training phase (backward and forward propagation).

![alt text](https://github.com/miguelmayhem92/udacity_projects/blob/main/DeepLearning_topics/screenshots/profiler_plots.jpg)

(the full report is in the profiler-report.html)

## Model Deployment

the end-point was deployed:

![alt text](https://github.com/miguelmayhem92/udacity_projects/blob/main/DeepLearning_topics/screenshots/endpoint.jpg)

to deploy the model:
 
 I used the compressed model that is available in S3 (in that way I do not depend on the model in active session). To deploy the model I followed the instructions in this official documentation https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#deploy-endpoints-from-model-data. In short a PytorchModel Object is going to upload the model in S3. Additional features are:
 
 * a unit test for the input data (raise one error if the input data is not a valid image)
 * Reshape step (because images do not have the same size and the model has an input shape!) 

![alt text](https://github.com/miguelmayhem92/udacity_projects/blob/main/DeepLearning_topics/screenshots/prediction.jpg)

## Standout Suggestions

* Model performance can improve if more training jobs are trigger (just 4 were done) so that better parameters could be found
* Model training time can be improved if GPU resources are used
* Maybe vanishing gradient hooks are making the task to take longer, (because the machine is assessing if the model weight optimization is doing well) I would suggest to remove this hook and see if the model trains faster
