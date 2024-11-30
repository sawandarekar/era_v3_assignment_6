# ERA V3 Session 6

## Situation:
  Tasked with an advanced assignment involving deep learning concepts covered in recent lectures, focusing on building a neural network that achieves high validation/test accuracy with specific constraints. 
  The assignment required integrating various techniques like Batch Normalization, Dropout, and managing parameters effectively.

## Task:
  Objective was to design a convolutional neural network that achieves 99.4% validation/test accuracy with less than 20,000 parameters, under 20 epochs, 
  while employing Batch Normalization and Dropout as part of the architecture. 
  Additionally, you needed to document your process and results in a public GitHub repository and ensure proper formatting and content in the README.md file.

## Action:
  Researched and integrated techniques such as:
  - Utilizing MaxPooling to reduce dimensionality and avoid overfitting.
  - Implementing 3x3 convolutions to learn features effectively.
  - Applying Batch Normalization after convolutions to stabilize learning and improve convergence.
  - Introducing Dropout layers to mitigate overfitting.
  - Optimizing the learning rate and batch size for better performance.
  - Setting up GitHub Actions to validate your model's parameters and techniques used.
  - Iterated on the model architecture, running multiple experiments to fine-tune the network, while keeping an eye on overfitting signs.

## Result:
  - Successfully designed a CNN and achieved 99.49% validation/test accuracy with 19,866 parameters with 20 epochs. 
  - Implemented GitHub Actions confirmed the use of Batch Normalization, Dropout, and a GAP


## Model Architecture
```
Net(
  (conv1): Sequential(
    (0): Conv2d(1, 32, kernel_size=(5, 5), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.1, inplace=False)
    (4): Conv2d(32, 24, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (5): ReLU()
    (6): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (7): Dropout(p=0.1, inplace=False)
  )
  (conv2): Sequential(
    (0): Conv2d(24, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (1): ReLU()
    (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.1, inplace=False)
  )
  (conv3): Sequential(
    (0): Conv2d(16, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (1): ReLU()
    (2): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.1, inplace=False)
  )
  (conv4): Sequential(
    (0): Conv2d(24, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  )
  (conv5): Sequential(
    (0): Conv2d(16, 10, kernel_size=(3, 3), stride=(1, 1))
  )
  (gap): AvgPool2d(kernel_size=5, stride=5, padding=0)
)
```


## Model Summary



Requirement already satisfied: torchsummary in /usr/local/lib/python3.10/dist-packages (1.5.1)
device: cuda

|        Layer (type)      |         Output Shape      |   Param #   |
| ------------------------ | ------------------------- | ----------- |
|            Conv2d-1      |       [-1, 32, 26, 26]    |       832     |
|              ReLU-2      |       [-1, 32, 26, 26]    |         0     |
|       BatchNorm2d-3      |       [-1, 32, 26, 26]    |        64     |
|           Dropout-4      |       [-1, 32, 26, 26]    |         0     |
|            Conv2d-5      |       [-1, 24, 26, 26]    |       6,936   |
|              ReLU-6      |       [-1, 24, 26, 26]    |         0     |
|       BatchNorm2d-7      |       [-1, 24, 26, 26]    |        48     |
|           Dropout-8      |       [-1, 24, 26, 26]    |         0     |
|            Conv2d-9      |       [-1, 16, 13, 13]    |       3,472   |
|             ReLU-10      |       [-1, 16, 13, 13]    |         0     |
|      BatchNorm2d-11      |       [-1, 16, 13, 13]    |        32     |
|          Dropout-12      |       [-1, 16, 13, 13]    |         0     |
|           Conv2d-13      |       [-1, 24, 7, 7]      |       3,480   |
|             ReLU-14      |       [-1, 24, 7, 7]      |         0     |
|      BatchNorm2d-15      |       [-1, 24, 7, 7]      |        48     |
|          Dropout-16      |       [-1, 24, 7, 7]      |         0     |
|           Conv2d-17      |       [-1, 16, 7, 7]      |       3,472   |
|             ReLU-18      |       [-1, 16, 7, 7]      |         0     |
|      BatchNorm2d-19      |       [-1, 16, 7, 7]      |        32     |
|           Conv2d-20      |       [-1, 10, 5, 5]      |       1,450   |
|        AvgPool2d-21      |       [-1, 10, 1, 1]      |         0     |

  ----------------------------------------------------------------

Total params: 19,866

Trainable params: 19,866

Non-trainable params: 0
Input size (MB): 0.00

Forward/backward pass size (MB): 1.29

Params size (MB): 0.08

Estimated Total Size (MB): 1.37

----------------------------------------------------------------


## Training logs

```shell
epoch=1 loss=0.06565283238887787 batch_id=468: 100%|██████████| 469/469 [00:26<00:00, 17.81it/s]

Test set: Average loss: 0.0863, Accuracy: 9756/10000 (97.56%)

epoch=2 loss=0.12914450466632843 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 23.76it/s]

Test set: Average loss: 0.0423, Accuracy: 9881/10000 (98.81%)

epoch=3 loss=0.09096860885620117 batch_id=468: 100%|██████████| 469/469 [00:21<00:00, 21.95it/s]

Test set: Average loss: 0.0390, Accuracy: 9879/10000 (98.79%)

epoch=4 loss=0.07025357335805893 batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 22.54it/s]

Test set: Average loss: 0.0325, Accuracy: 9902/10000 (99.02%)

epoch=5 loss=0.006744968239217997 batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 22.35it/s]

Test set: Average loss: 0.0303, Accuracy: 9902/10000 (99.02%)

epoch=6 loss=0.017363520339131355 batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 23.38it/s]

Test set: Average loss: 0.0283, Accuracy: 9904/10000 (99.04%)

epoch=7 loss=0.05886194482445717 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 23.46it/s]

Test set: Average loss: 0.0298, Accuracy: 9900/10000 (99.00%)

epoch=8 loss=0.036921724677085876 batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 22.49it/s]

Test set: Average loss: 0.0272, Accuracy: 9909/10000 (99.09%)

epoch=9 loss=0.01268548984080553 batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 22.43it/s]

Test set: Average loss: 0.0240, Accuracy: 9912/10000 (99.12%)

epoch=10 loss=0.030316432937979698 batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 23.02it/s]

Test set: Average loss: 0.0234, Accuracy: 9927/10000 (99.27%)

epoch=11 loss=0.020694730803370476 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 23.48it/s]

Test set: Average loss: 0.0238, Accuracy: 9926/10000 (99.26%)

epoch=12 loss=0.018838468939065933 batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 22.77it/s]

Test set: Average loss: 0.0229, Accuracy: 9927/10000 (99.27%)

epoch=13 loss=0.009510700590908527 batch_id=468: 100%|██████████| 469/469 [00:21<00:00, 22.20it/s]

Test set: Average loss: 0.0214, Accuracy: 9931/10000 (99.31%)

epoch=14 loss=0.013673587702214718 batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 22.56it/s]

Test set: Average loss: 0.0189, Accuracy: 9936/10000 (99.36%)

epoch=15 loss=0.02229400724172592 batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 22.59it/s]

Test set: Average loss: 0.0183, Accuracy: 9944/10000 (99.44%)

epoch=16 loss=0.018997881561517715 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 23.63it/s]

Test set: Average loss: 0.0202, Accuracy: 9937/10000 (99.37%)

epoch=17 loss=0.0058743711560964584 batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 22.57it/s]

Test set: Average loss: 0.0224, Accuracy: 9928/10000 (99.28%)

epoch=18 loss=0.017018208280205727 batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 22.48it/s]

Test set: Average loss: 0.0176, Accuracy: 9944/10000 (99.44%)

epoch=19 loss=0.010610316880047321 batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 23.33it/s]

Test set: Average loss: 0.0183, Accuracy: 9945/10000 (99.45%)

epoch=20 loss=0.010763414204120636 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 23.72it/s]

Test set: Average loss: 0.0191, Accuracy: 9949/10000 (99.49%)
```


[![Notebook Test Workflow](https://github.com/sawandarekar/era_v3_assignment_6/actions/workflows/notebook_test.yml/badge.svg)](https://github.com/{username}/{repository-name}/actions/workflows/notebook_test.yml)

This repository contains the assignment for Session 6 of ERA V3. The model implementation follows these specifications:

## Model Requirements Status
| Requirement | Status |
|------------|---------|
| Parameters < 20k | ![Test Status](https://github.com/sawandarekar/era_v3_assignment_6/actions/workflows/notebook_test.yml/badge.svg?event=push&label=parameters) |
| Batch Normalization | ![Test Status](https://github.com/sawandarekar/era_v3_assignment_6/actions/workflows/notebook_test.yml/badge.svg?event=push&label=batch-norm) |
| Dropout | ![Test Status](https://github.com/sawandarekar/era_v3_assignment_6/actions/workflows/notebook_test.yml/badge.svg?event=push&label=dropout) |
| GAP/FC Layer | ![Test Status](https://github.com/sawandarekar/era_v3_assignment_6/actions/workflows/notebook_test.yml/badge.svg?event=push&label=architecture) |
| Accuracy > 99.4% | ![Test Status](https://github.com/sawandarekar/era_v3_assignment_6/actions/workflows/notebook_test.yml/badge.svg?event=push&label=accuracy) |


