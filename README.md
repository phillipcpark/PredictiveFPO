# PredictiveFPO

## Overview
PredictiveFPO provides a GPU-accelerated implementation of a Bidirectional Message-Passing Graph Neural Network, for predictive recommendations of variables and function calls that may be tuned to lowered precision. For now, the domain is limited to static graph representations of floating-point programs. 

A pretrained Pytorch model is provided in the resources subdirectory, in addition to a small subset of the test programs used in the reported results. Due to the large size of validation inputs (10,000 per test program), only 128 programs may be provided in this repository. The full set of 4096 test programs used in evaluating the provided model can be made available, upon reasonble request. Since this work is not yet published, the full data used to train the model is not yet publically available. 

As well as these assets, an inference API is exposed via Flask, which are implemented in the drivers with the 'serve' and 'request' prefixes. 

## Installation
A DockerFile is provided for containerized installation, and can be executed simply through building the image and deploying. For local installation, a virtual environment, such as virtualenv, is highly recommended. Package managers such as pip3 may be used to install all dependencies.

Requirements:
* Python 3.x <br/>
* Numpy <br/>
* Pytorch <br/>
* Deep Graph Library <br/>
* Networkx <br/>
* mpmath <br/>

## Model Architecture
TODO

## Usage
### Graph Representation
Labeled program graphs have a flat file representation, delimited by empty CSV rows. Operations have an integer encoding, which are converted to 1-hot binary vectors during execution. Each graph has a corresponding input set in its own CSV file, which provides inputs to all program sources.

### Configuration
A JSON configuration file must be supplied at the command line using the *-cfg* flag, containing all fields used in the execution mode. The included *config.json* is a prototypical example, containing all fields used for both training and testing.

### Training
To start a training session, invoke the following at the command line
> python3 predictiveFPO.py -cfg \<path to config.json\> -ds <path to dataset directory> -m \<0:training, 1:testing\>

If a CUDA device is to be used, then the *use_gpu* attribute in the config should be set to 'true'. Checkpoints, validation indices, and test indices will be saved to the *experiment_path* directory, specified in the config.

### Testing
To test a model, the path to the checkpoint and test indices, must be specified in the config, in the nested *pretrained* dictionary. Invoking testing from the command line is identical to training, save for the appropriate mode specification.

Test examples are not batched, so CPU testing is recommended, via setting *use_gpu* to false in the config. This is due to the *mpmath* library being invoked for each test example, to simulate test programs per the precision return from inference.

## Results
TODO

## Inference API with Flask
TODO    

 

