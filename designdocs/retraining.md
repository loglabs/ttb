# Designdoc for Retraining Framework

Author: @shreyashankar

## Introduction

Many ML tasks on top of _streams_ of data require some retraining or continual learning to continuously exhibit high performance. Existing ML evaluation frameworks accept a trained model and a fixed hold-out dataset. We want to create a framework that allows users to easily evaluate their retraining policies.

## Solution Overview

We will provide a Python-based DSL for users to evaluate pipelines. Users can specify either scheduled retraining policies or react to .predefined "triggers."

## User Requirements

### User Characteristics  

Users are ML practitioners (e.g., Kaggle users, people who train models). They typically will use Pandas dataframes, PyTorch Datasets, or TF Data for data management. They typically will use sklearn, xgboost, PyTorch, or Tensorflow for modeling.

      
### User Interfaces   

The user will implement `mltrace` Component classes. Training and Serving components are required. The base Component class comes with abstract `beforeRun` and `afterRun` trigger stubs, where historical inputs & outputs are exposed to the user. When users fill out these stubs, they can optionally trigger another component to rerun.

**Question: do we need to integrate into a scheduler (e.g., Airflow) so usual dependencies are retriggered?**

An example user flow looks like this:

```python
from mltrace import Component, refresh

import numpy as np

class Serving(Component):

    # You can look at self.history.inputs and self.history.outputs

    def beforeRun(self):
        pass

    def afterRun(self):
        if np.mean(self.history.outputs[-100:]) < threshold:
            refresh("Training")

    def run(self):
        # Run serving logic

```

## System Requirements  

### Functional Requirements
*List here the functional requirements of the system. Functional requirements are requirements that specify __what__ the system should do and can be thought of as 'the system must do <requirement\>'. Implementation details for each requirement should be addressed in the system design document. An example of a functional requirement would be 'the system utilizes Java version...' This list can become quite extensive and for best practice each requirement should be issued its own unique name, number, and be accompanied by a description.*

### Degign Considerations


#### Assumptions
*In this section describe any assumptions, background, or dependencies of the software, its use, the operational environment, or significant project issues.*

#### Constraints
*In this section describe any constraints on the system that have a significant impact on the design of the system.*

#### System Environment
*In this section describe the system environment on which the software will be executing. Include any specific reasons why this system was chosen and if there are any plans to include new sections to the list of current ones.*

## Architecture

### Overview
*Provide here a descriptive overview of the software/system/application architecture.*


