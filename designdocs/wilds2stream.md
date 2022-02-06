# WILDS to Stream

Goal: given fixed datasets D_1, D_2, ..., D_n, can we create a "stream" that interpolates between these datasets?

## Interface

The user flow might look something like:

```python

from ttb import Dataset

dataset = Dataset("wilds/taskname", sampler="default", backend="pytorch")

dl = dataset.get_loader(batch_size = 16)

# Loop through dl

dataset.step()
```

## Work

At every time step t, we only want to provide access to the t - 1 time steps of information. Users may want to loop through epochs, motivating a manual "step" function for the dataset.

The `Dataset` object will have the following attributes:

* name
* timestep cutoff (including the step)
* ordered list of images representing the "stream"
* sample function

