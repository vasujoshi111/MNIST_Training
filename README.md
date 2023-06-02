# Training MNIST Data
> This repo helps you to understand the Receptive field concept and how the input channels are changing according to the size & kernels on simple MNIST(28\*28\*1) data.

Here we have 2 modules and one notebook which will call other two modules.

### 1. model.py
> This module has one class named Net in which the network architecture is defined and one function which has certain training craiterians are defined.
**In entire model only 3*3 kernels are used.**

This network here is simply defined in order to get the how the input chanelles are changing and receptive fields are changing after each layer in the forward.

The output channels are calculated using formula as:

*Nout = ((Nin + 2 * Padding - KernelSize) / Stride) + 1*

### 2. utils.py
> This module have dataset loaders, training and testing functions.

### 3. S5.ipynb
> This python notebook will call the other module files and train the model. We can visualize the results in notebook.
