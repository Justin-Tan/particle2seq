# particle2seq

A sequence modelling approach to event classification in particle physics, which naturally supports events with variable numbers of particles in the decay chain. This is designed to be used in conjunction with traditional approaches to background suppression based on high-level engineered features. `workshop.pdf` can be consulted for further details.

-----------------------------
## Usage
The code depends on [Tensorflow](https://www.tensorflow.org/)
```
# Check command line arguments
$ python3 train.py -h
# Run
$ python3 train.py -opt momentum --name my_network
```

## Results
Experiments run over 1 ab<sup>-1</sup> of simulated Belle II data with signal events defined as rare electroweak penguin decays and obscured by standard background processes.

![Alt text](show/loss_log.png?raw=true "Training/test loss")
```
graph showing convergence on high-multiplicity decay chain vs. standard dense nets
```
## Extensions
The network architecture is kept modular from the remainder of the computational graph. To swap out the network for your custom one, create a `@staticmethod` under the `Network` class in `network.py`:

```python
@staticmethod
def my_network(x, config, **kwargs):
    """
    Inputs:
    x: example data
    config: class defining hyperparameter values

    Returns:
    network logits
    """

    # To prevent overfitting, we don't even look at the inputs!
    return tf.random_normal([x.shape[0], config.n_classes], seed=42)
```
Now open model.py and edit the first line under the Model init:
```python
class Model():
    def __init__(self, **kwargs):

        arch = Network.my_network
        # Define the computational graph
```

### Dependencies
* Python 3.6
* [Pandas](https://pandas.pydata.org/)
* [TensorFlow 1.7](https://www.tensorflow.org/)

### Resources
* [Convolutional/recurrent networks for sequence modelling](https://arxiv.org/pdf/1803.01271.pdf)
* [Convolutional seq2seq learning](https://arxiv.org/pdf/1705.03122.pdf)

### Future Work
* Add embedding layer
* Experiment with 1D convolutions / TCNs
* Port to Pytorch

### Contact
Feel free to contact me at [justin.tan@coepp.org.au](mailto:justin.tan@coepp.org.au) for access to the dataset or questions about the model.
