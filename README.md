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
Experiments run over Belle II Monte Carlo data with signal events defined as rare electroweak penguin decays and obscured by standard background processes.
`graph showing recurrent convergence/loss vs. standard dense nets`
`graph showing conv convergence/loss vs. standard dense nets`

## Extensions
The network architecture is kept modular from the remainder of the computational graph. To swap out the network for your custom one, create a `@staticmethod` under the `Network` class in `network.py`:

```python
@staticmethod
def my_network(x, config, training, **kwargs):
    """
    Inputs:
    x: example data
    config: class defining hyperparameter values
    training: Placeholder boolean tensor to distinguish between training/prediction

    Returns:
    network logits
    """

    # To prevent overfitting, we don't even look at the inputs!
    return tf.random_uniform([x.shape[0]], minval=0, maxval=config.n_classes, dtype=tf.int32, seed=42)
```
Now open model.py and edit the first line under the Model init:
```python
class Model():
    def __init__(self, **kwargs):

        arch = Network.my_network
        # Define the rest of the computational graph
```

### Dependencies
* Python 3.6
* [Pandas](https://pandas.pydata.org/)
* [TensorFlow 1.7](https://www.tensorflow.org/)

### Resources
* [Convolutional/recurrent networks for sequence modelling](https://arxiv.org/pdf/1803.01271.pdf)
* [Convolutional seq2seq learning](https://arxiv.org/pdf/1705.03122.pdf)
