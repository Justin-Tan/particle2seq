# particle2seq

A sequence modelling approach to event classification in particle physics, which naturally supports events with variable numbers of particles in the decay chain. This is designed to be used in conjunction with traditional approaches to background suppression based on high-level engineered features. `workshop.pdf` can be consulted for further details.

-----------------------------
## Usage
The code depends on [Tensorflow 1.8](https://github.com/tensorflow/tensorflow)
```
# Check command line arguments
$ python3 train.py -h
# Run
$ python3 train.py -opt momentum --name my_network
```
To enable adversarial training mode based on a variant of the method proposed in [Louppe et. al.](https://arxiv.org/abs/1611.01046), use `adv_train.py` in place of `train.py` and enable `use_adverary = True` in the `config` file. 

## Extensions
The network architecture is kept modular from the remainder of the computational graph. For ease of experimentation, the codebase will support any arbitrary architecture that yields logits in the context of binary classification. In addition, the adversarial training procedure can interface with any arbitrary network architecture. To swap out the network for your custom one, create a `@staticmethod` under the `Network` class in `network.py`:

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
### Monitoring / checkpoints
Tensorboard summaries are written periodically to `tensorboard/` and checkpoints are saved every 10 epochs.

## Results
Experiments run over 1 ab<sup>-1</sup> of simulated Belle II data with signal events defined as rare electroweak penguin decays and obscured by standard background processes.

![Alt text](show/all_loss_log.png?raw=true "Training/test loss for dense/recurrent/deep conv. models")
```
graph showing convergence on high-multiplicity decay chain vs. standard dense nets
```

### Dependencies
* Python 3.6
* [Pandas](https://pandas.pydata.org/)
* [TensorFlow 1.8](https://github.com/tensorflow/tensorflow)

### Resources
* [Convolutional/recurrent networks for sequence modelling](https://arxiv.org/pdf/1803.01271.pdf)
* [Convolutional seq2seq learning](https://arxiv.org/pdf/1705.03122.pdf)
* [Learning to pivot with adversarial networks](https://arxiv.org/abs/1611.01046)

### Future Work
* Add embedding layer
* Experiment with 1D convolutions / TCNs
* Port to Pytorch

### Contact
Feel free to open an issue or contact me at [justin.tan@coepp.org.au](mailto:justin.tan@coepp.org.au) for access to the dataset or questions about the model.
