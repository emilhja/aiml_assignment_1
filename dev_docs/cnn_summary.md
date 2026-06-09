# CNN Summary

## Current CNN architecture

The current default model is a small CNN adapted for MNIST classification.

Layer sequence:

- `Conv2d(1, 32, kernel_size=3, padding=1)`
- `LeakyReLU`
- `MaxPool2d(2, 2)`
- `Conv2d(32, 64, kernel_size=3, padding=1)`
- `LeakyReLU`
- `MaxPool2d(2, 2)`
- `Flatten`
- `Linear(64 * 7 * 7, 128)`
- `LeakyReLU`
- `Linear(128, 10)`

## Tensor dimensions through the network

For one MNIST image:

- Input: `1 x 28 x 28`
- After first convolution: `32 x 28 x 28`
- After first max pooling: `32 x 14 x 14`
- After second convolution: `64 x 14 x 14`
- After second max pooling: `64 x 7 x 7`
- After flatten: `3136`
- Final output: `10` class scores

With batch size 64 during training, the tensors include the batch dimension as well, for example:

- `64 x 1 x 28 x 28`
- `64 x 32 x 28 x 28`
- `64 x 32 x 14 x 14`
- `64 x 64 x 14 x 14`
- `64 x 64 x 7 x 7`

## Why this is a CNN

This model uses convolutional layers instead of only fully connected layers. That allows it to detect local visual patterns such as edges, curves, and small digit shapes directly in the image. The `MaxPool2d` layers reduce the spatial dimensions and help keep the feature maps manageable for later layers.

Compared to the earlier FFN approach, this CNN is better suited for image data because it preserves spatial structure and is more robust to small translations in the input.

## Relation to ResNet50

This model is not similar to ResNet50 beyond the fact that both are convolutional neural networks.

The current model is:

- a small CNN
- only 2 convolution layers deep
- designed for MNIST
- easy to understand and fast to train

ResNet50 is:

- a much deeper network with about 50 layers
- built from many convolution blocks
- based on residual (skip) connections
- typically used for more difficult image recognition tasks

So the present model is best described as a classic introductory CNN, not a ResNet-style architecture.

## CNN variant comparison

| Model | Conv channels | Hidden layer | Activation | Dropout | Main purpose |
| --- | --- | --- | --- | --- | --- |
| `cnn_small` | `16 -> 32` | `64` | `ReLU` | `0.0` | Lightweight baseline with lower capacity |
| `cnn_medium` | `32 -> 64` | `128` | `LeakyReLU` | `0.0` | Balanced default model with strong performance |
| `cnn_dropout` | `32 -> 64` | `128` | `ReLU` | `0.3` | Tests whether extra regularization improves generalization |
| `cnn_deep_balanced` | `32 -> 64 -> 64` | `512` | `ReLU` | `0.0` | Adds a third convolution layer while staying in a similar parameter range |
| `cnn_deep_wide` | `32 -> 64 -> 128` | `256` | `ReLU` | `0.0` | Adds a third convolution layer with a wider final feature stage |

In short:

- `cnn_small` tests a smaller network.
- `cnn_medium` is the main balanced model.
- `cnn_dropout` tests the effect of regularization.
- `cnn_deep_balanced` tests a deeper CNN with a third convolution layer.
- `cnn_deep_wide` tests a deeper and slightly wider CNN.

## How to interpret later convolution layers

Typical pattern:

- The first convolution layer learns simple local features such as edges, stroke direction, and small blobs.
- Later convolution layers combine those simple patterns into more complex digit parts such as corners, loops, junctions, and curved stroke combinations.
- That is why deeper CNNs can sometimes classify digits better: later layers can represent more abstract structure than the first layer alone.

The comparison workflow now saves filter visualizations for the first and last convolution layer so these differences can be discussed directly from each trained model.
