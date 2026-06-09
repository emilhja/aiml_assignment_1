
### CNN layer details

All Part 2 CNN models take MNIST images with shape `1 x 28 x 28`, meaning one grayscale channel and a 28 by 28 pixel image. The convolutional part is used as a feature extractor, and the final dense classifier maps the extracted features to the 10 digit classes.

Each configurable CNN block follows this pattern:

`Conv2d -> optional BatchNorm2d -> activation -> MaxPool2d`

The convolution layers use `kernel_size=3` and `padding=1` in the main architecture comparison. Padding keeps the height and width unchanged during the convolution itself, so a `28 x 28` feature map is still `28 x 28` after the convolution. The `MaxPool2d(kernel_size=2, stride=2)` layer then halves the spatial size. This means:

- After one pool: `28 x 28 -> 14 x 14`
- After two pools: `14 x 14 -> 7 x 7`
- After three pools: `7 x 7 -> 3 x 3`

The channel count changes in the convolution layers. The spatial size changes in the pooling layers.

**`cnn_small`**

- Input: `1 x 28 x 28`
- Conv 1: `1 -> 16` channels, output `16 x 28 x 28`
- MaxPool 1: output `16 x 14 x 14`
- Conv 2: `16 -> 32` channels, output `32 x 14 x 14`
- MaxPool 2: output `32 x 7 x 7`
- Flatten: `32 * 7 * 7 = 1568` features
- Classifier: `1568 -> 64 -> 10`

This is the smallest CNN tested. It has fewer filters and a smaller dense layer, so it trains with fewer parameters while still using convolution and pooling to detect local digit features.

**`cnn_medium`**

- Input: `1 x 28 x 28`
- Conv 1: `1 -> 32` channels, output `32 x 28 x 28`
- MaxPool 1: output `32 x 14 x 14`
- Conv 2: `32 -> 64` channels, output `64 x 14 x 14`
- MaxPool 2: output `64 x 7 x 7`
- Flatten: `64 * 7 * 7 = 3136` features
- Classifier: `3136 -> 128 -> 10`

This is the baseline CNN. Compared with `cnn_small`, it doubles the channel counts and uses a larger hidden layer. It uses `LeakyReLU`, which keeps a small gradient for negative activations instead of setting them fully to zero.

**`cnn_dropout`**

- Same convolution shape as `cnn_medium`: `1 -> 32 -> 64`
- Same flatten size: `64 * 7 * 7 = 3136`
- Classifier: `3136 -> 128 -> Dropout(0.3) -> 10`

This model tests regularization through dropout. Dropout is placed after the classifier activation, so during training it randomly disables 30% of the hidden classifier features. The goal is to reduce overfitting by preventing the classifier from relying too heavily on a small set of activations.

**`cnn_deep_balanced`**

- Input: `1 x 28 x 28`
- Conv 1: `1 -> 32`, output `32 x 28 x 28`
- MaxPool 1: output `32 x 14 x 14`
- Conv 2: `32 -> 64`, output `64 x 14 x 14`
- MaxPool 2: output `64 x 7 x 7`
- Conv 3: `64 -> 64`, output `64 x 7 x 7`
- MaxPool 3: output `64 x 3 x 3`
- Flatten: `64 * 3 * 3 = 576` features
- Classifier: `576 -> 512 -> 10`

This model adds a third convolution layer but keeps the last channel count at 64. The extra convolution lets the model combine earlier low-level features into more complex digit features before classification. Because the third pooling step reduces the feature map to `3 x 3`, the dense classifier receives fewer spatial features than the 2-conv models.

**`cnn_deep_wide`**

- Input: `1 x 28 x 28`
- Conv 1: `1 -> 32`, output `32 x 28 x 28`
- MaxPool 1: output `32 x 14 x 14`
- Conv 2: `32 -> 64`, output `64 x 14 x 14`
- MaxPool 2: output `64 x 7 x 7`
- Conv 3: `64 -> 128`, output `128 x 7 x 7`
- MaxPool 3: output `128 x 3 x 3`
- Flatten: `128 * 3 * 3 = 1152` features
- Classifier: `1152 -> 256 -> 10`

This was the best architecture comparison result. It adds depth like `cnn_deep_balanced`, but also increases the last convolution layer to 128 channels. That gives the final feature extractor more filters for higher-level digit patterns while keeping the dense classifier smaller than `cnn_deep_balanced`.

**Batch-normalized and regularized variants**

The `cnn_batchnorm` and `cnn_regularized` models use the same 3-conv channel pattern as `cnn_deep_balanced`: `[32, 64, 64]`. The difference is that `BatchNorm2d` is inserted after each convolution and before the activation:

`Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d`

`cnn_batchnorm` uses dropout `0.1`, while `cnn_regularized` uses dropout `0.25`. These variants were mainly used in regularization and hyperparameter tuning experiments, not in the five-model architecture comparison table above.