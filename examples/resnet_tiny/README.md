# swin_tiny example

This example shows how to use a Swin Transformer V2 Tiny model for image classification.

## Installing the Dataset

See:
* https://github.com/BayesWatch/cinic-10
* https://datashare.ed.ac.uk/handle/10283/3192

1. Download the dataset, and unpack it.
2. Set the environment variable `CINIC10_PATH` to the path of the unpacked dataset.

## Running the Example

Run the training:

```bash
cargo run --release -p swin_tiny -- \
  --training-root $CINIC10_PATH/train \
  --validation-root $CINIC10_PATH/val 
```


