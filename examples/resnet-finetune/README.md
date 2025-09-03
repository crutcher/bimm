# resnet-finetune example

The dataload on this example is base on the tracel-ai models repository:
https://github.com/tracel-ai/models/blob/main/resnet-burn/examples/finetune/examples/finetune.rs

## Running the Example

This will download both model weights and a fine-tune dataset.

Run the training:

```bash
cargo run --release -p resnet_finetune -- \
  --drop-path-prob=0.1 \
  --drop-block-prob=0.2 \
  --num-epochs=20 \
  --batch-size=32 \
  --learning-rate=1e-4
```


