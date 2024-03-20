# tinygrad Image Models

A collection of vision models implemented in tinygrad, in a similar vein to [timm](https://github.com/huggingface/pytorch-image-models).

Mostly targeting models trained on Imagenet-1k, and other models that are fast on resource-constrained devices.

## Models

- ShuffleNetV2 - [paper](https://arxiv.org/abs/1807.11164) [code](/models/shufflenetv2.py)
- GhostNetV2 - [paper](https://arxiv.org/abs/2211.12905) [code](/models/ghostnetv2.py)
- FocalNet - [paper](https://arxiv.org/abs/2203.11926) [code](/models/focalnet.py)
- FastViT - [paper](https://arxiv.org/abs/2303.14189) [code](/models/fastvit.py)
- RepViT - [paper](https://arxiv.org/abs/2307.09283) [code](/models/repvit.py)

## TODO

- [ ] For models that can be reparameterized, add that functionality
- [ ] Training
