from tinygrad import Tensor, dtypes
from tinygrad.nn import Conv2d, Linear
from ..common.blocks import BatchNorm1d, BatchNorm2d, SE
from ..common.model import ModelReparam

class ConvNorm:
  def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride:int, padding:int, groups:int=1):
    self.c = Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
    self.bn = BatchNorm2d(out_channels)
  def __call__(self, x:Tensor) -> Tensor: return self.bn(self.c(x))

class LinearNorm:
  def __init__(self, in_features:int, out_features:int, bias:bool=True):
    self.bn = BatchNorm1d(in_features)
    self.l = Linear(in_features, out_features, bias)
  def __call__(self, x:Tensor) -> Tensor: return self.l(self.bn(x))

class RepVitStem:
  def __init__(self, embed_dims:int):
    self.conv1 = ConvNorm(3, embed_dims // 2, 3, 2, 1)
    self.conv2 = ConvNorm(embed_dims // 2, embed_dims, 3, 2, 1)
  def __call__(self, x:Tensor) -> Tensor: return self.conv2(self.conv1(x).gelu())

class RepVggDw:
  def __init__(self, dim:int):
    self.conv = ConvNorm(dim, dim, 3, 1, 1, groups=dim)
    self.conv1 = Conv2d(dim, dim, 1, 1, 0, groups=dim)
    self.bn = BatchNorm2d(dim)
  def __call__(self, x:Tensor) -> Tensor: return self.bn(self.conv(x) + self.conv1(x) + x)

class RepVitFFN:
  def __init__(self, dim:int, hidden_dim:int):
    self.conv1 = ConvNorm(dim, hidden_dim, 1, 1, 0)
    self.conv2 = ConvNorm(hidden_dim, dim, 1, 1, 0)
  def __call__(self, x:Tensor) -> Tensor: return self.conv2(self.conv1(x).gelu())

class RepVitBlock:
  def __init__(self, dim:int, use_se:bool):
    self.token_mixer = RepVggDw(dim)
    if use_se: self.se = SE(dim, 0.25, se_divisor=8)
    self.channel_mixer = RepVitFFN(dim, dim * 2)
  def __call__(self, x:Tensor) -> Tensor:
    x = self.token_mixer(x)
    if hasattr(self, "se"): x = self.se(x)
    return x + self.channel_mixer(x)

class RepVitDownsample:
  def __init__(self, in_dim:int, out_dim:int):
    self.pre_block = RepVitBlock(in_dim, False)
    self.spatial_downsample = ConvNorm(in_dim, in_dim, 3, 2, 1, groups=in_dim)
    self.channel_downsample = ConvNorm(in_dim, out_dim, 1, 1, 0)
    self.ffn = RepVitFFN(out_dim, out_dim * 2)
  def __call__(self, x:Tensor) -> Tensor:
    x = self.pre_block(x)
    x = self.spatial_downsample(x)
    x = self.channel_downsample(x)
    return x + self.ffn(x)

class RepVitStage:
  def __init__(self, in_dim:int, out_dim:int, depth:int, downsample:bool):
    if downsample: self.downsample = RepVitDownsample(in_dim, out_dim)
    self.blocks = []
    for i in range(depth): self.blocks.append(RepVitBlock(out_dim, i % 2 == 0))
  def __call__(self, x:Tensor) -> Tensor:
    if hasattr(self, "downsample"): x = self.downsample(x)
    return x.sequential(self.blocks)

class RepVitClassifier:
  def __init__(self, embed_dim:int, classes:int):
    self.head = LinearNorm(embed_dim, classes)
    self.head_dist = LinearNorm(embed_dim, classes)
  def __call__(self, x:Tensor) -> Tensor:
    x = x.mean((2, 3))
    x1 = self.head(x)
    x2 = self.head_dist(x)
    return (x1 + x2) / 2

class RepVit(ModelReparam):
  def __init__(self, classes:int=1000, size:str="1.0"):
    super().__init__(classes, size)

    match size:
      case "0.6": embed_dims, depths = [40, 80, 160, 320], [1, 1, 7, 1]
      case "0.9": embed_dims, depths = [48, 96, 192, 384], [2, 2, 14, 2]
      case "1.0": embed_dims, depths = [56, 112, 224, 448], [2, 2, 14, 2]
      case "1.1": embed_dims, depths = [64, 128, 256, 512], [2, 2, 12, 2]
      case "1.5": embed_dims, depths = [64, 128, 256, 512], [4, 4, 24, 4]
      case "2.3": embed_dims, depths = [80, 160, 320, 640], [6, 6, 34, 2]
      case _: raise ValueError(f"size {size} not supported")

    self.stem = RepVitStem(embed_dims[0])

    self.stages = []
    in_dim = embed_dims[0]
    for i in range(len(embed_dims)):
      self.stages.append(RepVitStage(in_dim, embed_dims[i], depths[i], i != 0))
      in_dim = embed_dims[i]

    self.head = RepVitClassifier(embed_dims[-1], classes)

  def forward_features(self, x: Tensor) -> list[Tensor]:
    x = self.stem(x)
    features = []
    for stage in self.stages:
      x = stage(x)
      features.append(x)
    return features

  def forward_head(self, xl: list[Tensor]) -> Tensor:
    return self.head(xl[-1])

def preprocess(x:Tensor) -> Tensor: return x.div(255)

if __name__ == "__main__":
  from tinygrad.nn.state import torch_load, safe_save, get_state_dict, get_parameters, load_state_dict
  from tinygrad import dtypes
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("--classes", type=int, default=1000, help="number of classes")
  parser.add_argument("--size", type=str, default="1.0", help="size of model")
  parser.add_argument("--input", type=str, required=True, help="path to model weights")
  parser.add_argument("--output", type=str, required=True, help="path to output model weights")
  args = parser.parse_args()

  net = RepVit(classes=args.classes, size=args.size)

  state_dict = torch_load(args.input)
  for key in list(state_dict.keys()):
    if "num_batches_tracked" in key: state_dict[key] = Tensor([state_dict[key].item()], dtype=dtypes.default_float)
  # rename se
  for key in list(state_dict.keys()):
    if "se.fc1" in key: state_dict[key.replace("se.fc1", "se.conv_reduce")] = state_dict.pop(key)
    if "se.fc2" in key: state_dict[key.replace("se.fc2", "se.conv_expand")] = state_dict.pop(key)

  for param in get_parameters(state_dict):
    param.replace(param.cast(dtypes.float32)).realize()

  load_state_dict(net, state_dict)

  # save state_dict
  safe_save(get_state_dict(net), args.output)
