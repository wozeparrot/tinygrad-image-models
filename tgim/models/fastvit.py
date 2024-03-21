# TODO: add support for the variants with attention and not just repmixer

from tinygrad import Tensor, dtypes
from tinygrad.nn import Conv2d
from ..common.blocks import BatchNorm2d, SE, LayerScale2d, SLClassifierHead
from ..common.model import ModelReparam

def num_groups(group_size, channels):
  if not group_size: return 1
  else:
    assert channels % group_size == 0
    return channels // group_size

class ConvNorm:
  def __init__(self, in_channels:int, out_channels:int, kernel_size:int, stride:int, padding:int, groups:int=1):
    self.conv = Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
    self.bn = BatchNorm2d(out_channels)
  def __call__(self, x:Tensor) -> Tensor: return self.bn(self.conv(x))

class MobileOneBlock:
  def __init__(self, cin:int, cout:int, kernel_size:int, stride:int=1, group_size:int=0, use_act:bool=True, use_se:bool=False, use_scale_branch:bool=True, num_conv_branches:int=1):
    groups = num_groups(group_size, cin)
    self.use_act = use_act

    if use_se: self.se = SE(cout, 1/16, se_divisor=1)

    if cin == cout and stride == 1: self.identity = BatchNorm2d(cin)
    if num_conv_branches > 0: self.conv_kxk = [ConvNorm(cin, cout, kernel_size, stride, kernel_size//2, groups=groups) for _ in range(num_conv_branches)]
    if kernel_size > 1 and use_scale_branch: self.conv_scale = ConvNorm(cin, cout, 1, stride, 0, groups=groups)
  def __call__(self, x:Tensor) -> Tensor:
    x_id = 0
    if hasattr(self, "identity"): x_id = self.identity(x)
    x_scale = 0
    if hasattr(self, "conv_scale"): x_scale = self.conv_scale(x)
    x_out = x_id + x_scale
    if hasattr(self, "conv_kxk"): x_out = x_out + sum([conv(x) for conv in self.conv_kxk])
    x = x_out # type: ignore
    if hasattr(self, "se"): x = self.se(x)
    if self.use_act: x = x.gelu()
    return x

def FastVitStem(cin:int, cout:int):
  return [
    MobileOneBlock(cin, cout, kernel_size=3, stride=2),
    MobileOneBlock(cout, cout, kernel_size=3, stride=2, group_size=1),
    MobileOneBlock(cout, cout, kernel_size=1, stride=1),
  ]

class RepLKConv:
  def __init__(self, cin:int, cout:int, kernel_size:int, stride:int, group_size:int, small_kernel:int):
    groups = num_groups(group_size, cin)
    self.large_conv = ConvNorm(cin, cout, kernel_size, stride, kernel_size//2, groups=groups)
    self.small_conv = ConvNorm(cin, cout, small_kernel, stride, small_kernel//2, groups=groups)
  def __call__(self, x:Tensor) -> Tensor: return self.large_conv(x) + self.small_conv(x)

class PatchEmbed:
  def __init__(self, cin:int, embed_dim:int):
    self.proj = [
      RepLKConv(cin, embed_dim, kernel_size=7, stride=2, group_size=1, small_kernel=3),
      MobileOneBlock(embed_dim, embed_dim, kernel_size=1, stride=1),
    ]
  def __call__(self, x:Tensor) -> Tensor: return x.sequential(self.proj)

class RepMixer:
  def __init__(self, dim):
    self.norm = MobileOneBlock(dim, dim, kernel_size=3, group_size=1, use_act=False, use_scale_branch=False, num_conv_branches=0)
    self.mixer = MobileOneBlock(dim, dim, kernel_size=3, group_size=1, use_act=False)
    self.layer_scale = LayerScale2d(dim)
  def __call__(self, x:Tensor) -> Tensor:
    return x + self.layer_scale(self.mixer(x) - self.norm(x))

class ConvFFN:
  def __init__(self, dim:int, hidden_dim:int):
    self.conv = ConvNorm(dim, dim, 7, 1, 3, groups=dim)
    self.fc1 = Conv2d(dim, hidden_dim, 1, 1, 0)
    self.fc2 = Conv2d(hidden_dim, dim, 1, 1, 0)
  def __call__(self, x:Tensor) -> Tensor: return self.fc2(self.fc1(self.conv(x)).gelu())

class RepMixerBlock:
  def __init__(self, dim:int, ffn_ratio:float):
    self.token_mixer = RepMixer(dim)
    self.ffn = ConvFFN(dim, int(dim * ffn_ratio))
    self.layer_scale = LayerScale2d(dim)
  def __call__(self, x:Tensor) -> Tensor:
    x = self.token_mixer(x)
    return x + self.layer_scale(self.ffn(x))

class FastVitStage:
  def __init__(self, in_dim:int, out_dim:int, depth:int, ffn_ratio:float, downsample:bool):
    if downsample: self.downsample = PatchEmbed(in_dim, out_dim)

    self.blocks = []
    for _ in range(depth):
      self.blocks.append(RepMixerBlock(out_dim, ffn_ratio))
  def __call__(self, x:Tensor) -> Tensor:
    if hasattr(self, "downsample"): x = self.downsample(x)
    return x.sequential(self.blocks)

class FastVit(ModelReparam):
  def __init__(self, classes:int=1000, size:str="t8"):
    super().__init__(classes, size)

    match size:
      case "t8": embed_dims, depths, ffn_ratios = [48, 96, 192, 384], [2, 2, 4, 2], [3, 3, 3, 3]
      case _: raise ValueError(f"size {size} not supported")

    self.stem = FastVitStem(3, embed_dims[0])

    self.stages = []
    in_dim = embed_dims[0]
    for i in range(len(embed_dims)):
      self.stages.append(FastVitStage(in_dim, embed_dims[i], depths[i], ffn_ratios[i], i != 0))
      in_dim = embed_dims[i]
    self.final_conv = MobileOneBlock(embed_dims[-1], embed_dims[-1] * 2, kernel_size=3, stride=1, group_size=1, use_se=True, num_conv_branches=1)

    self.head = SLClassifierHead(embed_dims[-1] * 2, classes)

  def forward_features(self, x:Tensor) -> list[Tensor]:
    x = x.sequential(self.stem)
    features = []
    for stage in self.stages:
      x = stage(x)
      features.append(x)
    x = self.final_conv(x)
    return features + [x]

  def forward_head(self, xl:list[Tensor]) -> Tensor:
    return self.head(xl[-1])

def preprocess(x:Tensor) -> Tensor: return x.div(255)

if __name__ == "__main__":
  from tinygrad.nn.state import torch_load, safe_save, get_state_dict, get_parameters, load_state_dict
  from tinygrad import dtypes
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("--classes", type=int, default=1000, help="number of classes")
  parser.add_argument("--size", type=str, default="t8", help="size of model")
  parser.add_argument("--input", type=str, required=True, help="path to model weights")
  parser.add_argument("--output", type=str, required=True, help="path to output model weights")
  args = parser.parse_args()

  net = FastVit(classes=args.classes, size=args.size)

  state_dict = torch_load(args.input)
  for key in list(state_dict.keys()):
    if "num_batches_tracked" in key: state_dict[key] = Tensor([state_dict[key].item()], dtype=dtypes.default_float)
  # rename se
  for key in list(state_dict.keys()):
    if "se.fc1" in key: state_dict[key.replace("se.fc1", "se.conv_reduce")] = state_dict.pop(key)
    if "se.fc2" in key: state_dict[key.replace("se.fc2", "se.conv_expand")] = state_dict.pop(key)
  # its ffn here
  for key in list(state_dict.keys()):
    if "mlp" in key: state_dict[key.replace("mlp", "ffn")] = state_dict.pop(key)

  for param in get_parameters(state_dict):
    param.replace(param.cast(dtypes.float32)).realize()

  load_state_dict(net, state_dict)

  # save state_dict
  safe_save(get_state_dict(net), args.output)
