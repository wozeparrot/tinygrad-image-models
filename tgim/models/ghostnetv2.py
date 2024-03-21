import math
from tinygrad import Tensor
from tinygrad.nn import Conv2d, Linear

from ..common.blocks import BatchNorm2d, SE, upsample_to_size, hardsigmoid
from ..common.model import Model

class GhostModuleV2:
  def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True, attn=False):
    self.oup, self.attn = oup, attn
    init_channels = math.ceil(oup / ratio)
    new_channels = init_channels * (ratio - 1)

    self.primary_conv = [
      Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
      BatchNorm2d(init_channels),
    ]
    if relu: self.primary_conv.append(Tensor.relu)
    self.cheap_operation = [
      Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
      BatchNorm2d(new_channels),
    ]
    if relu: self.cheap_operation.append(Tensor.relu)

    if attn:
      self.short_conv = [
        Conv2d(inp, oup, kernel_size, stride, kernel_size//2, bias=False),
        BatchNorm2d(oup),
        Conv2d(oup, oup, kernel_size=(1, 5), stride=1, padding=(0, 2), groups=oup, bias=False),
        BatchNorm2d(oup),
        Conv2d(oup, oup, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=oup, bias=False),
        BatchNorm2d(oup),
      ]
  def __call__(self, x:Tensor) -> Tensor:
    if self.attn: res = x.avg_pool2d(kernel_size=2, stride=2).sequential(self.short_conv)
    x1 = x.sequential(self.primary_conv)
    x2 = x1.sequential(self.cheap_operation)
    out = x1.cat(x2, dim=1)
    if self.attn: out = out * upsample_to_size(res.sigmoid(), *x.shape[2:])
    return out

class GhostBottleneckV2:
  def __init__(self, layer_id, cin, cmid, cout, dw_size=3, stride=1, se_ratio=0.):
    self.stride = stride

    self.ghost1 = GhostModuleV2(cin, cmid, relu=True, attn=layer_id > 1)

    if stride > 1:
      self.conv_dw = Conv2d(cmid, cmid, dw_size, stride, dw_size//2, groups=cmid, bias=False)
      self.bn_dw = BatchNorm2d(cmid)

    if se_ratio > 0:
      self.se = SE(cmid, se_ratio, gate=hardsigmoid)

    self.ghost2 = GhostModuleV2(cmid, cout, relu=False)

    if cin == cout and stride == 1:
      self.shortcut = []
    else:
      self.shortcut = [
        Conv2d(cin, cin, dw_size, stride=stride, padding=dw_size//2, groups=cin, bias=False),
        BatchNorm2d(cin),
        Conv2d(cin, cout, kernel_size=1, stride=1, padding=0, bias=False),
        BatchNorm2d(cout),
      ]
  def __call__(self, x:Tensor) -> Tensor:
    res = x
    x = self.ghost1(x)
    if self.stride > 1:
      x = self.bn_dw(self.conv_dw(x))
    if hasattr(self, "se"): x = self.se(x)
    x = self.ghost2(x)
    return x + res.sequential(self.shortcut)

class ConvBnAct:
  def __init__(self, cin, cout, kernel_size, stride=1):
    self.conv = Conv2d(cin, cout, kernel_size, stride, kernel_size//2, bias=False)
    self.bn1 = BatchNorm2d(cout)
  def __call__(self, x:Tensor) -> Tensor: return self.bn1(self.conv(x)).relu()

class GhostNetV2(Model):
  def __init__(self, classes:int=1000, size:str="1.0"):
    super().__init__(classes, size)

    self.conv_stem = Conv2d(3, 16, 3, 2, 1, bias=False)
    self.bn1 = BatchNorm2d(16)

    self.blocks = [
      # stage1
      [GhostBottleneckV2(0, cin=16, cmid=16, cout=16, dw_size=3, stride=1, se_ratio=0.)],
      # stage2
      [GhostBottleneckV2(1, cin=16, cmid=48, cout=24, dw_size=3, stride=2, se_ratio=0.)],
      [GhostBottleneckV2(2, cin=24, cmid=72, cout=24, dw_size=3, stride=1, se_ratio=0.)],
      # stage3
      [GhostBottleneckV2(3, cin=24, cmid=72, cout=40, dw_size=5, stride=2, se_ratio=0.25)],
      [GhostBottleneckV2(4, cin=40, cmid=120, cout=40, dw_size=5, stride=1, se_ratio=0.25)],
      # stage4
      [GhostBottleneckV2(5, cin=40, cmid=240, cout=80, dw_size=3, stride=2, se_ratio=0.)],
      [
        GhostBottleneckV2(6, cin=80, cmid=200, cout=80, dw_size=3, stride=1, se_ratio=0.),
        GhostBottleneckV2(7, cin=80, cmid=184, cout=80, dw_size=3, stride=1, se_ratio=0.),
        GhostBottleneckV2(8, cin=80, cmid=184, cout=80, dw_size=3, stride=1, se_ratio=0.),
        GhostBottleneckV2(9, cin=80, cmid=480, cout=112, dw_size=3, stride=1, se_ratio=0.25),
        GhostBottleneckV2(10, cin=112, cmid=672, cout=112, dw_size=3, stride=1, se_ratio=0.25),
      ],
      # stage5
      [GhostBottleneckV2(11, cin=112, cmid=672, cout=160, dw_size=5, stride=2, se_ratio=0.25)],
      [
        GhostBottleneckV2(12, cin=160, cmid=960, cout=160, dw_size=5, stride=1, se_ratio=0.),
        GhostBottleneckV2(13, cin=160, cmid=960, cout=160, dw_size=5, stride=1, se_ratio=0.25),
        GhostBottleneckV2(14, cin=160, cmid=960, cout=160, dw_size=5, stride=1, se_ratio=0.),
        GhostBottleneckV2(15, cin=160, cmid=960, cout=160, dw_size=5, stride=1, se_ratio=0.25),
      ],
      [ConvBnAct(160, 960, 1)],
    ]

    self.conv_head = Conv2d(960, 1280, 1, 1, 0)
    self.classifier = Linear(1280, classes)

  def forward_features(self, x: Tensor) -> list[Tensor]:
    x = self.bn1(self.conv_stem(x)).relu()
    features = []
    for i, block in enumerate(self.blocks):
      x = x.sequential(block)
      if i == len(self.blocks) - 1 or block[0].stride != 2: features.append(x)
    return features

  def forward_head(self, xl: list[Tensor]) -> Tensor:
    x = self.conv_head(xl[-1]).relu()
    x = x.mean((2, 3), keepdim=True)
    x = x.flatten(1)
    return self.classifier(x)

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

  net = GhostNetV2(classes=args.classes, size=args.size)

  state_dict = torch_load(args.input)
  for key in list(state_dict.keys()):
    if "num_batches_tracked" in key: state_dict[key] = Tensor([state_dict[key].item()], dtype=dtypes.default_float)

  for param in get_parameters(state_dict):
    param.replace(param.cast(dtypes.float32)).realize()

  load_state_dict(net, state_dict)

  # save state_dict
  safe_save(get_state_dict(net), args.output)
