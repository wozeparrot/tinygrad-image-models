from typing import Tuple
from tinygrad import Tensor, Device, dtypes
from tinygrad.nn import Conv2d, Linear
from ..common.blocks import BatchNorm2d
from ..common.model import Model

def channel_shuffle(x: Tensor) -> Tuple[Tensor, Tensor]:
  b, c, h, w = x.shape
  assert c % 4 == 0
  x = x.reshape(b * c // 2, 2, h * w).permute(1, 0, 2)
  x = x.reshape(2, -1, c // 2, h, w)
  return x[0], x[1]

class ShuffleV2Block:
  def __init__(self, inp: int, outp: int, c_mid: int, kernel_size: int, stride: int):
    assert stride in [1, 2]
    self.stride, self.inp, self.outp, self.c_mid = stride, inp, outp, c_mid
    pad, out = kernel_size // 2, outp - inp

    # pw
    self.cv1 = Conv2d(inp, c_mid, 1, 1, 0, bias=False)
    self.bn1 = BatchNorm2d(c_mid)
    # dw
    self.cv2 = Conv2d(c_mid, c_mid, kernel_size, stride, pad, groups=c_mid, bias=False)
    self.bn2 = BatchNorm2d(c_mid)
    # pw-linear
    self.cv3 = Conv2d(c_mid, out, 1, 1, 0, bias=False)
    self.bn3 = BatchNorm2d(out)

    if stride == 2:
      # dw
      self.cv4 = Conv2d(inp, inp, kernel_size, stride, pad, groups=inp, bias=False)
      self.bn4 = BatchNorm2d(inp)
      # pw-linear
      self.cv5 = Conv2d(inp, inp, 1, 1, 0, bias=False)
      self.bn5 = BatchNorm2d(inp)

  def __call__(self, x: Tensor) -> Tensor:
    if self.stride == 1:
      x_proj, x = channel_shuffle(x)
    elif self.stride == 2:
      x_proj = self.bn4(self.cv4(x))
      x_proj = self.bn5(self.cv5(x_proj)).relu()
    else: raise Exception("Invalid stride", self.stride)
    x = self.bn1(self.cv1(x)).relu()
    x = self.bn2(self.cv2(x))
    x = self.bn3(self.cv3(x)).relu()
    return x_proj.cat(x, dim=1)

class ShuffleNetV2(Model):
  def __init__(self, classes:int=1000, size:str="0.5"):
    super().__init__(classes, size)

    stage_repeats = [4, 8, 4]
    match size:
      case "0.5": stage_out_channels = [24, 48, 96, 192, 1024]
      case "1.0": stage_out_channels = [24, 116, 232, 464, 1024]
      case "1.5": stage_out_channels = [24, 176, 352, 704, 1024]
      case "2.0": stage_out_channels = [24, 244, 488, 976, 2048]
      case _: raise ValueError(f"Invalid size: {size}")

    self.stage1 = [Conv2d(3, stage_out_channels[0], 3, 2, 1, bias=False), BatchNorm2d(stage_out_channels[0]), Tensor.relu]
    self.stage2 = [ShuffleV2Block(stage_out_channels[0], stage_out_channels[1], stage_out_channels[1] // 2, kernel_size=3, stride=2)]
    self.stage2 += [ShuffleV2Block(stage_out_channels[1] // 2, stage_out_channels[1], stage_out_channels[1] // 2, 3, 1) for _ in range(stage_repeats[0] - 1)]
    self.stage3 = [ShuffleV2Block(stage_out_channels[1], stage_out_channels[2], stage_out_channels[2] // 2, kernel_size=3, stride=2)]
    self.stage3 += [ShuffleV2Block(stage_out_channels[2] // 2, stage_out_channels[2], stage_out_channels[2] // 2, 3, 1) for _ in range(stage_repeats[1] - 1)]
    self.stage4 = [ShuffleV2Block(stage_out_channels[2], stage_out_channels[3], stage_out_channels[3] // 2, kernel_size=3, stride=2)]
    self.stage4 += [ShuffleV2Block(stage_out_channels[3] // 2, stage_out_channels[3], stage_out_channels[3] // 2, 3, 1) for _ in range(stage_repeats[2] - 1)]
    self.stage5 = [Conv2d(stage_out_channels[3], stage_out_channels[4], 1, 1, 0, bias=False), BatchNorm2d(stage_out_channels[4]), Tensor.relu]

    self.classifier = [Linear(stage_out_channels[4], classes, bias=False)]
    if size == "2.0":
      self.classifier.insert(0, lambda x: x.dropout(0.2))

  def forward_features(self, x: Tensor) -> list[Tensor]:
    x = x.sequential(self.stage1).pad2d((1, 1, 1, 1)).max_pool2d(3, 2)
    x2 = x.sequential(self.stage2)
    x3 = x2.sequential(self.stage3)
    x4 = x3.sequential(self.stage4)
    x5 = x4.sequential(self.stage5)
    return [x2, x3, x4, x5]

  def forward_head(self, xl: list[Tensor]) -> Tensor:
    x = xl[-1].mean((2, 3)).flatten(1)
    return x.sequential(self.classifier)

if __name__ == "__main__":
  from tinygrad.nn.state import torch_load, safe_save, get_state_dict, get_parameters
  from tinygrad.helpers import get_child
  from tinygrad import dtypes
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("--classes", type=int, default=1000, help="number of classes")
  parser.add_argument("--size", type=str, default="0.5", help="size of model")
  parser.add_argument("--input", type=str, required=True, help="path to model weights")
  parser.add_argument("--output", type=str, required=True, help="path to output model weights")
  args = parser.parse_args()

  net = ShuffleNetV2(classes=args.classes, size=args.size)

  state_dict = torch_load(args.input)["state_dict"]
  # modify state_dict to match our model
  for key in list(state_dict.keys()):
    if "num_batches_tracked" in key: state_dict[key] = Tensor([state_dict[key].item()], dtype=dtypes.default_float)
  for key in list(state_dict.keys()):
    if "first_conv" in key:
      state_dict[key.replace("first_conv", "stage1")] = state_dict[key]
      del state_dict[key]
    if "conv_last" in key:
      state_dict[key.replace("conv_last", "stage5")] = state_dict[key]
      del state_dict[key]
  for key in list(state_dict.keys()):
    if "branch_main" in key:
      index = int(key.split(".")[4])
      if index == 0:
        state_dict[key.replace("branch_main.0", "cv1")] = state_dict[key]
        del state_dict[key]
      elif index == 1:
        state_dict[key.replace("branch_main.1", "bn1")] = state_dict[key]
        del state_dict[key]
      elif index == 3:
        state_dict[key.replace("branch_main.3", "cv2")] = state_dict[key]
        del state_dict[key]
      elif index == 4:
        state_dict[key.replace("branch_main.4", "bn2")] = state_dict[key]
        del state_dict[key]
      elif index == 5:
        state_dict[key.replace("branch_main.5", "cv3")] = state_dict[key]
        del state_dict[key]
      elif index == 6:
        state_dict[key.replace("branch_main.6", "bn3")] = state_dict[key]
        del state_dict[key]
    if "branch_proj" in key:
      index = int(key.split(".")[4])
      if index == 0:
        state_dict[key.replace("branch_proj.0", "cv4")] = state_dict[key]
        del state_dict[key]
      elif index == 1:
        state_dict[key.replace("branch_proj.1", "bn4")] = state_dict[key]
        del state_dict[key]
      elif index == 2:
        state_dict[key.replace("branch_proj.2", "cv5")] = state_dict[key]
        del state_dict[key]
      elif index == 3:
        state_dict[key.replace("branch_proj.3", "bn5")] = state_dict[key]
        del state_dict[key]
  for key in list(state_dict.keys()):
    if "features" in key:
      index = int(key.split(".")[2])
      if index in range(0, 4):
        state_dict[key.replace(f"features.{index}", f"stage2.{index}")] = state_dict[key]
        del state_dict[key]
      elif index in range(4, 12):
        state_dict[key.replace(f"features.{index}", f"stage3.{index - 4}")] = state_dict[key]
        del state_dict[key]
      elif index in range(12, 16):
        state_dict[key.replace(f"features.{index}", f"stage4.{index - 12}")] = state_dict[key]
        del state_dict[key]

  for key in list(state_dict.keys()):
    print(f"Loading {key}...")
    get_child(net, key.replace("module.", "")).replace(state_dict[key].to(Device.DEFAULT)).realize()

  for param in get_parameters(state_dict):
    param.replace(param.cast(dtypes.float32)).realize()

  # save state_dict
  safe_save(get_state_dict(net), args.output)
