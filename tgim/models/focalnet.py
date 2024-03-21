from tinygrad import Tensor, dtypes
from tinygrad.nn import Conv2d
from ..common.blocks import LayerNorm2d, SLClassifierHead
from ..common.model import Model

class Downsample:
  def __init__(self, cin:int, cout:int, stride:int=4):
    self.proj = Conv2d(cin, cout, kernel_size=stride, stride=stride)
    self.norm = LayerNorm2d(cout)
  def __call__(self, x:Tensor) -> Tensor: return self.norm(self.proj(x))

class FocalModulation:
  def __init__(self, dim:int, focal_window:int, focal_level:int):
    self.dim, self.focal_level = dim, focal_level

    self.f = Conv2d(dim, 2 * dim + (focal_level + 1), kernel_size=1)
    self.h = Conv2d(dim, dim, kernel_size=1)
    self.proj = Conv2d(dim, dim, kernel_size=1)

    self.focal_layers = []
    for k in range(focal_level):
      kernel_size = 2 * k + focal_window
      self.focal_layers.append([Conv2d(dim, dim, kernel_size=kernel_size, groups=dim, padding=kernel_size//2, bias=False), Tensor.gelu])
  def __call__(self, x:Tensor) -> Tensor:
    x = self.f(x)
    q, ctx, gates = x.split([self.dim, self.dim, self.focal_level + 1], dim=1)

    ctx_all = 0
    for l, focal_layer in enumerate(self.focal_layers):
      ctx = ctx.sequential(focal_layer)
      ctx_all = ctx_all + ctx * gates[:, l:l+1]
    ctx_global = ctx.mean((2, 3), keepdim=True).gelu()
    ctx_all = ctx_all + ctx_global * gates[:, self.focal_level:]

    x_out = q * self.h(ctx_all)
    return self.proj(x_out)

class FFN:
  def __init__(self, dim:int, hidden_dim:int):
    self.fc1 = Conv2d(dim, hidden_dim, 1, 1, 0)
    self.fc2 = Conv2d(hidden_dim, dim, 1, 1, 0)
  def __call__(self, x:Tensor) -> Tensor: return self.fc2(self.fc1(x).gelu())

class FocalNetBlock:
  def __init__(self, dim:int, focal_level:int):
    self.norm1 = LayerNorm2d(dim)
    self.modulation = FocalModulation(dim, focal_window=3, focal_level=focal_level)
    self.norm2 = LayerNorm2d(dim)
    self.ffn = FFN(dim, dim * 4)
  def __call__(self, x:Tensor) -> Tensor:
    x = x + self.modulation(self.norm1(x))
    return x + self.ffn(self.norm2(x))

class FocalNetLayer:
  def __init__(self, dim:int, out_dim:int, depth:int, focal_level:int, use_downsample:bool):
    if use_downsample: self.downsample = Downsample(dim, out_dim, stride=2)
    self.blocks = [FocalNetBlock(out_dim, focal_level) for _ in range(depth)]
  def __call__(self, x:Tensor) -> Tensor:
    if hasattr(self, "downsample"): x = self.downsample(x)
    return x.sequential(self.blocks)

class FocalNet(Model):
  def __init__(self, classes:int=1000, size:str="tiny_srf"):
    super().__init__(classes, size)

    match size:
      case "tiny_srf": embed_dim, depths, focal_levels = 96, [2, 2, 6, 2], [2, 2, 2, 2]
      case "small_srf": embed_dim, depths, focal_levels = 96, [2, 2, 18, 2], [2, 2, 2, 2]
      case "base_srf": embed_dim, depths, focal_levels = 128, [2, 2, 18, 2], [2, 2, 2, 2]
      case "tiny_lrf": embed_dim, depths, focal_levels = 96, [2, 2, 6, 2], [3, 3, 3, 3]
      case "small_lrf": embed_dim, depths, focal_levels = 96, [2, 2, 18, 2], [3, 3, 3, 3]
      case "base_lrf": embed_dim, depths, focal_levels = 128, [2, 2, 18, 2], [3, 3, 3, 3]
      case _: raise ValueError(f"size {size} not supported")

    embed_dims = [embed_dim * (2 ** i) for i in range(len(depths))]
    self.stem = Downsample(3, embed_dims[0])

    self.layers= []
    in_dim = embed_dims[0]
    for i in range(len(embed_dims)):
      self.layers.append(FocalNetLayer(in_dim, embed_dims[i], depths[i], focal_levels[i], i != 0))
      in_dim = embed_dims[i]

    self.norm = LayerNorm2d(embed_dims[-1])
    self.head = SLClassifierHead(embed_dims[-1], classes)

  def forward_features(self, x: Tensor) -> list[Tensor]:
    x = self.stem(x)
    features = []
    for layer in self.layers:
      x = layer(x)
      features.append(x)
    return features

  def forward_head(self, xl: list[Tensor]) -> Tensor:
    return self.head(self.norm(xl[-1]))

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

  net = FocalNet(classes=args.classes, size=args.size)

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
