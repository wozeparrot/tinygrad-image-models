from tinygrad import Tensor, nn, dtypes
from tinygrad.nn import Conv2d, Linear

# *** layers ***

# can reshape before the batchnorm so it can work on 1d tensors
class BatchNorm1d(nn.BatchNorm2d):
  def __init__(self, dim:int, eps=1e-5): super().__init__(dim, eps)
  def __call__(self, x:Tensor) -> Tensor: return super().__call__(x.reshape(-1, x.shape[-1], 1, 1).float()).cast(dtypes.default_float).reshape(x.shape)

class BatchNorm2d(nn.BatchNorm2d):
  def __init__(self, dim:int, eps=1e-5): super().__init__(dim, eps)
  def __call__(self, x:Tensor) -> Tensor: return super().__call__(x.float()).cast(dtypes.default_float)

class LayerNorm2d(nn.LayerNorm2d):
  def __init__(self, shape:int | tuple[int, ...], eps=1e-5): super().__init__(shape, eps)
  def __call__(self, x:Tensor) -> Tensor: return super().__call__(x.float()).cast(dtypes.default_float)

class SE:
  def __init__(self, dim:int, se_ratio=0.25, se_divisor=4, gate=Tensor.sigmoid):
    reduced = make_divisible(dim * se_ratio, se_divisor)
    self.conv_reduce = Conv2d(dim, reduced, kernel_size=1)
    self.conv_expand = Conv2d(reduced, dim, kernel_size=1)
    self.gate = gate
  def __call__(self, x:Tensor):
    xx = x.mean((2, 3), keepdim=True)
    xx = self.conv_reduce(xx).relu()
    xx = self.gate(self.conv_expand(xx))
    return x * xx

class LayerScale2d:
  def __init__(self, dim:int, init_value:float=1e-5):
    self.gamma = Tensor.full((dim, 1, 1), init_value)
  def __call__(self, x:Tensor) -> Tensor: return x * self.gamma.reshape(1, -1, 1, 1)

class SLClassifierHead:
  def __init__(self, dim:int, classes:int):
    self.fc = Linear(dim, classes)
  def __call__(self, x:Tensor) -> Tensor: return self.fc(x.mean((2, 3)))


# *** activations ***

def hardsigmoid(x:Tensor) -> Tensor: return (x + 3).relu6() / 6

# *** utils ***

def make_divisible(x:float, divisible_by:int, round_limit=0.9) -> int:
  new_x = int(x + divisible_by / 2) // divisible_by * divisible_by
  return new_x + divisible_by if new_x < round_limit * x else new_x

# nearest neighbor upsample
def upsample(x:Tensor, scale:int):
  bs, c, py, px = x.shape
  return x.reshape(bs, c, py, 1, px, 1).expand(bs, c, py, scale, px, scale).reshape(bs, c, py * scale, px * scale)

# TODO: this really only works in the very specific case for ghostnetv2 dfc upsampling
def upsample_to_size(x:Tensor, height:int, width:int):
  bs, c, py, px = x.shape
  assert isinstance(py, int) and isinstance(px, int), "symbolic shape not supported"

  # upsample to larger size
  height_scale, width_scale = (height // py) * 2, (width // px) * 2
  x = x.reshape(bs, c, py, 1, px, 1).expand(bs, c, py, height_scale, px, width_scale) # upsample
  x = x.reshape(bs, c, py * height_scale, px * width_scale)

  # pool to downsample
  if py * height_scale > height:
    for _ in range(height_scale // 2 - 2): x = x.pad2d((0, 0, 1, 1)).avg_pool2d(kernel_size=(3, 1), stride=(2, 1))
    padding = (height - py * (height_scale // 2) + 1)
    x = x.pad2d((0, 0, padding, padding))
    x = x.avg_pool2d(kernel_size=(3, 1), stride=(2, 1))
  if px * width_scale > width:
    for _ in range(width_scale // 2 - 2): x = x.pad2d((1, 1, 0, 0)).avg_pool2d(kernel_size=(1, 3), stride=(1, 2))
    padding = (width - px * (width_scale // 2) + 1)
    x = x.pad2d((padding, padding, 0, 0))
    x = x.avg_pool2d(kernel_size=(1, 3), stride=(1, 2))
  return x
