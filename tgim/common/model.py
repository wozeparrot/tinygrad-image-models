from tinygrad import Tensor

class Model:
  def __init__(self, cin:int, classes:int, size:str): self.cin, self.classes, self.size = cin, classes, size
  def forward_features(self, x:Tensor) -> list[Tensor]: raise NotImplementedError
  def forward_head(self, xl:list[Tensor]) -> Tensor: raise NotImplementedError
  def __call__(self, x:Tensor) -> Tensor:
    return self.forward_head(self.forward_features(x))

class ModelReparam(Model):
  def reparam(self): raise NotImplementedError
