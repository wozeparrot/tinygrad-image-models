from tinygrad import Tensor, TinyJit, GlobalCounters
from tinygrad.helpers import fetch
from tinygrad.nn.state import load_state_dict, safe_load
import argparse, importlib, inspect, time, ast, sys
from pathlib import Path
from PIL import Image
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
def load_model(name:str):
  members = inspect.getmembers(importlib.import_module(f"tgim.models.{name}"))
  members = {k.lower(): v for k, v in members}
  return {
    "preprocess": members.get("preprocess", lambda x: x),
    "model": members[name.lower()],
  }

if __name__ == "__main__":
  Tensor.training = False
  Tensor.no_grad = True

  parser = argparse.ArgumentParser()
  parser.add_argument("--model", type=str, required=True, help="name of model to run inference on")
  parser.add_argument("--classes", type=int, default=1000, help="number of classes")
  parser.add_argument("--size", type=str, required=True, help="size of model")
  parser.add_argument("--weights", type=str, required=True, help="path to model weights")
  parser.add_argument("--input", type=str, default=str(Path(__file__).parent / "chicken.jpg"), help="path to input image")
  args = parser.parse_args()

  # load model
  model = load_model(args.model)
  model_instance = model["model"](classes=args.classes, size=args.size)
  load_state_dict(model_instance, safe_load(args.weights))

  @TinyJit
  def run(img): return model_instance(model["preprocess"](img))

  # load image
  img = Image.open(args.input)
  img = img.resize((224, 224))
  img = img.convert("RGB")
  imgt = Tensor(np.array(img)).float().unsqueeze(0).permute(0, 3, 1, 2).contiguous().realize()

  # run inference
  out = run(imgt)
  out_class, out_prob = out.argmax(1).item(), out.max(1).item()

  # benchmark
  times = []
  for _ in range(20):
    GlobalCounters.reset()
    st = time.perf_counter()
    run(imgt)
    times.append(time.perf_counter() - st)
  print(f"best time: {min(times):.5f}ms")

  # print result
  lbls = ast.literal_eval(fetch("https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt").read_text())
  print(f"predicted class: {out_class}, label: {lbls[out_class]}, prob: {out_prob:.5f}")
