from pathlib import Path
from setuptools import setup

directory = Path(__file__).resolve().parent
with open(directory / "README.md", "r") as file: long_description = file.read()
setup(
  name="tgim",
  version="0.0.0",
  description="tinygrad Image Models",
  long_description=long_description,
  long_description_content_type="text/markdown",
  author="wozeparrot",
  license="Apache-2.0",
  packages=["tgim.models", "tgim.common"],
  install_requires=["tinygrad"],
  python_requires=">=3.10",
  include_package_data=True,
)
