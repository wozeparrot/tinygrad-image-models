{
  description = "";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    gitignore.url = "github:hercules-ci/gitignore.nix";
    tinygrad.url = "github:wozeparrot/tinygrad-nix";
  };

  outputs = inputs @ {
    nixpkgs,
    flake-utils,
    gitignore,
    ...
  }: let
    inherit (gitignore.lib) gitignoreSource;
    overlay = final: prev: {
      pythonPackagesExtensions =
        prev.pythonPackagesExtensions
        ++ [
          (
            python-final: python-prev: {
              tgim = python-prev.buildPythonPackage {
                pname = "tgim";
                version = "0.0.0";
                pyproject = true;
                src = gitignoreSource ./.;

                nativeBuildInputs = with python-prev; [
                  setuptools
                  wheel
                ];

                propagatedBuildInputs = with python-final; [
                  tinygrad
                ];

                doCheck = false;
              };
            }
          )
        ];
    };
  in
    flake-utils.lib.eachDefaultSystem
    (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            inputs.tinygrad.overlays.default
            overlay
          ];
        };
      in {
        packages = rec {
          inherit (pkgs.python3Packages) tgim;
          default = tgim;
        };

        devShell = pkgs.mkShell {
          packages = let
            python-packages = p:
              with p; [
                pillow
                tinygrad
                numpy

                setuptools
                build
                twine
              ];
            python = pkgs.python311;
          in [
            (python.withPackages python-packages)
          ];
        };
      }
    )
    // {
      overlays.default = overlay;
    };
}
