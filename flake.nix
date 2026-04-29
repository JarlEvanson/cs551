{
  description = "revm";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
  let
    system = "x86_64-linux";
    pkgs = import nixpkgs { inherit system; };
  in rec {
    devShells.${system} = {
      default = pkgs.mkShellNoCC {
        packages = [
          pkgs.uv
          pkgs.python313
        ];

        shellHook = ''
          export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH

          # Creating and entering the virtual environment.
          uv venv --allow-existing
          source .venv/bin/activate
        '';
      };
    };
  };
}
