# flake.nix
# run a python script with nix-shell using: sudo nix develop
# WARN: Don't forget to add the flake.nix to git!
{
  description = "Python 3.11 environment";
  
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = {nixpkgs, ...}: let
    system = "x86_64-linux";
    #       ↑ Swap it for your system if needed
    #       "aarch64-linux" / "x86_64-darwin" / "aarch64-darwin"
    pkgs = nixpkgs.legacyPackages.${system};
    my-python-packages = ps: with ps; [
      # python packages
      pandas
      requests
      datetime
      graphviz
      python-dotenv
      pygraphviz
      networkx
    ];
    my-python = pkgs.python311.withPackages my-python-packages;
  in {
    devShells.${system}.default = pkgs.mkShell {
      packages = [
        my-python
      ];
    };

  };
}
