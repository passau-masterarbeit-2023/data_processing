{ pkgs ? import <nixpkgs> {} }:

let
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
in
pkgs.mkShell {
  packages = [
    # packages
    my-python
  ];
}