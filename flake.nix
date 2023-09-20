{
  description = "Data processing for my research project (PhdTrack-Masterarbeit).";
  
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    research-base.url = "github:0nyr/research-base";
  };

  outputs = { self, nixpkgs, research-base, ...}: let
    system = "x86_64-linux";
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

      # custom packages
      research-base
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
