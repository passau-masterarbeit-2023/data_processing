# Feature engineering

## commands

`pytest -s`: run tests with print from each test being output to the terminal (otherwise, those prints are omitted.

`nix flake update`: update flake channels.

### optimization

It appears that annotation the graph is much faster in sequence that in parallel. Tests have been done on the server, for file `Performance_Test_7572-1650972667-heap.raw_dot.gv` that contains 5 keys, and a total of 8867 nodes.

In sequential:

```
Loading annotated graph took: 52.7257080078125 seconds
Setting labels in seqential...
Setting labels in seqential took: 0.0029768943786621094 seconds
Checking if all nodes have a label...
Checking if all nodes have a label took: 0.0029401779174804688 seconds
All nodes have a label. Nb of nodes: 8867
```

In parallel:

```
Loading annotated graph took: 53.18473720550537 seconds
Setting labels in parallel...
Setting labels in parallel took: 0.30648255348205566 seconds
Checking if all nodes have a label...
Checking if all nodes have a label took: 0.009880781173706055 seconds
All nodes have a label. Nb of nodes: 8867
```

Hence, parallel run took 0.3s, and sequential took 0.003s, so is 100x faster.

## Nix

[search nixos packages](https://search.nixos.org)

``
