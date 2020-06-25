# GraLSP

This is an example implementation of 

>  GraLSP: Graph Neural Networks with Local Structural Patterns (AAAI 2020)

in TensorFlow. 

The GraLSP learns representations for nodes in networks according to their local structural patterns as well its neighboring nodes. 

## Dependencies

Tensorflow 1.15.0rc0

Networkx 2.3

Python 3.7

I am not sure whether other versions are compatible, but I think as long as you use py3 and fix some issues in tensorflow, it will be fine. 

## How to Use

If you want to use a custom dataset named X, please provide the following files and put them under the path `data/X`:

`data/X/edges`: formatted as a file where each line (a, b) describes a link from a to b. a and b are node ids which should be continuous in [0, num_nodes)

`data/X/features.npy`: A numpy array describing node features, where `features[a]` describes the feature vector for node id a in the `data/X/edges` file. 

You can run the code in path `./` with the following script: 

`python main.py --model gralsp --dataset_name YOURDATASET`

Of course there are other command line arguments which are listed in `main.py`. 

## Cite

```
@article{jin2019gralsp,
  title={GraLSP: Graph Neural Networks with Local Structural Patterns},
  author={Jin, Yilun and Song, Guojie and Shi, Chuan},
  journal={arXiv preprint arXiv:1911.07675},
  year={2019}
}
```

If you use related codes or mention our work, you can cite us via the above bibtex. 

The formal bibtex will be provided after the conference proceedings. 

