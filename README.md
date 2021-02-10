# Information content of samples
The author implementation of smooth unique information proposed in the [paper](https://arxiv.org/abs/2101.06640) "Estimating informativeness of samples with Smooth Unique Information" by Hrayr Harutyunyan, Alessandro Achille, Giovanni Paolini, Orchid Majumder, Avinash Ravichandran, Rahul Bhotika, and Stefano Soatto.
This work defines and estimates smooth unique information of samples with respect to classifier weights and predictions. We compute these quantities for linearized neural networks.

To cite the paper please use the following BibTeX:
```text
@inproceedings{harutyunyan2021estimating,
  title={Estimating informativeness of samples with Smooth Unique Information},
  author={Hrayr Harutyunyan and Alessandro Achille and Giovanni Paolini and Orchid Majumder and Avinash Ravichandran and Rahul Bhotika and Stefano Soatto},
  booktitle={International Conference on Learning Representations},
  year={2021},
  url={https://openreview.net/forum?id=kEnBH98BGs5}
}
```

# About the repository
* `nnlib` contains useful general tools for training and working with neural networks. [Source](https://github.com/hrayrhar/nnlib/tree/master/nnlib).
* `sample_info` is the main directory to look at.
* `sample_info/methods/` implements standard and linearized classifiers. The latter ones are mainly used for testing and debugging purposes. They are never used in the main experiments.
* `sample_info/configs/` lists neural network architectures.
* `sample_info/modules` contains is the most important subdirectory
  * `data_utils.py` contains simple datasets and tools for creating datasets.
  * `influence_functions.py` implements influence functions.
  * `misc.py` contains a few tools.
  * `nn_utils.py` extents the corresponding file from `nnlib` and is used to parse neural networks from architecture configs.
  * `ntk.py` is one of the most important files and implements needed functions for working with linearized neural networks, such as computing Jacobians, predicting weights, training and test predictions at custom times.
  * `sgd.py` implements computation of the SGD noise covariance matrix and its diagonal.
  * `stability.py` is implements the proposed methods -- computing information with weights or activations.
* `sample_info/notebooks` contains some Jupyter notebooks. The content of these can be ignored, as most of them are not up do date and were used only for initial experiments.
* `sample_info/scripts` contains scripts of experiments and codes for generating commands.
* `tests` implements some unit tests, which mainly tests NTK tools and stability measures.

## Requirements
* Standard libraries like numpy, scipy, tqdm, scikit-leearn, matplotlib
* Pytorch 1.4
* Tensorboad

To run the tests:
```bash
nosetests test
```

Additionally, you need to have $DATA_DIR in environment, pointing to the directory where all data is stored.

## Experiments
Most experiment commands can be generated from the script `sample_info/scripts/generate_commands.py`.
Here is an example how to run the experiment of computing correlations of informativeness scores with ground truth in 
the case of MNIST 4 vs 9 classification with MLP.

### MNIST 4 vs 9, full batch, mlp 1024
*  Generate and run the commands for getting the ground truth effects, informativeness scores using our method,
and influence functions:
```bash
python -um sample_info.scripts.generate_commands --exp_names mnist4vs9_fullbatch_noreg_small_cnn_ground_truth \
    mnist4vs9_fullbatch_noreg_small_cnn_informativeness \
    mnist4vs9_fullbatch_noreg_small_cnn_influence_functions
```
* Aggregate the results
```bash
python -um sample_info.scripts.aggregate_ground_truth_results --exp_name mnist4vs9_fullbatch_noreg_small_cnn -n 1000
```

## License
This project is licensed under the Apache-2.0 License.
