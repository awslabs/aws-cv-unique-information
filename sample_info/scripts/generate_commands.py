# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import argparse

import numpy as np

from nnlib.nnlib.data_utils.base import register_parser as register_fn


local_functions = {}  # storage for registering the functions below


#######################################################################################
#
#     MNIST 4 vs 9 with 1000 examples, full batch size, no weight decay, 1-hidden 1024
#
# exp_name: mnist4vs9_fullbatch_noreg_n1024_mlp
#######################################################################################

@register_fn(local_functions, "mnist4vs9_fullbatch_noreg_n1024_mlp_ground_truth")
def foo():
    config_file = 'sample_info/configs/1hidden-mlp-n1024-binary-mnist.json'
    dataset = 'mnist4vs9'
    num_train_examples = 1000
    lr = 0.001
    n_epochs = 2000
    batch_size = 2**20
    exp_name = 'mnist4vs9_fullbatch_noreg_n1024_mlp'

    command_prefix = f"python -um sample_info.scripts.ground_truth_effects -c {config_file} -e {n_epochs} " \
                     f"--num_train_examples {num_train_examples} --exp_name {exp_name} --batch_size {batch_size} "\
                     f"--lr {lr} -D {dataset} -d cuda --cache_dataset"
    print(command_prefix + ";")

    for sample_idx in range(num_train_examples):
        cur_command = command_prefix + f" --exclude_index {sample_idx}"
        print(cur_command + ";")


@register_fn(local_functions, "mnist4vs9_fullbatch_noreg_n1024_mlp_informativeness")
def foo():
    config_file = 'sample_info/configs/1hidden-mlp-n1024-binary-mnist.json'
    dataset = 'mnist4vs9'
    num_train_examples = 1000
    lr = 0.001
    exp_name = 'mnist4vs9_fullbatch_noreg_n1024_mlp'
    t = 2000

    command_prefix = f"python -um sample_info.scripts.compute_informativeness -c {config_file} " \
                     f"--num_train_examples {num_train_examples} --exp_name {exp_name} " \
                     f"--lr {lr} -D {dataset} -d cuda --which_measures weights-plain predictions " \
                     f"-t {t} --projection none --return_change_vectors --cache_dataset"

    print(command_prefix + ";")


@register_fn(local_functions, "mnist4vs9_fullbatch_noreg_n1024_mlp_influence_functions")
def foo():
    config_file = 'sample_info/configs/1hidden-mlp-n1024-binary-mnist.json'
    dataset = 'mnist4vs9'
    num_train_examples = 1000
    exp_name = 'mnist4vs9_fullbatch_noreg_n1024_mlp'
    recursion_depth = 1000
    damping = 0.01
    scale = 1000.0

    command_prefix = f"python -um sample_info.scripts.compute_influence_functions -c {config_file} " \
                     f"--num_train_examples {num_train_examples} --exp_name {exp_name} " \
                     f"-D {dataset} -d cuda --recursion_depth {recursion_depth} --damping {damping} " \
                     f"--scale {scale} --cache_dataset"

    print(command_prefix + ";")





#######################################################################################
#
#     MNIST 4 vs 9 with 1000 examples, full batch size, with weight decay, 1-hidden 1024
#
# exp_name: mnist4vs9_fullbatch_reg_n1024_mlp
#######################################################################################

@register_fn(local_functions, "mnist4vs9_fullbatch_reg_n1024_mlp_ground_truth")
def foo():
    config_file = 'sample_info/configs/1hidden-mlp-n1024-binary-mnist.json'
    dataset = 'mnist4vs9'
    num_train_examples = 1000
    lr = 0.001
    n_epochs = 2000
    batch_size = 2**20
    exp_name = 'mnist4vs9_fullbatch_reg_n1024_mlp'
    l2_reg_coef = 1.0

    command_prefix = f"python -um sample_info.scripts.ground_truth_effects -c {config_file} -e {n_epochs} " \
                     f"--num_train_examples {num_train_examples} --exp_name {exp_name} --batch_size {batch_size} "\
                     f"--lr {lr} -D {dataset} -d cuda --l2_reg_coef {l2_reg_coef} --cache_dataset"
    print(command_prefix + ";")

    for sample_idx in range(num_train_examples):
        cur_command = command_prefix + f" --exclude_index {sample_idx}"
        print(cur_command + ";")


@register_fn(local_functions, "mnist4vs9_fullbatch_reg_n1024_mlp_informativeness")
def foo():
    config_file = 'sample_info/configs/1hidden-mlp-n1024-binary-mnist.json'
    dataset = 'mnist4vs9'
    num_train_examples = 1000
    lr = 0.001
    exp_name = 'mnist4vs9_fullbatch_reg_n1024_mlp'
    t = 2000
    l2_reg_coef = 1.0

    command_prefix = f"python -um sample_info.scripts.compute_informativeness -c {config_file} " \
                     f"--num_train_examples {num_train_examples} --exp_name {exp_name} " \
                     f"--lr {lr} -D {dataset} -d cuda --which_measures weights-plain predictions " \
                     f"-t {t} --projection none --return_change_vectors --l2_reg_coef {l2_reg_coef} --cache_dataset"

    print(command_prefix + ";")


@register_fn(local_functions, "mnist4vs9_fullbatch_reg_n1024_mlp_influence_functions")
def foo():
    config_file = 'sample_info/configs/1hidden-mlp-n1024-binary-mnist.json'
    dataset = 'mnist4vs9'
    num_train_examples = 1000
    exp_name = 'mnist4vs9_fullbatch_reg_n1024_mlp'
    recursion_depth = 1000
    damping = 0.0
    scale = 1000.0
    l2_reg_coef = 1.0

    command_prefix = f"python -um sample_info.scripts.compute_influence_functions -c {config_file} " \
                     f"--num_train_examples {num_train_examples} --exp_name {exp_name} " \
                     f"-D {dataset} -d cuda --recursion_depth {recursion_depth} --damping {damping} " \
                     f"--scale {scale} --l2_reg_coef {l2_reg_coef} --cache_dataset"

    print(command_prefix + ";")





#######################################################################################
#
#     MNIST 4 vs 9 with 1000 examples, full batch size, no weight decay, 4 layer CNN
#
# exp_name: mnist4vs9_fullbatch_noreg_cnn
#######################################################################################

@register_fn(local_functions, "mnist4vs9_fullbatch_noreg_cnn_ground_truth")
def foo():
    config_file = 'sample_info/configs/binary-mnist-4layer-cnn.json'
    dataset = 'mnist4vs9'
    num_train_examples = 1000
    lr = 0.01
    n_epochs = 2000
    batch_size = 2**20
    exp_name = 'mnist4vs9_fullbatch_noreg_cnn'

    command_prefix = f"python -um sample_info.scripts.ground_truth_effects -c {config_file} -e {n_epochs} " \
                     f"--num_train_examples {num_train_examples} --exp_name {exp_name} --batch_size {batch_size} "\
                     f"--lr {lr} -D {dataset} -d cuda --cache_dataset"
    print(command_prefix + ";")

    for sample_idx in range(num_train_examples):
        cur_command = command_prefix + f" --exclude_index {sample_idx}"
        print(cur_command + ";")


@register_fn(local_functions, "mnist4vs9_fullbatch_noreg_cnn_informativeness")
def foo():
    config_file = 'sample_info/configs/binary-mnist-4layer-cnn.json'
    dataset = 'mnist4vs9'
    num_train_examples = 1000
    lr = 0.01
    t = 2000
    exp_name = 'mnist4vs9_fullbatch_noreg_cnn'

    command_prefix = f"python -um sample_info.scripts.compute_informativeness -c {config_file} " \
                     f"--num_train_examples {num_train_examples} --exp_name {exp_name} " \
                     f"--lr {lr} -D {dataset} -d cuda --which_measures weights-plain predictions " \
                     f"-t {t} --projection none --return_change_vectors --cache_dataset"

    print(command_prefix + ";")


@register_fn(local_functions, "mnist4vs9_fullbatch_noreg_cnn_influence_functions")
def foo():
    config_file = 'sample_info/configs/binary-mnist-4layer-cnn.json'
    dataset = 'mnist4vs9'
    num_train_examples = 1000
    exp_name = 'mnist4vs9_fullbatch_noreg_cnn'
    recursion_depth = 1000
    damping = 0.01
    scale = 1000.0

    command_prefix = f"python -um sample_info.scripts.compute_influence_functions -c {config_file} " \
                     f"--num_train_examples {num_train_examples} --exp_name {exp_name} " \
                     f"-D {dataset} -d cuda --recursion_depth {recursion_depth} --damping {damping} " \
                     f"--scale {scale} --cache_dataset"

    print(command_prefix + ";")





#######################################################################################
#
#     MNIST 4 vs 9 with 1000 examples, full batch size, with weight decay, 4 layer CNN
#
# exp_name: mnist4vs9_fullbatch_reg_cnn
#######################################################################################

@register_fn(local_functions, "mnist4vs9_fullbatch_reg_cnn_ground_truth")
def foo():
    config_file = 'sample_info/configs/binary-mnist-4layer-cnn.json'
    dataset = 'mnist4vs9'
    num_train_examples = 1000
    lr = 0.01
    n_epochs = 2000
    batch_size = 2**20
    exp_name = 'mnist4vs9_fullbatch_reg_cnn'
    l2_reg_coef = 0.1

    command_prefix = f"python -um sample_info.scripts.ground_truth_effects -c {config_file} -e {n_epochs} " \
                     f"--num_train_examples {num_train_examples} --exp_name {exp_name} --batch_size {batch_size} "\
                     f"--lr {lr} -D {dataset} -d cuda --l2_reg_coef {l2_reg_coef} --cache_dataset"
    print(command_prefix + ";")

    for sample_idx in range(num_train_examples):
        cur_command = command_prefix + f" --exclude_index {sample_idx}"
        print(cur_command + ";")


@register_fn(local_functions, "mnist4vs9_fullbatch_reg_cnn_informativeness")
def foo():
    config_file = 'sample_info/configs/binary-mnist-4layer-cnn.json'
    dataset = 'mnist4vs9'
    num_train_examples = 1000
    lr = 0.01
    t = 2000
    exp_name = 'mnist4vs9_fullbatch_reg_cnn'
    l2_reg_coef = 0.1

    command_prefix = f"python -um sample_info.scripts.compute_informativeness -c {config_file} " \
                     f"--num_train_examples {num_train_examples} --exp_name {exp_name} " \
                     f"--lr {lr} -D {dataset} -d cuda --which_measures weights-plain predictions " \
                     f"-t {t} --projection none --return_change_vectors --l2_reg_coef {l2_reg_coef} --cache_dataset"

    print(command_prefix + ";")


@register_fn(local_functions, "mnist4vs9_fullbatch_reg_cnn_influence_functions")
def foo():
    config_file = 'sample_info/configs/binary-mnist-4layer-cnn.json'
    dataset = 'mnist4vs9'
    num_train_examples = 1000
    exp_name = 'mnist4vs9_fullbatch_reg_cnn'
    recursion_depth = 1000
    damping = 0.0
    scale = 1000.0
    l2_reg_coef = 0.1

    command_prefix = f"python -um sample_info.scripts.compute_influence_functions -c {config_file} " \
                     f"--num_train_examples {num_train_examples} --exp_name {exp_name} " \
                     f"-D {dataset} -d cuda --recursion_depth {recursion_depth} --damping {damping} " \
                     f"--scale {scale} --l2_reg_coef {l2_reg_coef} --cache_dataset"

    print(command_prefix + ";")






#######################################################################################
#
#     MNIST 4 vs 9 with 1000 examples, full batch size, no weight decay, 4 layer CNN,
#                                     pretrained
#
# exp_name: mnist4vs9_fullbatch_noreg_cnn_pretrained
#######################################################################################

@register_fn(local_functions, "mnist4vs9_fullbatch_noreg_cnn_pretrained_ground_truth")
def foo():
    config_file = 'sample_info/configs/binary-mnist-4layer-cnn-pretrained.json'
    dataset = 'mnist4vs9'
    num_train_examples = 1000
    lr = 0.002
    n_epochs = 2000
    batch_size = 2**20
    exp_name = 'mnist4vs9_fullbatch_noreg_cnn_pretrained'

    command_prefix = f"python -um sample_info.scripts.ground_truth_effects -c {config_file} -e {n_epochs} " \
                     f"--num_train_examples {num_train_examples} --exp_name {exp_name} --batch_size {batch_size} "\
                     f"--lr {lr} -D {dataset} -d cuda --cache_dataset"
    print(command_prefix + ";")

    for sample_idx in range(num_train_examples):
        cur_command = command_prefix + f" --exclude_index {sample_idx}"
        print(cur_command + ";")


@register_fn(local_functions, "mnist4vs9_fullbatch_noreg_cnn_pretrained_informativeness")
def foo():
    config_file = 'sample_info/configs/binary-mnist-4layer-cnn-pretrained.json'
    dataset = 'mnist4vs9'
    num_train_examples = 1000
    lr = 0.002
    t = 2000
    exp_name = 'mnist4vs9_fullbatch_noreg_cnn_pretrained'

    command_prefix = f"python -um sample_info.scripts.compute_informativeness -c {config_file} " \
                     f"--num_train_examples {num_train_examples} --exp_name {exp_name} " \
                     f"--lr {lr} -D {dataset} -d cuda --which_measures weights-plain predictions " \
                     f"-t {t} --projection none --return_change_vectors --cache_dataset"

    print(command_prefix + ";")


@register_fn(local_functions, "mnist4vs9_fullbatch_noreg_cnn_pretrained_influence_functions")
def foo():
    config_file = 'sample_info/configs/binary-mnist-4layer-cnn-pretrained.json'
    dataset = 'mnist4vs9'
    num_train_examples = 1000
    exp_name = 'mnist4vs9_fullbatch_noreg_cnn_pretrained'
    recursion_depth = 1000
    damping = 0.01
    scale = 1000.0

    command_prefix = f"python -um sample_info.scripts.compute_influence_functions -c {config_file} " \
                     f"--num_train_examples {num_train_examples} --exp_name {exp_name} " \
                     f"-D {dataset} -d cuda --recursion_depth {recursion_depth} --damping {damping} " \
                     f"--scale {scale} --cache_dataset"

    print(command_prefix + ";")








#######################################################################################
#
#     MNIST 4 vs 9 with 1000 examples, full batch size, with weight decay, 4 layer CNN,
#                                     pretrained
#
# exp_name: mnist4vs9_fullbatch_reg_cnn_pretrained
#######################################################################################

@register_fn(local_functions, "mnist4vs9_fullbatch_reg_cnn_pretrained_ground_truth")
def foo():
    config_file = 'sample_info/configs/binary-mnist-4layer-cnn-pretrained.json'
    dataset = 'mnist4vs9'
    num_train_examples = 1000
    lr = 0.002
    n_epochs = 2000
    batch_size = 2**20
    exp_name = 'mnist4vs9_fullbatch_reg_cnn_pretrained'
    l2_reg_coef = 1.0

    command_prefix = f"python -um sample_info.scripts.ground_truth_effects -c {config_file} -e {n_epochs} " \
                     f"--num_train_examples {num_train_examples} --exp_name {exp_name} --batch_size {batch_size} "\
                     f"--lr {lr} -D {dataset} -d cuda --cache_dataset --l2_reg_coef {l2_reg_coef}"
    print(command_prefix + ";")

    for sample_idx in range(num_train_examples):
        cur_command = command_prefix + f" --exclude_index {sample_idx}"
        print(cur_command + ";")


@register_fn(local_functions, "mnist4vs9_fullbatch_reg_cnn_pretrained_informativeness")
def foo():
    config_file = 'sample_info/configs/binary-mnist-4layer-cnn-pretrained.json'
    dataset = 'mnist4vs9'
    num_train_examples = 1000
    lr = 0.002
    t = 2000
    exp_name = 'mnist4vs9_fullbatch_reg_cnn_pretrained'
    l2_reg_coef = 1.0

    command_prefix = f"python -um sample_info.scripts.compute_informativeness -c {config_file} " \
                     f"--num_train_examples {num_train_examples} --exp_name {exp_name} " \
                     f"--lr {lr} -D {dataset} -d cuda --which_measures weights-plain predictions " \
                     f"-t {t} --projection none --return_change_vectors --cache_dataset --l2_reg_coef {l2_reg_coef}"

    print(command_prefix + ";")


@register_fn(local_functions, "mnist4vs9_fullbatch_reg_cnn_pretrained_influence_functions")
def foo():
    config_file = 'sample_info/configs/binary-mnist-4layer-cnn-pretrained.json'
    dataset = 'mnist4vs9'
    num_train_examples = 1000
    exp_name = 'mnist4vs9_fullbatch_reg_cnn_pretrained'
    recursion_depth = 1000
    damping = 0.01
    scale = 1000.0
    l2_reg_coef = 1.0

    command_prefix = f"python -um sample_info.scripts.compute_influence_functions -c {config_file} " \
                     f"--num_train_examples {num_train_examples} --exp_name {exp_name} " \
                     f"-D {dataset} -d cuda --recursion_depth {recursion_depth} --damping {damping} " \
                     f"--scale {scale} --cache_dataset --l2_reg_coef {l2_reg_coef}"

    print(command_prefix + ";")






#######################################################################################
#
#     CIFAR cat vs dog 1000 examples, 2 batches of 500, no weight decay
#                        pretrained resnet18
#
# exp_name: cifar10_cat_vs_dog_noreg_pretrained_resnet18
#######################################################################################

@register_fn(local_functions, "cifar10_cat_vs_dog_noreg_pretrained_resnet18_ground_truth")
def foo():
    config_file = 'sample_info/configs/binary-cifar10-resnet18-pretrained.json'
    dataset = 'cifar10-cat-vs-dog'
    num_train_examples = 1000
    lr = 0.001
    n_epochs = 500
    batch_size = 500
    exp_name = 'cifar10_cat_vs_dog_noreg_pretrained_resnet18'

    command_prefix = f"python -um sample_info.scripts.ground_truth_effects -c {config_file} -e {n_epochs} " \
                     f"--num_train_examples {num_train_examples} --exp_name {exp_name} --batch_size {batch_size} "\
                     f"--lr {lr} -D {dataset} -d cuda --resize_to_imagenet --cache_dataset"
    print(command_prefix + ";")

    for sample_idx in range(num_train_examples):
        cur_command = command_prefix + f" --exclude_index {sample_idx}"
        print(cur_command + ";")


@register_fn(local_functions, "cifar10_cat_vs_dog_noreg_pretrained_resnet18_informativeness")
def foo():
    config_file = 'sample_info/configs/binary-cifar10-resnet18-pretrained.json'
    dataset = 'cifar10-cat-vs-dog'
    num_train_examples = 1000
    lr = 0.001
    exp_name = 'cifar10_cat_vs_dog_noreg_pretrained_resnet18'
    t = 1000

    command_prefix = f"python -um sample_info.scripts.compute_informativeness -c {config_file} " \
                     f"--num_train_examples {num_train_examples} --exp_name {exp_name} " \
                     f"--lr {lr} -D {dataset} -d cuda --which_measures weights-plain predictions " \
                     f"-t {t} --cpu --large_model_regime " \
                     f"--return_change_vectors --resize_to_imagenet --cache_dataset"

    print(command_prefix + ";")


@register_fn(local_functions, "cifar10_cat_vs_dog_noreg_pretrained_resnet18_influence_functions")
def foo():
    config_file = 'sample_info/configs/binary-cifar10-resnet18-pretrained.json'
    dataset = 'cifar10-cat-vs-dog'
    num_train_examples = 1000
    exp_name = 'cifar10_cat_vs_dog_noreg_pretrained_resnet18'
    recursion_depth = 500
    damping = 0.01
    scale = 1000.0

    command_prefix = f"python -um sample_info.scripts.compute_influence_functions -c {config_file} " \
                     f"--num_train_examples {num_train_examples} --exp_name {exp_name} " \
                     f"-D {dataset} -d cuda --recursion_depth {recursion_depth} --damping {damping} " \
                     f"--scale {scale} --cpu --cache_dataset --resize_to_imagenet"

    print(command_prefix + ";")






#######################################################################################
#
#     CIFAR cat vs dog 1000 examples, 2 batches of 500, with weight decay
#                         pretrained resnet18
#
# exp_name: cifar10_cat_vs_dog_reg_pretrained_resnet18
#######################################################################################

@register_fn(local_functions, "cifar10_cat_vs_dog_reg_pretrained_resnet18_ground_truth")
def foo():
    config_file = 'sample_info/configs/binary-cifar10-resnet18-pretrained.json'
    dataset = 'cifar10-cat-vs-dog'
    num_train_examples = 1000
    lr = 0.001
    n_epochs = 500
    batch_size = 500
    exp_name = 'cifar10_cat_vs_dog_reg_pretrained_resnet18'
    l2_reg_coef = 1.0

    command_prefix = f"python -um sample_info.scripts.ground_truth_effects -c {config_file} -e {n_epochs} " \
                     f"--num_train_examples {num_train_examples} --exp_name {exp_name} --batch_size {batch_size} "\
                     f"--lr {lr} -D {dataset} -d cuda --resize_to_imagenet --l2_reg_coef {l2_reg_coef} --cache_dataset"
    print(command_prefix + ";")

    for sample_idx in range(num_train_examples):
        cur_command = command_prefix + f" --exclude_index {sample_idx}"
        print(cur_command + ";")


@register_fn(local_functions, "cifar10_cat_vs_dog_reg_pretrained_resnet18_informativeness")
def foo():
    config_file = 'sample_info/configs/binary-cifar10-resnet18-pretrained.json'
    dataset = 'cifar10-cat-vs-dog'
    num_train_examples = 1000
    lr = 0.001
    exp_name = 'cifar10_cat_vs_dog_reg_pretrained_resnet18'
    l2_reg_coef = 1.0
    t = 1000

    command_prefix = f"python -um sample_info.scripts.compute_informativeness -c {config_file} " \
                     f"--num_train_examples {num_train_examples} --exp_name {exp_name} " \
                     f"--lr {lr} -D {dataset} -d cuda --which_measures weights-plain predictions " \
                     f"-t {t} --cpu --large_model_regime " \
                     f"--return_change_vectors --resize_to_imagenet --l2_reg_coef {l2_reg_coef} --cache_dataset"

    print(command_prefix + ";")


@register_fn(local_functions, "cifar10_cat_vs_dog_reg_pretrained_resnet18_influence_functions")
def foo():
    config_file = 'sample_info/configs/binary-cifar10-resnet18-pretrained.json'
    dataset = 'cifar10-cat-vs-dog'
    num_train_examples = 1000
    exp_name = 'cifar10_cat_vs_dog_reg_pretrained_resnet18'
    l2_reg_coef = 1.0
    recursion_depth = 500
    damping = 0.0
    scale = 1000.0

    command_prefix = f"python -um sample_info.scripts.compute_influence_functions -c {config_file} " \
                     f"--num_train_examples {num_train_examples} --exp_name {exp_name} " \
                     f"-D {dataset} -d cuda --recursion_depth {recursion_depth} --damping {damping} " \
                     f"--scale {scale} --cpu --l2_reg_coef {l2_reg_coef} --cache_dataset --resize_to_imagenet"

    print(command_prefix + ";")



#######################################################################################
#
#     Data summarization: mnist 4 vs 9, mlp 1024, no decay
#
# exp_name: mnist4vs9_noreg
#######################################################################################

@register_fn(local_functions, "data_sum_mnist4vs9_random")
def foo():
    config_file = 'sample_info/configs/1hidden-mlp-n1024-binary-mnist.json'
    dataset = 'mnist4vs9'
    num_train_examples = 1000
    baseline_name = 'random'
    exp_name = 'mnist4vs9_noreg'
    lr = 0.001
    optimizer = 'sgd'
    n_epochs = 2000
    batch_size = 2**20
    percents = np.linspace(0.0, 0.95, 21)
    n_runs = 5

    for p in percents:
        for run_id in range(n_runs):
            command = f"python -um sample_info.scripts.data_summarization -c {config_file} -e {n_epochs} " \
                      f"-D {dataset} --num_train_examples {num_train_examples} "\
                      f"--batch_size {batch_size} --lr {lr} -d cuda --baseline_name {baseline_name} "\
                      f"--exp_name {exp_name} --random_baseline_seed {run_id} --exclude_ratio {p} "\
                      f"--optimizer {optimizer} --cache_dataset"
            print(command + ";")


@register_fn(local_functions, "prep_informativeness_orders_mnist4vs9")
def foo():
    config_file = 'sample_info/configs/1hidden-mlp-n1024-binary-mnist.json'
    dataset = 'mnist4vs9'
    num_train_examples = 1000
    exp_name = 'mnist4vs9_noreg'
    lr = 0.001
    t = 2000

    # weights
    command = f"python -um sample_info.scripts.prepare_informativeness_orders_for_data_summarization " \
              f"-c {config_file} -D {dataset} --num_train_examples {num_train_examples} "\
              f"--lr {lr} -d cuda --cpu --exp_name {exp_name} -t {t} "\
              f"--which_measure weights-plain"
    print(command + ";")

    # predictions
    command = f"python -um sample_info.scripts.prepare_informativeness_orders_for_data_summarization " \
              f"-c {config_file} -D {dataset} --num_train_examples {num_train_examples} " \
              f"--lr {lr} -d cuda --cpu --exp_name {exp_name} -t {t} " \
              f"--which_measure predictions"

    print(command + ";")


@register_fn(local_functions, "data_sum_mnist4vs9_non_iterative_informativeness")
def foo():
    config_file = 'sample_info/configs/1hidden-mlp-n1024-binary-mnist.json'
    dataset = 'mnist4vs9'
    num_train_examples = 1000
    exp_name = 'mnist4vs9_noreg'
    lr = 0.001
    optimizer = 'sgd'
    n_epochs = 2000
    batch_size = 2**20
    percents = np.linspace(0.0, 0.95, 21)

    for p in percents:
        # weights top
        baseline_name = 'weights-plain-top'
        command = f"python -um sample_info.scripts.data_summarization -c {config_file} -e {n_epochs} " \
                  f"-D {dataset} --num_train_examples {num_train_examples} " \
                  f"--batch_size {batch_size} --lr {lr} -d cuda --baseline_name {baseline_name} " \
                  f"--exp_name {exp_name} --exclude_ratio {p} " \
                  f"--sample_ranking_file sample_info/results/data-summarization/orders/{exp_name}/iter0-weights.pkl " \
                  f"--optimizer {optimizer} --exclude_side top --cache_dataset"
        print(command + ";")

        # weights bottom
        baseline_name = 'weights-plain-bottom'
        command = f"python -um sample_info.scripts.data_summarization -c {config_file} -e {n_epochs} " \
                  f"-D {dataset} --num_train_examples {num_train_examples} " \
                  f"--batch_size {batch_size} --lr {lr} -d cuda --baseline_name {baseline_name} " \
                  f"--exp_name {exp_name} --exclude_ratio {p} " \
                  f"--sample_ranking_file sample_info/results/data-summarization/orders/{exp_name}/iter0-weights.pkl " \
                  f"--optimizer {optimizer} --exclude_side bottom --cache_dataset"
        print(command + ";")

        # preds top
        baseline_name = 'predictions-top'
        command = f"python -um sample_info.scripts.data_summarization -c {config_file} -e {n_epochs} " \
                  f"-D {dataset} --num_train_examples {num_train_examples} " \
                  f"--batch_size {batch_size} --lr {lr} -d cuda --baseline_name {baseline_name} " \
                  f"--exp_name {exp_name} --exclude_ratio {p} " \
                  f"--sample_ranking_file sample_info/results/data-summarization/orders/{exp_name}/iter0-predictions.pkl " \
                  f"--optimizer {optimizer} --exclude_side top --cache_dataset"
        print(command + ";")

        # preds bottom
        baseline_name = 'predictions-bottom'
        command = f"python -um sample_info.scripts.data_summarization -c {config_file} -e {n_epochs} " \
                  f"-D {dataset} --num_train_examples {num_train_examples} " \
                  f"--batch_size {batch_size} --lr {lr} -d cuda --baseline_name {baseline_name} " \
                  f"--exp_name {exp_name} --exclude_ratio {p} " \
                  f"--sample_ranking_file sample_info/results/data-summarization/orders/{exp_name}/iter0-predictions.pkl " \
                  f"--optimizer {optimizer} --exclude_side bottom --cache_dataset"
        print(command + ";")


@register_fn(local_functions, "data_sum_mnist4vs9_iterative_informativeness")
def foo():
    config_file = 'sample_info/configs/1hidden-mlp-n1024-binary-mnist.json'
    dataset = 'mnist4vs9'
    num_train_examples = 1000
    exp_name = 'mnist4vs9_noreg'
    lr = 0.001
    optimizer = 'sgd'
    n_epochs = 2000
    batch_size = 2**20

    iter_idx = 0
    n_excluded = 0
    while n_excluded / num_train_examples < 0.95:
        p = n_excluded / num_train_examples

        # weights
        baseline_name = 'weights-plain-iterative'
        command = f"python -um sample_info.scripts.data_summarization -c {config_file} -e {n_epochs} " \
                  f"-D {dataset} --num_train_examples {num_train_examples} " \
                  f"--batch_size {batch_size} --lr {lr} -d cuda --baseline_name {baseline_name} " \
                  f"--exp_name {exp_name} --exclude_ratio {p} " \
                  f"--sample_ranking_file sample_info/results/data-summarization/orders/{exp_name}/iter{iter_idx}-weights.pkl " \
                  f"--optimizer {optimizer} --exclude_side bottom --cache_dataset"
        print(command + ";")

        # preds
        baseline_name = 'predictions-iterative'
        command = f"python -um sample_info.scripts.data_summarization -c {config_file} -e {n_epochs} " \
                  f"-D {dataset} --num_train_examples {num_train_examples} " \
                  f"--batch_size {batch_size} --lr {lr} -d cuda --baseline_name {baseline_name} " \
                  f"--exp_name {exp_name} --exclude_ratio {p} " \
                  f"--sample_ranking_file sample_info/results/data-summarization/orders/{exp_name}/iter{iter_idx}-predictions.pkl " \
                  f"--optimizer {optimizer} --exclude_side bottom --cache_dataset"
        print(command + ";")

        exclude_count = int(0.05 * (num_train_examples - n_excluded))
        n_excluded += exclude_count
        iter_idx += 1



#######################################################################################
#
#     Data summarization: CIFAR 4 vs 9, pretrained resnet18, no reg
#
# exp_name: cifar10_cat_vs_dog_noreg_pretrained_resnet18
#######################################################################################

@register_fn(local_functions, "data_sum_cifar_cat_vs_dog_random")
def foo():
    config_file = 'sample_info/configs/binary-cifar10-resnet18-pretrained.json'
    dataset = 'cifar10-cat-vs-dog'
    num_train_examples = 1000
    baseline_name = 'random'
    exp_name = 'cifar10_cat_vs_dog_noreg_pretrained_resnet18'
    lr = 0.001
    optimizer = 'sgd'
    n_epochs = 500
    batch_size = 500
    percents = np.linspace(0.0, 0.95, 21)
    n_runs = 3

    for p in percents:
        for run_id in range(n_runs):
            command = f"python -um sample_info.scripts.data_summarization -c {config_file} -e {n_epochs} " \
                      f"-D {dataset} --num_train_examples {num_train_examples} "\
                      f"--batch_size {batch_size} --lr {lr} -d cuda --baseline_name {baseline_name} "\
                      f"--exp_name {exp_name} --random_baseline_seed {run_id} --exclude_ratio {p} "\
                      f"--optimizer {optimizer} --cache_dataset --resize_to_imagenet"
            print(command + ";")


@register_fn(local_functions, "prep_informativeness_orders_cifar10_cat_vs_dog")
def foo():
    config_file = 'sample_info/configs/binary-cifar10-resnet18-pretrained.json'
    dataset = 'cifar10-cat-vs-dog'
    num_train_examples = 1000
    exp_name = 'cifar10_cat_vs_dog_noreg_pretrained_resnet18'
    lr = 0.001
    t = 1000

    # weights
    # command = f"python -um sample_info.scripts.prepare_informativeness_orders_for_data_summarization " \
    #           f"-c {config_file} -D {dataset} --num_train_examples {num_train_examples} "\
    #           f"--lr {lr} -d cuda --cpu --exp_name {exp_name} -t {t} "\
    #           f"--which_measure weights-plain --cache_dataset --resize_to_imagenet "\
    #           f"--cpu --large_model_regime --projection random-subset"
    # print(command + ";")

    # predictions
    command = f"python -um sample_info.scripts.prepare_informativeness_orders_for_data_summarization " \
              f"-c {config_file} -D {dataset} --num_train_examples {num_train_examples} " \
              f"--lr {lr} -d cuda --cpu --exp_name {exp_name} -t {t} " \
              f"--which_measure predictions --cache_dataset --resize_to_imagenet " \
              f"--cpu --large_model_regime --projection random-subset"

    print(command + ";")


@register_fn(local_functions, "data_sum_cifar10_cat_vs_dog_non_iterative_informativeness")
def foo():
    config_file = 'sample_info/configs/binary-cifar10-resnet18-pretrained.json'
    dataset = 'cifar10-cat-vs-dog'
    num_train_examples = 1000
    exp_name = 'cifar10_cat_vs_dog_noreg_pretrained_resnet18'
    lr = 0.001
    optimizer = 'sgd'
    n_epochs = 500
    batch_size = 500
    percents = np.linspace(0.0, 0.95, 21)

    for p in percents:
        # # weights top
        # baseline_name = 'weights-plain-top'
        # command = f"python -um sample_info.scripts.data_summarization -c {config_file} -e {n_epochs} " \
        #           f"-D {dataset} --num_train_examples {num_train_examples} " \
        #           f"--batch_size {batch_size} --lr {lr} -d cuda --baseline_name {baseline_name} " \
        #           f"--exp_name {exp_name} --exclude_ratio {p} " \
        #           f"--sample_ranking_file sample_info/results/data-summarization/orders/{exp_name}/iter0-weights.pkl " \
        #           f"--optimizer {optimizer} --exclude_side top --cache_dataset --resize_to_imagenet"
        # print(command + ";")
        #
        # # weights bottom
        # baseline_name = 'weights-plain-bottom'
        # command = f"python -um sample_info.scripts.data_summarization -c {config_file} -e {n_epochs} " \
        #           f"-D {dataset} --num_train_examples {num_train_examples} " \
        #           f"--batch_size {batch_size} --lr {lr} -d cuda --baseline_name {baseline_name} " \
        #           f"--exp_name {exp_name} --exclude_ratio {p} " \
        #           f"--sample_ranking_file sample_info/results/data-summarization/orders/{exp_name}/iter0-weights.pkl " \
        #           f"--optimizer {optimizer} --exclude_side bottom --cache_dataset --resize_to_imagenet"
        # print(command + ";")

        # preds top
        baseline_name = 'predictions-top'
        command = f"python -um sample_info.scripts.data_summarization -c {config_file} -e {n_epochs} " \
                  f"-D {dataset} --num_train_examples {num_train_examples} " \
                  f"--batch_size {batch_size} --lr {lr} -d cuda --baseline_name {baseline_name} " \
                  f"--exp_name {exp_name} --exclude_ratio {p} " \
                  f"--sample_ranking_file sample_info/results/data-summarization/orders/{exp_name}/iter0-predictions.pkl " \
                  f"--optimizer {optimizer} --exclude_side top --cache_dataset --resize_to_imagenet"
        print(command + ";")

        # preds bottom
        baseline_name = 'predictions-bottom'
        command = f"python -um sample_info.scripts.data_summarization -c {config_file} -e {n_epochs} " \
                  f"-D {dataset} --num_train_examples {num_train_examples} " \
                  f"--batch_size {batch_size} --lr {lr} -d cuda --baseline_name {baseline_name} " \
                  f"--exp_name {exp_name} --exclude_ratio {p} " \
                  f"--sample_ranking_file sample_info/results/data-summarization/orders/{exp_name}/iter0-predictions.pkl " \
                  f"--optimizer {optimizer} --exclude_side bottom --cache_dataset --resize_to_imagenet"
        print(command + ";")


@register_fn(local_functions, "data_sum_cifar10_cat_vs_dog_iterative_informativeness")
def foo():
    config_file = 'sample_info/configs/binary-cifar10-resnet18-pretrained.json'
    dataset = 'cifar10-cat-vs-dog'
    num_train_examples = 1000
    exp_name = 'cifar10_cat_vs_dog_noreg_pretrained_resnet18'
    lr = 0.001
    optimizer = 'sgd'
    n_epochs = 500
    batch_size = 500

    iter_idx = 0
    n_excluded = 0
    while n_excluded / num_train_examples < 0.95:
        p = n_excluded / num_train_examples

        # # weights
        # baseline_name = 'weights-plain-iterative'
        # command = f"python -um sample_info.scripts.data_summarization -c {config_file} -e {n_epochs} " \
        #           f"-D {dataset} --num_train_examples {num_train_examples} " \
        #           f"--batch_size {batch_size} --lr {lr} -d cuda --baseline_name {baseline_name} " \
        #           f"--exp_name {exp_name} --exclude_ratio {p} " \
        #           f"--sample_ranking_file sample_info/results/data-summarization/orders/{exp_name}/iter{iter_idx}-weights.pkl " \
        #           f"--optimizer {optimizer} --exclude_side bottom --cache_dataset --resize_to_imagenet"
        # print(command + ";")

        # preds
        baseline_name = 'predictions-iterative'
        command = f"python -um sample_info.scripts.data_summarization -c {config_file} -e {n_epochs} " \
                  f"-D {dataset} --num_train_examples {num_train_examples} " \
                  f"--batch_size {batch_size} --lr {lr} -d cuda --baseline_name {baseline_name} " \
                  f"--exp_name {exp_name} --exclude_ratio {p} " \
                  f"--sample_ranking_file sample_info/results/data-summarization/orders/{exp_name}/iter{iter_idx}-predictions.pkl " \
                  f"--optimizer {optimizer} --exclude_side bottom --cache_dataset --resize_to_imagenet"
        print(command + ";")

        exclude_count = int(0.05 * (num_train_examples - n_excluded))
        n_excluded += exclude_count
        iter_idx += 1



#######################################################################################
#
#                                    MISC commands
#
#######################################################################################

@register_fn(local_functions, "pretrain_cnn_on_emnist_letters")
def foo():
    command = "python -um sample_info.scripts.train_classifier -c sample_info/configs/emnist-4layer-cnn.json "\
              "-d cuda:0 -b 256 -e 300 -s 1000000 -l sample_info/logs/emnist-cnn-StandardClassifier-adam -D emnist"
    command += '\nmkdir sample_info/modules/resources/'
    command += '\ncp sample_info/logs/emnist-cnn-StandardClassifier-adam/checkpoints/best_val_accuracy.mdl ' \
               'sample_info/modules/resources/emnist_letters_cnn_pretrained.mdl'
    print(command)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_names', '-E', type=str, nargs='+', required=True)
    args = parser.parse_args()

    for exp_name in args.exp_names:
        assert exp_name in local_functions
        local_functions[exp_name]()


if __name__ == '__main__':
    main()
