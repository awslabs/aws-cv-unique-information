from collections import defaultdict
import re

from tqdm import tqdm
from torch.utils.data import Subset, DataLoader
import torch

from . import utils


def get_hook(storage, key, cpu=True):
    # storage is a defaultdict(list)
    def hook(grad):
        if cpu:
            storage[key].append(utils.to_cpu(grad))
        else:
            storage[key].append(grad)
    return hook


def get_activation_gradients(model, dataset, batch_size=256, cpu=True, description="",
                             output_keys_regexp='.*', max_num_examples=2**30,
                             num_workers=0, **kwargs):
    model.eval()

    if num_workers > 0:
        torch.multiprocessing.set_sharing_strategy('file_system')
        torch.multiprocessing.set_start_method('spawn', force=True)

    n_examples = min(len(dataset), max_num_examples)
    loader = DataLoader(dataset=Subset(dataset, range(n_examples)),
                        batch_size=batch_size, shuffle=False,
                        num_workers=num_workers)

    activation_gradients = defaultdict(list)

    for inputs_batch, labels_batch in tqdm(loader, desc=description):
        if isinstance(inputs_batch, torch.Tensor):
            inputs_batch = [inputs_batch]
        if not isinstance(labels_batch, list):
            labels_batch = [labels_batch]

        outputs = model.forward(inputs=inputs_batch, labels=labels_batch,
                                grad_enabled=True, loader=loader, **kwargs)
        batch_losses, batch_info = model.compute_loss(inputs=inputs_batch, labels=labels_batch, outputs=outputs,
                                                      loader=loader, grad_enabled=True, **kwargs)

        # compute total loss
        total_loss = sum([v for k, v in batch_losses.items()])

        # register hooks
        for k, v in batch_info.items():
            # skip if not a variable of interest
            if re.fullmatch(output_keys_regexp, k) is None:
                continue

            # register the hook
            hook_fn = get_hook(storage=activation_gradients, key=k, cpu=cpu)
            batch_info[k].register_hook(hook_fn)

        # backward
        utils.zero_grad(model)
        total_loss.backward()

    for k in activation_gradients:
        activation_gradients[k] = torch.cat(activation_gradients[k], dim=0)
        assert len(activation_gradients[k]) == n_examples

    return activation_gradients


def get_weight_gradients(model, dataset, cpu=True, description="", output_keys_regexp='.*',
                         max_num_examples=2**30, num_workers=0, **kwargs):
    model.eval()

    if num_workers > 0:
        torch.multiprocessing.set_sharing_strategy('file_system')
        torch.multiprocessing.set_start_method('spawn', force=True)

    n_examples = min(len(dataset), max_num_examples)
    loader = DataLoader(dataset=Subset(dataset, range(n_examples)),
                        batch_size=1, shuffle=False, num_workers=num_workers)

    weight_gradients = defaultdict(list)

    # register hooks
    handlers = dict()
    for param_name, param in model.named_parameters():
        # skip if not a variable of interest
        if re.fullmatch(output_keys_regexp, param_name) is None:
            continue

        # register the hook
        hook_fn = get_hook(storage=weight_gradients, key=param_name, cpu=cpu)
        handler = param.register_hook(hook_fn)
        handlers[param_name] = handler

    # loop over the dataset
    for inputs_batch, labels_batch in tqdm(loader, desc=description):
        if isinstance(inputs_batch, torch.Tensor):
            inputs_batch = [inputs_batch]
        if not isinstance(labels_batch, list):
            labels_batch = [labels_batch]

        outputs = model.forward(inputs=inputs_batch, labels=labels_batch,
                                grad_enabled=True, loader=loader, **kwargs)
        batch_losses, batch_info = model.compute_loss(inputs=inputs_batch, labels=labels_batch, outputs=outputs,
                                                      loader=loader, grad_enabled=True, **kwargs)

        # compute total loss
        total_loss = sum([v for k, v in batch_losses.items()])

        # backward
        utils.zero_grad(model)
        total_loss.backward()

    # remove hooks
    for param_name, handler in handlers.items():
        handler.remove()

    for k in weight_gradients:
        weight_gradients[k] = torch.stack(weight_gradients[k])
        assert len(weight_gradients[k]) == n_examples

    return weight_gradients
