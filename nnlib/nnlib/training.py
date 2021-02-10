from collections import defaultdict
import os
import pickle
import time
import copy
import logging

from torch.utils.tensorboard import SummaryWriter
from torch import optim
from tqdm import tqdm
import numpy as np
import torch

from . import visualizations as vis
from . import utils


def add_weight_decay(named_params, weight_decay_rate):
    decay = []
    no_decay = []
    for name, param in named_params:
        if len(param.shape) == 1 or name.endswith(".bias"):  # BatchNorm1D or bias
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.0},
            {'params': decay, 'weight_decay': weight_decay_rate}]


def build_optimizer(named_params, optimization_args):
    optimization_args = copy.deepcopy(optimization_args)  # as we are modifying it below
    args = optimization_args['optimizer']

    # add weight decay if needed
    weight_decay_rate = args.pop('weight_decay', None)
    if weight_decay_rate is not None:
        params = add_weight_decay(named_params, weight_decay_rate)
    else:
        params = [param for name, param in named_params]

    optimizer = None
    name = args.pop('name', 'adam')
    if name == 'adam':
        optimizer = optim.Adam(params, **args)
    elif name == 'sgd':
        optimizer = optim.SGD(params, **args)
    else:
        raise ValueError(f"Optimizer with name '{name}' is not supported")
    return optimizer


def build_scheduler(optimizer, optimization_args):
    args = optimization_args.get('scheduler', {})
    name = args.get('name', 'StepLR')
    scheduler = None
    if name == 'StepLR':
        step_size = args.get('step_size', 1)
        gamma = args.get('gamma', 1.0)
        print(f"Using StepLR scheduler with step_size={step_size} and gamma={gamma}")
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif name == 'MultiStepLR':
        milestones = args.get('milestones', [])
        gamma = args.get('gamma', 1.0)
        print(f"Using MultiStepLR scheduler with milestones={milestones} and gamma={gamma}")
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    else:
        raise ValueError(f"Scheduler with name '{name}' is not supported")
    return scheduler


def run_partition(model, epoch, tensorboard, optimizer, loader, partition, training, metrics,
                  data_parallel_model=None, num_accumulation_steps=1):
    # call on_epoch_start callbacks
    if hasattr(model, 'on_epoch_start'):
        model.on_epoch_start(epoch=epoch, tensorboard=tensorboard, partition=partition, loader=loader)
    for metric in metrics:
        metric.on_epoch_start(epoch=epoch, partition=partition)

    losses = defaultdict(list)
    total_number_of_samples = 0
    current_step_idx = 0
    current_total_number_of_samples = 0

    for (batch_data, batch_labels) in tqdm(loader, desc='{} batches'.format(partition)):
        # make the input and labels lists
        if isinstance(batch_data, torch.Tensor):
            batch_data = [batch_data]
        if isinstance(batch_labels, torch.Tensor):
            batch_labels = [batch_labels]

        # compute actual batch size
        actual_batch_size = len(batch_data[0])

        # zero gradients in training phase
        if training and current_step_idx == 0:
            current_total_number_of_samples = 0
            optimizer.zero_grad()

        # forward pass
        forward_model = (model if data_parallel_model is None else data_parallel_model)
        with torch.set_grad_enabled(training):
            outputs = forward_model.forward(inputs=batch_data, labels=batch_labels, partition=partition,
                                            grad_enabled=training, loader=loader, dataset=loader.dataset,
                                            epoch=epoch)

            batch_losses, outputs = model.compute_loss(inputs=batch_data, labels=batch_labels, outputs=outputs,
                                                       grad_enabled=training, loader=loader, dataset=loader.dataset,
                                                       epoch=epoch, partition=partition)
            batch_total_loss = sum([loss for name, loss in batch_losses.items()])
            batch_total_loss_adjusted = batch_total_loss * actual_batch_size
            current_total_number_of_samples += actual_batch_size

        if training:
            # backward pass
            batch_total_loss_adjusted.backward()

            # update the parameters
            if current_step_idx == num_accumulation_steps - 1:
                # adjust the weight of gradients
                for v in model.parameters():
                    if v.requires_grad:
                        v.grad /= current_total_number_of_samples

                # some models might need to do something before applying gradients (e.g. clipping or adding noise)
                # TODO: if data parallelism is on, each model should call its before_weight_update
                if hasattr(model, 'before_weight_update'):
                    model.before_weight_update()

                optimizer.step()

        # call on_iteration_end callbacks
        if hasattr(model, 'on_iteration_end'):
            model.on_iteration_end(outputs=outputs, batch_losses=batch_losses, batch_labels=batch_labels,
                                   partition=partition, tensorboard=tensorboard)
        for metric in metrics:
            metric.on_iteration_end(outputs=outputs, batch_labels=batch_labels, partition=partition)

        # collect all losses
        if len(batch_losses) > 1:
            batch_losses['total'] = batch_total_loss
        for k, v in batch_losses.items():
            losses['{}_{}'.format(partition, k)].append(actual_batch_size * utils.to_numpy(v))
        total_number_of_samples += actual_batch_size

        # update the step counter
        current_step_idx = (current_step_idx + 1) % num_accumulation_steps

    for k, v in losses.items():
        losses[k] = np.sum(v) / total_number_of_samples
        tensorboard.add_scalar('losses/{}'.format(k), losses[k], epoch)

    # if some gradient is left to apply
    if training and current_step_idx > 0:
        logging.warning('The number of training steps in one epoch is not a multiple of '
                        'number of accumulation steps')
        # adjust the weight of gradients
        for v in model.parameters():
            if v.requires_grad:
                v.grad /= current_total_number_of_samples

        if hasattr(model, 'before_weight_update'):
            model.before_weight_update()

        optimizer.step()

    # call on_epoch_end callbacks
    if hasattr(model, 'on_epoch_end'):
        model.on_epoch_end(epoch=epoch, tensorboard=tensorboard, partition=partition, loader=loader)
    for metric in metrics:
        metric.on_epoch_end(epoch=epoch, tensorboard=tensorboard, partition=partition)

    return losses


def make_markdown_table_from_dict(params_dict):
    table = "| param | value |  \n|:----|:-----|  \n"
    for k, v in params_dict.items():
        table += "| {} | {} |  \n".format(k, v)
    return table


def train(model, train_loader, val_loader, epochs, save_iter=10, vis_iter=4,
          optimization_args=None, log_dir=None, args_to_log=None, metrics=None,
          callbacks=None, stopper=None, device_ids=None, num_accumulation_steps=1):
    """ Trains the model. Validation loader can be None.
    Assumptions:
    1. loaders return (batch_inputs, batch_labels), where both can be lists or torch.Tensors
    2. models are inheriting from method_utils.Method.
    3. callback and metrics are inheriting from their abstract classes described in callbacks.py and metrics.py

    :param num_accumulation_steps: an integer that tells how many step gradients should be averaged before
                                   updating the parameters.
    """

    # print the architecture of the model, helps to notice mistakes
    print(model)

    # if there are at least two devices, we use distributed data training using torch.nn.DataParallel
    # note that PyTorch requires and we rely on the fact that the first device should match with model.device
    data_parallel_model = None
    if (device_ids is not None) and len(device_ids) >= 2:
        print(f"Using multiple GPUs: {device_ids}")
        data_parallel_model = torch.nn.DataParallel(model, device_ids=device_ids)

    # if log_dir is not given, logging will be done a new directory in 'logs/' directory
    if log_dir is None:
        log_root = 'logs/'
        utils.make_path(log_root)
        last_run = max([0] + [int(k) for k in os.listdir(log_root) if k.isdigit()])
        log_dir = os.path.join(log_root, '{0:04d}'.format(last_run + 1))
        utils.make_path(log_dir)

    tensorboard = SummaryWriter(log_dir)
    print("Visualize logs using: tensorboard --logdir={0}".format(log_dir))

    # add args_to_log to tensorboard, but also store it separately for easier access
    if args_to_log is not None:
        tensorboard.add_text('script arguments table', make_markdown_table_from_dict(vars(args_to_log)))
        with open(os.path.join(log_dir, 'args.pkl'), 'wb') as f:
            pickle.dump(args_to_log, f)

    optimizer = build_optimizer(model.named_parameters(), optimization_args)
    scheduler = build_scheduler(optimizer, optimization_args)

    # convert metrics to list
    if metrics is None:
        metrics = []
    assert isinstance(metrics, (list, tuple))

    # convert callbacks to list
    if callbacks is None:
        callbacks = []
    assert isinstance(callbacks, (list, tuple))

    for epoch in range(epochs):
        t0 = time.time()

        model.train()
        if data_parallel_model is not None:
            data_parallel_model.train()
        train_losses = run_partition(model=model, epoch=epoch, tensorboard=tensorboard, optimizer=optimizer,
                                     loader=train_loader, partition='train', training=True, metrics=metrics,
                                     data_parallel_model=data_parallel_model,
                                     num_accumulation_steps=num_accumulation_steps)

        val_losses = {}
        if val_loader is not None:
            model.eval()
            if data_parallel_model is not None:
                data_parallel_model.eval()
            val_losses = run_partition(model=model, epoch=epoch, tensorboard=tensorboard, optimizer=optimizer,
                                       loader=val_loader, partition='val', training=False, metrics=metrics,
                                       data_parallel_model=data_parallel_model,
                                       num_accumulation_steps=1)

        # log some statistics
        t = time.time()
        log_string = 'Epoch: {}/{}'.format(epoch, epochs)
        for k, v in list(train_losses.items()) + list(val_losses.items()):
            log_string += ', {}: {:0.6f}'.format(k, v)
        log_string += ', Time: {:0.1f}s'.format(t - t0)
        print(log_string)

        # add visualizations
        if (epoch + 1) % vis_iter == 0 and hasattr(model, 'visualize'):
            visualizations = model.visualize(train_loader, val_loader, tensorboard=tensorboard, epoch=epoch)
            # visualizations is a dictionary containing figures in (name, fig) format.
            # there are visualizations created using matplotlib rather than tensorboard
            for (name, fig) in visualizations.items():
                tensorboard.add_figure(name, fig, epoch)

        # save the model according to our schedule
        if (epoch + 1) % save_iter == 0:
            utils.save(model=model, optimizer=optimizer, scheduler=scheduler,
                       path=os.path.join(log_dir, 'checkpoints', 'epoch{}.mdl'.format(epoch)))

        # Call callbacks. These can be used to save the best model so far or initiate testing.
        for callback in callbacks:
            callback.call(epoch=epoch, model=model, optimizer=optimizer, scheduler=scheduler, log_dir=log_dir)

        # check whether the training should be ended
        if (stopper is not None) and stopper.call(epoch=epoch):
            print(f"Finishing the training at epoch {epoch}...")
            break

        # update the learning rate
        scheduler.step()

    # enable testing mode
    model.eval()

    # save the final version of the network
    utils.save(model=model, optimizer=optimizer, scheduler=scheduler,
               path=os.path.join(log_dir, 'checkpoints', 'final.mdl'))

    # do final visualizations
    if hasattr(model, 'visualize'):
        visualizations = model.visualize(train_loader, val_loader, tensorboard=tensorboard, epoch=epochs)
        for (name, fig) in visualizations.items():
            tensorboard.add_figure(name, fig, epochs)
            vis.savefig(fig, os.path.join(log_dir, name, 'final.png'))
