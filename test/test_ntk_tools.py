import pickle
import os

import torch
from torch.utils.data import TensorDataset

from sample_info.methods.classifiers import ClassifierL2
import sample_info.modules.ntk
from .utils import assert_equal_dictionaries


n_train_samples = 100
n_test_samples = 20
n_dim = 30
n_outputs = 2

architecture_args = {
    "classifier": [
        {"type": "flatten"},
        {"type": "fc", "dim": 100, "activation": "relu", "batch_norm": False},
        {"type": "fc", "dim": n_outputs}
    ]
}


def get_data(n_samples, seed=42):
    torch.manual_seed(seed)
    x = torch.randn(size=(n_samples, n_dim))
    y = torch.rand(size=(n_samples, n_outputs))
    return TensorDataset(x, y)


def get_model():
    model = ClassifierL2(input_shape=(n_dim,),
                         architecture_args=architecture_args,
                         device='cpu',
                         seed=42)
    return model


def test_get_jacobians():
    dataset = get_data(n_samples=n_train_samples)
    model = get_model()

    # check all three projections
    for projection in ['none', 'random-subset', 'very-sparse']:
        ret = sample_info.modules.ntk.prepare_needed_items(model=model, train_data=dataset,
                                                           projection=projection)
        jacobians = ret['train_jacobians']

        for k, v in jacobians.items():
            assert v.shape[0] == n_train_samples * n_outputs

        file_path = None
        if projection == 'none':
            file_path = os.path.join('test/resources', 'jacobians.pkl')
        if projection == 'random-subset':
            file_path = os.path.join('test/resources', 'jacobians-random-subset-approx.pkl')
        if projection == 'very-sparse':
            file_path = os.path.join('test/resources', 'jacobians-very-sparse-approx.pkl')

        with open(file_path, 'rb') as f:
            saved_jacobians = pickle.load(f)

        assert_equal_dictionaries(jacobians, saved_jacobians)


def test_compute_ntk():
    dataset = get_data(n_samples=n_train_samples)
    model = get_model()
    ret = sample_info.modules.ntk.prepare_needed_items(model=model, train_data=dataset,
                                                       projection='none')
    ntk = ret['ntk']

    assert ntk.ndim == 2
    assert ntk.shape[0] == n_train_samples * n_outputs
    assert ntk.shape[1] == n_train_samples * n_outputs

    file_path = os.path.join('test/resources/ntk.pkl')
    with open(file_path, 'rb') as f:
        saved_ntk = pickle.load(f)

    assert torch.allclose(ntk, saved_ntk, atol=1e-5)


def test_compute_test_train_ntk():
    train_data = get_data(n_samples=n_train_samples, seed=42)
    test_data = get_data(n_samples=n_test_samples, seed=43)
    model = get_model()
    ret = sample_info.modules.ntk.prepare_needed_items(model=model, train_data=train_data,
                                                       test_data=test_data, projection='none')
    ntk = ret['test_train_ntk']

    assert ntk.ndim == 2
    assert ntk.shape[0] == n_test_samples * n_outputs
    assert ntk.shape[1] == n_train_samples * n_outputs

    file_path = os.path.join('test/resources/train-test-ntk.pkl')
    with open(file_path, 'rb') as f:
        saved_ntk = pickle.load(f)

    assert torch.allclose(ntk, saved_ntk, atol=1e-5)


def test_compute_exp_matrix():
    dataset = get_data(n_samples=n_train_samples)
    model = get_model()
    ret = sample_info.modules.ntk.prepare_needed_items(model=model, train_data=dataset,
                                                       projection='none')
    ntk = ret['ntk']
    t = 100
    eta = 0.001 / n_train_samples

    # check both continuous and discrete cases
    for continuous in [False, True]:
        exp_matrix = sample_info.modules.ntk.compute_exp_matrix(t=t, eta=eta, ntk=ntk, continuous=continuous)

        assert exp_matrix.ndim == 2
        assert exp_matrix.shape[0] == exp_matrix.shape[1]
        assert exp_matrix.shape[0] == n_train_samples * n_outputs

        file_path = os.path.join('test/resources', f'exp-matrix-continuous-{continuous}.pkl')
        with open(file_path, 'rb') as f:
            saved_exp_matrix = pickle.load(f)

        assert torch.allclose(exp_matrix, saved_exp_matrix, atol=1e-5)


def test_get_predictions_at_time_t():
    dataset = get_data(n_samples=n_train_samples)
    model = get_model()
    ret = sample_info.modules.ntk.prepare_needed_items(model=model, train_data=dataset,
                                                       projection='none')
    t = 100
    eta = 0.001 / n_train_samples

    # check both continuous and discrete cases
    for continuous in [False, True]:
        preds = sample_info.modules.ntk.get_predictions_at_time_t(t=t, eta=eta, ntk=ret['ntk'],
                                                                  init_preds=ret['train_init_preds'],
                                                                  Y=ret['train_Y'], continuous=continuous)

        assert preds.ndim == 2
        assert preds.shape[0] == n_train_samples
        assert preds.shape[1] == n_outputs

        file_path = os.path.join('test/resources', f'train-pred-continuous-{continuous}.pkl')
        with open(file_path, 'rb') as f:
            saved_preds = pickle.load(f)

        assert torch.allclose(preds, saved_preds, atol=1e-5)


def test_get_test_predictions_at_time_t():
    train_data = get_data(n_samples=n_train_samples, seed=42)
    test_data = get_data(n_samples=n_test_samples, seed=43)
    model = get_model()
    ret = sample_info.modules.ntk.prepare_needed_items(model=model, train_data=train_data,
                                                       test_data=test_data, projection='none')

    t = 100
    eta = 0.001 / n_train_samples

    # check both continuous and discrete cases
    for continuous in [False, True]:
        preds = sample_info.modules.ntk.get_test_predictions_at_time_t(t=t, eta=eta,
                                                                       ntk=ret['ntk'],
                                                                       test_train_ntk=ret['test_train_ntk'],
                                                                       train_init_preds=ret['train_init_preds'],
                                                                       test_init_preds=ret['test_init_preds'],
                                                                       train_Y=ret['train_Y'],
                                                                       continuous=continuous)

        file_path = os.path.join('test/resources', f'test-pred-continuous-{continuous}.pkl')
        with open(file_path, 'rb') as f:
            saved_preds = pickle.load(f)

        assert torch.allclose(preds, saved_preds, atol=1e-5)


def test_get_weights_at_time_t():
    dataset = get_data(n_samples=n_train_samples)
    model = get_model()
    ret = sample_info.modules.ntk.prepare_needed_items(model=model, train_data=dataset,
                                                       projection='none')
    t = 100
    eta = 0.001 / n_train_samples

    # check both continuous and discrete cases
    for continuous in [False, True]:
        weights = sample_info.modules.ntk.get_weights_at_time_t(t=t, eta=eta, ntk=ret['ntk'],
                                                                init_params=ret['init_params'],
                                                                jacobians=ret['train_jacobians'],
                                                                init_preds=ret['train_init_preds'],
                                                                Y=ret['train_Y'], continuous=continuous)

        file_path = os.path.join('test/resources', f'weights-continuous-{continuous}.pkl')
        with open(file_path, 'rb') as f:
            saved_weights = pickle.load(f)

        assert_equal_dictionaries(weights, saved_weights)

    # test also the case then jacobians=None, so it need to iterate over samples and do the Jacobian * vector trick.
    weights = sample_info.modules.ntk.get_weights_at_time_t(t=t, eta=eta, ntk=ret['ntk'],
                                                            init_params=ret['init_params'],
                                                            init_preds=ret['train_init_preds'],
                                                            Y=ret['train_Y'], continuous=continuous,
                                                            model=model,
                                                            dataset=dataset,
                                                            large_model_regime=True)
    file_path = os.path.join('test/resources', f'weights-continuous-{continuous}.pkl')
    with open(file_path, 'rb') as f:
        saved_weights = pickle.load(f)

    assert_equal_dictionaries(weights, saved_weights)
