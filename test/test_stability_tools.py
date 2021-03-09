import pickle
import os

import torch
from torch.utils.data import TensorDataset

from sample_info.methods.classifiers import ClassifierL2
import sample_info.modules.ntk
import sample_info.modules.stability
from .utils import assert_equal_list_of_tensors, assert_equal_dictionaries


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


def test_training_loss_stability():
    dataset = get_data(n_samples=n_train_samples)
    model = get_model()
    ret = sample_info.modules.ntk.prepare_needed_items(model=model, train_data=dataset,
                                                       projection='none')

    ts = [1, 10, 100]
    vectors, quantities = sample_info.modules.stability.training_loss_stability(ts=ts,
                                                                                n=n_train_samples,
                                                                                eta=0.001 / n_train_samples,
                                                                                ntk=ret['ntk'],
                                                                                init_preds=ret['train_init_preds'],
                                                                                Y=ret['train_Y'],
                                                                                continuous=False)
    assert len(vectors) == n_train_samples
    assert len(quantities) == n_train_samples

    file_path = os.path.join('test/resources', 'train_loss_stability_vector_quantities.pkl')
    with open(file_path, 'rb') as f:
        saved = pickle.load(f)

    assert_equal_list_of_tensors(vectors, saved['vectors'])
    assert_equal_list_of_tensors(quantities, saved['quantities'])


def test_training_pred_stability():
    dataset = get_data(n_samples=n_train_samples)
    model = get_model()
    ret = sample_info.modules.ntk.prepare_needed_items(model=model, train_data=dataset,
                                                       projection='none')

    vectors, quantities = sample_info.modules.stability.training_pred_stability(t=100,
                                                                                n=n_train_samples,
                                                                                eta=0.001 / n_train_samples,
                                                                                ntk=ret['ntk'],
                                                                                init_preds=ret['train_init_preds'],
                                                                                Y=ret['train_Y'],
                                                                                continuous=False)
    assert len(vectors) == n_train_samples
    assert len(quantities) == n_train_samples

    file_path = os.path.join('test/resources', 'train_pred_stability_vector_quantities.pkl')

    with open(file_path, 'rb') as f:
        saved = pickle.load(f)

    assert_equal_list_of_tensors(vectors, saved['vectors'])
    assert_equal_list_of_tensors(quantities, saved['quantities'])


def test_test_pred_stability():
    train_data = get_data(n_samples=n_train_samples, seed=42)
    test_data = get_data(n_samples=n_test_samples, seed=43)
    model = get_model()

    ret = sample_info.modules.ntk.prepare_needed_items(model=model, train_data=train_data,
                                                       test_data=test_data, projection='none')

    vectors, quantities = sample_info.modules.stability.test_pred_stability(t=100,
                                                                            n=n_train_samples,
                                                                            eta=0.001 / n_train_samples,
                                                                            ntk=ret['ntk'],
                                                                            test_train_ntk=ret['test_train_ntk'],
                                                                            train_init_preds=ret['train_init_preds'],
                                                                            test_init_preds=ret['test_init_preds'],
                                                                            train_Y=ret['train_Y'],
                                                                            continuous=False)
    assert len(vectors) == n_train_samples
    assert len(quantities) == n_train_samples

    file_path = os.path.join('test/resources', 'test_pred_stability_vector_quantities.pkl')

    with open(file_path, 'rb') as f:
        saved = pickle.load(f)

    assert_equal_list_of_tensors(vectors, saved['vectors'])
    assert_equal_list_of_tensors(quantities, saved['quantities'])


def test_weight_stability_small_data_regime():
    dataset = get_data(n_samples=n_train_samples)
    model = get_model()
    ret = sample_info.modules.ntk.prepare_needed_items(model=model, train_data=dataset,
                                                       projection='none')

    # check both without_sgd=False and without_sgd=True
    for without_sgd in [True]:  # TODO: add False
        vectors, quantities = sample_info.modules.stability.weight_stability(t=100,
                                                                             n=n_train_samples,
                                                                             eta=0.001 / n_train_samples,
                                                                             init_params=ret['init_params'],
                                                                             jacobians=ret['train_jacobians'],
                                                                             ntk=ret['ntk'],
                                                                             init_preds=ret['train_init_preds'],
                                                                             Y=ret['train_Y'],
                                                                             continuous=False,
                                                                             without_sgd=without_sgd,
                                                                             model=model,
                                                                             dataset=dataset)

        assert len(vectors) == n_train_samples
        assert len(quantities) == n_train_samples

        file_path = os.path.join('test/resources', f'weight_without_sgd_{without_sgd}_stability_vector_quantities.pkl')

        with open(file_path, 'rb') as f:
            saved = pickle.load(f)

        for idx in range(n_train_samples):
            assert_equal_dictionaries(vectors[idx], saved['vectors'][idx])
        assert_equal_list_of_tensors(quantities, saved['quantities'])


def test_weight_stability_large_data_regime():
    dataset = get_data(n_samples=n_train_samples)
    model = get_model()
    ret = sample_info.modules.ntk.prepare_needed_items(model=model, train_data=dataset,
                                                       projection='none')

    # check only without_sgd=True, since without_sgd=False and large_model_regime=True are incompatible
    vectors, quantities = sample_info.modules.stability.weight_stability(t=100,
                                                                         n=n_train_samples,
                                                                         eta=0.001 / n_train_samples,
                                                                         init_params=ret['init_params'],
                                                                         jacobians=None,
                                                                         ntk=ret['ntk'],
                                                                         init_preds=ret['train_init_preds'],
                                                                         Y=ret['train_Y'],
                                                                         continuous=False,
                                                                         without_sgd=True,
                                                                         large_model_regime=True,
                                                                         model=model,
                                                                         dataset=dataset)
    assert len(vectors) == n_train_samples
    assert len(quantities) == n_train_samples

    file_path = os.path.join('test/resources', f'weight_without_sgd_True_stability'
                                               f'_large_data_regime_vector_quantities.pkl')

    with open(file_path, 'rb') as f:
        saved = pickle.load(f)

    for idx in range(n_train_samples):
        assert_equal_dictionaries(vectors[idx], saved['vectors'][idx])
    assert_equal_list_of_tensors(quantities, saved['quantities'])
