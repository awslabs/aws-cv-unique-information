import torch


def assert_equal_dictionaries(a, b):
    for k in a.keys():
        assert k in b.keys()

    for k in b.keys():
        assert k in a.keys()

    for k in a.keys():
        assert torch.allclose(a[k], b[k], atol=1e-5)


def assert_equal_list_of_tensors(a, b):
    assert len(a) == len(b)
    for idx in range(len(a)):
        assert torch.allclose(a[idx], b[idx], atol=1e-5)
