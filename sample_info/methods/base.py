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

from nnlib.nnlib import visualizations as vis
from nnlib.nnlib.method_utils import Method


class BaseClassifier(Method):
    """ Abstract class for classifiers.
    """
    def __init__(self, **kwargs):
        super(BaseClassifier, self).__init__(**kwargs)

    def visualize(self, train_loader, val_loader, **kwargs):
        visualizations = {}

        # visualize pred
        fig, _ = vis.plot_predictions(self, train_loader, key='pred')
        visualizations['predictions/pred-train'] = fig
        if val_loader is not None:
            fig, _ = vis.plot_predictions(self, val_loader, key='pred')
            visualizations['predictions/pred-val'] = fig

        return visualizations


class BaseAutoEncoder(Method):
    """ Abstract class for auto-encoders.
    """
    def __init__(self, **kwargs):
        super(BaseAutoEncoder, self).__init__(**kwargs)

    def visualize(self, train_loader, val_loader, **kwargs):
        visualizations = {}

        # reconstruction plot
        fig, _ = vis.reconstruction_plot(self, train_loader.dataset, val_loader.dataset)
        visualizations['reconstruction'] = fig

        return visualizations
