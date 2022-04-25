import abc

import numpy as np
import torch
from alibi_detect.cd import (ChiSquareDrift, ClassifierDrift,
                             ClassifierUncertaintyDrift, CVMDrift, FETDrift,
                             KSDrift, LearnedKernelDrift, LSDDDrift, MMDDrift,
                             SpotTheDiffDrift, TabularDrift)
from alibi_detect.utils.pytorch.kernels import DeepKernel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from torch import nn

# TODO: implement preprocessing for feature reduction in all models
class ModelWrapperInterface(abc.ABC):
    '''
        Drift detector wrapper.
    '''
    @abc.abstractmethod
    def fit(self, x_fit, y_fit):
        '''
            Initializes a drift detector with some combination of
            x_fit, y_fit. x_fit, y_fit can be of any type, depending
            on the data loader implementation.
        '''
        pass
    @abc.abstractmethod
    def infer(self, x_infer, y_infer) -> float:
        '''
            Returns p-value corresponding to drift for the drift
            detector initialized in fit method.
        '''
        pass

@ModelWrapperInterface.register
class CVMModel:
    def fit(self, x_fit, y_fit):
        self.model = CVMDrift(x_fit)
    def infer(self, x_infer, y_infer) -> float:
        return self.model.predict(x_infer)['data']['p_val'][0]

@ModelWrapperInterface.register
class MMDModel:
    def fit(self, x_fit, y_fit):
        self.model = MMDDrift(x_fit)
    def infer(self, x_infer, y_infer) -> float:
        return self.model.predict(x_infer)['data']['p_val']

@ModelWrapperInterface.register
class TabularModel:
    def fit(self, x_fit_annotated, y_fit_annotated):
        '''
            Because TabularModel chooses between KS and Chi-Squared
            feature-wise drift detection depending on user labels,
            each argument should be composed of a numpy array of features
            and a boolean numpy array of whether each feature is continuous.
        '''
        x_fit, x_is_continuous_annotations = x_fit_annotated
        tabular_continuous_annotation = {}
        for i in range(x_is_continuous_annotations.shape[0]):
            if not x_is_continuous_annotations[i]:
                tabular_continuous_annotation[i] = None
        self.model = TabularDrift(x_fit,
            categories_per_feature = tabular_continuous_annotation)
    def infer(self, x_infer_annotated, y_infer_annotated) -> float:
        x_infer, _ = x_infer_annotated
        return self.model.predict(x_infer)['data']['p_val'][0]

@ModelWrapperInterface.register
class FETModel:
    def __init__(self, classifier_model = None, train_drift_frac = .7):
        if classifier_model is None:
            classifier_model = RandomForestClassifier()
        self.classifier_model = classifier_model
        self.train_drift_frac = train_drift_frac
    def fit(self, x_fit, y_fit) -> float:
        shuffled_indices = np.random.permutation(x_fit.shape[0])
        split_index = round(x_fit.shape[0] * self.train_drift_frac)
        x_fit_class, x_fit_fet = np.split(x_fit, [split_index,])
        y_fit_class, y_fit_fet = np.split(y_fit.reshape(-1), [split_index,])
        self.classifier_model.fit(x_fit_class, y_fit_class)
        x_fit_classifier_output = np.array(
            self.classifier_model.predict(x_fit_fet), dtype='bool')
        self.model = FETDrift(x_fit_classifier_output, alternative='two-sided')
    def infer(self, x_infer, y_infer) -> float:
        x_infer_classified = self.classifier_model.predict(x_infer)
        return self.model.predict(x_infer_classified)['data']['p_val'][0]

@ModelWrapperInterface.register
class LearnedKernelModel:
    def __init__(self, kernel:DeepKernel):
        self.kernel = kernel
    def fit(self, x_fit, y_fit):
        # pytorch models accept float32 by default.
        x_fit = x_fit.astype(np.float32)
        self.model = LearnedKernelDrift(x_fit, self.kernel, backend='pytorch')
    def infer(self, x_infer, y_infer) -> float:
        x_infer = x_infer.astype(np.float32)
        return self.model.predict(x_infer)['data']['p_val']

@ModelWrapperInterface.register
class ClassifierDriftModel:
    def __init__(self, classifier_model:nn.Module):
        self.classifier_model = classifier_model
    def fit(self, x_fit, y_fit):
        # pytorch models accept float32 by default.
        x_fit = x_fit.astype(np.float32)
        self.model = ClassifierDrift(
            x_fit, self.classifier_model, backend='pytorch')
    def infer(self, x_infer, y_infer) -> float:
        x_infer = x_infer.astype(np.float32)
        return self.model.predict(x_infer)['data']['p_val']

@ModelWrapperInterface.register
class SpotTheDiffModel:
    def fit(self, x_fit, y_fit):
        x_fit = x_fit.astype(np.float32)
        self.model = SpotTheDiffDrift(x_fit, backend='pytorch')
    def infer(self, x_infer, y_infer) -> float:
        x_infer = x_infer.astype(np.float32)
        return self.model.predict(x_infer)['data']['p_val']

@ModelWrapperInterface.register
class ClassifierUncertaintyModel:
    def __init__(self, classifier_model = None, train_fxn = None, train_drift_frac = .7):
        if classifier_model is None:
            classifier_model = nn.Sequential(
                nn.Linear(18,10),
                nn.ReLU(),
                nn.Linear(10,2))
        if train_fxn is None:
            train_fxn = lambda *args: None
        self.train_fxn = train_fxn
        self.classifier_model = classifier_model
        self.train_drift_frac = train_drift_frac
    def fit(self, x_fit, y_fit):
        shuffled_indices = np.random.permutation(x_fit.shape[0])
        split_index = round(x_fit.shape[0] * self.train_drift_frac)
        x_fit_class, x_fit_uncertainty = np.split(x_fit, [split_index,])
        y_fit_class, y_fit_uncertainty = np.split(y_fit.reshape(-1), [split_index,])
        self.train_fxn(self.classifier_model, x_fit_class, y_fit_class)
        x_fit_uncertainty = x_fit_uncertainty.astype(np.float32)
        self.model = ClassifierUncertaintyDrift(
            x_fit_uncertainty, self.classifier_model, preds_type='logits', backend='pytorch')
    def infer(self, x_infer, y_infer) -> float:
        x_infer = x_infer.astype(np.float32)
        return self.model.predict(x_infer)['data']['p_val']

# TODO: track the runtime
@ModelWrapperInterface.register
class CVMModel:
    def fit(self, x_fit, y_fit):
        self.model = CVMDrift(x_fit)
    def infer(self, x_infer, y_infer) -> float:
        return self.model.predict(x_infer)['data']['p_val'][0]

@ModelWrapperInterface.register
class OutputClassifierAccuracyBaselineModel:
    '''
        Wrapper for sklearn classifier model.
    '''
    def __init__(self, model):
        self.model = model
    def fit(self, x_fit, y_fit):
        self.model.fit(x_fit, y_fit.reshape(-1))
    def infer(self, x_infer, y_infer) -> float:
        return self.model.score(x_infer, y_infer.reshape(-1))

@ModelWrapperInterface.register
class OutputClassifierF1ScoreBaselineModel:
    '''
        Wrapper for sklearn classifier model.
    '''
    def __init__(self, model):
        self.model = model
    def fit(self, x_fit, y_fit):
        self.model.fit(x_fit, y_fit.reshape(-1))
    def infer(self, x_infer, y_infer) -> float:
        y_pred = self.model.predict(x_infer)
        return f1_score(y_infer.reshape(-1), y_pred, average='micro')