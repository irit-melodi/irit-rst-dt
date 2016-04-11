"""Configuration helpers for using perceptron based learners
"""

# import numpy as np
from sklearn import linear_model as sk

from attelo.harness.config import (Keyed)

from attelo.learning.perceptron import (Perceptron,
                                        PassiveAggressive,
                                        StructuredPerceptron,
                                        StructuredPassiveAggressive)
from attelo.learning.local import (SklearnAttachClassifier,
                                   SklearnLabelClassifier)


VERBOSE = 2  # verbosity level
# parameters for local perceptrons
LOCAL_N_ITER = 20
# deal with imbalanced classes
# "balanced" seems to be beneficial to PA but detrimental to perceptron
LOCAL_CLASS_WEIGHT = None  # {None, "balanced"}
LOCAL_LOSS = "hinge"  # {"hinge", "squared_hinge"}
LOCAL_AVG = True  # NB: sklearn perceptrons don't offer this option
LOCAL_USE_PROB = False  # NB: True would currently have no effect
# parameter for local passive-aggressive
LOCAL_C = 1.0  # was: np.inf

# parameters for structured perceptrons
STRUC_N_ITER = 20  # was 50
STRUC_LOSS = "hinge"  # {"hinge", "squared_hinge"}
STRUC_COST = "dtree"  # {"dtree", "ctree"}
STRUC_AVG = True  # NB: ibid
STRUC_USE_PROB = False  # NB: ibid
# parameter for structured passive-aggressive
STRUC_C = 1.0  # was: np.inf

# ---------------------------------------------------------------------
# scikit
# ---------------------------------------------------------------------


def attach_learner_perc():
    "return a keyed instance of perceptron learner"
    learner = sk.Perceptron(n_iter=LOCAL_N_ITER,
                            class_weight=LOCAL_CLASS_WEIGHT)
    return Keyed('perc', SklearnAttachClassifier(learner))


def label_learner_perc():
    "return a keyed instance of perceptron learner"
    learner = sk.Perceptron(n_iter=LOCAL_N_ITER,
                            class_weight=LOCAL_CLASS_WEIGHT)
    return Keyed('perc', SklearnLabelClassifier(learner))


def attach_learner_pa():
    "return a keyed instance of passive aggressive learner"
    learner = sk.PassiveAggressiveClassifier(C=LOCAL_C,
                                             n_iter=LOCAL_N_ITER,
                                             class_weight=LOCAL_CLASS_WEIGHT)
    return Keyed('pa', SklearnAttachClassifier(learner))


def label_learner_pa():
    "return a keyed instance of passive aggressive learner"
    learner = sk.PassiveAggressiveClassifier(C=LOCAL_C,
                                             n_iter=LOCAL_N_ITER,
                                             class_weight=LOCAL_CLASS_WEIGHT)
    return Keyed('pa', SklearnLabelClassifier(learner))


# ---------------------------------------------------------------------
# dp
# ---------------------------------------------------------------------

def attach_learner_dp_perc():
    "return a keyed instance of perceptron learner"
    return Keyed('dp-perc',
                 SklearnAttachClassifier(
                     Perceptron(n_iter=LOCAL_N_ITER,
                                verbose=VERBOSE,
                                average=LOCAL_AVG,
                                use_prob=LOCAL_USE_PROB)))


def label_learner_dp_perc():
    "return a keyed instance of perceptron learner"
    return Keyed('dp-perc',
                 SklearnLabelClassifier(
                     Perceptron(n_iter=LOCAL_N_ITER,
                                verbose=VERBOSE,
                                average=LOCAL_AVG,
                                use_prob=LOCAL_USE_PROB)))


def attach_learner_dp_pa():
    "return a keyed instance of passive aggressive learner"
    return Keyed('dp-pa',
                 SklearnAttachClassifier(
                     PassiveAggressive(C=LOCAL_C,
                                       n_iter=LOCAL_N_ITER,
                                       verbose=VERBOSE,
                                       average=LOCAL_AVG,
                                       use_prob=LOCAL_USE_PROB)))


def label_learner_dp_pa():
    "return a keyed instance of passive aggressive learner"
    return Keyed('dp-pa',
                 SklearnLabelClassifier(
                     PassiveAggressive(C=LOCAL_C,
                                       n_iter=LOCAL_N_ITER,
                                       verbose=VERBOSE,
                                       average=LOCAL_AVG,
                                       use_prob=LOCAL_USE_PROB)))


def attach_learner_dp_struct_perc(decoder):
    "structured perceptron learning"
    learner = StructuredPerceptron(decoder,
                                   n_iter=STRUC_N_ITER,
                                   verbose=VERBOSE,
                                   cost=STRUC_COST,
                                   average=STRUC_AVG,
                                   use_prob=STRUC_USE_PROB)
    return Keyed('dp-struct-perc', learner)


def attach_learner_dp_struct_pa(decoder):
    "structured passive-aggressive learning"
    learner = StructuredPassiveAggressive(decoder,
                                          C=STRUC_C,
                                          n_iter=STRUC_N_ITER,
                                          verbose=VERBOSE,
                                          loss=STRUC_LOSS,
                                          cost=STRUC_COST,
                                          average=STRUC_AVG,
                                          use_prob=STRUC_USE_PROB)
    return Keyed('dp-struct-pa', learner)
