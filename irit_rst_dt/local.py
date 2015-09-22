"""
Paths and settings used for this experimental harness
In the future we may move this to a proper configuration file.
"""

# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

from __future__ import print_function
from os import path as fp
import itertools as itr

from attelo.harness.config import (LearnerConfig,
                                   Keyed)
# from attelo.decoding.astar import (AstarArgs,
#                                    AstarDecoder,
#                                    Heuristic,
#                                    RfcConstraint)
from attelo.decoding.eisner import EisnerDecoder
from attelo.decoding.mst import (MstDecoder, MstRootStrategy)
from attelo.learning.local import (SklearnAttachClassifier,
                                   SklearnLabelClassifier)
from attelo.parser.intra import (IntraInterPair,
                                 HeadToHeadParser,
                                 # SentOnlyParser,
                                 SoftParser)

from sklearn.linear_model import (LogisticRegression)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


from .config.intra import (combine_intra)
from .config.perceptron import (attach_learner_dp_pa,
                                attach_learner_dp_perc,
                                attach_learner_dp_struct_pa,
                                attach_learner_dp_struct_perc,
                                attach_learner_pa,
                                attach_learner_perc,
                                label_learner_dp_pa,
                                label_learner_dp_perc,
                                label_learner_pa,
                                label_learner_perc)
from .config.common import (ORACLE,
                            combined_key,
                            decoder_last,
                            decoder_local,
                            mk_joint,
                            mk_post)

# PATHS

CONFIG_FILE = fp.splitext(__file__)[0] + '.py'


LOCAL_TMP = 'TMP'
"""Things we may want to hold on to (eg. for weeks), but could
live with throwing away as needed"""

SNAPSHOTS = 'SNAPSHOTS'
"""Results over time we are making a point of saving"""

# TRAINING_CORPUS = 'tiny'
# TRAINING_CORPUS = 'corpus/RSTtrees-WSJ-main-1.0/TRAINING'
TRAINING_CORPUS = 'corpus/RSTtrees-WSJ-double-1.0'
"""Corpora for use in building/training models and running our
incremental experiments. Later on we should consider using the
held-out test data for something, but let's make a point of
holding it out for now.

Note that by convention, corpora are identified by their basename.
Something like `corpus/RSTtrees-WSJ-main-1.0/TRAINING` would result
in a corpus named "TRAINING". This could be awkward if the basename
is used in more than one corpus, but we can revisit this scheme as
needed.
"""

# TEST_CORPUS = None
TEST_CORPUS = 'corpus/RSTtrees-WSJ-main-1.0/TEST'
"""Corpora for use in FINAL testing.

You should probably leave this set to None until you've tuned and
tweaked things to the point of being able to write up a paper.
Wouldn't want to overfit to the test corpus, now would we?

(If you leave this set to None, we will perform 10-fold cross
validation on the training data)
"""

TEST_EVALUATION_KEY = None
# TEST_EVALUATION_KEY = 'maxent-AD.L-jnt-mst'
# TEST_EVALUATION_KEY = 'maxent-AD.L-jnt-eisner'
"""Evaluation to use for testing.

Leave this to None until you think it's OK to look at the test data.
The key should be the evaluation key from one of your EVALUATIONS,
eg. 'maxent-C0.9-AD.L_jnt-mst'

(HINT: you can join them together from the report headers)
"""

PTB_DIR = 'ptb3'
"""
Where to read the Penn Treebank from (should be dir corresponding to
parsed/mrg/wsj)
"""

FEATURE_SET = 'dev'  # one of ['dev', 'eyk', 'li2014']
"""
Which feature set to use for feature extraction
"""

FIXED_FOLD_FILE = None
# FIXED_FOLD_FILE = 'folds-TRAINING.json'
"""
Set this to a file path if you *always* want to use it for your corpus
folds. This is for long standing evaluation experiments; we want to
ensure that we have the same folds across different evaluate experiments,
and even across different runs of gather.

NB. It's up to you to ensure that the folds file makes sense
"""


DECODER_LOCAL = decoder_local(0.2)
"local decoder should accept above this score"


def decoder_eisner():
    "our instantiation of the Eisner decoder"
    return Keyed('eisner', EisnerDecoder(use_prob=True))


def decoder_mst():
    "our instantiation of the mst decoder"
    return Keyed('mst', MstDecoder(MstRootStrategy.fake_root, True))


def attach_learner_maxent():
    "return a keyed instance of maxent learner"
    return Keyed('maxent', SklearnAttachClassifier(LogisticRegression()))


def label_learner_maxent():
    "return a keyed instance of maxent learner"
    return Keyed('maxent', SklearnLabelClassifier(LogisticRegression()))


def attach_learner_dectree():
    "return a keyed instance of decision tree learner"
    return Keyed('dectree', SklearnAttachClassifier(DecisionTreeClassifier()))


def label_learner_dectree():
    "return a keyed instance of decision tree learner"
    return Keyed('dectree',
                 SklearnLabelClassifier(DecisionTreeClassifier()))


def attach_learner_rndforest():
    "return a keyed instance of random forest learner"
    return Keyed('rndforest',
                 SklearnAttachClassifier(RandomForestClassifier()))


def label_learner_rndforest():
    "return a keyed instance of decision tree learner"
    return Keyed('rndforest', SklearnLabelClassifier(RandomForestClassifier()))


_LOCAL_LEARNERS = [
    #    ORACLE,
    #    LearnerConfig(attach=attach_learner_maxent(),
    #                  label=label_learner_maxent()),
    LearnerConfig(attach=attach_learner_maxent(),
                  label=label_learner_maxent()),
    #    LearnerConfig(attach=attach_learner_maxent(),
    #                  label=label_learner_oracle()),
    #    LearnerConfig(attach=attach_learner_rndforest(),
    #                  label=label_learner_rndforest()),
    #    LearnerConfig(attach=attach_learner_perc(),
    #                  label=label_learner_maxent()),
    #    LearnerConfig(attach=attach_learner_pa(),
    #                  label=label_learner_maxent()),
    #    LearnerConfig(attach=attach_learner_dp_perc(),
    #                  label=label_learner_maxent()),
    #    LearnerConfig(attach=attach_learner_dp_pa(),
    #                  label=label_learner_maxent()),
]
"""Straightforward attelo learner algorithms to try

It's up to you to choose values for the key field that can distinguish
between different configurations of your learners.

"""


def _structured(klearner):
    """learner configuration pair for a structured learner

    (parameterised on a decoder)"""
    return lambda d: LearnerConfig(attach=klearner(d),
                                   label=label_learner_maxent())


_STRUCTURED_LEARNERS = [
    #    _structured(attach_learner_dp_struct_perc),
    #    _structured(attach_learner_dp_struct_pa),
]
"""Attelo learners that take decoders as arguments.
We assume that they cannot be used relation modelling
"""


def _core_parsers(klearner, unique_real_root=True):
    """Our basic parser configurations
    """
    # joint
    if ((not klearner.attach.payload.can_predict_proba or
         not klearner.label.payload.can_predict_proba)):
        joint = []
    else:
        joint = [
            mk_joint(klearner, d) for d in [
                # decoder_last(),
                # DECODER_LOCAL,
                # decoder_mst(),
                Keyed('eisner',
                      EisnerDecoder(unique_real_root=unique_real_root,
                                    use_prob=True)),
            ]
        ]

    # postlabeling
    post = [
        mk_post(klearner, d) for d in [
            # decoder_last() ,
            # DECODER_LOCAL,
            # decoder_mst(),
            Keyed('eisner',
                  EisnerDecoder(unique_real_root=unique_real_root,
                                use_prob=True)),
        ]
    ]

    return joint + post


_INTRA_INTER_CONFIGS = [
    Keyed('iheads', HeadToHeadParser),
    # Keyed('ionly', SentOnlyParser),
    Keyed('isoft', SoftParser),
]


# -------------------------------------------------------------------------------
# maybe less to edit below but still worth having a glance
# -------------------------------------------------------------------------------

HARNESS_NAME = 'irit-rst-dt'


# possibly obsolete
def _mk_basic_intras(klearner, kconf):
    """Intra/inter parser based on a single core parser
    """
    # NEW intra parsers are explicitly authorized to have more than one
    # real root (necessary for the Eisner decoder, maybe other decoders too)
    parsers = [IntraInterPair(intra=x, inter=y) for x, y in
               zip(_core_parsers(klearner, unique_real_root=False),
                   _core_parsers(klearner))]
    return [combine_intra(p, kconf) for p in parsers]


def _mk_sorc_intras(klearner, kconf):
    """Intra/inter parsers based on a single core parser
    and a sentence oracle
    """
    parsers = [IntraInterPair(intra=x, inter=y) for x, y in
               zip(_core_parsers(ORACLE, unique_real_root=False),
                   _core_parsers(klearner))]
    return [combine_intra(p, kconf, primary='inter') for p in parsers]


def _mk_dorc_intras(klearner, kconf):
    """Intra/inter parsers based on a single core parser
    and a document oracle
    """
    parsers = [IntraInterPair(intra=x, inter=y) for x, y in
               zip(_core_parsers(klearner, unique_real_root=False),
                   _core_parsers(ORACLE))]
    return [combine_intra(p, kconf, primary='intra') for p in parsers]


def _mk_last_intras(klearner, kconf):
    """Parsers using "last" for intra and a core decoder for inter.
    """
    if ((not klearner.attach.payload.can_predict_proba or
         not klearner.label.payload.can_predict_proba)):
        return []

    kconf = Keyed(key=combined_key('last', kconf),
                  payload=kconf.payload)
    econf_last = mk_joint(klearner, decoder_last())
    parsers = [IntraInterPair(intra=econf_last, inter=y) for y in
               _core_parsers(klearner)]
    return [combine_intra(p, kconf, primary='inter') for p in parsers]
# end of possibly obsolete


def _is_junk(econf):
    """
    Any configuration for which this function returns True
    will be silently discarded
    """
    # intrasential head to head mode only works with mst for now
    has = econf.settings
    kids = econf.settings.children
    has_intra_oracle = has.intra and (kids.intra.oracle or kids.inter.oracle)
    has_any_oracle = has.oracle or has_intra_oracle

    decoder_name = econf.parser.key[len(has.key) + 1:]
    # last with last-based intra decoders is a bit redundant
    if has.intra and decoder_name == 'last':
        return True

    # oracle would be redundant with sentence/doc oracles
    if has.oracle and has_intra_oracle:
        return True

    # toggle or comment to enable filtering in/out oracles
    if has_any_oracle:
        return True

    return False


def _evaluations():
    "the evaluations we want to run"
    res = []

    # == one-step (global) parsers ==
    learners = []
    learners.extend(_LOCAL_LEARNERS)
    # current structured learners don't do probs, hence non-prob decoders
    nonprob_eisner = EisnerDecoder(use_prob=False)
    learners.extend(l(nonprob_eisner) for l in _STRUCTURED_LEARNERS)
    # MST is disabled by default, as it does not output projective trees
    # nonprob_mst = MstDecoder(MstRootStrategy.fake_root, False)
    # learners.extend(l(nonprob_mst) for l in _STRUCTURED_LEARNERS)
    global_parsers = itr.chain.from_iterable(_core_parsers(l)
                                             for l in learners)
    res.extend(global_parsers)

    # == two-step parsers: intra then inter-sentential ==
    ii_learners = []  # (intra, inter) learners
    ii_learners.extend((klearner, klearner)
                       for klearner in _LOCAL_LEARNERS)
    # structured learners, cf. supra
    intra_nonprob_eisner = EisnerDecoder(use_prob=False,
                                         unique_real_root=False)
    inter_nonprob_eisner = EisnerDecoder(use_prob=False,
                                         unique_real_root=True)
    ii_learners.extend((l(intra_nonprob_eisner), l(inter_nonprob_eisner))
                       for l in _STRUCTURED_LEARNERS)
    # couples of learners with either sentence- or document-level oracle
    sorc_ii_learners = [
        (ORACLE, inter_lnr) for intra_lnr, inter_lnr in ii_learners
    ]
    dorc_ii_learners = [
        (intra_lnr, ORACLE) for intra_lnr, inter_lnr in ii_learners
    ]
    # enumerate pairs of (intra, inter) parsers
    ii_pairs = []
    for intra_lnr, inter_lnr in itr.chain(ii_learners,
                                          sorc_ii_learners,
                                          dorc_ii_learners):
        # NEW intra parsers are explicitly authorized (in fact, expected)
        # to have more than one real root ; this is necessary for the
        # Eisner decoder and probably others, with "hard" strategies
        ii_pairs.extend(IntraInterPair(intra=x, inter=y) for x, y in
                        zip(_core_parsers(intra_lnr, unique_real_root=False),
                            _core_parsers(inter_lnr, unique_real_root=True)))
    # cross-product: pairs of parsers x intra-/inter- configs
    ii_parsers = [combine_intra(p, kconf,
                                primary=('inter' if p.intra.settings.oracle
                                         else 'intra'))
                  for p, kconf
                  in itr.product(ii_pairs, _INTRA_INTER_CONFIGS)]
    res.extend(ii_parsers)

    return [x for x in res if not _is_junk(x)]


EVALUATIONS = _evaluations()


GRAPH_DOCS = [
    'wsj_1184.out',
    'wsj_1120.out',
]
"""Just the documents that you want to graph.
Set to None to graph everything
"""


def _want_details(econf):
    "true if we should do detailed reporting on this configuration"

    if isinstance(econf.learner, IntraInterPair):
        learners = [econf.learner.intra, econf.learner.inter]
    else:
        learners = [econf.learner]
    has_maxent = any('maxent' in l.key for l in learners)
    has = econf.settings
    kids = econf.settings.children
    has_intra_oracle = has.intra and (kids.intra.oracle or kids.inter.oracle)
    return (has_maxent and
            ('mst' in econf.parser.key or 'astar' in econf.parser.key or
             'eisner' in econf.parser.key) and
            not has_intra_oracle)

DETAILED_EVALUATIONS = [e for e in EVALUATIONS if _want_details(e)]
"""
Any evalutions that we'd like full reports and graphs for.
You could just set this to EVALUATIONS, but this sort of
thing (mostly the graphs) takes time and space to build

HINT: set to empty list for no graphs whatsoever
"""


def print_evaluations():
    """
    Print out the name of each evaluation in our config
    """
    for econf in EVALUATIONS:
        print(econf)
        print()
    print("\n".join(econf.key for econf in EVALUATIONS))

if __name__ == '__main__':
    print_evaluations()
