from .registration import registry, ChallengeSpec
import fewshot.datasets as fd
import fewshot.stores
import fewshot.samplers as fs


_bao_stores = {
    'newsgroup': fewshot.stores.NewsgroupStoreCfg,
    'reuters': fewshot.stores.ReutersStoreCfg,
    'huffpost': fewshot.stores.HuffpostStoreCfg,
    'fewrel': fewshot.stores.FewrelStoreCfg,
    'amazon': fewshot.stores.AmazonStoreCfg,
}
_glue_train_val_stores = {
    'glue_mnli': fewshot.stores.GlueMnliCfg,
    'glue_mrpc': fewshot.stores.GlueMrpcCfg,
    'glue_qnli': fewshot.stores.GlueQnliCfg,
    'glue_qqp': fewshot.stores.GlueQqpCfg,
    'glue_rte': fewshot.stores.GlueRteCfg,
    'glue_sst2': fewshot.stores.GlueSst2Cfg,
}
_gao_stores = {
    'snli': {
        'store': fewshot.stores.SNLICfg(split='validation'),
        'way': 3,
    },
    'trec': {
        # Gao et al tested on test only, but the class with the fewest examples
        # has only 9 examples; we add the training set to get a more representative
        # sample of this class.
        'store': fewshot.stores.TrecCfg(split='train+test'),
        'way': 6,
    },
    'mr': {
        'store': fewshot.stores.MRCfg(split='train'),
        'way': 2,
    },
    'cr': {
        'store': fewshot.stores.CRCfg(split='train'),
        'way': 2,
    },
    'subj': {
        'store': fewshot.stores.SubjCfg(split='train'),
        'way': 2,
    }
}
_bansal_test_stores = {
    'scitail': {
        'store': fewshot.stores.SciTailCfg(split='test'),
        'way': 2,
    },
    'conll': {
        'store': fewshot.stores.ConllCfg(split='test'),
        'way': 4,
    },
}

_bao_train_stores = {k: cfg(split='train') for k, cfg in _bao_stores.items()}
_bao_val_stores = {k: cfg(split='validation') for k, cfg in _bao_stores.items()}
_glue_train_stores = {k: cfg(split='train') for k, cfg in _glue_train_val_stores.items()}
_glue_val_stores = {k: cfg(split=(
    'validation_matched+validation_mismatched' if 'mnli' in k
    else 'validation'
)) for k, cfg in _glue_train_val_stores.items()}
_combined_train_stores = {
    **_bao_train_stores,
    **_glue_train_stores,
}
_combined_val_stores = {
    **_bao_val_stores,
    **_glue_val_stores,
    # Hold out for meta-validation task transfer
    'glue_cola': fewshot.stores.GlueColaCfg(split='train+validation'),
    # Hold out for meta-validation distribution transfer
    'glue_wnli': fewshot.stores.GlueWnliCfg(split='train+validation'),
}


_flex_episodes_per_reported_config = 90
registry.register(ChallengeSpec(
    id='flex',
    hash='29332397efd53c29e77c4fe66405d13ae061306a',
    num_tasks=_flex_episodes_per_reported_config * 12 * 2,  # episodes x datasets x mix_or_zero_shot
    train_stores=_combined_train_stores,
    val_stores=_combined_val_stores,
    metadatasampler=fd.MetadataSamplerCfg(
        seed=0,
        datasets=[
            *[
                fd.DatasetCfg(
                    labeled_store=cfg(split='test'),
                    sampler=fs.FlexTestCfg(
                        min_way=5,
                        max_zero_shot_episodes=_flex_episodes_per_reported_config,
                    ),
                    total_samples=_flex_episodes_per_reported_config * 2,
                    name=name,
                )
                for name, cfg in _bao_stores.items()
            ],
            *[
                fd.DatasetCfg(
                    labeled_store=cfg['store'],
                    sampler=fs.FlexTestCfg(
                        way=cfg['way'],
                        max_zero_shot_episodes=_flex_episodes_per_reported_config,
                    ),
                    total_samples=_flex_episodes_per_reported_config * 2,
                    name=name,
                )
                for name, cfg in {**_gao_stores, **_bansal_test_stores}.items()
            ]
        ]
    )
))


_gao_shots = (16, )
registry.register(ChallengeSpec(
    id='gao',
    hash='95e0bd71d00fe562878a1cbe7591368b1b338034',
    num_tasks=_flex_episodes_per_reported_config *
    len(_gao_shots) * len(_gao_stores),  # episodes x shots x datasets
    train_stores=None,
    val_stores=None,
    metadatasampler=fd.MetadataSamplerCfg(
        seed=0,
        datasets=[
            *[
                fd.DatasetCfg(
                    labeled_store=cfg['store'],
                    sampler=fs.GaoTestCfg(
                        way=cfg['way'],
                        num_support_samples=_gao_shots,
                    ),
                    total_samples=_flex_episodes_per_reported_config * len(_gao_shots),
                    name=name,
                )
                for name, cfg in _gao_stores.items()
            ]
        ]
    )
))

_bansal_shots = (4, 8, 16)
registry.register(ChallengeSpec(
    id='bansal',
    hash='31823db1a92462cd4760526ba23c244789170968',
    num_tasks=_flex_episodes_per_reported_config *
    len(_bansal_shots) * len(_bansal_test_stores),  # episodes x shots x datasets
    train_stores=_glue_train_stores,
    val_stores=_glue_val_stores,
    metadatasampler=fd.MetadataSamplerCfg(
        seed=0,
        datasets=[
            *[
                fd.DatasetCfg(
                    labeled_store=cfg['store'],
                    sampler=fs.GaoTestCfg(
                        way=cfg['way'],
                        num_support_samples=_bansal_shots,
                    ),
                    total_samples=_flex_episodes_per_reported_config * len(_bansal_shots),
                    name=name,
                )
                for name, cfg in _bansal_test_stores.items()
            ]
        ]
    )
))

_bao_shots = (1, 5)
registry.register(ChallengeSpec(
    id='bao',
    hash='5207d3b32f0b2a957badca3cc297e80b7dec18b9',
    num_tasks=_flex_episodes_per_reported_config * len(_bao_shots) * len(_bao_stores),  # episodes x shots x datasets
    train_stores=_bao_train_stores,
    val_stores=_bao_val_stores,
    metadatasampler=fd.MetadataSamplerCfg(
        seed=0,
        datasets=[
            *[
                fd.DatasetCfg(
                    labeled_store=cfg(split='test'),
                    sampler=fs.BaoTestCfg(num_support_samples=_bao_shots),
                    total_samples=_flex_episodes_per_reported_config * len(_bao_shots),
                    name=name,
                )
                for name, cfg in _bao_stores.items()
            ]
        ]
    )
))

