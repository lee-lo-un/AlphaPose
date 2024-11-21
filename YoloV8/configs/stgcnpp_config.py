_base_ = ['configs/_base_/default_runtime.py']

# 모델 설정
model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCNPP',
        in_channels=3,
        base_channels=64,
        num_stages=10,
        inflate_stages=[5, 8],
        down_stages=[5, 8],
        edge_importance_weighting=True,
        data_bn=True,
        num_person=2),
    cls_head=dict(
        type='STGCNHead',
        num_classes=60,
        in_channels=256,
        loss_cls=dict(type='CrossEntropyLoss')),
    data_preprocessor=dict(
        type='ActionDataPreprocessor',
        mean=[0.5],
        std=[0.5],
        format_shape='NCTHW'))

# 데이터셋 설정
dataset_type = 'PoseDataset'
ann_file = 'none'

# 테스트 파이프라인 설정
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['bm']),
    dict(
        type='UniformSampleFrames', clip_len=100, num_clips=10,
        test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='PackActionInputs')
]

# 데이터 로더 설정
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=test_pipeline,
        test_mode=True))

# 평가 설정
test_evaluator = dict(type='AccMetric')

# 런타임 설정
default_scope = 'mmaction'
default_hooks = dict(
    runtime_info=dict(type='RuntimeInfoHook'),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

# 환경 설정
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# 로그 설정
log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)
log_level = 'INFO'

# 시각화 설정
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='ActionVisualizer', vis_backends=vis_backends)

# 체크포인트 설정
load_from = None
resume = False