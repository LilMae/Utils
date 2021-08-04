model = dict(
    type='MaskRCNN',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=7,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=7,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))

dataset_type = 'COCODataset'
data_root = './data_dataset_converted/'
img_norm_cfg = dict(type='Normalize',
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    img_norm_cfg,
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            img_norm_cfg,
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
classes = ('microcystis', 'synedra', 'staurastrum', 'pediastrum',
           'oscillatoria', 'anabaena', 'aphanizomenon')
"""
학습에 사용되는 클래스 정보
"""

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root+'annotations.json',
        img_prefix=data_root,
        pipeline=train_pipeline,
        classes=classes
        ),
    val=dict(
        type=dataset_type,
        ann_file=data_root+'annotations.json',
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes=classes
        ),
    test=dict(
        type=dataset_type,
        ann_file=data_root+'annotations.json',
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes=classes
        )
)


optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
"""
파이토치에서 사용할 수 있는 optimzer와 동일
https://pytorch.org/docs/stable/optim.html?highlight=optim#module-torch.optim
를 참고하여
dict(type='사용하고자 하는 optimizer', 해당 optimizer가 가지는 파라미터들)
"""

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 22])
"""
학습 스케줄러
1. wrmup 파라미터
    Args:
        by_epoch (bool): epoch에 따라 가중치를 조정 유무(warmup 유무)
        warmup (string): 가중치 warmup하는 방식
            'constant', 'linear','exp'
        warmup_iters (int): wramup이 지속되는 epoch 길이
        warmup_ratio (float): wramup학습에서 시작 lr은 warmup_ratio * initial_lr
        warmup_by_epoch (bool): 매 학습마다 warmup을 수행 유무

2. 학습 방법 파라미터
dict(policy='사용하고자하는 스케줄러', 해당 스케줄러가 가지는 파리미터들)
ex)

2.1 Step
dict(policy='Step', step, gamma=0.1, min_lr=None)
Args:
        step (int | list[int]): Step to decay the LR. If an int value is given,
            regard it as the decay interval. If a list is given, decay LR at
            these steps.
        gamma (float, optional): Decay LR ratio. Default: 0.1.
        min_lr (float, optional): Minimum LR value to keep. If LR after decay
            is lower than `min_lr`, it will be clipped to this value. If None
            is given, we don't perform lr clipping. Default: None.
            
2.2 Poly
dict(policy='Poly', power=1., min_lr=0.)

2.3 Inv
dict(policy='Inv', gamma, power=1.)

2.4 CosineAnnealing
dict(policy='CosineAnnealing', min_lr=None, min_lr_ratio=None)

2.5 FlatCosineAnnealing
dict(policy='FlatCosineAnnealing', start_percent=0.75, min_lr=None, min_lr_ratio=None)
    Modified from https://github.com/fastai/fastai/blob/master/fastai/callback/schedule.py#L128 # noqa: E501
Args:
    start_percent (float): When to start annealing the learning rate
        after the percentage of the total training steps.
        The value should be in range [0, 1).
        Default: 0.75
    min_lr (float, optional): The minimum lr. Default: None.
    min_lr_ratio (float, optional): The ratio of minimum lr to the base lr.
        Either `min_lr` or `min_lr_ratio` should be specified.
        Default: None.

2.6 CosineRestart
dict(policy='CosineRestart', periods, restart_weights=[1], min_lr=None, min_lr_ratio=None)
Args:
        periods (list[int]): Periods for each cosine anneling cycle.
        restart_weights (list[float], optional): Restart weights at each
            restart iteration. Default: [1].
        min_lr (float, optional): The minimum lr. Default: None.
        min_lr_ratio (float, optional): The ratio of minimum lr to the base lr.
            Either `min_lr` or `min_lr_ratio` should be specified.
            Default: None.

2.7 Cyclic
dict(policy='Cyclic', by_epoch=False, target_ratio=(10, 1e-4), cyclic_times=1, step_ratio_up=0.4, anneal_strategy='cos')
Args:
        by_epoch (bool): Whether to update LR by epoch.
        target_ratio (tuple[float]): Relative ratio of the highest LR and the
            lowest LR to the initial LR.
        cyclic_times (int): Number of cycles during training
        step_ratio_up (float): The ratio of the increasing process of LR in
            the total cycle.
        anneal_strategy (str): {'cos', 'linear'}
            Specifies the annealing strategy: 'cos' for cosine annealing,
            'linear' for linear annealing. Default: 'cos'.

2.8 OneCycle
dict(policy='OneCycle', max_lr, total_steps=None, pct_start=0.3, anneal_strategy='cos', div_factor=25, final_div_factor=1e4, three_phase=False)
Args:
        max_lr (float or list): Upper learning rate boundaries in the cycle
            for each parameter group.
        total_steps (int, optional): The total number of steps in the cycle.
            Note that if a value is not provided here, it will be the max_iter
            of runner. Default: None.
        pct_start (float): The percentage of the cycle (in number of steps)
            spent increasing the learning rate.
            Default: 0.3
        anneal_strategy (str): {'cos', 'linear'}
            Specifies the annealing strategy: 'cos' for cosine annealing,
            'linear' for linear annealing.
            Default: 'cos'
        div_factor (float): Determines the initial learning rate via
            initial_lr = max_lr/div_factor
            Default: 25
        final_div_factor (float): Determines the minimum learning rate via
            min_lr = initial_lr/final_div_factor
            Default: 1e4
        three_phase (bool): If three_phase is True, use a third phase of the
            schedule to annihilate the learning rate according to
            final_div_factor instead of modifying the second phase (the first
            two phases will be symmetrical about the step indicated by
            pct_start).
            Default: False
"""

runner = dict(type='EpochBasedRunner', max_epochs=24)

checkpoint_config = dict(interval=1)
"""Save checkpoints periodically.
    Args:
        interval (int): The saving period. If ``by_epoch=True``, interval
            indicates epochs, otherwise it indicates iterations.
            Default: -1, which means "never".
        by_epoch (bool): Saving checkpoints by epoch or by iteration.
            Default: True.
        save_optimizer (bool): Whether to save optimizer state_dict in the
            checkpoint. It is usually used for resuming experiments.
            Default: True.
        out_dir (str, optional): The directory to save checkpoints. If not
            specified, ``runner.work_dir`` will be used by default.
        max_keep_ckpts (int, optional): The maximum checkpoints to keep.
            In some cases we want only the latest few checkpoints and would
            like to delete old ones to save the disk space.
            Default: -1, which means unlimited.
"""

log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
"""
Args:
    interval (int) – Logging interval (every k iterations).
    ignore_last (bool) – Ignore the log of last iterations in each epoch if less than interval.
    reset_flag (bool) – Whether to clear the output buffer after logging.
    by_epoch (bool) – Whether EpochBasedRunner is used.
    hooks (dict) 아래에 있는 양식으로 넣어준다.
    
1. WandbLoggerHook
dict(type='WandbLoggerHook', init_kwargs=None, interval=10, ignore_last=True, reset_flag=False, commit=True, by_epoch=True, with_step=True)

2. TextLoggerHook
dict(type='TextLoggerHook', by_epoch=True, interval=10, ignore_last=True, reset_flag=False, interval_exp_name=1000)

3. TensorboardLoggerHook
dict(type='TensorboardLoggerHook', log_dir=None, interval=10, ignore_last=True, reset_flag=False, by_epoch=True)

4. PaviLoggerHook
dict(type='PaviLoggerHook',init_kwargs=None, add_graph=False, add_last_ckpt=False, interval=10, ignore_last=True, reset_flag=False, by_epoch=True, img_key='img_info')

5. NeptuneLoggerHook
dict(type='NeptuneLoggerHook', init_kwargs=None, interval=10, ignore_last=True, reset_flag=True, with_step=True, by_epoch=True)

6. MlflowLoggerHook
dict(type='MlflowLoggerHook', exp_name=None, tags=None, log_model=True, interval=10, ignore_last=True, reset_flag=False, by_epoch=True)

7. DvcliveLoggerHook
dict(type='DvcliveLoggerHook', path, interval=10, ignore_last=True, reset_flag=True, by_epoch=True)
"""

load_from = 'work_dir/epoch_24.pth'
"""
학습된 모델 불러오기(다운로드 주소 or .pth)
"""
resume_from = None
"""
Resume checkpoints from a given path, the training
"""
workflow = [('train', 1)]
"""
list(phase,epochs)
phase가 몇번의 epochs의 간격을 두고 반복적으로 수행되는지를 결정
ex)
    [('train', 1),('val', 1)]
"""

work_dir = 'work_dir'
"""
log, checkpopint등이 저장될 폴더 위치
"""