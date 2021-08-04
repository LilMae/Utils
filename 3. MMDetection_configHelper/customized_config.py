_base_ = './configs/mask_rcnn/mask_rcnn_r50_fpn_2x_coco.py'

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=7),
        mask_head=dict(num_classes=7),
    ),
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                        checkpoint='torchvision://resnet101')
    ))

dataset_type = 'COCODataset'
classes = ('microcystis',
    'synedra',
    'staurastrum',
    'pediastrum',
    'oscillatoria',
    'anabaena',
    'aphanizomenon')

data = dict(
    train=dict(
        img_prefix='./data_dataset_converted/',
        classes=classes,
        ann_file='./data_dataset_converted/annotations.json'),
    val=dict(
        img_prefix='./data_dataset_converted/',
        classes=classes,
        ann_file='./data_dataset_converted/annotations.json'),
    test=dict(
        img_prefix='./data_dataset_converted/',
        classes=classes,
        ann_file='./data_dataset_converted/annotations.json'))

work_dir = 'work_dir'
#load_from = 'work_dir/epoch_24.pth'
runner = dict(type='EpochBasedRunner', max_epochs=24)

#checkpoint_config