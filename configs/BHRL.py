# model settings
dataset_split = 1
test_seen_classes = False

model = dict(
    type='BHRL', #检测器（detector）名称
    pretrained=None,
    # 主干网络的类别
    backbone=dict(
        type='ResNet',
        depth=50,#resnet50
        num_stages=4,#主干网络状态（stages）的数目，这些状态产生的特征图作为后续head输入
        out_indices=(0, 1, 2, 3),#4个stage的输出都需要
        frozen_stages=1,#第一个状态的权重被冻结 -1表示全部可学习 0表示stem权重固定 2表示stem和前面两个stage权重固定
        norm_cfg=dict(type='BN', requires_grad=True),#归一化层的配置项 训练归一化里面的gamma和beta
        norm_eval=True,# backbone 所有的 BN 层的均值和方差都直接采用全局预训练值，不进行更新
        style='pytorch'),# 主干网络的风格，'pytorch' 意思是步长为2的层为 3x3 卷积，'caffe' 意思是步长为2的层为 1x1 卷积。
    neck=dict(
        type='FPN',#检测器的neck是FPN，我们同样支持 'NASFPN', 'PAFPN' 等
        in_channels=[256, 512, 1024, 2048],#输入通道数，这与主干网络的输出通道一致
        out_channels=256,#金字塔特征图每一层的输出通道
        num_outs=5),#fpn输出特征图数量
    rpn_head=dict(
        type='RPNHead',
        in_channels=384, # 每个特征图输入通道
        feat_channels=256,# head卷基层的特征通道
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],# 锚点的基本比例，特征图某一位置的锚点面积为 scale * base_sizes
            # anchor宽高比
            ratios=[0.5, 1.0, 2.0],
            # 特征图对应的 stride，必须特征图 stride 一致，不可以随意更改
            strides=[4, 8, 16, 32, 64]),
        # 在训练和测试期间对框进行编码和解码 
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',# 框编码器的类别
            target_means=[0.0, 0.0, 0.0, 0.0],# 用于编码和解码框的目标均值
            target_stds=[1.0, 1.0, 1.0, 1.0]),# 用于编码和解码框的标准差
        # 分类分支的损失函数配置 二分类所以sigmoid为True
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        # 回归分支的损失函数配置
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # RoIHead 封装了两步(two-stage)/级联(cascade)检测器的第二步
    roi_head=dict(
        type='BHRLRoIHead',#RoI head的类型
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',# RoI特征提取器的类型
            # out_size 特征图的输出大小
            # 提取RoI特征时的采样率 0表示自适应比率
            roi_layer=dict(type='RoIAlign', out_size=7, sampling_ratio=0),
            # 提取特征的输出通道
            out_channels=256,
            # 多尺度特征图的步幅
            featmap_strides=[4, 8, 16, 32]),
        #RoIHead 中 box head 的配置
        bbox_head=[
            dict(
                type='BHRLConvFCBBoxHead',
                use_shared_fc = True,
                num_fcs=2,
                in_channels=384,
                fc_out_channels=1024,
                roi_feat_size=7,# 候选区域的大小
                num_classes=1,# 分类的类别数量
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,# 回归是否与类别无关
                ihr = dict(
                    metric_module_in_channel=256,
                    metric_module_out_channel=384,
                ),
                # 可平衡分类损失
                loss_cls=dict(
                    type='RPLoss', use_sigmoid=False, loss_weight=1.0,alpha=0.25),
                loss_bbox=dict(type='L1Loss', loss_weight=1.0))]),

    train_cfg = dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,# IoU >= 0.7(阈值) 被视为正样本
                neg_iou_thr=0.3,# IoU < 0.3(阈值) 被视为负样本
                min_pos_iou=0.3,# 将框作为正样本的最小 IoU 阈值
                match_low_quality=True, # 是否匹配低质量的框
                ignore_iof_thr=-1),# 忽略 bbox 的 IoF 阈值
            # 正/负采样器(sampler)的配置
            sampler=dict(
                type='RandomSampler',
                num=256, #样本数量
                pos_fraction=0.5,#正样本占总样本的比例
                neg_pos_ub=-1,# 基于正样本数量的负样本上限
                add_gt_as_proposals=False),# 采样后是否添加 GT 作为 proposal
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            # nms_across_levels=False,
            nms_pre=2000,#nms前的box数
            max_per_img=1000,#nms后要保留的box数
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        # roi head配置
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False)]),
    test_cfg = dict(
        rpn=dict(
            # nms_across_levels=False,
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,#bbox分数阈值
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))

# dataset settings
dataset_type = 'OneShotMyDataset' #数据集类型
#图像归一化配置
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# We use the same image size as the paper (One-Shot Instance Segmentation). It is the first to study one-shot object detection.
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),#从文件路径里加载图像
    dict(type='LoadAnnotations', with_bbox=True, with_mask=False),#对于当前图像，加载注释信息
    #dict(type='Rotate',level=5,img_fill_val=(124, 116, 104),prob=0.5,scale=1),#随机旋转-15～15
    dict(type='Resize', img_scale=(640, 640), keep_ratio=True),#图像的最大规模 是否保持图像长宽比
    dict(type='RandomFlip', flip_ratio=0.5),#随机水平翻转
    dict(type='PhotoMetricDistortion'),#随机颜色变换
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(640, 640)),#填充图像到指定大小
    dict(type='DefaultFormatBundle'),#to tensor, to DataContainer
    dict(type='LoadSiameseReference'),
    dict(type='ReferenceTransform', img_scale=(128, 128), keep_ratio=True, **img_norm_cfg),
    dict(type='SiameseFormatBundle'),
    # 决定数据中哪些键应该传递给检测器的流程
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),  # 考虑到 RandomFlip 已经被添加到流程里，当 flip=False 时它将不被使用。
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(640, 640)),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='LoadSiameseReference'),
            dict(type='ReferenceTransform', img_scale=(128, 128), keep_ratio=True, **img_norm_cfg),
            dict(type='SiameseFormatBundle'),
            dict(type='Collect', keys=['img'], meta_keys=['img_info', 'filename', 'ori_shape',
                                                          'img_shape', 'pad_shape', 'scale_factor',
                                                          'flip', 'img_norm_cfg', 'label']),
        ])
]

infer_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),  # 考虑到 RandomFlip 已经被添加到流程里，当 flip=False 时它将不被使用。
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(640, 640)),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='LoadSiameseReferenceInfer'),
            dict(type='ReferenceTransform', img_scale=(128, 128), keep_ratio=True, **img_norm_cfg),
            dict(type='SiameseFormatBundle'),
            dict(type='Collect', keys=['img'], meta_keys=['img_info', 'filename', 'ori_shape',
                                                          'img_shape', 'pad_shape', 'scale_factor',
                                                          'flip', 'img_norm_cfg']),
        ])
]

data_root = '/home/mgtv/instance_retrieval/BHRL/data/openlogo/'
# 不同的数据集可以不用合并相同类别，但要注意如果类别名字相同但是实际的类别不同不可以
data = dict(
    samples_per_gpu=24, # 单个gpu的batchsize
    workers_per_gpu=16, # 单个gpu分配的workers
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'Annotations/',
        img_prefix=data_root + 'JPEGImages/',
        test_images_num=1000, #用来测试的图片数量
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'Annotations/',
        img_prefix=data_root + 'JPEGImages/',
        test_images_num=1000,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        dataset_split=dataset_split,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline,
        test_seen_classes=test_seen_classes,
        position=0))

# 保存最佳权重文件
evaluation = dict(interval=1, metric='mAP', save_best='auto', rule='greater')

# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 18])#学习率发生变化的epoch
runner = dict(type='EpochBasedRunner', max_epochs=20)
checkpoint_config = dict(interval=1)#保存间隔
# yapf:disable
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
#log 和训练好的模型保存路径
work_dir = '/home/mgtv/instance_retrieval/BHRL/work_dirs/BHRLv4'
#load_from = '/home/mgtv/instance_retrieval/BHRL/checkpoints/model_split1.pth'
#load_from = '/home/mgtv/instance_retrieval/BHRL/work_dirs/BHRLv1/epoch_9.pth' #只有openlogo数据集
resume_from = '/home/mgtv/instance_retrieval/BHRL/work_dirs/BHRLv3/best_AP80_epoch_8.pth' #添加logodet3k数据集
workflow = [('train', 1), ('val', 1)]#只有一个工作流 且工作流只执行一次