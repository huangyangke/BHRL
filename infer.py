# mmdet常见的推理api
from mmdet.apis import init_detector, show_result_pyplot
# mmcv是mmdet之类的通用库
import mmcv
import numpy as np
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmdet.datasets.pipelines import Compose
import torch

def inference_detector(model, query_imgs, imgs):

    if isinstance(imgs, (list, tuple)):
        is_batch = True
    else:
        imgs = [imgs]
        query_imgs = [query_imgs]
        is_batch = False

    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    test_pipeline = Compose(cfg.infer_pipeline)

    datas = []
    for query_img, img in zip(query_imgs,imgs):
        # prepare data
        # add information into dict
        data = dict(img_info=dict(filename=img),rf_img_info=dict(file_name=query_img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # forward the model
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)

    if not is_batch:
        return results[0]
    else:
        return results

# 配置文件路径
config_file = '/home/mgtv/instance_retrieval/BHRL/configs/BHRL.py'
# 模型权重文件
#checkpoint_file = '/home/mgtv/instance_retrieval/BHRL/work_dirs/BHRLv3/latest.pth'
checkpoint_file = '/home/mgtv/instance_retrieval/BHRL/work_dirs/BHRLv3/best_AP80_epoch_8.pth'
#checkpoint_file = '/home/mgtv/instance_retrieval/BHRL/work_dirs/BHRLv1/latest.pth'
#checkpoint_file = '/home/mgtv/instance_retrieval/BHRL/checkpoints/model_split1.pth'
# 模型初始化
model = init_detector(config_file, checkpoint_file, device='cuda:0') 

# 输入图片
img = '/home/mgtv/instance_retrieval/dst_frames/wj02.png'
query_img = f'/home/mgtv/instance_retrieval/#U6d4b#U8bd5#U7528#U4f8b/#U4f8b#U5b50/pictures/wj010120.png'

#img = '/home/mgtv/instance_retrieval/LogoDeTracker/detection_sample/douyin/template15.png'
#query_img = '/home/mgtv/instance_retrieval/LogoDeTracker/detection_sample/douyin/template6.png'

#img = '/home/mgtv/instance_retrieval/LogoDeTracker/detection_sample/mgtv1.jpg'
#query_img = '/home/mgtv/instance_retrieval/LogoDeTracker/detection_sample/mgtv.jpg'


# 返回结果
result = inference_detector(model, query_img, img)
print(result)
# 在原图上绘制结果
if hasattr(model, 'module'):
    model = model.module
model.show_result(
    img,
    result,
    out_file='/home/mgtv/result1.jpg',
    score_thr=0.5,
    show=False,
    wait_time=0,
    win_name='result',
    bbox_color=(72, 101, 241),
    text_color=(72, 101, 241))