import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict
from mmdet.core import eval_map, eval_recalls

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable
# from pycocotools.coco import COCO
from mmdet.core import eval_recalls
from .api_wrappers import COCO, COCOeval
from .builder import DATASETS
from .custom import CustomDataset
import copy
import pickle
import random
import glob
import xml.etree.ElementTree as ET
import os

# 会过滤宽或高小于32的图片，但是不会过滤gt_box为空的图片
# 目前测试使用了1000张图片，每张图片固定类别，每个类别固定query图片 后续可以改进
@DATASETS.register_module()
class OneShotMyDataset(CustomDataset):

    CLASSES = ('box')

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 test_seen_classes=False,
                 ref_ann_file=None,
                 test_images_num = 1000,
                 position=0):
        self.test_image_num = test_images_num
        self.position = position
        #self.ref_ann_file = ref_ann_file #ref denotes the query in the paper. 
        # 类别名称到id的映射
        self.cat2label = {cat_id: i for i, cat_id in enumerate(OneShotMyDataset.CLASSES)}
        classes = None 
        super(OneShotMyDataset,
              self).__init__(ann_file, pipeline, classes, data_root, img_prefix,
              seg_prefix, proposal_file, test_mode) 

    # 获得boxes和对应的类别
    # 如果传入qclass不为空则只获得指定类别的boxes
    def get_boxes(self, annotation, qclass=''):
        boxes = []
        classes = []
        for obj in annotation.findall('object'):
            #xmin, xmax, ymin, ymax = [int(obj.find('bndbox').find(tag).text) - 1 for tag in ["xmin", "xmax", "ymin", "ymax"]]
            xmin, xmax, ymin, ymax = [int(obj.find('bndbox').find(tag).text) for tag in ["xmin", "xmax", "ymin", "ymax"]]
            tlabel = obj.find('name').text.lower().strip() 
            if qclass != '':  # check query class
                if tlabel == qclass:
                    # 左上 宽高
                    #boxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
                    # 左上 右下
                    boxes.append([xmin, ymin, xmax, ymax])
                    classes.append(tlabel)
            else:
                # Otherwise append boxes for all classes in the image
                # 左上 宽高
                #boxes.append([xmin, ymin, xmax - xmin, ymax - ymin])
                # 左上 右下
                boxes.append([xmin, ymin, xmax, ymax])
                classes.append(tlabel)
        return boxes, classes

    def load_annotations(self, ann_file_dir): 
        print('>>开始加载标注文件')
        data_infos = [] 
        self.query_infos = {}
        
        xml_paths = glob.glob(ann_file_dir + '*.xml')
        #random.shuffle(xml_paths)
        self.cates = []#用于测试
        if self.test_mode:
            xml_paths = xml_paths[-self.test_image_num:]
        else:
            xml_paths = xml_paths[:-self.test_image_num]

        # 遍历每张图的标签
        for xml_path in xml_paths:
            annot = ET.parse(xml_path)
            # 图片文件名
            filename = os.path.basename(xml_path).replace('.xml', '.jpg')
            #filename = annot.find('filename').text.strip() #可能改名了不一定对
            # boxes和类别名称
            boxes, classes = self.get_boxes(annot)
            # 图片宽高
            width = int(annot.find('size').find('width').text.lower().strip())
            height = int(annot.find('size').find('height').text.lower().strip())

            #gt_bboxes = []
            #gt_labels = []
            # 当前图片boxes类别到boxes的映射
            gt_cats2ann = {}
            #gt_cats = []
            # 遍历当前图的bbox
            # 过滤不合理的box
            for box, cat in zip(boxes, classes):
                x1, y1, x2, y2 = box
                inter_w = max(0, min(x2, width) - max(x1, 0))
                inter_h = max(0, min(y2, height) - max(y1, 0))
                if inter_w * inter_h == 0:
                    continue
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                if x1 < 0 or x1 > width or y1 < 0 or y1 > height:
                    continue
                #gt_bboxes.append(box)
                # 所有box标签均为0
                #gt_labels.append(0)
                #gt_cats.append(cat)
                box = np.array(box, dtype=np.float32).reshape(1, 4)
                label = np.array([0], dtype=np.int64)
                if gt_cats2ann.get(cat, None):
                    gt_cats2ann[cat]['bboxes'] = np.concatenate([gt_cats2ann[cat]['bboxes'], box])
                    gt_cats2ann[cat]['labels'] = np.concatenate([gt_cats2ann[cat]['labels'], label])
                else:
                    gt_cats2ann[cat] = dict()
                    gt_cats2ann[cat]['bboxes'] = box
                    gt_cats2ann[cat]['labels'] = label

                if self.query_infos.get(cat, None):
                    self.query_infos[cat].append((filename, box))
                else:
                    self.query_infos[cat] = [(filename, box)]
            # if gt_bboxes:
            #     gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            #     gt_labels = np.array(gt_labels, dtype=np.int64)
            # else:
            #     gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            #     gt_labels = np.array([], dtype=np.int64)

            self.cates.append(cat)# 每张图只取一个类别

            data_infos.append(
                dict(
                    filename=filename,
                    width=width,
                    height=height,
                    ann = gt_cats2ann,
                    #ann=dict(bboxes=gt_bboxes, labels=gt_labels),
                    #category_id=[self.cat2label[cat] for cat in gt_cats]
                    )
                )
        print('>>完成标注文件加载')
        return data_infos

    # 从当前图的bbox的所有类别中随机抽取一个类别
    def random_cate(self, ann_info): 
        #index = np.random.randint(len(ann_info))
        #cate = ann_info[index]['category_id']
        index = np.random.randint(len(ann_info.keys()))
        cate = list(ann_info.keys())[index]
        return cate

    # idx图片的标注信息 只返回当前query的类别
    def get_ann_info(self, idx, cate=None): 
        # 对应的bbox标注
        ann_info = self.data_infos[idx]['ann']
        # 如果没有指定类别 随机抽取一个类
        if cate is None: 
            cate = self.random_cate(ann_info) 
        return self.data_infos[idx]['ann'][cate], cate

    # 可能出现query图片是被检测图片的一部分的情况
    def prepare_train_ref_img(self, cate):
        rf_img_info = dict()
        rand_index = np.random.randint(len(self.query_infos[cate]))
        filename, box = self.query_infos[cate][rand_index]
        rf_img_info['box'] = box
        rf_img_info['file_name'] = filename
        return rf_img_info

    # 随机从img 中crop 一个patch图片 
    # 返回crop的patch, 左上点坐标、宽、高
    def get_random_patch_from_img(self, img_w, img_h, min_pixel=32):#8
        # 要求img_w >= 2*min_pixel img_h >= 2*min_pixel
        w, h = img_w, img_h
        min_w, max_w = min_pixel, w - min_pixel
        min_h, max_h = min_pixel, h - min_pixel
        sw, sh = np.random.randint(min_w, max_w + 1), np.random.randint(min_h, max_h + 1)
        x, y = np.random.randint(w - sw) if sw != w else 0, np.random.randint(h - sh) if sh != h else 0
        box = np.array([x, y, x + sw, y + sh], dtype=np.float32).reshape(1, 4)
        return box

    def prepare_train_img(self, idx):
        img_info = self.data_infos[idx]

        min_pixel=32
        random_patch = False
        img_h, img_w = img_info['height'], img_info['width']
        # 是否满足随机扣的条件
        if img_h >= 2 * min_pixel and img_w >= 2 * min_pixel:
            random_patch = True
        # 随机扣patch
        if random.random() > 0.9 and random_patch:
            box = self.get_random_patch_from_img(img_w, img_h, min_pixel = min_pixel)
            ann_info = dict()
            ann_info['bboxes'] = box
            ann_info['labels'] = np.array([0], dtype=np.int64)
            rf_img_info = dict()
            rf_img_info['box'] = box
            rf_img_info['file_name'] = img_info['filename']
        else:
            ann_info, cate = self.get_ann_info(idx) 
            rf_img_info = self.prepare_train_ref_img(cate)
        #print(rf_img_info)
        results = dict(img_info=img_info,
                    ann_info=ann_info,
                    rf_img_info=rf_img_info)
        # print(img_info)
        # print(ann_info)
        # print(rf_img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        # 当前图片信息
        img_info = self.data_infos[idx]
        # 当前图片指定类别信息
        ann_info, cate = self.get_ann_info(idx, self.cates[idx])
        rf_img_info = self.prepare_test_ref_img(cate)
        results = dict(img_info=img_info,
                       ann_info=ann_info,
                       rf_img_info=rf_img_info,
                       label=0)
        self.pre_pipeline(results)
        return self.pipeline(results)

    # 测试参考图片
    # 这里每个类别都固定取第一个query图片
    def prepare_test_ref_img(self, cate):
        rf_img_info = dict()
        filename, box = self.query_infos[cate][0]
        rf_img_info['box'] = box
        rf_img_info['file_name'] = filename
        return rf_img_info   

    # 模型评估
    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.8,
                 scale_ranges=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
        """

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')

        annotations = [self.get_ann_info(i, self.cates[i])[0] for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, _ = eval_map(
                    results,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    #dataset=self.CLASSES,
                    dataset=None,
                    logger=logger)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
      
        elif metric == 'recall':
            gt_bboxes = [ann['bboxes'] for ann in annotations]
            recalls = eval_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thrs):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        return eval_results    