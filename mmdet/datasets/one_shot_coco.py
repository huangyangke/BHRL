import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict

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

@DATASETS.register_module()
class OneShotCocoDataset(CustomDataset):

    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

    def __init__(self,
                 ann_file,
                 pipeline,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 proposal_file=None,
                 test_mode=False,
                 test_seen_classes=False,
                 dataset_split=4,
                 ref_ann_file=None,
                 position=0):
        if dataset_split == 4:
            dataset_split = 0
        self.split = dataset_split
        self.test_seen_classes = test_seen_classes 
        self.ref_ann_file = ref_ann_file #ref denotes the query in the paper. 
        self.position = position
        classes = None 
        super(OneShotCocoDataset,
              self).__init__(ann_file, pipeline, classes, data_root, img_prefix,
                             seg_prefix, proposal_file, test_mode) 
    # 加载标注文件
    def load_annotations(self, ann_file): 
        self.coco = COCO(ann_file)
        # 获得类别id 1 - 90 的数字 总共80个
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        # 获得类别id到label的映射 
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.split_cats()
        # 每张图片的ids
        self.img_ids = self.coco.get_img_ids()
        """
        img_infos中的元素
        {'license': 3, 'file_name': '000000391895.jpg', 
        'coco_url': 'http://images.cocoda...391895.jpg', 
        'height': 360, 'width': 640, 
        'date_captured': '2013-11-14 11:18:45',
        'flickr_url': 'http://farm9.staticf...8349_z.jpg', 
        'id': 391895, 'filename': '000000391895.jpg'}
        """
        img_infos, img_cates = self.generate_infos()
        self.cates = img_cates#训练时为None
        # self.ref_coco[5] = [{'category_id': 5, 'id': 157224}]
        self.ref_coco = pickle.load(open(self.ref_ann_file, 'rb'), encoding='utf-8')
        return img_infos

    def get_ann_info(self, idx, cate=None): 
        # 图片id
        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        # 当前图片对应的bbox标注
        ann_info = self.coco.load_anns(ann_ids) 
        # 如果没有指定类别 随机抽取一个类
        if cate is None: 
            cate = self.random_cate(ann_info) 
        return self._parse_ann_info(self.data_infos[idx], ann_info, cate)

    # 按照类别分成5份
    def split_cats(self): 
        self.train_cat = []#训练类别
        self.test_cat = []#测试类别
        for i in range(len(self.cat_ids)):
            if (i + 1) % 4 == self.split:
                self.test_cat.append(self.cat_ids[i])
            else:
                self.train_cat.append(self.cat_ids[i])
        if self.test_seen_classes:
            self.test_cat = self.train_cat

    def generate_infos(self):
        img_infos = []
        img_cates = []
        for i in self.img_ids:
            if not self.test_mode:
                img_infos, img_cates = self.generate_train(i, img_infos)
            else:
                img_infos, img_cates = self.generate_test(i, img_infos, img_cates)
        return img_infos, img_cates

    # img_infos存储图片信息 其中图片可能重复，因为是以box为单位，一张图片有多少个box图片就重复多少次
    def generate_train(self, i, img_infos): 
        info = self.coco.load_imgs([i])[0]# 当前图片信息
        info['filename'] = info['file_name']
        img_anns_ids = self.coco.get_ann_ids(img_ids=[i])
        img_anns = self.coco.load_anns(img_anns_ids)# 当前图片的box标注
        for img_ann in img_anns:
            if img_ann['category_id'] in self.train_cat:
                img_infos.append(info)
                break 
        return img_infos, None

    def generate_test(self, i, img_infos, img_cates): 
        info = self.coco.loadImgs([i])[0]
        info['filename'] = info['file_name']
        img_anns_ids = self.coco.getAnnIds(imgIds=i)
        img_anns = self.coco.loadAnns(img_anns_ids)
        img_cats = list()
        for img_ann in img_anns:
            if img_ann['category_id'] in img_cats:
                continue
            elif img_ann['category_id'] in self.test_cat:
                img_cats.append(img_ann['category_id']) 
                img_infos.append(info)
                img_cates.append(img_ann['category_id'])
            else:
                continue
        return img_infos, img_cates

    # 过滤太小的图片
    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco.cat_img_map[class_id])
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    # 随机抽取一个类别
    def random_cate(self, ann_info): 
        index = np.random.randint(len(ann_info))
        cate = ann_info[index]['category_id']

        if not self.test_mode:
            cates = self.train_cat
        else:
            cates = self.test_cat

        while cate not in cates:
            index = np.random.randint(len(ann_info))
            cate = ann_info[index]['category_id']
        return cate

    # 合并图像和bbox
    # 过滤不合理的bbox
    def _parse_ann_info(self, img_info, ann_info, cate_select): 
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info): 
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']#左上 宽高
            # 过滤不合理的box
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            #左上宽高=>左上 右下
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            elif ann['category_id'] == cate_select: 
                gt_bboxes.append(bbox)
                gt_labels.append(0)
                gt_masks_ann.append(ann['segmentation'])
            else:
                continue

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(bboxes=gt_bboxes,
                labels=gt_labels,
                bboxes_ignore=gt_bboxes_ignore,
                masks=gt_masks_ann,
                seg_map=seg_map,
                cate=cate_select)

        return ann

    def prepare_train_ref_img(self, idx, cate):
        rf_img_info = dict()
        img_id = self.data_infos[idx]['id']
        # 固定随机种子
        np.random.seed(img_id) 
        while True:
            if isinstance(self.ref_coco, dict):
                # 随机选该类的一个box 由于前面固定了随机种子 所以每次取的是同一个
                rand_index = np.random.randint(len(self.ref_coco[cate]))
                rf_anns = self.coco.loadAnns(self.ref_coco[cate][rand_index]['id'])[0]
                rf_img_info['ann'] = rf_anns
                rf_img_info['file_name'] = self.coco.loadImgs(rf_anns['image_id'])[0]['file_name']
                rf_img_info['img_info'] = self.coco.loadImgs(rf_anns['image_id'])[0]
                if len(self.ref_coco[cate]) == 1:
                    break
                if rf_anns['image_id'] == img_id or rf_anns['iscrowd']:
                    continue
                else:
                    break
        return rf_img_info

    def prepare_test_ref_img(self, idx, cate):
        rf_img_info = dict()
        img_id = self.data_infos[idx]['id']
        all_ref = self.ref_coco[cate]
        # 随机打乱query图片索引
        random.seed(img_id)
        l = list(range(len(all_ref)))
        random.shuffle(l)
        
        # 测试query图片的索引
        position = l[self.position % len(l)]
        ref = all_ref[position]

        rf_anns = self.coco.loadAnns(ref['id'])[0]
        rf_img_info['ann'] = rf_anns
        rf_img_info['file_name'] = self.coco.loadImgs(rf_anns['image_id'])[0]['file_name']
        rf_img_info['img_info'] = self.coco.loadImgs(rf_anns['image_id'])[0]
        return rf_img_info

    def prepare_train_img(self, idx):
        """
        {'file_name': '000000581839.jpg',
        'height': 640, 'width': 466,'id':581839, 
        'filename':'000000581839.jpg'}
        """
        img_info = self.data_infos[idx]
        """
        'bboxes': array([[175.69, 354....e=float32), 'labels': array([0, 0]), 
        'bboxes_ignore': array([], shape=(0, ...e=float32), 'masks': [[...], [...]], 
        'seg_map': '000000581839.png',
        'cate':43
        """
        ann_info = self.get_ann_info(idx) 
        """
        参考图片相关信息
        ann
        file_name
        img_info
        """
        rf_img_info = self.prepare_train_ref_img(idx, ann_info['cate'])
        results = dict(img_info=img_info,
                    ann_info=ann_info,
                    rf_img_info=rf_img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx, self.cates[idx])
        rf_img_info = self.prepare_test_ref_img(idx, self.cates[idx])
        results = dict(img_info=img_info,
                       ann_info=ann_info,
                       rf_img_info=rf_img_info,
                       label=self.cat2label[self.cates[idx]])
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        return self.pipeline(results)