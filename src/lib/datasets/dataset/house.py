from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pycocotools.coco import COCO
# from tools.cocoeval import COCOeval
from pycocotools.cocoeval import COCOeval
import glob
import copy
import numpy as np
import json
import os
import time
from collections import defaultdict

import torch.utils.data as data

class AnRuanTools:
  def __init__(self, data_dir, split):
    self.data_dir = data_dir
    self.cids = [i.split('.jpg')[0] for i in os.listdir(self.data_dir) if i.endswith('.jpg')]
    self.anns = dict()
    self.imgIds = []
    self.annIds = []
    self.dataset = {'images': [], 'categories': [{'id': 1, 'name': 'grass'}, {'id': 2, 'name': 'iron'}]}
    stats = defaultdict(int)
    for cid in self.cids:
      self.dataset['images'].append({'file_name':cid + '.jpg', 'height': 512, 'width': 512, 'id': cid})
      self.imgIds.append(cid)
      self.anns[cid] = []
      ann_file = os.path.join(self.data_dir, '{}.npy'.format(cid))
      data = np.load(ann_file, allow_pickle=True).item()
      bboxes = data['bboxes']

      for aid, box in enumerate(bboxes):
        x, y, w, h, cat_id = box
        self.annIds.append((cid, aid))
        self.anns[cid].append((int(cat_id), float(x), float(y), float(w), float(h), len(self.imgIds) - 1, len(self.annIds) - 1))
        stats[cat_id] += 1

    print('stats', stats)

  def getImgIds(self, catIds=[]):
    result = []
    catIds = set(catIds)
    for iid, cid in enumerate(self.imgIds):
      cats = set([ann[0] for ann in self.anns[cid]])
      if catIds<=cats:
        result.append(iid)
    return result

  def getCatIds(self):
    return [1,2]

  def loadImgs(self, ids):
    '''
    [{'file_name':..., ...}]
    '''
    assert type(ids)!=str, 'ids cannot be str'
    result = []
    for iid in ids:
      cid = self.imgIds[iid]
      result.append({'file_name': os.path.join(self.data_dir, cid + '.jpg')})
    return result

  def getAnnIds(self, imgIds, catIds=None):
    result = []
    # catIds = set(catIds)
    for id in imgIds:
      result.extend([i[-1] for i in self.anns[self.imgIds[id]]])
    return result

  def loadAnns(self, ids):
    result = []
    for aid in ids:
      ann = self.anns[self.annIds[aid][0]][self.annIds[aid][1]]
      result.append({'id': aid,
      'image_id':ann[5],
      'category_id':ann[0],
      'bbox':[ann[1], ann[2], ann[3], ann[4]],
      'iscrowd':0,
      'area': ann[3]*ann[4]})
    return result

  def loadRes(self, resFile):
    res = COCO()
    res.dataset['images'] = [img for img in self.dataset['images']]
    print('Loading and preparing results...')
    tic = time.time()
    anns = json.load(open(resFile))
    assert type(anns) == list, 'results in not an array of objects'
    annsImgIds = [ann['image_id'] for ann in anns]
    assert set(annsImgIds) == (set(annsImgIds) & set(self.getImgIds())), \
          'Results do not correspond to current coco set'
    res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
    for id, ann in enumerate(anns):
      bb = ann['bbox']
      x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
      if not 'segmentation' in ann:
          ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
      ann['area'] = bb[2]*bb[3]
      ann['id'] = id+1
      ann['iscrowd'] = 0
    res.dataset['annotations'] = anns
    res.createIndex()
    return res



class House(data.Dataset):
  num_classes = 3
  default_resolution = [512, 512]
  mean = np.array([0.522, 0.537, 0.558],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.180, 0.174, 0.171],
                   dtype=np.float32).reshape(1, 1, 3)

  def __init__(self, opt, split):
    super(House, self).__init__()
    self.data_dir = os.path.join(opt.data_dir, 'house_detection', 'preprocess', split)
    self.img_dir = os.path.join(self.data_dir, '{}'.format(split))
    self.max_objs = 128
    self.class_name = [
      '__background__', 'grass', 'iron']
    self._valid_ids = [1, 2, 3]
    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
    self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                      for v in range(1, self.num_classes + 1)]
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

    self.split = split
    self.opt = opt

    print('==> initializing house {} data.'.format(split))
    self.coco = AnRuanTools(self.data_dir, split)
    self.images = set()
    for id in self._valid_ids:
      self.images |= set(self.coco.getImgIds(catIds=[id]))
    self.images = list(self.images)
    self.num_samples = len(self.images)

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    # import pdb; pdb.set_trace()
    detections = []
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        category_id = self._valid_ids[cls_ind - 1]
        for bbox in all_bboxes[image_id][cls_ind]:
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          score = bbox[4]
          bbox_out  = list(map(self._to_float, bbox[0:4]))

          detection = {
              "image_id": image_id,
              "category_id": int(category_id),
              "bbox": bbox_out,
              "score": float("{:.2f}".format(score))
          }
          if len(bbox) > 5:
              extreme_points = list(map(self._to_float, bbox[5:13]))
              detection["extreme_points"] = extreme_points
          detections.append(detection)
    return detections

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    # json.dump(results,
    #             open('{}/results.json'.format(save_dir), 'w'))
    json.dump(self.convert_eval_format(results),
                open('{}/results.json'.format(save_dir), 'w'))

  def run_eval(self, results, save_dir):
    # result_json = os.path.join(save_dir, "results.json")
    # detections  = self.convert_eval_format(results)
    # json.dump(detections, open(result_json, "w"))
    self.save_results(results, save_dir)
    #print(save_dir)
    #raise ValueError()
    coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
    coco_eval = COCOeval(self.coco, coco_dets, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()