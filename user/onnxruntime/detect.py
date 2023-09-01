import math
import os
import cv2
import torch
import time
from nanodet.util import cfg, load_config
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
import numpy as np
import onnxruntime as ort
from nanodet.model.head.gfl_head import Integral
from nanodet.data.batch_process import stack_batch_img
from nanodet.util import (
    distance2bbox,
    visualization,
)
from nanodet.model.module.nms import multiclass_nms
from nanodet.data.transform.warp import warp_boxes


class Predictor(object):
    def __init__(self, config, onnx_path, device="cuda:0"):
        self.config = config
        self.num_classes = len(config.class_names)
        self.device = device
        self.pipeline = Pipeline(config.data.val.pipeline, config.data.val.keep_ratio)
        self.ort_session = ort.InferenceSession(onnx_path)
        self.reg_max = 7
        self.strides = [8, 16, 32, 64]
        self.distribution_project = Integral(self.reg_max)

    def get_single_level_center_point(
        self, featmap_size, stride, dtype, device, flatten=True
    ):
        """
        Generate pixel centers of a single stage feature map.
        :param featmap_size: height and width of the feature map
        :param stride: down sample stride of the feature map
        :param dtype: data type of the tensors
        :param device: device of the tensors
        :param flatten: flatten the x and y tensors
        :return: y and x of the center points
        """
        h, w = featmap_size
        x_range = (torch.arange(w, dtype=dtype, device=device) + 0.5) * stride
        y_range = (torch.arange(h, dtype=dtype, device=device) + 0.5) * stride
        y, x = torch.meshgrid(y_range, x_range)
        if flatten:
            y = y.flatten()
            x = x.flatten()
        return y, x

    def get_bboxes(self, cls_preds, reg_preds, img_metas):
        """Decode the outputs to bboxes.
        Args:
            cls_preds (Tensor): Shape (num_imgs, num_points, num_classes).
            reg_preds (Tensor): Shape (num_imgs, num_points, 4 * (regmax + 1)).
            img_metas (dict): Dict of image info.

        Returns:
            results_list (list[tuple]): List of detection bboxes and labels.
        """
        device = cls_preds.device
        b = cls_preds.shape[0]
        input_height, input_width = img_metas["img"].shape[2:]
        input_shape = (input_height, input_width)

        featmap_sizes = [
            (math.ceil(input_height / stride), math.ceil(input_width) / stride)
            for stride in self.strides
        ]
        # get grid cells of one image
        mlvl_center_priors = []
        for i, stride in enumerate(self.strides):
            y, x = self.get_single_level_center_point(
                featmap_sizes[i], stride, torch.float32, device
            )
            strides = x.new_full((x.shape[0],), stride)
            proiors = torch.stack([x, y, strides, strides], dim=-1)
            mlvl_center_priors.append(proiors.unsqueeze(0).repeat(b, 1, 1))

        center_priors = torch.cat(mlvl_center_priors, dim=1)
        dis_preds = self.distribution_project(reg_preds) * center_priors[..., 2, None]
        bboxes = distance2bbox(center_priors[..., :2], dis_preds, max_shape=input_shape)
        scores = cls_preds.sigmoid()
        result_list = []
        for i in range(b):
            # add a dummy background class at the end of all labels
            # same with mmdetection2.0
            score, bbox = scores[i], bboxes[i]
            padding = score.new_zeros(score.shape[0], 1)
            score = torch.cat([score, padding], dim=1)
            results = multiclass_nms(
                bbox,
                score,
                score_thr=0.05,
                nms_cfg=dict(type="nms", iou_threshold=0.6),
                max_num=100,
            )
            result_list.append(results)
        return result_list

    def post_process(self, preds, meta):
        cls_scores, bbox_preds = preds.split(
            [self.num_classes, 4 * (self.reg_max + 1)], dim=-1
        )
        result_list = self.get_bboxes(cls_scores, bbox_preds, meta)
        det_results = {}
        warp_matrixes = (
            meta["warp_matrix"]
            if isinstance(meta["warp_matrix"], list)
            else meta["warp_matrix"]
        )
        img_heights = (
            meta["img_info"]["height"].cpu().numpy()
            if isinstance(meta["img_info"]["height"], torch.Tensor)
            else meta["img_info"]["height"]
        )
        img_widths = (
            meta["img_info"]["width"].cpu().numpy()
            if isinstance(meta["img_info"]["width"], torch.Tensor)
            else meta["img_info"]["width"]
        )
        img_ids = (
            meta["img_info"]["id"].cpu().numpy()
            if isinstance(meta["img_info"]["id"], torch.Tensor)
            else meta["img_info"]["id"]
        )

        for result, img_width, img_height, img_id, warp_matrix in zip(
                result_list, img_widths, img_heights, img_ids, warp_matrixes
        ):
            det_result = {}
            det_bboxes, det_labels = result
            det_bboxes = det_bboxes.detach().cpu().numpy()
            det_bboxes[:, :4] = warp_boxes(
                det_bboxes[:, :4], np.linalg.inv(warp_matrix), img_width, img_height
            )
            classes = det_labels.detach().cpu().numpy()
            for i in range(self.num_classes):
                inds = classes == i
                det_result[i] = np.concatenate(
                    [
                        det_bboxes[inds, :4].astype(np.float32),
                        det_bboxes[inds, 4:5].astype(np.float32),
                    ],
                    axis=1,
                ).tolist()
            det_results[img_id] = det_result
        return det_results

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline(meta, self.config.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)
        # debug
        print(f"img_shape:{meta['img'].shape}")
        # onnx 格式
        results = self.ort_session.run(
            ['output'],
            {'data': meta["img"].cpu().numpy()},
        )
        results = torch.from_numpy(results[0]).to(self.device)
        results = self.post_process(preds=results, meta=meta)
        return meta, results

    def visualize(self, dets, meta, class_names, score_thres, wait=0):
        time1 = time.time()
        result_img = visualization.overlay_bbox_cv(meta["raw_img"][0], dets, self.config.class_names, score_thres)
        cv2.imshow("det", result_img)
        print("viz time: {:.3f}s".format(time.time() - time1))
        return result_img


def get_img_meta(img, config):
    img_info = {"id": 0}
    if isinstance(img, str):
        img_info["file_name"] = os.path.basename(img)
        img = cv2.imread(img)
    else:
        img_info["file_name"] = None

    height, width = img.shape[:2]
    img_info["height"] = height
    img_info["width"] = width
    meta = dict(img_info=img_info, raw_img=img, img=img)
    meta = Pipeline(meta, config.data.val.input_size)
    meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to("cuda:0")
    meta = naive_collate([meta])
    meta["img"] = stack_batch_img(meta["img"], divisible=32)


if __name__ == '__main__':
    # 导入模型
    image_path = './images/image1.jpg'
    config_path = '../config/nanodet_custom_xml_dataset.yml'
    onnx_path = '../models/nanodet.onnx'
    load_config(cfg, config_path)
    predictor = Predictor(cfg, onnx_path, device="cuda:0")
    meta, res = predictor.inference(image_path)
    result_image = predictor.visualize(res[0], meta, cfg.class_names, 0.60)
    cv2.waitKey(0)
    # debug
    print('END')