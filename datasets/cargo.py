# encoding: utf-8

import os
import os.path as osp
import glob
import re
from .bases import BaseImageDataset
from collections import defaultdict
import pickle
# from fastreid.data.datasets import DATASET_REGISTRY
# from fastreid.data.datasets.bases import ImageDataset

import pdb

__all__ = ['CARGO', ]


# @DATASET_REGISTRY.register()
class CARGO(BaseImageDataset):
    dataset_dir = "CARGO"
    dataset_name = 'cargo'

    def __init__(self, root='',verbose=True,pid_begin = 0, **kwargs):
        super(CARGO, self).__init__()
        self.root = root

        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        self._check_before_run()
        self.pid_begin = pid_begin
        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Cargo loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))
    def process_dir(self, dir_path, relabel = False):
        img_paths = []
        pid_container = set()
        for cam_index in range(13):
            img_paths = img_paths + glob.glob(osp.join(dir_path, f'Cam{cam_index + 1}', '*.jpg'))

        dataset = []
        for img_path in img_paths:
            pid = int(img_path.split('/')[-1].split('_')[2])
            pid_container.add(pid)
            # camid = int(img_path.split('/')[-1].split('_')[0][3:])
            # viewid = 'Aerial' if camid <= 5 else 'Ground'

            # camid -= 1  # index starts from 0


        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        for img_path in img_paths:
            pid = int(img_path.split('/')[-1].split('_')[2])
            camid = int(img_path.split('/')[-1].split('_')[0][3:])
            # viewid = 'Aerial' if camid <= 5 else 'Ground'
            viewid = 1 if camid <= 5 else 0 #1表示天空，0表示地面
            # camid = 1 if camid <= 5 else 2#ag数据集时候采用这个
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            # print(pid)
            # dataset.append((img_path, self.pid_begin + pid, camid, 0))#后期可以修改，让0表示天空，1表示地面这种
            dataset.append((img_path, self.pid_begin + pid, camid, viewid))#后期可以修改，让0表示天空，1表示地面这种
        return dataset

        #     if is_train:
        #         pid = self.dataset_name + "_" + str(pid)
        #         camid = self.dataset_name + "_" + str(camid)
        #     data.append((img_path, pid, camid, viewid))
        # return data


# @DATASET_REGISTRY.register()
# class CARGO_AA(ImageDataset):
#     dataset_dir = "CARGO"
#     dataset_name = 'cargo_aa'
#
#     def __init__(self, root='datasets', **kwargs):
#         self.root = root
#         self.data_dir = 'XXX'
#
#         self.train_dir = osp.join(self.data_dir, 'train')
#         self.query_dir = osp.join(self.data_dir, 'query')
#         self.gallery_dir = osp.join(self.data_dir, 'gallery')
#
#         train = self.process_dir(self.train_dir, is_train=True)
#         query = self.process_dir(self.query_dir, is_train=False)
#         gallery = self.process_dir(self.gallery_dir, is_train=False)
#
#         super().__init__(train, query, gallery, **kwargs)
#
#     def process_dir(self, dir_path, is_train=True):
#         img_paths = []
#         for cam_index in range(13):
#             img_paths = img_paths + glob.glob(osp.join(dir_path, f'Cam{cam_index + 1}', '*.jpg'))
#
#         data = []
#         for img_path in img_paths:
#             pid = int(img_path.split('/')[-1].split('_')[2])
#             camid = int(img_path.split('/')[-1].split('_')[0][3:])
#             viewid = 'Aerial' if camid <= 5 else 'Ground'
#             camid -= 1  # index starts from 0
#             if viewid == 'Ground':
#                 continue
#
#             if is_train:
#                 pid = self.dataset_name + "_" + str(pid)
#                 camid = self.dataset_name + "_" + str(camid)
#             data.append((img_path, pid, camid, viewid))
#         return data
#
#
# @DATASET_REGISTRY.register()
# class CARGO_GG(ImageDataset):
#     dataset_dir = "CARGO"
#     dataset_name = 'cargo_gg'
#
#     def __init__(self, root='datasets', **kwargs):
#         self.root = root
#         self.data_dir = 'XXX'
#
#         self.train_dir = osp.join(self.data_dir, 'train')
#         self.query_dir = osp.join(self.data_dir, 'query')
#         self.gallery_dir = osp.join(self.data_dir, 'gallery')
#
#         train = self.process_dir(self.train_dir, is_train=True)
#         query = self.process_dir(self.query_dir, is_train=False)
#         gallery = self.process_dir(self.gallery_dir, is_train=False)
#
#         super().__init__(train, query, gallery, **kwargs)
#
#     def process_dir(self, dir_path, is_train=True):
#         img_paths = []
#         for cam_index in range(13):
#             img_paths = img_paths + glob.glob(osp.join(dir_path, f'Cam{cam_index + 1}', '*.jpg'))
#
#         data = []
#         for img_path in img_paths:
#             pid = int(img_path.split('/')[-1].split('_')[2])
#             camid = int(img_path.split('/')[-1].split('_')[0][3:])
#             viewid = 'Aerial' if camid <= 5 else 'Ground'
#             if viewid == 'Aerial':
#                 continue
#             camid -= 1  # index starts from 0
#
#             if is_train:
#                 pid = self.dataset_name + "_" + str(pid)
#                 camid = self.dataset_name + "_" + str(camid)
#             data.append((img_path, pid, camid, viewid))
#         return data
#
#
# @DATASET_REGISTRY.register()
# class CARGO_AG(ImageDataset):
#     dataset_dir = "CARGO"
#     dataset_name = 'cargo_ag'
#
#     def __init__(self, root='datasets', **kwargs):
#         self.root = root
#         self.data_dir = 'XXX'
#
#         self.train_dir = osp.join(self.data_dir, 'train')
#         self.query_dir = osp.join(self.data_dir, 'query')
#         self.gallery_dir = osp.join(self.data_dir, 'gallery')
#
#         train = self.process_dir(self.train_dir, is_train=True)
#         query = self.process_dir(self.query_dir, is_train=False)
#         gallery = self.process_dir(self.gallery_dir, is_train=False)
#
#         super().__init__(train, query, gallery, **kwargs)
#
#     def process_dir(self, dir_path, is_train=True):
#         img_paths = []
#         for cam_index in range(13):
#             img_paths = img_paths + glob.glob(osp.join(dir_path, f'Cam{cam_index + 1}', '*.jpg'))
#
#         data = []
#         for img_path in img_paths:
#             pid = int(img_path.split('/')[-1].split('_')[2])
#             camid = int(img_path.split('/')[-1].split('_')[0][3:])
#             viewid = 'Aerial' if camid <= 5 else 'Ground'
#             camid = 1 if camid <= 5 else 2
#             camid -= 1  # index starts from 0
#
#             if is_train:
#                 pid = self.dataset_name + "_" + str(pid)
#                 camid = self.dataset_name + "_" + str(camid)
#             data.append((img_path, pid, camid, viewid))
#         return data
