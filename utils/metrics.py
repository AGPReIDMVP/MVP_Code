import torch
import numpy as np
import os
from utils.reranking import re_ranking


def euclidean_distance(qf, gf):
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(1, -2, qf, gf.t())
    return dist_mat.cpu().numpy()

def cosine_similarity(qf, gf):
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    """Evaluation with market1501 metric
        Key: for each query identity, its gallery images from the same camera view are discarded.
        """
    num_q, num_g = distmat.shape
    # distmat g
    #    q    1 3 2 4
    #         4 1 2 3
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    #  0 2 1 3
    #  1 2 3 0
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]  # select one row
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
        tmp_cmc = tmp_cmc / y
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):  # called once for each batch
        feat, pid, camid = output
        self.feats.append(feat.cpu())
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))




    def compute(self):  # called after each epoch
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test feature is normalized")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])

        g_camids = np.asarray(self.camids[self.num_query:])
        if self.reranking:
            print('=> Enter reranking')
            # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
            distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

        else:
            print('=> Computing DistMat with euclidean_distance')
            distmat = euclidean_distance(qf, gf)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)
        # print(q_pids)

        return cmc, mAP, distmat, self.pids, self.camids, qf, gf



# import torch
# import numpy as np
# import os
# from utils.reranking import re_ranking
# import matplotlib.pyplot as plt
# from PIL import Image

# def euclidean_distance(qf, gf):
#     m = qf.shape[0]
#     n = gf.shape[0]
#     dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
#                torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
#     dist_mat.addmm_(1, -2, qf, gf.t())
#     return dist_mat.cpu().numpy()

# def cosine_similarity(qf, gf):
#     epsilon = 0.00001
#     dist_mat = qf.mm(gf.t())
#     qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
#     gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
#     qg_normdot = qf_norm.mm(gf_norm.t())

#     dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
#     dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
#     dist_mat = np.arccos(dist_mat)
#     return dist_mat


# def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
#     """Evaluation with market1501 metric
#         Key: for each query identity, its gallery images from the same camera view are discarded.
#         """
#     num_q, num_g = distmat.shape
#     # distmat g
#     #    q    1 3 2 4
#     #         4 1 2 3
#     if num_g < max_rank:
#         max_rank = num_g
#         print("Note: number of gallery samples is quite small, got {}".format(num_g))
#     indices = np.argsort(distmat, axis=1)
#     #  0 2 1 3
#     #  1 2 3 0
#     matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
#     # compute cmc curve for each query
#     all_cmc = []
#     all_AP = []
#     num_valid_q = 0.  # number of valid query
#     for q_idx in range(num_q):
#         # get query pid and camid
#         q_pid = q_pids[q_idx]
#         q_camid = q_camids[q_idx]

#         # remove gallery samples that have the same pid and camid with query
#         order = indices[q_idx]  # select one row
#         remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
#         keep = np.invert(remove)

#         # compute cmc curve
#         # binary vector, positions with value 1 are correct matches
#         orig_cmc = matches[q_idx][keep]
#         if not np.any(orig_cmc):
#             # this condition is true when query identity does not appear in gallery
#             continue

#         cmc = orig_cmc.cumsum()
#         cmc[cmc > 1] = 1

#         all_cmc.append(cmc[:max_rank])
#         num_valid_q += 1.

#         # compute average precision
#         # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
#         num_rel = orig_cmc.sum()
#         tmp_cmc = orig_cmc.cumsum()
#         #tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
#         y = np.arange(1, tmp_cmc.shape[0] + 1) * 1.0
#         tmp_cmc = tmp_cmc / y
#         tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
#         AP = tmp_cmc.sum() / num_rel
#         all_AP.append(AP)

#     assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

#     all_cmc = np.asarray(all_cmc).astype(np.float32)
#     all_cmc = all_cmc.sum(0) / num_valid_q
#     mAP = np.mean(all_AP)

#     return all_cmc, mAP


# class R1_mAP_eval():
#     def __init__(self, num_query, max_rank=50, feat_norm=True, reranking=False):
#         super(R1_mAP_eval, self).__init__()
#         self.num_query = num_query
#         self.max_rank = max_rank
#         self.feat_norm = feat_norm
#         self.reranking = reranking

#     def reset(self):
#         self.feats = []
#         self.pids = []
#         self.camids = []
#         self.img = []

#     def update(self, output):  # called once for each batch
#         feat, pid, camid = output
#         self.feats.append(feat.cpu())
#         self.pids.extend(np.asarray(pid))
#         self.camids.extend(np.asarray(camid))
#         # self.img.extend(img.cpu())  # 将图像数据加入到列表中
         

#     # def visualize_retrieval_results(self,query_idx, distmat, qf, gf, q_pids,g_pids, q_img,g_img,num_results=5):
#     #     """
#     #     可视化检索结果
#     #     :param query_idx: 查询图像的索引
#     #     :param distmat: 查询图像与图库图像的距离矩阵
#     #     :param qf: 查询图像的特征
#     #     :param gf: 图库图像的特征
#     #     :param pids: 图像的ID列表
#     #     :param num_results: 显示前多少个匹配结果
#     #     """
#     #     # 获取查询图像的前 num_results 个最匹配的图库图像的索引
#     #     query_pid = q_pids[query_idx]
#     #     query_dist = distmat[query_idx]
#     #     sorted_idx = np.argsort(query_dist)  # 按照距离排序，距离小的排前面
#     #     print(f"g_img size: {g_img.size()}")
#     #     print(f"q_img size: {q_img.size()}")
#     #     # 准备图像显示
#     #     fig, axes = plt.subplots(1, num_results + 1, figsize=(15, 5))
#     #     query_img = (q_img[query_idx].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
#     #     axes[0].imshow(query_img)  # 查询图像
#     #     axes[0].set_title(f"Query: {query_pid}")
#     #     axes[0].axis('off')

#     #     for i in range(num_results):
#     #         match_idx = sorted_idx[i]
#     #         match_pid = g_pids[match_idx]
#     #         # 同样对匹配图像进行转换
#     #         match_img = (g_img[match_idx].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)  # 转换为 NumPy 格式并调整维度
#     #         axes[i + 1].imshow(match_img)  # 匹配图像
#     #         axes[i + 1].set_title(f"Rank {i+1}: {match_pid}")
#     #         axes[i + 1].axis('off')

#     #     plt.show()
#     def visualize_retrieval_results(self, query_idx, distmat, qf, gf, q_pids, g_pids, q_path,g_path,num_results=5):
#         """
#         可视化检索结果
#         :param query_idx: 查询图像的索引
#         :param distmat: 查询图像与图库图像的距离矩阵
#         :param qf: 查询图像的特征
#         :param gf: 图库图像的特征
#         :param q_pids: 查询图像的ID列表
#         :param g_pids: 图库图像的ID列表
#         :param q_img: 查询图像列表
#         :param g_img: 图库图像列表
#         :param num_results: 显示前多少个匹配结果
#         """
#         # 获取查询图像的前 num_results 个最匹配的图库图像的索引
#         query_pid = q_pids[query_idx]
#         query_dist = distmat[query_idx]
#         sorted_idx = np.argsort(query_dist)  # 按照距离排序，距离小的排前面
#         # print(f"g_img size: {g_img.size()}")  # 打印 g_img 的大小
#         # print(f"q_img size: {q_img.size()}")  # 打印 q_img 的大小
        
#         # 准备图像显示
#         fig, axes = plt.subplots(1, num_results + 1, figsize=(15, 5))

#         # 将查询图像从 GPU 转移到 CPU，并转换为 NumPy 格式
#         # query_img = (q_img[query_idx].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
#         print(q_path[query_idx])
#         # axes[0].imshow(query_img)  # 显示查询图像
#         axes[0].set_title(f"Query: {query_pid}")
#         axes[0].axis('off')

#         # 保存查询图像
#         # plt.imsave(f"query_image_{query_pid}.png", query_img)

#         for i in range(num_results):
#             match_idx = sorted_idx[i]
#             match_pid = g_pids[match_idx]
#             # 将匹配图像从 GPU 转移到 CPU，并转换为 NumPy 格式
#             # match_img = (g_img[match_idx].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
#             # axes[i + 1].imshow(match_img)  # 显示匹配图像
#             axes[i + 1].set_title(f"Rank {i+1}: {match_pid}")
#             axes[i + 1].axis('off')

#             # 保存每个匹配图像
#             # plt.imsave(f"match_image_{i+1}_{match_pid}.png", match_img)
#             print(f"Rank{i}",g_path[match_idx])

#         # 显示所有图像
#         plt.show()


#     def compute(self,img_path_list):  # called after each epoch
#         feats = torch.cat(self.feats, dim=0)
#         if self.feat_norm:
#             print("The test feature is normalized")
#             feats = torch.nn.functional.normalize(feats, dim=1, p=2)  # along channel
#         # query
#         qf = feats[:self.num_query]
#         q_pids = np.asarray(self.pids[:self.num_query])
#         q_camids = np.asarray(self.camids[:self.num_query])
#         # q_img = img[:self.num_query]
#         q_path = img_path_list[:self.num_query]
#         # print(q_img)
#         # gallery
#         gf = feats[self.num_query:]
#         g_pids = np.asarray(self.pids[self.num_query:])

#         g_camids = np.asarray(self.camids[self.num_query:])

#         # g_img = img[self.num_query:]
#         g_path = img_path_list[self.num_query:]
#         if self.reranking:
#             print('=> Enter reranking')
#             # distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
#             distmat = re_ranking(qf, gf, k1=50, k2=15, lambda_value=0.3)

#         else:
#             print('=> Computing DistMat with euclidean_distance')
#             distmat = euclidean_distance(qf, gf)
#         cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)

#         query_idx = 25
#         # self.visualize_retrieval_results(query_idx, distmat, qf, gf, q_pids,g_pids,q_img,g_img,q_path,g_path)
#         self.visualize_retrieval_results(query_idx, distmat, qf, gf, q_pids,g_pids,q_path,g_path)


#         return cmc, mAP, distmat, self.pids, self.camids, qf, gf


