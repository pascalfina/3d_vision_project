# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# MASt3R Sparse Global Alignement
# --------------------------------------------------------
from tqdm import tqdm
import roma
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from collections import namedtuple
from functools import lru_cache
from scipy import sparse as sp
import copy
import scipy.cluster.hierarchy as sch
import gc
import time
import pickle
import gzip

from mast3r.utils.misc import mkdir_for, hash_md5
from mast3r.cloud_opt.utils.losses import gamma_loss
from mast3r.cloud_opt.utils.schedules import linear_schedule, cosine_schedule
from mast3r.fast_nn import fast_reciprocal_NNs, merge_corres

import mast3r.utils.path_to_dust3r  # noqa
from dust3r.utils.geometry import inv, geotrf  # noqa
from dust3r.utils.device import to_cpu, to_numpy, todevice  # noqa
from dust3r.utils.image import load_images as dust3r_load_images  # noqa
from dust3r.post_process import estimate_focal_knowing_depth  # noqa
from dust3r.optim_factory import adjust_learning_rate_by_lr  # noqa
from dust3r.cloud_opt.base_opt import clean_pointcloud
from dust3r.viz import SceneViz


def _log_host_ram(label):
    try:
        import psutil
        proc = psutil.Process()
        rss_gb = proc.memory_info().rss / (1024 ** 3)
        avail_gb = psutil.virtual_memory().available / (1024 ** 3)
        print(f"[MASt3R-SfM] {label} host_rss={rss_gb:.2f}G avail={avail_gb:.2f}G", flush=True)
    except Exception:
        pass


def _stage_log(label):
    print(f"[MASt3R-SfM] {label}", flush=True)
    _log_host_ram(label)


class SparseGA():
    def __init__(self, img_paths, pairs_in, res_fine, anchors, canonical_paths=None):
        def fetch_img(im):
            def torgb(x): return (x[0].permute(1, 2, 0).numpy() * .5 + .5).clip(min=0., max=1.)
            for im1, im2 in pairs_in:
                if im1['instance'] == im:
                    entry = _materialize_img_entry(im1)
                    return torgb(entry['img'])
                if im2['instance'] == im:
                    entry = _materialize_img_entry(im2)
                    return torgb(entry['img'])
        self.canonical_paths = canonical_paths
        self.img_paths = img_paths
        self.imgs = [fetch_img(img) for img in img_paths]
        self.intrinsics = res_fine['intrinsics']
        self.cam2w = res_fine['cam2w']
        self.depthmaps = res_fine['depthmaps']
        self.pts3d = res_fine['pts3d']
        self.pts3d_colors = []
        self.working_device = self.cam2w.device
        for i in range(len(self.imgs)):
            im = self.imgs[i]
            x, y = anchors[i][0][..., :2].detach().cpu().numpy().T
            self.pts3d_colors.append(im[y, x])
            assert self.pts3d_colors[-1].shape == self.pts3d[i].shape
        self.n_imgs = len(self.imgs)

    def get_focals(self):
        return torch.tensor([ff[0, 0] for ff in self.intrinsics]).to(self.working_device)

    def get_principal_points(self):
        return torch.stack([ff[:2, -1] for ff in self.intrinsics]).to(self.working_device)

    def get_im_poses(self):
        return self.cam2w

    def get_sparse_pts3d(self):
        return self.pts3d

    def get_dense_pts3d(self, clean_depth=True, subsample=8):
        assert self.canonical_paths, 'cache_path is required for dense 3d points'
        device = self.cam2w.device
        confs = []
        base_focals = []
        anchors = {}
        for i, canon_path in enumerate(self.canonical_paths):
            (canon, canon2, conf), focal = _load_cache(canon_path, map_location=device)
            confs.append(conf)
            base_focals.append(focal)

            H, W = conf.shape
            pixels = torch.from_numpy(np.mgrid[:W, :H].T.reshape(-1, 2)).float().to(device)
            idxs, offsets = anchor_depth_offsets(canon2, {i: (pixels, None)}, subsample=subsample)
            anchors[i] = (pixels, idxs[i], offsets[i])

        # densify sparse depthmaps
        pts3d, depthmaps = make_pts3d(anchors, self.intrinsics, self.cam2w, [
                                      d.ravel() for d in self.depthmaps], base_focals=base_focals, ret_depth=True)

        if clean_depth:
            confs = clean_pointcloud(confs, self.intrinsics, inv(self.cam2w), depthmaps, pts3d)

        return pts3d, depthmaps, confs

    def get_pts3d_colors(self):
        return self.pts3d_colors

    def get_depthmaps(self):
        return self.depthmaps

    def get_masks(self):
        return [slice(None, None) for _ in range(len(self.imgs))]

    def show(self, show_cams=True):
        pts3d, _, confs = self.get_dense_pts3d()
        show_reconstruction(self.imgs, self.intrinsics if show_cams else None, self.cam2w,
                            [p.clip(min=-50, max=50) for p in pts3d],
                            masks=[c > 1 for c in confs])


def convert_dust3r_pairs_naming(imgs, pairs_in):
    for pair_id in range(len(pairs_in)):
        for i in range(2):
            pairs_in[pair_id][i]['instance'] = imgs[pairs_in[pair_id][i]['idx']]
    return pairs_in


def _materialize_img_entry(img):
    tensor = img.get('img')
    if tensor is not None:
        return img
    source_path = img.get('source_path')
    load_size = int(img.get('load_size', 512))
    if source_path is None:
        raise KeyError("Missing 'source_path' for lazy MASt3R image loading")
    loaded = dust3r_load_images([source_path], size=load_size, verbose=False)[0]
    return {
        **img,
        'img': loaded['img'],
        'true_shape': loaded['true_shape'],
    }


def sparse_global_alignment(imgs, pairs_in, cache_path, model, subsample=8, desc_conf='desc_conf',
                            kinematic_mode='hclust-ward', device='cuda', dtype=torch.float32, shared_intrinsics=False, **kw):
    """ Sparse alignment with MASt3R
        imgs: list of image paths
        cache_path: path where to dump temporary files (str)

        lr1, niter1: learning rate and #iterations for coarse global alignment (3D matching)
        lr2, niter2: learning rate and #iterations for refinement (2D reproj error)

        lora_depth: smart dimensionality reduction with depthmaps
    """
    kw = dict(kw)
    loss_dust3r_w = float(kw.get("loss_dust3r_w", 0.01))
    keep_preds_21 = loss_dust3r_w > 0.0
    streaming_condense = bool(kw.pop("streaming_condense", True))

    # Convert pair naming convention from dust3r to mast3r
    pairs_in = convert_dust3r_pairs_naming(imgs, pairs_in)
    # forward pass
    pairs, cache_path = forward_mast3r(pairs_in, model,
                                       cache_path=cache_path, subsample=subsample,
                                       desc_conf=desc_conf, device=device)
    # The MASt3R network is only needed for the forward cache generation.
    # Free it before the canonical/BA stages, which can be the peak-memory
    # phase on 16 GB GPUs.
    del model
    if str(device).startswith('cuda'):
        torch.cuda.empty_cache()

    if streaming_condense:
        # Extract canonical pointmaps and directly condense them into the final BA
        # structures instead of materializing the full canonical_views dict for all
        # images at once. This keeps host RAM well below the old peak.
        pairwise_scores, canonical_paths, imsizes, pps, base_focals, core_depth, anchors, pair_slices, preds_21 = \
            prepare_condensed_data(
                imgs,
                pairs,
                subsample,
                cache_path=cache_path,
                mode='avg-angle',
                device=device,
                keep_preds_21=keep_preds_21,
                dtype=dtype,
            )
        gc.collect()
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()
        _stage_log("after_prepare_condensed_data")
    else:
        _stage_log("using_legacy_condense_path")
        tmp_pairs, pairwise_scores, canonical_views, canonical_paths, preds_21 = \
            prepare_canonical_data(
                imgs,
                pairs,
                subsample,
                cache_path=cache_path,
                mode='avg-angle',
                device=device,
                keep_preds_21=keep_preds_21,
            )
        gc.collect()
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()
        _stage_log("after_prepare_canonical_data")
        imsizes, pps, base_focals, core_depth, anchors, pair_slices, preds_21 = \
            condense_data(imgs, tmp_pairs, canonical_views, preds_21, dtype)
        gc.collect()
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()
        _stage_log("after_condense_data")

    # Build kinematic chain
    if kinematic_mode == 'mst':
        # compute minimal spanning tree
        mst = compute_min_spanning_tree(pairwise_scores)

    elif kinematic_mode.startswith('hclust'):
        mode, linkage = kinematic_mode.split('-')

        # Convert the affinity matrix to a distance matrix (if needed)
        n_patches = (imsizes // subsample).prod(dim=1)
        max_n_corres = 3 * torch.minimum(n_patches[:,None], n_patches[None,:])
        pws = (pairwise_scores.clone() / max_n_corres).clip(max=1)
        pws.fill_diagonal_(1)
        pws = to_numpy(pws)
        distance_matrix = np.where(pws, 1 - pws, 2)

        # Compute the condensed distance matrix
        condensed_distance_matrix = sch.distance.squareform(distance_matrix)

        # Perform hierarchical clustering using the linkage method
        Z = sch.linkage(condensed_distance_matrix, method=linkage)
        # dendrogram = sch.dendrogram(Z)

        tree = np.eye(len(imgs))
        new_to_old_nodes = {i:i for i in range(len(imgs))}
        for i, (a, b) in enumerate(Z[:,:2].astype(int)):
            # given two nodes to be merged, we choose which one is the best representant
            a = new_to_old_nodes[a]
            b = new_to_old_nodes[b]
            tree[a,b] = tree[b,a] = 1
            best = a if pws[a].sum() > pws[b].sum() else b
            new_to_old_nodes[len(imgs)+i] = best
            pws[best] = np.maximum(pws[a], pws[b]) # update the node

        pairwise_scores = torch.from_numpy(tree) # this output just gives 1s for connected edges and zeros for other, i.e. no scores or priority
        mst = compute_min_spanning_tree(pairwise_scores)

    else:
        raise ValueError(f'bad {kinematic_mode=}')

    # remove all edges not in the spanning tree?
    # min_spanning_tree = {(imgs[i],imgs[j]) for i,j in mst[1]}
    # tmp_pairs = {(a,b):v for (a,b),v in tmp_pairs.items() if {(a,b),(b,a)} & min_spanning_tree}

    imgs, res_coarse, res_fine = sparse_scene_optimizer(
        imgs, subsample, imsizes, pps, base_focals, core_depth, anchors, pair_slices, preds_21, canonical_paths, mst,
        shared_intrinsics=shared_intrinsics, cache_path=cache_path, device=device, dtype=dtype, **kw)

    return SparseGA(imgs, pairs_in, res_fine or res_coarse, anchors, canonical_paths)


def sparse_scene_optimizer(imgs, subsample, imsizes, pps, base_focals, core_depth, anchors, pair_slices,
                           preds_21, canonical_paths, mst, cache_path,
                           lr1=0.07, niter1=300, loss1=gamma_loss(1.5),
                           lr2=0.01, niter2=300, loss2=gamma_loss(0.5),
                           lossd=gamma_loss(1.1),
                           opt_pp=True, opt_depth=True,
                           schedule=cosine_schedule, depth_mode='add', exp_depth=False,
                           lora_depth=False,  # dict(k=96, gamma=15, min_norm=.5),
                           shared_intrinsics=False,
                           init={}, device='cuda', dtype=torch.float32,
                           matching_conf_thr=5., loss_dust3r_w=0.01,
                           loss3d_backward_chunk_pairs=64,
                           loss2d_backward_chunk_pairs=64,
                           loss_dust3r_backward_chunk_pairs=32,
                           fast_loss_path=False,
                           verbose=True, dbg=()):
    _stage_log("enter_sparse_scene_optimizer")
    print(
        "[MASt3R-SfM] optimizer_flags "
        f"fast_loss_path={bool(fast_loss_path)} "
        f"loss3d_chunks={loss3d_backward_chunk_pairs} "
        f"loss2d_chunks={loss2d_backward_chunk_pairs} "
        f"dust3r_chunks={loss_dust3r_backward_chunk_pairs}",
        flush=True,
    )
    init = copy.deepcopy(init)
    # extrinsic parameters
    vec0001 = torch.tensor((0, 0, 0, 1), dtype=dtype, device=device)
    quats = [nn.Parameter(vec0001.clone()) for _ in range(len(imgs))]
    trans = [nn.Parameter(torch.zeros(3, device=device, dtype=dtype)) for _ in range(len(imgs))]

    # initialize
    ones = torch.ones((len(imgs), 1), device=device, dtype=dtype)
    median_depths = torch.ones(len(imgs), device=device, dtype=dtype)
    for img in imgs:
        idx = imgs.index(img)
        init_values = init.setdefault(img, {})
        if verbose and init_values:
            print(f' >> initializing img=...{img[-25:]} [{idx}] for {set(init_values)}')

        K = init_values.get('intrinsics')
        if K is not None:
            K = K.detach()
            focal = K[:2, :2].diag().mean()
            pp = K[:2, 2]
            base_focals[idx] = focal
            pps[idx] = pp
        pps[idx] /= imsizes[idx]  # default principal_point would be (0.5, 0.5)

        depth = init_values.get('depthmap')
        if depth is not None:
            core_depth[idx] = depth.detach()

        median_depths[idx] = med_depth = core_depth[idx].median()
        core_depth[idx] /= med_depth

        cam2w = init_values.get('cam2w')
        if cam2w is not None:
            rot = cam2w[:3, :3].detach()
            cam_center = cam2w[:3, 3].detach()
            quats[idx].data[:] = roma.rotmat_to_unitquat(rot)
            trans_offset = med_depth * torch.cat((imsizes[idx] / base_focals[idx] * (0.5 - pps[idx]), ones[:1, 0]))
            trans[idx].data[:] = cam_center + rot @ trans_offset
            del rot
            assert False, 'inverse kinematic chain not yet implemented'

    # intrinsics parameters
    if shared_intrinsics:
        # Optimize a single set of intrinsics for all cameras. Use averages as init.
        confs = torch.stack([_load_cache(pth)[0][2].mean() for pth in canonical_paths]).to(pps)
        weighting = confs / confs.sum()
        pp = nn.Parameter((weighting @ pps).to(dtype))
        pps = [pp for _ in range(len(imgs))]
        focal_m = weighting @ base_focals
        log_focal = nn.Parameter(focal_m.view(1).log().to(dtype))
        log_focals = [log_focal for _ in range(len(imgs))]
    else:
        pps = [nn.Parameter(pp.to(dtype)) for pp in pps]
        log_focals = [nn.Parameter(f.view(1).log().to(dtype)) for f in base_focals]

    diags = imsizes.float().norm(dim=1)
    min_focals = 0.25 * diags  # diag = 1.2~1.4*max(W,H) => beta >= 1/(2*1.2*tan(fov/2)) ~= 0.26
    max_focals = 10 * diags

    assert len(mst[1]) == len(pps) - 1

    def make_K_cam_depth(log_focals, pps, trans, quats, log_sizes, core_depth):
        # make intrinsics
        focals = torch.cat(log_focals).exp().clip(min=min_focals, max=max_focals)
        pps = torch.stack(pps)
        K = torch.eye(3, dtype=dtype, device=device)[None].expand(len(imgs), 3, 3).clone()
        K[:, 0, 0] = K[:, 1, 1] = focals
        K[:, 0:2, 2] = pps * imsizes
        if trans is None:
            return K

        # security! optimization is always trying to crush the scale down
        sizes = torch.cat(log_sizes).exp()
        global_scaling = 1 / sizes.min()

        # compute distance of camera to focal plane
        # tan(fov) = W/2 / focal
        z_cameras = sizes * median_depths * focals / base_focals

        # make extrinsic
        rel_cam2cam = torch.eye(4, dtype=dtype, device=device)[None].expand(len(imgs), 4, 4).clone()
        rel_cam2cam[:, :3, :3] = roma.unitquat_to_rotmat(F.normalize(torch.stack(quats), dim=1))
        rel_cam2cam[:, :3, 3] = torch.stack(trans)

        # camera are defined as a kinematic chain
        tmp_cam2w = [None] * len(K)
        tmp_cam2w[mst[0]] = rel_cam2cam[mst[0]]
        for i, j in mst[1]:
            # i is the cam_i_to_world reference, j is the relative pose = cam_j_to_cam_i
            tmp_cam2w[j] = tmp_cam2w[i] @ rel_cam2cam[j]
        tmp_cam2w = torch.stack(tmp_cam2w)

        # smart reparameterizaton of cameras
        trans_offset = z_cameras.unsqueeze(1) * torch.cat((imsizes / focals.unsqueeze(1) * (0.5 - pps), ones), dim=-1)
        new_trans = global_scaling * (tmp_cam2w[:, :3, 3:4] - tmp_cam2w[:, :3, :3] @ trans_offset.unsqueeze(-1))
        cam2w = torch.cat((torch.cat((tmp_cam2w[:, :3, :3], new_trans), dim=2),
                          vec0001.view(1, 1, 4).expand(len(K), 1, 4)), dim=1)

        depthmaps = []
        for i in range(len(imgs)):
            core_depth_img = core_depth[i]
            if exp_depth:
                core_depth_img = core_depth_img.exp()
            if lora_depth:  # compute core_depth as a low-rank decomposition of 3d points
                core_depth_img = lora_depth_proj[i] @ core_depth_img
            if depth_mode == 'add':
                core_depth_img = z_cameras[i] + (core_depth_img - 1) * (median_depths[i] * sizes[i])
            elif depth_mode == 'mul':
                core_depth_img = z_cameras[i] * core_depth_img
            else:
                raise ValueError(f'Bad {depth_mode=}')
            depthmaps.append(global_scaling * core_depth_img)

        return K, (inv(cam2w), cam2w), depthmaps

    K = make_K_cam_depth(log_focals, pps, None, None, None, None)

    if shared_intrinsics:
        print('init focal (shared) = ', to_numpy(K[0, 0, 0]).round(2))
    else:
        print('init focals =', to_numpy(K[:, 0, 0]))

    # spectral low-rank projection of depthmaps
    if lora_depth:
        core_depth, lora_depth_proj = spectral_projection_of_depthmaps(
            imgs, K, core_depth, subsample, cache_path=cache_path, **lora_depth)
    if exp_depth:
        core_depth = [d.clip(min=1e-4).log() for d in core_depth]
    core_depth = [nn.Parameter(d.ravel().to(dtype)) for d in core_depth]
    log_sizes = [nn.Parameter(torch.zeros(1, dtype=dtype, device=device)) for _ in range(len(imgs))]

    # Pair-level correspondence slices shared by both losses. Keeping a single
    # compact structure here avoids duplicating the same match tensors again in
    # a separate corres2d aggregate.
    imgs_slices = pair_slices

    # Define which pairs are fine to use with matching
    def matching_check(x): return x.max() > matching_conf_thr
    is_matching_ok = {}
    for s in imgs_slices:
        is_matching_ok[s.img1, s.img2] = matching_check(s.confs)

    # Prepare slices and corres for losses
    dust3r_slices = [s for s in imgs_slices if not is_matching_ok[s.img1, s.img2]]
    loss3d_slices = [s for s in imgs_slices if is_matching_ok[s.img1, s.img2]]
    fast_loss2d_terms = None
    if fast_loss_path:
        fast_loss2d_terms = [[] for _ in range(len(imgs))]
        for s in loss3d_slices:
            fast_loss2d_terms[s.img1].append((s.pix1, s.confs, s.img2, s.slice2))
            fast_loss2d_terms[s.img2].append((s.pix2, s.confs, s.img1, s.slice1))
    dust3r_anchor_idxs_cpu = None
    dust3r_compact_cache = None
    if preds_21 is not None and dust3r_slices:
        dust3r_anchor_idxs_cpu = {
            idx: anchors[idx][1].detach().cpu()
            for idx in range(len(imgs))
        }
        dust3r_compact_cache = {}

    def _get_compact_dust3r_pair(src_img_idx, tgt_img_idx):
        key = (src_img_idx, tgt_img_idx)
        cached = dust3r_compact_cache.get(key)
        if cached is not None:
            return cached

        pred_full, conf_full = preds_21[imgs[src_img_idx]][imgs[tgt_img_idx]]
        idxs_cpu = dust3r_anchor_idxs_cpu[tgt_img_idx]
        compact = (
            pred_full.index_select(0, idxs_cpu),
            conf_full.index_select(0, idxs_cpu),
        )
        dust3r_compact_cache[key] = compact
        preds_21[imgs[src_img_idx]][imgs[tgt_img_idx]] = compact
        return compact

    def loss_dust3r(cam2w, pts3d, pix_loss):
        if loss_dust3r_w <= 0.0 or preds_21 is None:
            return 0.
        # In the case no correspondence could be established, fallback to DUSt3R GA regression loss formulation (sparsified)
        loss = 0.
        cf_sum = 0.
        for s in dust3r_slices:
            if init[imgs[s.img1]].get('freeze') and init[imgs[s.img2]].get('freeze'):
                continue
            # fallback to dust3r regression
            tgt_pts, tgt_confs = _get_compact_dust3r_pair(s.img2, s.img1)
            tgt_pts = tgt_pts.to(device=cam2w.device, dtype=pts3d[s.img1].dtype, non_blocking=True)
            tgt_confs = tgt_confs.to(device=cam2w.device, dtype=pts3d[s.img1].dtype, non_blocking=True)
            tgt_pts = geotrf(cam2w[s.img2], tgt_pts)
            cf_sum += tgt_confs.sum()
            loss += tgt_confs @ pix_loss(pts3d[s.img1], tgt_pts)
        return loss / cf_sum if cf_sum != 0. else 0.

    def loss_3d(K, w2cam, pts3d, pix_loss):
        # For each correspondence, we have two 3D points (one for each image of the pair).
        # For each 3D point, we have 2 reproj errors
        if fast_loss_path:
            active_slices = []
            if any(v.get('freeze') for v in init.values()):
                for s in loss3d_slices:
                    if init[imgs[s.img1]].get('freeze') and init[imgs[s.img2]].get('freeze'):
                        continue
                    active_slices.append(s)
            else:
                active_slices = loss3d_slices

            if not active_slices:
                return torch.tensor(0., device=device, dtype=dtype)

            confs = torch.cat([s.confs for s in active_slices])
            pts3d_1 = torch.cat([pts3d[s.img1][s.slice1] for s in active_slices])
            pts3d_2 = torch.cat([pts3d[s.img2][s.slice2] for s in active_slices])
            return (confs @ pix_loss(pts3d_1, pts3d_2)) / confs.sum()

        loss = None
        cf_sum = None
        any_frozen = any(v.get('freeze') for v in init.values())

        for s in loss3d_slices:
            if any_frozen and init[imgs[s.img1]].get('freeze') and init[imgs[s.img2]].get('freeze'):
                continue

            pts3d_1 = pts3d[s.img1][s.slice1]
            pts3d_2 = pts3d[s.img2][s.slice2]
            confs = s.confs
            pair_loss = confs @ pix_loss(pts3d_1, pts3d_2)
            pair_cf_sum = confs.sum()

            if loss is None:
                loss = pair_loss
                cf_sum = pair_cf_sum
            else:
                loss = loss + pair_loss
                cf_sum = cf_sum + pair_cf_sum

        if loss is None:
            return torch.tensor(0., device=device, dtype=dtype)
        return loss / cf_sum

    def loss_2d(K, w2cam, pts3d, pix_loss):
        # For each correspondence, we have two 3D points (one for each image of the pair).
        # For each 3D point, we have 2 reproj errors
        proj_matrix = K @ w2cam[:, :3]
        if fast_loss_path:
            loss = None
            npix = None
            for img_idx, img_terms in enumerate(fast_loss2d_terms):
                if init[imgs[img_idx]].get('freeze', 0) >= 1 or not img_terms:
                    continue
                pix = torch.cat([pix_a for pix_a, _, _, _ in img_terms])
                confs = torch.cat([confs_a for _, confs_a, _, _ in img_terms])
                pts3d_in_img = torch.cat([pts3d[img_b][slice_b] for _, _, img_b, slice_b in img_terms])
                img_loss = confs @ pix_loss(pix, reproj2d(proj_matrix[img_idx], pts3d_in_img))
                img_npix = confs.sum()
                if loss is None:
                    loss = img_loss
                    npix = img_npix
                else:
                    loss = loss + img_loss
                    npix = npix + img_npix
            if loss is None:
                return torch.tensor(0., device=device, dtype=dtype)
            return loss / npix

        loss = npix = 0
        for s in loss3d_slices:
            if init[imgs[s.img1]].get('freeze', 0) < 1:
                pts3d_in_img1 = pts3d[s.img2][s.slice2]
                loss += s.confs @ pix_loss(s.pix1, reproj2d(proj_matrix[s.img1], pts3d_in_img1))
                npix += s.confs.sum()
            if init[imgs[s.img2]].get('freeze', 0) < 1:
                pts3d_in_img2 = pts3d[s.img1][s.slice1]
                loss += s.confs @ pix_loss(s.pix2, reproj2d(proj_matrix[s.img2], pts3d_in_img2))
                npix += s.confs.sum()

        return loss / npix if npix != 0 else 0.

    def _chunk_ranges(n_items, chunk_size):
        if chunk_size is None or chunk_size <= 0 or chunk_size >= n_items:
            return [(0, n_items)]
        return [(start, min(start + chunk_size, n_items)) for start in range(0, n_items, chunk_size)]

    def backward_loss3d_chunked(cam2w, pts3d, pix_loss_3d, pix_loss_dust3r):
        active_loss3d_slices = [
            s for s in loss3d_slices
            if not (any(v.get('freeze') for v in init.values())
                    and init[imgs[s.img1]].get('freeze')
                    and init[imgs[s.img2]].get('freeze'))
        ]
        if not active_loss3d_slices:
            return 0.0

        loss3d_norm = sum(float(s.confs.sum().detach().cpu()) for s in active_loss3d_slices)
        if loss3d_norm == 0.0:
            return 0.0

        dust_chunks = []
        dust_norm = 0.0
        if loss_dust3r_w > 0.0 and preds_21 is not None:
            active_dust_slices = [
                s for s in dust3r_slices
                if not (init[imgs[s.img1]].get('freeze') and init[imgs[s.img2]].get('freeze'))
            ]
            dust_chunks = _chunk_ranges(len(active_dust_slices), loss_dust3r_backward_chunk_pairs)
            for s in active_dust_slices:
                _, tgt_confs = _get_compact_dust3r_pair(s.img2, s.img1)
                dust_norm += float(tgt_confs.sum().detach().cpu())
        else:
            active_dust_slices = []

        loss3d_chunks = _chunk_ranges(len(active_loss3d_slices), loss3d_backward_chunk_pairs)
        remaining_calls = len(loss3d_chunks) + (len(dust_chunks) if dust_norm > 0.0 else 0)
        total_loss_value = 0.0

        for start, end in loss3d_chunks:
            chunk_loss = None
            for s in active_loss3d_slices[start:end]:
                pts3d_1 = pts3d[s.img1][s.slice1]
                pts3d_2 = pts3d[s.img2][s.slice2]
                pair_loss = s.confs @ pix_loss_3d(pts3d_1, pts3d_2)
                chunk_loss = pair_loss if chunk_loss is None else chunk_loss + pair_loss
            scaled_chunk = chunk_loss / loss3d_norm
            total_loss_value += float(scaled_chunk.detach().cpu())
            remaining_calls -= 1
            scaled_chunk.backward(retain_graph=remaining_calls > 0)

        if dust_norm > 0.0:
            for start, end in dust_chunks:
                chunk_loss = None
                for s in active_dust_slices[start:end]:
                    tgt_pts, tgt_confs = _get_compact_dust3r_pair(s.img2, s.img1)
                    tgt_pts = tgt_pts.to(device=cam2w.device, dtype=pts3d[s.img1].dtype, non_blocking=True)
                    tgt_confs = tgt_confs.to(device=cam2w.device, dtype=pts3d[s.img1].dtype, non_blocking=True)
                    tgt_pts = geotrf(cam2w[s.img2], tgt_pts)
                    pair_loss = tgt_confs @ pix_loss_dust3r(pts3d[s.img1], tgt_pts)
                    chunk_loss = pair_loss if chunk_loss is None else chunk_loss + pair_loss
                scaled_chunk = loss_dust3r_w * (chunk_loss / dust_norm)
                total_loss_value += float(scaled_chunk.detach().cpu())
                remaining_calls -= 1
                scaled_chunk.backward(retain_graph=remaining_calls > 0)

        return total_loss_value

    def backward_loss2d_chunked(K, w2cam, pts3d, pix_loss):
        proj_matrix = K @ w2cam[:, :3]
        active_terms = []
        for s in loss3d_slices:
            if init[imgs[s.img1]].get('freeze', 0) < 1:
                active_terms.append((s.img1, s.pix1, s.confs, s.img2, s.slice2))
            if init[imgs[s.img2]].get('freeze', 0) < 1:
                active_terms.append((s.img2, s.pix2, s.confs, s.img1, s.slice1))

        if not active_terms:
            return 0.0

        norm = sum(float(confs.sum().detach().cpu()) for _, _, confs, _, _ in active_terms)
        if norm == 0.0:
            return 0.0

        chunks = _chunk_ranges(len(active_terms), loss2d_backward_chunk_pairs)
        remaining_calls = len(chunks)
        total_loss_value = 0.0

        for start, end in chunks:
            chunk_loss = None
            for img_a, pix_a, confs, img_b, slice_b in active_terms[start:end]:
                pts3d_in_img = pts3d[img_b][slice_b]
                pair_loss = confs @ pix_loss(pix_a, reproj2d(proj_matrix[img_a], pts3d_in_img))
                chunk_loss = pair_loss if chunk_loss is None else chunk_loss + pair_loss
            scaled_chunk = chunk_loss / norm
            total_loss_value += float(scaled_chunk.detach().cpu())
            remaining_calls -= 1
            scaled_chunk.backward(retain_graph=remaining_calls > 0)

        return total_loss_value

    def optimize_loop(loss_func, lr_base, niter, pix_loss, lr_end=0):
        # create optimizer
        params = pps + log_focals + quats + trans + log_sizes + core_depth
        deduped_params = []
        seen_param_ids = set()
        for p in params:
            pid = id(p)
            if pid in seen_param_ids:
                continue
            seen_param_ids.add(pid)
            deduped_params.append(p)
        optimizer = torch.optim.Adam(deduped_params, lr=1, weight_decay=0, betas=(0.9, 0.9))
        ploss = pix_loss if 'meta' in repr(pix_loss) else (lambda a: pix_loss)
        last_state = None
        last_loss = float('nan')

        with tqdm(total=niter) as bar:
            for iter in range(niter or 1):
                K, (w2cam, cam2w), depthmaps = make_K_cam_depth(log_focals, pps, trans, quats, log_sizes, core_depth)
                pts3d = make_pts3d(anchors, K, cam2w, depthmaps, base_focals=base_focals)
                if niter == 0:
                    last_state = dict(
                        intrinsics=K.detach(),
                        cam2w=cam2w.detach(),
                        depthmaps=[d.detach() for d in depthmaps],
                        pts3d=[p.detach() for p in pts3d],
                    )
                    break

                alpha = (iter / niter)
                lr = schedule(alpha, lr_base, lr_end)
                adjust_learning_rate_by_lr(optimizer, lr)
                pix_loss = ploss(1 - alpha)
                optimizer.zero_grad(set_to_none=True)
                use_chunked_loss3d = loss_func is loss_3d and loss3d_backward_chunk_pairs > 0
                use_chunked_loss2d = loss_func is loss_2d and loss2d_backward_chunk_pairs > 0
                if use_chunked_loss3d:
                    loss = backward_loss3d_chunked(cam2w, pts3d, pix_loss, lossd)
                elif use_chunked_loss2d:
                    loss = backward_loss2d_chunked(K, w2cam, pts3d, pix_loss)
                elif loss_func is loss_3d:
                    loss = loss_func(K, w2cam, pts3d, pix_loss) + loss_dust3r_w * loss_dust3r(cam2w, pts3d, lossd)
                    loss.backward()
                else:
                    loss = loss_func(K, w2cam, pts3d, pix_loss)
                    loss.backward()
                optimizer.step()

                # make sure the pose remains well optimizable
                for i in range(len(imgs)):
                    quats[i].data[:] /= quats[i].data.norm()

                loss = float(loss)
                last_loss = loss
                last_state = dict(
                    intrinsics=K.detach(),
                    cam2w=cam2w.detach(),
                    depthmaps=[d.detach() for d in depthmaps],
                    pts3d=[p.detach() for p in pts3d],
                )
                if loss != loss:
                    break  # NaN loss
                bar.set_postfix_str(f'{lr=:.4f}, {loss=:.3f}')
                bar.update(1)

                del pts3d
                del depthmaps
                del K
                del w2cam
                del cam2w
                if use_chunked_loss3d or use_chunked_loss2d:
                    gc.collect()
                    if str(device).startswith('cuda'):
                        torch.cuda.empty_cache()

        if niter:
            print(f'>> final loss = {last_loss}')
        return last_state

    # at start, don't optimize 3d points
    for i, img in enumerate(imgs):
        trainable = not (init[img].get('freeze'))
        pps[i].requires_grad_(False)
        log_focals[i].requires_grad_(False)
        quats[i].requires_grad_(trainable)
        trans[i].requires_grad_(trainable)
        log_sizes[i].requires_grad_(trainable)
        core_depth[i].requires_grad_(False)

    res_coarse = optimize_loop(loss_3d, lr_base=lr1, niter=niter1, pix_loss=loss1)

    res_fine = None
    if niter2:
        # now we can optimize 3d points
        for i, img in enumerate(imgs):
            if init[img].get('freeze', 0) >= 1:
                continue
            pps[i].requires_grad_(bool(opt_pp))
            log_focals[i].requires_grad_(True)
            core_depth[i].requires_grad_(opt_depth)

        # refinement with 2d reproj
        res_fine = optimize_loop(loss_2d, lr_base=lr2, niter=niter2, pix_loss=loss2)

    K = make_K_cam_depth(log_focals, pps, None, None, None, None)
    if shared_intrinsics:
        print('Final focal (shared) = ', to_numpy(K[0, 0, 0]).round(2))
    else:
        print('Final focals =', to_numpy(K[:, 0, 0]))

    return imgs, res_coarse, res_fine


@lru_cache
def mask110(device, dtype):
    return torch.tensor((1, 1, 0), device=device, dtype=dtype)


def proj3d(inv_K, pixels, z):
    if pixels.shape[-1] == 2:
        pixels = torch.cat((pixels, torch.ones_like(pixels[..., :1])), dim=-1)
    return z.unsqueeze(-1) * (pixels * inv_K.diag() + inv_K[:, 2] * mask110(z.device, z.dtype))


def make_pts3d(anchors, K, cam2w, depthmaps, base_focals=None, ret_depth=False):
    focals = K[:, 0, 0]
    invK = inv(K)
    all_pts3d = []
    depth_out = []

    for img, (pixels, idxs, offsets) in anchors.items():
        # from depthmaps to 3d points
        if base_focals is None:
            pass
        else:
            # compensate for focal
            # depth + depth * (offset - 1) * base_focal / focal
            # = depth * (1 + (offset - 1) * (base_focal / focal))
            offsets = 1 + (offsets - 1) * (base_focals[img] / focals[img])

        pts3d = proj3d(invK[img], pixels, depthmaps[img][idxs] * offsets)
        if ret_depth:
            depth_out.append(pts3d[..., 2])  # before camera rotation

        # rotate to world coordinate
        pts3d = geotrf(cam2w[img], pts3d)
        all_pts3d.append(pts3d)

    if ret_depth:
        return all_pts3d, depth_out
    return all_pts3d


def make_dense_pts3d(intrinsics, cam2w, depthmaps, canonical_paths, subsample, device='cuda'):
    base_focals = []
    anchors = {}
    confs = []
    for i, canon_path in enumerate(canonical_paths):
        (canon, canon2, conf), focal = _load_cache(canon_path, map_location=device)
        confs.append(conf)
        base_focals.append(focal)
        H, W = conf.shape
        pixels = torch.from_numpy(np.mgrid[:W, :H].T.reshape(-1, 2)).float().to(device)
        idxs, offsets = anchor_depth_offsets(canon2, {i: (pixels, None)}, subsample=subsample)
        anchors[i] = (pixels, idxs[i], offsets[i])

    # densify sparse depthmaps
    pts3d, depthmaps_out = make_pts3d(anchors, intrinsics, cam2w, [
                                      d.ravel() for d in depthmaps], base_focals=base_focals, ret_depth=True)

    return pts3d, depthmaps_out, confs


_CACHE_MAGIC = b"OBJECTX_MAST3R_CACHE_V1\n"
_CACHE_MAGIC_GZIP = b"OBJECTX_MAST3R_CACHE_V2GZ\n"


def _maybe_compact_array(arr: np.ndarray):
    arr = np.ascontiguousarray(arr)
    meta = {"orig_dtype": arr.dtype.str}
    if arr.size == 0:
        return arr, meta

    # Preserve floating-point tensors losslessly. The earlier float16/float32
    # cache compaction reduced storage substantially, but it can also degrade
    # downstream geometry quality because these cached tensors are reused during
    # canonical-view construction and the DUSt3R fallback loss.
    if np.issubdtype(arr.dtype, np.floating):
        return arr, meta

    if arr.dtype in (np.int64, np.int32, np.int16):
        arr_min = int(arr.min())
        arr_max = int(arr.max())
        if arr_min >= 0 and arr_max <= np.iinfo(np.uint16).max:
            return arr.astype(np.uint16), meta
        if np.iinfo(np.int32).min <= arr_min and arr_max <= np.iinfo(np.int32).max:
            return arr.astype(np.int32), meta
        if np.iinfo(np.int16).min <= arr_min and arr_max <= np.iinfo(np.int16).max:
            return arr.astype(np.int16), meta

    return arr, meta


def _cache_pack(obj):
    if torch.is_tensor(obj):
        arr, meta = _maybe_compact_array(obj.detach().cpu().contiguous().numpy())
        return {"__cache_type__": "tensor", "array": arr, **meta}
    if isinstance(obj, np.ndarray):
        arr, meta = _maybe_compact_array(obj)
        return {"__cache_type__": "ndarray", "array": arr, **meta}
    if isinstance(obj, dict):
        return {
            "__cache_type__": "dict",
            "items": {k: _cache_pack(v) for k, v in obj.items()},
        }
    if isinstance(obj, list):
        return {"__cache_type__": "list", "items": [_cache_pack(v) for v in obj]}
    if isinstance(obj, tuple):
        return {"__cache_type__": "tuple", "items": [_cache_pack(v) for v in obj]}
    return obj


def _cache_unpack(obj, map_location='cpu'):
    if isinstance(obj, dict) and "__cache_type__" in obj:
        kind = obj["__cache_type__"]
        if kind in {"tensor", "ndarray"}:
            arr = obj["array"]
            orig_dtype = np.dtype(obj.get("orig_dtype", arr.dtype.str))
            if arr.dtype != orig_dtype:
                arr = arr.astype(orig_dtype, copy=False)
            tensor = torch.from_numpy(arr)
            return tensor.to(map_location) if map_location is not None else tensor
        if kind == "dict":
            return {k: _cache_unpack(v, map_location=map_location) for k, v in obj["items"].items()}
        if kind == "list":
            return [_cache_unpack(v, map_location=map_location) for v in obj["items"]]
        if kind == "tuple":
            return tuple(_cache_unpack(v, map_location=map_location) for v in obj["items"])
    return obj


def _load_cache(path, map_location='cpu'):
    with open(path, "rb") as f:
        magic = f.read(max(len(_CACHE_MAGIC), len(_CACHE_MAGIC_GZIP)))
        if magic[:len(_CACHE_MAGIC_GZIP)] == _CACHE_MAGIC_GZIP:
            with gzip.GzipFile(fileobj=f, mode="rb") as gz:
                payload = pickle.load(gz)
            return _cache_unpack(payload, map_location=map_location)
        if magic[:len(_CACHE_MAGIC)] == _CACHE_MAGIC:
            payload = pickle.load(f)
            return _cache_unpack(payload, map_location=map_location)
    return torch.load(path, map_location=map_location)


def _save_atomic(obj, path, retries=8):
    """Atomic torch.save with retry + exponential back-off.

    MASt3R's cache files are transient intermediate artifacts, so we always use
    the legacy serializer here. In practice the zip serializer in torch 2.x has
    been the least reliable part of this pipeline on large tmpfs-backed caches
    and can surface as misleading "unexpected pos X vs Y" write failures even
    when /tmp has ample space left.
    """
    import errno as _errno
    import tempfile
    import shutil

    last_exc = None
    for attempt in range(retries):
        parent = os.path.dirname(path) or "."
        os.makedirs(parent, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            prefix=os.path.basename(path) + ".tmp.",
            dir=parent,
        )
        try:
            payload = _cache_pack(obj)
            with os.fdopen(fd, "wb") as f:
                f.write(_CACHE_MAGIC_GZIP)
                with gzip.GzipFile(fileobj=f, mode="wb", compresslevel=6) as gz:
                    pickle.dump(payload, gz, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            return
        except Exception as exc:
            last_exc = exc
            try:
                os.close(fd)
            except OSError:
                pass
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass

            is_quota = (isinstance(exc, OSError) and
                        getattr(exc, 'errno', None) == _errno.EDQUOT)
            delay = min(2.0 ** attempt, 30.0)

            if attempt == 0 or is_quota:
                try:
                    disk = shutil.disk_usage(parent)
                    disk_msg = (f"disk_total={disk.total/1e9:.1f}G "
                                f"disk_used={disk.used/1e9:.1f}G "
                                f"disk_free={disk.free/1e9:.1f}G")
                except Exception:
                    disk_msg = "disk_usage=unavailable"
                try:
                    import psutil
                    mem = psutil.virtual_memory()
                    mem_msg = (f"ram_total={mem.total/1e9:.1f}G "
                               f"ram_avail={mem.available/1e9:.1f}G")
                except Exception:
                    mem_msg = "ram=unavailable"
                print(f"[_save_atomic] attempt {attempt+1}/{retries} FAILED "
                      f"{'EDQUOT' if is_quota else type(exc).__name__}: {exc}\n"
                      f"  path={path}\n  {disk_msg} {mem_msg}\n"
                      f"  retrying in {delay:.1f}s with legacy_pickle=True")

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()
            time.sleep(delay)

    raise RuntimeError(
        f"_save_atomic failed after {retries} retries for {path}: {last_exc}"
    ) from last_exc


def _pair_cache_instances(img1_instance, img2_instance):
    if img1_instance <= img2_instance:
        return (img1_instance, img2_instance), False
    return (img2_instance, img1_instance), True


def _pair_cache_path(cache_path, img1_instance, img2_instance, desc_conf, subsample):
    (inst_a, inst_b), _ = _pair_cache_instances(img1_instance, img2_instance)
    idx_a = hash_md5(inst_a)
    idx_b = hash_md5(inst_b)
    return (
        cache_path
        + f"/pair_cache_conf={desc_conf}_{subsample=}/{idx_a}/{idx_a}-{idx_b}.pth"
    )


def _build_compact_pair_payload(
    img1_instance,
    img2_instance,
    X11,
    C11,
    X21,
    C21,
    X22,
    C22,
    X12,
    C12,
    corres,
    matching_score,
    subsample,
):
    (inst_a, inst_b), flipped = _pair_cache_instances(img1_instance, img2_instance)
    xy1, xy2, confs = corres

    if not flipped:
        a_self_X, a_self_C = X11, C11
        a_other_X, a_other_C = X21, C21
        b_self_X, b_self_C = X22, C22
        b_other_X, b_other_C = X12, C12
        a_xy, b_xy = xy1, xy2
    else:
        a_self_X, a_self_C = X22, C22
        a_other_X, a_other_C = X12, C12
        b_self_X, b_self_C = X11, C11
        b_other_X, b_other_C = X21, C21
        a_xy, b_xy = xy2, xy1

    def _safe_cpu(t):
        return t.detach().contiguous().cpu().contiguous()

    return {
        "instances": (inst_a, inst_b),
        "matching_score": tuple(float(v) for v in matching_score),
        "corres": {
            "a_xy": _safe_cpu(a_xy),
            "b_xy": _safe_cpu(b_xy),
            "confs": _safe_cpu(confs),
        },
        "views": {
            "a": {
                "X": _safe_cpu(a_self_X),
                "C": _safe_cpu(a_self_C),
                "X_other_sub": _safe_cpu(
                    a_other_X[::subsample, ::subsample].reshape(-1, 3)
                ),
                "C_other_sub": _safe_cpu(
                    a_other_C[::subsample, ::subsample].ravel()
                ),
            },
            "b": {
                "X": _safe_cpu(b_self_X),
                "C": _safe_cpu(b_self_C),
                "X_other_sub": _safe_cpu(
                    b_other_X[::subsample, ::subsample].reshape(-1, 3)
                ),
                "C_other_sub": _safe_cpu(
                    b_other_C[::subsample, ::subsample].ravel()
                ),
            },
        },
    }


def _load_compact_pair_for_direction(
    pair_cache_path,
    img1_instance,
    img2_instance,
    device,
    min_conf_thr=0,
):
    payload = _load_cache(pair_cache_path, map_location="cpu")
    inst_a, inst_b = payload["instances"]
    if (img1_instance, img2_instance) == (inst_a, inst_b):
        view_key = "a"
        xy1 = payload["corres"]["a_xy"]
        xy2 = payload["corres"]["b_xy"]
    elif (img1_instance, img2_instance) == (inst_b, inst_a):
        view_key = "b"
        xy1 = payload["corres"]["b_xy"]
        xy2 = payload["corres"]["a_xy"]
    else:
        raise ValueError(
            f"pair cache {pair_cache_path} does not match requested "
            f"direction ({img1_instance}, {img2_instance})"
        )

    confs = payload["corres"]["confs"]
    valid = confs > min_conf_thr if min_conf_thr else slice(None)

    view = payload["views"][view_key]
    return (
        payload["matching_score"],
        (
            xy1[valid].to(device),
            xy2[valid].to(device),
            confs[valid].to(device),
        ),
        (
            view["X"].to(device),
            view["C"].to(device),
            view["X_other_sub"],
            view["C_other_sub"],
        ),
    )


@torch.no_grad()
def forward_mast3r(pairs, model, cache_path, desc_conf='desc_conf',
                   device='cuda', subsample=8, **matching_kw):
    res_paths = {}

    for img1, img2 in tqdm(pairs):
        pair_cache = _pair_cache_path(
            cache_path,
            img1['instance'],
            img2['instance'],
            desc_conf,
            subsample,
        )

        if not os.path.isfile(pair_cache):
            if model is None:
                continue
            res = symmetric_inference(model, img1, img2, device=device)
            X11, X21, X22, X12 = [r['pts3d'][0] for r in res]
            C11, C21, C22, C12 = [r['conf'][0] for r in res]
            descs = [r['desc'][0] for r in res]
            qonfs = [r[desc_conf][0] for r in res]

            # perform reciprocal matching
            corres = extract_correspondences(descs, qonfs, device=device, subsample=subsample)

            conf_score = (C11.mean() * C12.mean() * C21.mean() * C22.mean()).sqrt().sqrt()
            matching_score = (float(conf_score), float(corres[2].sum()), len(corres[2]))
            if cache_path is not None:
                compact_payload = _build_compact_pair_payload(
                    img1['instance'],
                    img2['instance'],
                    X11,
                    C11,
                    X21,
                    C21,
                    X22,
                    C22,
                    X12,
                    C12,
                    corres,
                    matching_score,
                    subsample,
                )
                _save_atomic(compact_payload, mkdir_for(pair_cache))

        res_paths[img1['instance'], img2['instance']] = pair_cache

    del model
    torch.cuda.empty_cache()

    return res_paths, cache_path


def symmetric_inference(model, img1, img2, device):
    img1 = _materialize_img_entry(img1)
    img2 = _materialize_img_entry(img2)
    shape1 = torch.from_numpy(img1['true_shape']).to(device, non_blocking=True)
    shape2 = torch.from_numpy(img2['true_shape']).to(device, non_blocking=True)
    img1 = img1['img'].to(device, non_blocking=True)
    img2 = img2['img'].to(device, non_blocking=True)

    # compute encoder only once
    feat1, feat2, pos1, pos2 = model._encode_image_pairs(img1, img2, shape1, shape2)

    def decoder(feat1, feat2, pos1, pos2, shape1, shape2):
        dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
        with torch.cuda.amp.autocast(enabled=False):
            res1 = model._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = model._downstream_head(2, [tok.float() for tok in dec2], shape2)
        return res1, res2

    # decoder 1-2
    res11, res21 = decoder(feat1, feat2, pos1, pos2, shape1, shape2)
    # decoder 2-1
    res22, res12 = decoder(feat2, feat1, pos2, pos1, shape2, shape1)

    return (res11, res21, res22, res12)


def extract_correspondences(feats, qonfs, subsample=8, device=None, ptmap_key='pred_desc'):
    feat11, feat21, feat22, feat12 = feats
    qonf11, qonf21, qonf22, qonf12 = qonfs
    assert feat11.shape[:2] == feat12.shape[:2] == qonf11.shape == qonf12.shape
    assert feat21.shape[:2] == feat22.shape[:2] == qonf21.shape == qonf22.shape

    if '3d' in ptmap_key:
        opt = dict(device='cpu', workers=32)
    else:
        opt = dict(device=device, dist='dot', block_size=2**13)

    # matching the two pairs
    idx1 = []
    idx2 = []
    qonf1 = []
    qonf2 = []
    # TODO add non symmetric / pixel_tol options
    for A, B, QA, QB in [(feat11, feat21, qonf11.cpu(), qonf21.cpu()),
                         (feat12, feat22, qonf12.cpu(), qonf22.cpu())]:
        nn1to2 = fast_reciprocal_NNs(A, B, subsample_or_initxy1=subsample, ret_xy=False, **opt)
        nn2to1 = fast_reciprocal_NNs(B, A, subsample_or_initxy1=subsample, ret_xy=False, **opt)

        idx1.append(np.r_[nn1to2[0], nn2to1[1]])
        idx2.append(np.r_[nn1to2[1], nn2to1[0]])
        qonf1.append(QA.ravel()[idx1[-1]])
        qonf2.append(QB.ravel()[idx2[-1]])

    # merge corres from opposite pairs
    H1, W1 = feat11.shape[:2]
    H2, W2 = feat22.shape[:2]
    cat = np.concatenate

    xy1, xy2, idx = merge_corres(cat(idx1), cat(idx2), (H1, W1), (H2, W2), ret_xy=True, ret_index=True)
    corres = (xy1.copy(), xy2.copy(), np.sqrt(cat(qonf1)[idx] * cat(qonf2)[idx]))

    return todevice(corres, device)


@torch.no_grad()
def prepare_canonical_data(imgs, tmp_pairs, subsample, order_imgs=False, min_conf_thr=0,
                           cache_path=None, device='cuda', keep_preds_21=True, **kw):
    canonical_views = {}
    pairwise_scores = torch.zeros((len(imgs), len(imgs)), device=device)
    canonical_paths = []
    preds_21 = {} if keep_preds_21 else None

    for img in tqdm(imgs):
        cache = None
        if cache_path:
            cache = os.path.join(cache_path, 'canon_views', hash_md5(img) + f'_{subsample=}_{kw=}.pth')
            canonical_paths.append(cache)
        if cache is not None:
            try:
                (canon, canon2, cconf), focal = _load_cache(cache, map_location=device)
            except IOError:
                # cache does not exist yet, we create it!
                canon = focal = None
        else:
            canon = focal = None

        # collect all pred1
        n_pairs = sum((img in pair) for pair in tmp_pairs)

        ptmaps11 = None
        pixels = {}
        n = 0
        for (img1, img2), pair_cache_path in tmp_pairs.items():
            score = None
            if img == img1:
                score, (xy1, xy2, confs), (X, C, X2_sub, C2_sub) = _load_compact_pair_for_direction(
                    pair_cache_path,
                    img1,
                    img2,
                    device=device,
                    min_conf_thr=min_conf_thr,
                )
                pixels[img2] = xy1, confs
                if keep_preds_21:
                    if img not in preds_21:
                        preds_21[img] = {}
                    preds_21[img][img2] = to_cpu((
                        X2_sub.to(torch.float32),
                        C2_sub.to(torch.float32),
                    ))

            if img == img2:
                score, (xy1, xy2, confs), (X, C, X2_sub, C2_sub) = _load_compact_pair_for_direction(
                    pair_cache_path,
                    img2,
                    img1,
                    device=device,
                    min_conf_thr=min_conf_thr,
                )
                pixels[img1] = xy1, confs
                if keep_preds_21:
                    if img not in preds_21:
                        preds_21[img] = {}
                    preds_21[img][img1] = to_cpu((
                        X2_sub.to(torch.float32),
                        C2_sub.to(torch.float32),
                    ))

            if score is not None:
                i, j = imgs.index(img1), imgs.index(img2)
                # score = score[0]
                # score = np.log1p(score[2])
                score = score[2]
                pairwise_scores[i, j] = score
                pairwise_scores[j, i] = score

                if canon is not None:
                    continue
                if ptmaps11 is None:
                    H, W = C.shape
                    ptmaps11 = torch.empty((n_pairs, H, W, 3), device=device)
                    confs11 = torch.empty((n_pairs, H, W), device=device)

                ptmaps11[n] = X
                confs11[n] = C
                n += 1

        if canon is None:
            if ptmaps11 is None or n == 0:
                raise RuntimeError(
                    f"prepare_canonical_data found no pair pointmaps for image {img!r}; "
                    "this usually means all correspondences were filtered out or the pair cache is incomplete"
                )
            if n != n_pairs:
                raise RuntimeError(
                    f"prepare_canonical_data pair count mismatch for image {img!r}: "
                    f"expected {n_pairs}, collected {n}"
                )
            canon, canon2, cconf = canonical_view(ptmaps11, confs11, subsample, **kw)
            del ptmaps11
            del confs11

        # compute focals
        H, W = canon.shape[:2]
        pp = torch.tensor([W / 2, H / 2], device=device)
        if focal is None:
            focal = estimate_focal_knowing_depth(canon[None], pp, focal_mode='weiszfeld', min_focal=0.5, max_focal=3.5)
            if cache:
                _save_atomic(to_cpu(((canon, canon2, cconf), focal)), mkdir_for(cache))

        # extract depth offsets with correspondences
        core_depth = canon[subsample // 2::subsample, subsample // 2::subsample, 2]
        idxs, offsets = anchor_depth_offsets(canon2, pixels, subsample=subsample)

        canonical_views[img] = (pp, (H, W), focal.view(1), core_depth, pixels, idxs, offsets)

    return tmp_pairs, pairwise_scores, canonical_views, canonical_paths, preds_21


@torch.no_grad()
def prepare_condensed_data(imgs, tmp_pairs, subsample, order_imgs=False, min_conf_thr=0,
                           cache_path=None, device='cuda', keep_preds_21=True,
                           dtype=torch.float32, **kw):
    _stage_log(
        f"prepare_condensed_data start n_imgs={len(imgs)} n_pairs={len(tmp_pairs)} "
        f"subsample={subsample} keep_preds_21={keep_preds_21}"
    )
    pairwise_scores = torch.zeros((len(imgs), len(imgs)), device=device)
    canonical_paths = []
    preds_21 = {} if keep_preds_21 else None

    img_to_idx = {img: idx for idx, img in enumerate(imgs)}
    set_imgs = set(imgs)
    pair_counts = {img: 0 for img in imgs}
    for img1, img2 in tmp_pairs:
        if img1 in pair_counts:
            pair_counts[img1] += 1
        if img2 in pair_counts:
            pair_counts[img2] += 1

    principal_points = []
    shapes = []
    focals = []
    core_depth = []
    img_anchors = {}
    tmp_pixels = {}

    for idx1, img in enumerate(tqdm(imgs)):
        if idx1 == 0 or (idx1 + 1) % 50 == 0 or (idx1 + 1) == len(imgs):
            _stage_log(f"prepare_condensed_data image_loop idx={idx1 + 1}/{len(imgs)}")
        cache = None
        if cache_path:
            cache = os.path.join(cache_path, 'canon_views', hash_md5(img) + f'_{subsample=}_{kw=}.pth')
            canonical_paths.append(cache)
        if cache is not None:
            try:
                (canon, canon2, cconf), focal = _load_cache(cache, map_location=device)
            except IOError:
                canon = focal = None
        else:
            canon = focal = None

        ptmaps11 = None
        confs11 = None
        pixels = {}
        n = 0
        n_pairs = pair_counts[img]

        for (img1, img2), pair_cache_path in tmp_pairs.items():
            score = None
            if img == img1:
                score, (xy1, xy2, confs), (X, C, X2_sub, C2_sub) = _load_compact_pair_for_direction(
                    pair_cache_path,
                    img1,
                    img2,
                    device=device,
                    min_conf_thr=min_conf_thr,
                )
                pixels[img2] = xy1, confs
                if keep_preds_21:
                    preds_21.setdefault(img, {})[img2] = to_cpu((
                        X2_sub.to(torch.float32),
                        C2_sub.to(torch.float32),
                    ))

            if img == img2:
                score, (xy1, xy2, confs), (X, C, X2_sub, C2_sub) = _load_compact_pair_for_direction(
                    pair_cache_path,
                    img2,
                    img1,
                    device=device,
                    min_conf_thr=min_conf_thr,
                )
                pixels[img1] = xy1, confs
                if keep_preds_21:
                    preds_21.setdefault(img, {})[img1] = to_cpu((
                        X2_sub.to(torch.float32),
                        C2_sub.to(torch.float32),
                    ))

            if score is not None:
                i = img_to_idx[img1]
                j = img_to_idx[img2]
                pairwise_scores[i, j] = score[2]
                pairwise_scores[j, i] = score[2]

                if canon is not None:
                    continue
                if ptmaps11 is None:
                    H, W = C.shape
                    ptmaps11 = torch.empty((n_pairs, H, W, 3), device=device)
                    confs11 = torch.empty((n_pairs, H, W), device=device)

                ptmaps11[n] = X
                confs11[n] = C
                n += 1

        if canon is None:
            if ptmaps11 is None or n == 0:
                raise RuntimeError(
                    f"prepare_condensed_data found no pair pointmaps for image {img!r}; "
                    "this usually means all correspondences were filtered out or the pair cache is incomplete"
                )
            if n != n_pairs:
                raise RuntimeError(
                    f"prepare_condensed_data pair count mismatch for image {img!r}: "
                    f"expected {n_pairs}, collected {n}"
                )
            _stage_log(f"prepare_condensed_data canonical_view_build idx={idx1 + 1}/{len(imgs)} pairs={n_pairs}")
            canon, canon2, cconf = canonical_view(ptmaps11, confs11, subsample, **kw)
            del ptmaps11
            del confs11

        H, W = canon.shape[:2]
        pp = torch.tensor([W / 2, H / 2], device=device)
        if focal is None:
            _stage_log(f"prepare_condensed_data focal_estimate idx={idx1 + 1}/{len(imgs)}")
            focal = estimate_focal_knowing_depth(canon[None], pp, focal_mode='weiszfeld', min_focal=0.5, max_focal=3.5)
            if cache:
                _save_atomic(to_cpu(((canon, canon2, cconf), focal)), mkdir_for(cache))

        depth_anchor = canon[subsample // 2::subsample, subsample // 2::subsample, 2]
        idxs, offsets = anchor_depth_offsets(canon2, pixels, subsample=subsample)

        principal_points.append(pp)
        shapes.append((H, W))
        focals.append(focal.view(1))
        core_depth.append(depth_anchor)

        img_uv1 = []
        img_idxs = []
        img_offs = []
        cur_n = [0]

        for img2, (pixels_xy, match_confs) in pixels.items():
            if img2 not in set_imgs:
                continue
            assert len(pixels_xy) == len(idxs[img2]) == len(offsets[img2])
            img_uv1.append(torch.cat((pixels_xy, torch.ones_like(pixels_xy[:, :1])), dim=-1))
            img_idxs_cur = idxs[img2]
            img_offs_cur = offsets[img2]
            img_idxs.append(img_idxs_cur)
            img_offs.append(img_offs_cur)
            cur_n.append(cur_n[-1] + len(pixels_xy))
            tmp_pixels[img, img2] = pixels_xy.to(dtype), match_confs.to(dtype), slice(*cur_n[-2:])

        if img_uv1:
            img_anchors[idx1] = (torch.cat(img_uv1), torch.cat(img_idxs), torch.cat(img_offs))
        else:
            img_anchors[idx1] = (
                torch.empty((0, 3), device=pp.device, dtype=torch.float32),
                torch.empty((0,), device=pp.device, dtype=torch.long),
                torch.empty((0,), device=pp.device, dtype=depth_anchor.dtype),
            )
        del pixels, idxs, offsets, canon, canon2, cconf
        gc.collect()

    imgs_slices = []
    _stage_log("prepare_condensed_data building_pair_slices")

    for pair_idx, (img1, img2) in enumerate(tmp_pairs, start=1):
        if pair_idx == 1 or pair_idx % 2000 == 0 or pair_idx == len(tmp_pairs):
            _stage_log(f"prepare_condensed_data pair_slice_loop idx={pair_idx}/{len(tmp_pairs)}")
        try:
            pix1, confs1, slice1 = tmp_pixels[img1, img2]
            pix2, confs2, slice2 = tmp_pixels[img2, img1]
        except KeyError:
            continue
        img1_idx = img_to_idx[img1]
        img2_idx = img_to_idx[img2]
        confs = (confs1 * confs2).sqrt()

        imgs_slices.append(PairOfSlices(
            img1_idx, slice1, pix1,
            img2_idx, slice2, pix2,
            confs,
        ))

    imsizes = torch.tensor([(W, H) for H, W in shapes], device=principal_points[0].device)  # (W,H)
    principal_points = torch.stack(principal_points)
    focals = torch.cat(focals)

    del tmp_pixels
    gc.collect()
    _stage_log(
        f"prepare_condensed_data ready_for_return n_pair_slices={len(imgs_slices)} "
        f"n_preds21_imgs={0 if preds_21 is None else len(preds_21)}"
    )

    return pairwise_scores, canonical_paths, imsizes, principal_points, focals, core_depth, img_anchors, imgs_slices, preds_21


def load_corres(path_corres, device, min_conf_thr):
    score, (xy1, xy2, confs) = _load_cache(path_corres, map_location=device)
    valid = confs > min_conf_thr if min_conf_thr else slice(None)
    # valid = (xy1 > 0).all(dim=1) & (xy2 > 0).all(dim=1) & (xy1 < 512).all(dim=1) & (xy2 < 512).all(dim=1)
    # print(f'keeping {valid.sum()} / {len(valid)} correspondences')
    return score, (xy1[valid], xy2[valid], confs[valid])


PairOfSlices = namedtuple(
    'ImgPair', 'img1, slice1, pix1, img2, slice2, pix2, confs')


def condense_data(imgs, tmp_paths, canonical_views, preds_21, dtype=torch.float32):
    # aggregate all data properly
    set_imgs = set(imgs)

    principal_points = []
    shapes = []
    focals = []
    core_depth = []
    img_anchors = {}
    tmp_pixels = {}

    for idx1, img1 in enumerate(imgs):
        # Consume the bulky per-image correspondence maps and immediately drop
        # them from canonical_views so we do not keep both the source structure
        # and the condensed representation alive at the same time.
        pp, shape, focal, anchors, pixels_confs, idxs, offsets = canonical_views.pop(img1)

        principal_points.append(pp)
        shapes.append(shape)
        focals.append(focal)
        core_depth.append(anchors)

        img_uv1 = []
        img_idxs = []
        img_offs = []
        cur_n = [0]

        for img2, (pixels, match_confs) in pixels_confs.items():
            if img2 not in set_imgs:
                continue
            assert len(pixels) == len(idxs[img2]) == len(offsets[img2])
            img_uv1.append(torch.cat((pixels, torch.ones_like(pixels[:, :1])), dim=-1))
            img_idxs_cur = idxs[img2]
            img_offs_cur = offsets[img2]
            img_idxs.append(img_idxs_cur)
            img_offs.append(img_offs_cur)
            cur_n.append(cur_n[-1] + len(pixels))
            # store the position of 3d points
            tmp_pixels[img1, img2] = pixels.to(dtype), match_confs.to(dtype), slice(*cur_n[-2:])
        if img_uv1:
            img_anchors[idx1] = (torch.cat(img_uv1), torch.cat(img_idxs), torch.cat(img_offs))
        else:
            img_anchors[idx1] = (
                torch.empty((0, 3), device=pp.device, dtype=torch.float32),
                torch.empty((0,), device=pp.device, dtype=torch.long),
                torch.empty((0,), device=pp.device, dtype=anchors.dtype),
            )
        del pixels_confs, idxs, offsets

    imgs_slices = []

    for img1, img2 in tmp_paths:
        try:
            pix1, confs1, slice1 = tmp_pixels[img1, img2]
            pix2, confs2, slice2 = tmp_pixels[img2, img1]
        except KeyError:
            continue
        img1 = imgs.index(img1)
        img2 = imgs.index(img2)
        confs = (confs1 * confs2).sqrt()

        imgs_slices.append(PairOfSlices(
            img1, slice1, pix1,
            img2, slice2, pix2,
            confs,
        ))

    imsizes = torch.tensor([(W, H) for H, W in shapes], device=pp.device)  # (W,H)
    principal_points = torch.stack(principal_points)
    focals = torch.cat(focals)

    # Reindex preds_21 in-place to the final anchor set instead of duplicating
    # the whole nested dictionary into another large CPU structure.
    gc.collect()

    return imsizes, principal_points, focals, core_depth, img_anchors, imgs_slices, preds_21


def canonical_view(ptmaps11, confs11, subsample, mode='avg-angle'):
    assert len(ptmaps11) == len(confs11) > 0, 'not a single view1 for img={i}'

    # canonical pointmap is just a weighted average
    confs11 = confs11.unsqueeze(-1) - 0.999
    canon = (confs11 * ptmaps11).sum(0) / confs11.sum(0)

    canon_depth = ptmaps11[..., 2].unsqueeze(1)
    S = slice(subsample // 2, None, subsample)
    center_depth = canon_depth[:, :, S, S]
    center_depth = torch.clip(center_depth, min=torch.finfo(center_depth.dtype).eps)

    stacked_depth = F.pixel_unshuffle(canon_depth, subsample)
    stacked_confs = F.pixel_unshuffle(confs11[:, None, :, :, 0], subsample)

    if mode == 'avg-reldepth':
        rel_depth = stacked_depth / center_depth
        stacked_canon = (stacked_confs * rel_depth).sum(dim=0) / stacked_confs.sum(dim=0)
        canon2 = F.pixel_shuffle(stacked_canon.unsqueeze(0), subsample).squeeze()

    elif mode == 'avg-angle':
        xy = ptmaps11[..., 0:2].permute(0, 3, 1, 2)
        stacked_xy = F.pixel_unshuffle(xy, subsample)
        B, _, H, W = stacked_xy.shape
        stacked_radius = (stacked_xy.view(B, 2, -1, H, W) - xy[:, :, None, S, S]).norm(dim=1)
        stacked_radius.clip_(min=1e-8)

        stacked_angle = torch.arctan((stacked_depth - center_depth) / stacked_radius)
        avg_angle = (stacked_confs * stacked_angle).sum(dim=0) / stacked_confs.sum(dim=0)

        # back to depth
        stacked_depth = stacked_radius.mean(dim=0) * torch.tan(avg_angle)

        canon2 = F.pixel_shuffle((1 + stacked_depth / canon[S, S, 2]).unsqueeze(0), subsample).squeeze()
    else:
        raise ValueError(f'bad {mode=}')

    confs = (confs11.square().sum(dim=0) / confs11.sum(dim=0)).squeeze()
    return canon, canon2, confs


def anchor_depth_offsets(canon_depth, pixels, subsample=8):
    device = canon_depth.device

    # create a 2D grid of anchor 3D points
    H1, W1 = canon_depth.shape
    yx = np.mgrid[subsample // 2:H1:subsample, subsample // 2:W1:subsample]
    H2, W2 = yx.shape[1:]
    cy, cx = yx.reshape(2, -1)
    core_depth = canon_depth[cy, cx]
    assert (core_depth > 0).all()

    # slave 3d points (attached to core 3d points)
    core_idxs = {}  # core_idxs[img2] = {corr_idx:core_idx}
    core_offs = {}  # core_offs[img2] = {corr_idx:3d_offset}

    for img2, (xy1, _confs) in pixels.items():
        px, py = xy1.long().T

        # find nearest anchor == block quantization
        core_idx = (py // subsample) * W2 + (px // subsample)
        core_idxs[img2] = core_idx.to(device)

        # compute relative depth offsets w.r.t. anchors
        ref_z = core_depth[core_idx]
        pts_z = canon_depth[py, px]
        offset = pts_z / ref_z
        core_offs[img2] = offset.detach().to(device)

    return core_idxs, core_offs


def spectral_clustering(graph, k=None, normalized_cuts=False):
    graph.fill_diagonal_(0)

    # graph laplacian
    degrees = graph.sum(dim=-1)
    laplacian = torch.diag(degrees) - graph
    if normalized_cuts:
        i_inv = torch.diag(degrees.sqrt().reciprocal())
        laplacian = i_inv @ laplacian @ i_inv

    # compute eigenvectors!
    eigval, eigvec = torch.linalg.eigh(laplacian)
    return eigval[:k], eigvec[:, :k]


def sim_func(p1, p2, gamma):
    diff = (p1 - p2).norm(dim=-1)
    avg_depth = (p1[:, :, 2] + p2[:, :, 2])
    rel_distance = diff / avg_depth
    sim = torch.exp(-gamma * rel_distance.square())
    return sim


def backproj(K, depthmap, subsample):
    H, W = depthmap.shape
    uv = np.mgrid[subsample // 2:subsample * W:subsample, subsample // 2:subsample * H:subsample].T.reshape(H, W, 2)
    xyz = depthmap.unsqueeze(-1) * geotrf(inv(K), todevice(uv, K.device), ncol=3)
    return xyz


def spectral_projection_depth(K, depthmap, subsample, k=64, cache_path='',
                              normalized_cuts=True, gamma=7, min_norm=5):
    try:
        if cache_path:
            cache_path = cache_path + f'_{k=}_norm={normalized_cuts}_{gamma=}.pth'
        lora_proj = _load_cache(cache_path, map_location=K.device)

    except IOError:
        # reconstruct 3d points in camera coordinates
        xyz = backproj(K, depthmap, subsample)

        # compute all distances
        xyz = xyz.reshape(-1, 3)
        graph = sim_func(xyz[:, None], xyz[None, :], gamma=gamma)
        _, lora_proj = spectral_clustering(graph, k, normalized_cuts=normalized_cuts)

        if cache_path:
            _save_atomic(lora_proj.cpu(), mkdir_for(cache_path))

    lora_proj, coeffs = lora_encode_normed(lora_proj, depthmap.ravel(), min_norm=min_norm)

    # depthmap ~= lora_proj @ coeffs
    return coeffs, lora_proj


def lora_encode_normed(lora_proj, x, min_norm, global_norm=False):
    # encode the pointmap
    coeffs = torch.linalg.pinv(lora_proj) @ x

    # rectify the norm of basis vector to be ~ equal
    if coeffs.ndim == 1:
        coeffs = coeffs[:, None]
    if global_norm:
        lora_proj *= coeffs[1:].norm() * min_norm / coeffs.shape[1]
    elif min_norm:
        lora_proj *= coeffs.norm(dim=1).clip(min=min_norm)
    # can have rounding errors here!
    coeffs = (torch.linalg.pinv(lora_proj.double()) @ x.double()).float()

    return lora_proj.detach(), coeffs.detach()


@torch.no_grad()
def spectral_projection_of_depthmaps(imgs, intrinsics, depthmaps, subsample, cache_path=None, **kw):
    # recover 3d points
    core_depth = []
    lora_proj = []

    for i, img in enumerate(tqdm(imgs)):
        cache = os.path.join(cache_path, 'lora_depth', hash_md5(img)) if cache_path else None
        depth, proj = spectral_projection_depth(intrinsics[i], depthmaps[i], subsample,
                                                cache_path=cache, **kw)
        core_depth.append(depth)
        lora_proj.append(proj)

    return core_depth, lora_proj


def reproj2d(Trf, pts3d):
    res = (pts3d @ Trf[:3, :3].transpose(-1, -2)) + Trf[:3, 3]
    clipped_z = res[:, 2:3].clip(min=1e-3)  # make sure we don't have nans!
    uv = res[:, 0:2] / clipped_z
    return uv.clip(min=-1000, max=2000)


def bfs(tree, start_node):
    order, predecessors = sp.csgraph.breadth_first_order(tree, start_node, directed=False)
    ranks = np.arange(len(order))
    ranks[order] = ranks.copy()
    return ranks, predecessors


def compute_min_spanning_tree(pws):
    sparse_graph = sp.dok_array(pws.shape)
    for i, j in pws.nonzero().cpu().tolist():
        sparse_graph[i, j] = -float(pws[i, j])
    msp = sp.csgraph.minimum_spanning_tree(sparse_graph)

    # now reorder the oriented edges, starting from the central point
    ranks1, _ = bfs(msp, 0)
    ranks2, _ = bfs(msp, ranks1.argmax())
    ranks1, _ = bfs(msp, ranks2.argmax())
    # this is the point farther from any leaf
    root = np.minimum(ranks1, ranks2).argmax()

    # find the ordered list of edges that describe the tree
    order, predecessors = sp.csgraph.breadth_first_order(msp, root, directed=False)
    order = order[1:]  # root not do not have a predecessor
    edges = [(predecessors[i], i) for i in order]

    return root, edges


def show_reconstruction(shapes_or_imgs, K, cam2w, pts3d, gt_cam2w=None, gt_K=None, cam_size=None, masks=None, **kw):
    viz = SceneViz()

    cc = cam2w[:, :3, 3]
    cs = cam_size or float(torch.cdist(cc, cc).fill_diagonal_(np.inf).min(dim=0).values.median())
    colors = 64 + np.random.randint(255 - 64, size=(len(cam2w), 3))

    if isinstance(shapes_or_imgs, np.ndarray) and shapes_or_imgs.ndim == 2:
        cam_kws = dict(imsizes=shapes_or_imgs[:, ::-1], cam_size=cs)
    else:
        imgs = shapes_or_imgs
        cam_kws = dict(images=imgs, cam_size=cs)
    if K is not None:
        viz.add_cameras(to_numpy(cam2w), to_numpy(K), colors=colors, **cam_kws)

    if gt_cam2w is not None:
        if gt_K is None:
            gt_K = K
        viz.add_cameras(to_numpy(gt_cam2w), to_numpy(gt_K), colors=colors, marker='o', **cam_kws)

    if pts3d is not None:
        for i, p in enumerate(pts3d):
            if not len(p):
                continue
            if masks is None:
                viz.add_pointcloud(to_numpy(p), color=tuple(colors[i].tolist()))
            else:
                viz.add_pointcloud(to_numpy(p), mask=masks[i], color=imgs[i])
    viz.show(**kw)
