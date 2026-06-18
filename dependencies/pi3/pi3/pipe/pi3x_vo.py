from ..utils.geometry import homogenize_points, depth_edge
import torch
import torch.nn.functional as F


class Pi3XVO:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def __call__(
        self,
        imgs,
        chunk_size=16,
        overlap=6,
        conf_thre=0.05,
        inject_condition=None,
        dtype=torch.bfloat16,
        align_mode="se3",
        anchor_indices=None,
        intrinsics=None,
    ):
        """
        inject_condition: list of strings, e.g. ['pose', 'depth']
        """
        if inject_condition is None:
            inject_condition = []
            
        B, T, C, H, W = imgs.shape
        print(
            f"[PiXVO] Total frames: {T}, Chunk size: {chunk_size}, "
            f"Overlap: {overlap}, Align: {align_mode}"
        )

        if anchor_indices is not None:
            anchor_indices = sorted({
                int(idx) for idx in anchor_indices if 0 <= int(idx) < T
            })
            if len(anchor_indices) >= 2 and len(anchor_indices) < min(chunk_size, T):
                print(
                    "[PiXVO] Anchor-aligned chunking enabled: "
                    f"anchors={anchor_indices}"
                )
                return self._run_anchor_aligned(
                    imgs=imgs,
                    chunk_size=chunk_size,
                    conf_thre=conf_thre,
                    dtype=dtype,
                    align_mode=align_mode,
                    anchor_indices=anchor_indices,
                    intrinsics=intrinsics,
                )

        merged_points, merged_poses, merged_confs, merged_local_points = [], [], [], []
        
        prev_global_pts_overlap, prev_global_mask_overlap = None, None
        prev_aligned_poses_overlap = None
        prev_local_depth_overlap, prev_local_conf_overlap = None, None

        for start_idx in range(0, T, chunk_size - overlap):
            end_idx = min(start_idx + chunk_size, T)
            chunk_imgs = imgs[:, start_idx:end_idx]
            current_len = end_idx - start_idx

            print(f"  > Inference chunk: [{start_idx} : {end_idx}] (Length: {current_len})")
            
            if current_len <= overlap and start_idx > 0:
                break

            model_kwargs = {'with_prior': False}
            if intrinsics is not None:
                model_kwargs['intrinsics'] = intrinsics[:, start_idx:end_idx]
                model_kwargs['with_prior'] = True
            
            if start_idx > 0:
                if 'pose' in inject_condition and prev_aligned_poses_overlap is not None:
                    prior_poses = torch.eye(4, device=imgs.device).repeat(B, current_len, 1, 1)
                    prior_poses[:, :overlap] = prev_aligned_poses_overlap
                    
                    mask_pose = torch.zeros((B, current_len), dtype=torch.bool, device=imgs.device)
                    mask_pose[:, :overlap] = True
                    
                    model_kwargs['poses'] = prior_poses
                    model_kwargs['mask_add_pose'] = mask_pose
                    model_kwargs['with_prior'] = True

                if 'depth' in inject_condition and prev_local_depth_overlap is not None:
                    prior_depths = torch.zeros((B, current_len, H, W), device=imgs.device)
                    prior_depths[:, :overlap] = prev_local_depth_overlap
                    
                    mask_depth = torch.zeros((B, current_len), dtype=torch.bool, device=imgs.device)
                    mask_depth[:, :overlap] = True
                    
                    if prev_local_conf_overlap is not None:
                        valid_mask = prev_local_conf_overlap > conf_thre
                        prior_depths[:, :overlap][~valid_mask] = 0

                    model_kwargs['depths'] = prior_depths
                    model_kwargs['mask_add_depth'] = mask_depth
                    model_kwargs['with_prior'] = True

                if ('ray' in inject_condition or 'intrinsic' in inject_condition) and prev_local_depth_overlap is not None:
                    prior_rays = torch.zeros((B, current_len, H, W, 3), device=imgs.device)
                    prior_rays[:, :overlap] = prev_rays_overlap
                    
                    mask_ray = torch.zeros((B, current_len), dtype=torch.bool, device=imgs.device)
                    mask_ray[:, :overlap] = True
                    
                    model_kwargs['rays'] = prior_rays
                    model_kwargs['mask_add_ray'] = mask_ray
                    model_kwargs['with_prior'] = True

            with torch.amp.autocast('cuda', dtype=dtype):
                pred = self.model(chunk_imgs, **model_kwargs)
            
            curr_local_depth = pred['local_points'][..., 2] 
            curr_local_points = pred['local_points']
            curr_pts = pred['points']
            curr_poses = pred['camera_poses']
            curr_conf = torch.sigmoid(pred['conf'])[..., 0]
            curr_rays = pred['rays']
            
            edge = depth_edge(curr_local_depth, rtol=0.03)
            curr_conf[edge] = 0

            curr_mask = curr_conf > conf_thre
            
            if curr_mask.sum() < 10:
                flat_conf = curr_conf.view(B, current_len, -1)
                k = int(flat_conf.shape[-1] * 0.1)
                topk_vals, _ = torch.topk(flat_conf, k, dim=-1)
                min_vals = topk_vals[..., -1].unsqueeze(-1).unsqueeze(-1)
                curr_mask = curr_conf >= min_vals

            if start_idx == 0:
                aligned_pts = curr_pts
                aligned_poses = curr_poses
                aligned_local_points = curr_local_points
            else:
                src_pts = curr_pts[:, :overlap]
                src_mask = curr_mask[:, :overlap]
                tgt_pts = prev_global_pts_overlap
                tgt_mask = prev_global_mask_overlap
                
                transform_matrix = self._compute_alignment_umeyama_masked(
                    src_pts,
                    tgt_pts,
                    src_mask,
                    tgt_mask,
                    align_mode=align_mode,
                )
                self._log_alignment_summary(
                    prefix=f"chain start={start_idx}",
                    src_points=src_pts,
                    tgt_points=tgt_pts,
                    src_mask=src_mask,
                    tgt_mask=tgt_mask,
                    sim3=transform_matrix,
                )
                
                aligned_pts = self._apply_sim3_to_points(curr_pts, transform_matrix)
                aligned_poses = self._apply_sim3_to_poses(curr_poses, transform_matrix)
                aligned_local_points = self._apply_sim3_to_local_points(
                    curr_local_points, transform_matrix
                )

            if start_idx == 0:
                merged_points.append(aligned_pts)
                merged_poses.append(aligned_poses)
                merged_confs.append(curr_conf)
                merged_local_points.append(aligned_local_points)
            else:
                merged_points.append(aligned_pts[:, overlap:])
                merged_poses.append(aligned_poses[:, overlap:])
                merged_confs.append(curr_conf[:, overlap:])
                merged_local_points.append(aligned_local_points[:, overlap:])
            
            prev_global_pts_overlap = aligned_pts[:, -overlap:]
            prev_global_mask_overlap = curr_mask[:, -overlap:]

            prev_aligned_poses_overlap = aligned_poses[:, -overlap:]
            prev_local_depth_overlap = aligned_local_points[:, -overlap:, ..., 2]
            prev_local_conf_overlap = curr_conf[:, -overlap:]
            prev_rays_overlap = curr_rays[:, -overlap:]
            
            del pred, curr_pts, curr_poses, curr_mask, curr_local_depth, curr_local_points, curr_conf, curr_rays
            if 'poses' in model_kwargs: del model_kwargs['poses']
            if 'depths' in model_kwargs: del model_kwargs['depths']
            if 'rays' in model_kwargs: del model_kwargs['rays']
            torch.cuda.empty_cache()

            if end_idx == T:
                break
        
        return {
            'points': torch.cat(merged_points, dim=1),
            'camera_poses': torch.cat(merged_poses, dim=1),
            'conf': torch.cat(merged_confs, dim=1),
            'local_points': torch.cat(merged_local_points, dim=1),
        }

    def _predict_geometry(self, chunk_imgs, model_kwargs, dtype, conf_thre):
        with torch.amp.autocast('cuda', dtype=dtype):
            pred = self.model(chunk_imgs, **model_kwargs)

        local_depth = pred['local_points'][..., 2]
        local_points = pred['local_points']
        points = pred['points']
        poses = pred['camera_poses']
        conf = torch.sigmoid(pred['conf'])[..., 0]
        rays = pred['rays']

        edge = depth_edge(local_depth, rtol=0.03)
        conf[edge] = 0

        mask = conf > conf_thre
        if mask.sum() < 10:
            B, current_len = conf.shape[:2]
            flat_conf = conf.view(B, current_len, -1)
            k = int(flat_conf.shape[-1] * 0.1)
            topk_vals, _ = torch.topk(flat_conf, k, dim=-1)
            min_vals = topk_vals[..., -1].unsqueeze(-1).unsqueeze(-1)
            mask = conf >= min_vals

        return points, poses, conf, local_points, local_depth, rays, mask

    def _run_anchor_aligned(
        self,
        imgs,
        chunk_size,
        conf_thre,
        dtype,
        align_mode,
        anchor_indices,
        intrinsics=None,
    ):
        B, T, C, H, W = imgs.shape
        device = imgs.device
        anchor_count = len(anchor_indices)
        block_capacity = max(1, chunk_size - anchor_count)
        anchor_set = set(anchor_indices)

        frame_points = [None] * T
        frame_poses = [None] * T
        frame_confs = [None] * T
        frame_local_points = [None] * T

        anchor_tensor = torch.as_tensor(anchor_indices, device=device, dtype=torch.long)
        print(
            "  > Anchor reference chunk: "
            f"{anchor_indices} (Length: {anchor_count})"
        )
        ref_imgs = imgs.index_select(1, anchor_tensor)
        ref_kwargs = {'with_prior': False}
        if intrinsics is not None:
            ref_kwargs['intrinsics'] = intrinsics.index_select(1, anchor_tensor)
            ref_kwargs['with_prior'] = True
        (
            ref_points,
            ref_poses,
            ref_confs,
            ref_local_points,
            _ref_local_depth,
            _ref_rays,
            ref_mask,
        ) = self._predict_geometry(ref_imgs, ref_kwargs, dtype, conf_thre)

        for pos, frame_idx in enumerate(anchor_indices):
            frame_points[frame_idx] = ref_points[:, pos]
            frame_poses[frame_idx] = ref_poses[:, pos]
            frame_confs[frame_idx] = ref_confs[:, pos]
            frame_local_points[frame_idx] = ref_local_points[:, pos]

        non_anchor_indices = [idx for idx in range(T) if idx not in anchor_set]
        for block_start in range(0, len(non_anchor_indices), block_capacity):
            block_indices = non_anchor_indices[block_start:block_start + block_capacity]
            chunk_indices = sorted(anchor_indices + block_indices)
            anchor_positions = [chunk_indices.index(idx) for idx in anchor_indices]
            chunk_tensor = torch.as_tensor(chunk_indices, device=device, dtype=torch.long)
            anchor_pos_tensor = torch.as_tensor(
                anchor_positions, device=device, dtype=torch.long
            )
            print(
                "  > Anchor chunk: "
                f"anchors={anchor_count} frames={block_indices[0]}:{block_indices[-1] + 1} "
                f"(Length: {len(chunk_indices)})"
            )

            chunk_imgs = imgs.index_select(1, chunk_tensor)
            chunk_kwargs = {'with_prior': False}
            if intrinsics is not None:
                chunk_kwargs['intrinsics'] = intrinsics.index_select(1, chunk_tensor)
                chunk_kwargs['with_prior'] = True
            (
                curr_pts,
                curr_poses,
                curr_conf,
                curr_local_points,
                _curr_local_depth,
                _curr_rays,
                curr_mask,
            ) = self._predict_geometry(chunk_imgs, chunk_kwargs, dtype, conf_thre)

            curr_anchor_pts = curr_pts.index_select(1, anchor_pos_tensor)
            curr_anchor_mask = curr_mask.index_select(1, anchor_pos_tensor)
            transform_matrix = self._compute_alignment_umeyama_masked(
                curr_anchor_pts,
                ref_points,
                curr_anchor_mask,
                ref_mask,
                align_mode=align_mode,
            )
            self._log_alignment_summary(
                prefix=f"anchor block={block_indices[0]}:{block_indices[-1] + 1}",
                src_points=curr_anchor_pts,
                tgt_points=ref_points,
                src_mask=curr_anchor_mask,
                tgt_mask=ref_mask,
                sim3=transform_matrix,
            )

            aligned_pts = self._apply_sim3_to_points(curr_pts, transform_matrix)
            aligned_poses = self._apply_sim3_to_poses(curr_poses, transform_matrix)
            aligned_local_points = self._apply_sim3_to_local_points(
                curr_local_points, transform_matrix
            )

            for pos, frame_idx in enumerate(chunk_indices):
                if frame_idx in anchor_set:
                    continue
                frame_points[frame_idx] = aligned_pts[:, pos]
                frame_poses[frame_idx] = aligned_poses[:, pos]
                frame_confs[frame_idx] = curr_conf[:, pos]
                frame_local_points[frame_idx] = aligned_local_points[:, pos]

            del curr_pts, curr_poses, curr_conf, curr_local_points, curr_mask
            torch.cuda.empty_cache()

        missing = [idx for idx, value in enumerate(frame_points) if value is None]
        if missing:
            raise RuntimeError(f"Pi3XVO anchor mode missed frames: {missing[:20]}")

        return {
            'points': torch.stack(frame_points, dim=1),
            'camera_poses': torch.stack(frame_poses, dim=1),
            'conf': torch.stack(frame_confs, dim=1),
            'local_points': torch.stack(frame_local_points, dim=1),
        }

    def _compute_alignment_umeyama_masked(
        self,
        src_points,
        tgt_points,
        src_mask,
        tgt_mask,
        align_mode="se3",
    ):
        B = src_points.shape[0]
        device = src_points.device
        
        src = src_points.reshape(B, -1, 3)
        tgt = tgt_points.reshape(B, -1, 3)
        
        mask = (src_mask.reshape(B, -1) & tgt_mask.reshape(B, -1)).float().unsqueeze(-1)
        valid_cnt = mask.sum(dim=1).squeeze(-1)
        eps = 1e-6
        
        bad_mask = valid_cnt < 10
        if bad_mask.all():
            return torch.eye(4, device=device).repeat(B, 1, 1)

        src_mean = (src * mask).sum(dim=1, keepdim=True) / (valid_cnt.view(B, 1, 1) + eps)
        tgt_mean = (tgt * mask).sum(dim=1, keepdim=True) / (valid_cnt.view(B, 1, 1) + eps)
        
        src_centered = (src - src_mean) * mask
        tgt_centered = (tgt - tgt_mean) * mask
        
        H = torch.bmm(src_centered.transpose(1, 2), tgt_centered)
        U, S, V = torch.svd(H)
        
        R = torch.bmm(V, U.transpose(1, 2))
        
        det = torch.det(R)
        diag = torch.ones(B, 3, device=device)
        diag[:, 2] = torch.sign(det)
        R = torch.bmm(torch.bmm(V, torch.diag_embed(diag)), U.transpose(1, 2))
        
        src_var = (src_centered ** 2).sum(dim=2) * mask.squeeze(-1)
        src_var = src_var.sum(dim=1) / (valid_cnt + eps)
        
        corrected_S = S.clone()
        corrected_S[:, 2] *= diag[:, 2]
        trace_S = corrected_S.sum(dim=1)
        
        if align_mode == "sim3":
            scale = trace_S / (src_var * valid_cnt + eps)
        else:
            # Pi3X already predicts approximately metric geometry. Allowing an
            # independent scale per chunk caused large piecewise scale jumps in
            # long sequences, which then fractured the downstream point cloud.
            # For VO stitching we therefore default to rigid alignment.
            scale = torch.ones_like(trace_S)
        scale = scale.view(B, 1, 1)
        
        t = tgt_mean.transpose(1, 2) - scale * torch.bmm(R, src_mean.transpose(1, 2))
        
        sim3 = torch.eye(4, device=device).repeat(B, 1, 1)
        sim3[:, :3, :3] = scale * R
        sim3[:, :3, 3] = t.squeeze(2)
        
        if bad_mask.any():
            identity = torch.eye(4, device=device).repeat(B, 1, 1)
            sim3[bad_mask] = identity[bad_mask]
            
        return sim3
    
    def _apply_sim3_to_points(self, points, sim3):
        B, T, H, W, C = points.shape
        flat_pts = points.reshape(B, -1, 3)
        R_s = sim3[:, :3, :3]
        t = sim3[:, :3, 3].unsqueeze(1)
        out_pts = torch.bmm(flat_pts, R_s.transpose(1, 2)) + t
        return out_pts.reshape(B, T, H, W, 3)

    def _apply_sim3_to_poses(self, poses, sim3):
        A = sim3[:, :3, :3]
        scale = torch.linalg.norm(A, dim=1).mean(dim=1).clamp_min(1e-8)
        R_align = A / scale.view(-1, 1, 1)
        t_align = sim3[:, :3, 3]

        out_poses = poses.clone()
        out_poses[..., :3, :3] = torch.matmul(
            R_align.unsqueeze(1), poses[..., :3, :3]
        )
        out_poses[..., :3, 3] = (
            scale.view(-1, 1, 1)
            * torch.matmul(
                R_align.unsqueeze(1), poses[..., :3, 3].unsqueeze(-1)
            ).squeeze(-1)
            + t_align.unsqueeze(1)
        )
        return out_poses

    def _apply_sim3_to_local_points(self, local_points, sim3):
        A = sim3[:, :3, :3]
        scale = torch.linalg.norm(A, dim=1).mean(dim=1).clamp_min(1e-8)
        return local_points * scale.view(-1, 1, 1, 1, 1)

    def _log_alignment_summary(
        self,
        prefix,
        src_points,
        tgt_points,
        src_mask,
        tgt_mask,
        sim3,
    ):
        with torch.no_grad():
            valid = (src_mask.reshape(src_mask.shape[0], -1)
                     & tgt_mask.reshape(tgt_mask.shape[0], -1))
            valid_count = int(valid.sum().item())
            A = sim3[:, :3, :3]
            scale = torch.linalg.norm(A, dim=1).mean(dim=1)
            if valid_count <= 0:
                print(
                    f"[PiXVO] align {prefix}: valid=0 "
                    f"scale={float(scale.median().item()):.4f}"
                )
                return

            aligned_src = self._apply_sim3_to_points(src_points, sim3)
            residual = torch.linalg.norm(
                aligned_src.reshape(src_points.shape[0], -1, 3)
                - tgt_points.reshape(tgt_points.shape[0], -1, 3),
                dim=-1,
            )[valid]
            if residual.numel() == 0:
                print(
                    f"[PiXVO] align {prefix}: valid={valid_count} "
                    f"scale={float(scale.median().item()):.4f}"
                )
                return
            residual = residual.float()
            med = torch.quantile(residual, 0.50)
            p95 = torch.quantile(residual, 0.95)
            print(
                f"[PiXVO] align {prefix}: valid={valid_count} "
                f"scale={float(scale.median().item()):.4f} "
                f"resid_med={float(med.item()):.4f} "
                f"resid_p95={float(p95.item()):.4f}"
            )
