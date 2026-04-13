import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SEG_ROOT = REPO_ROOT / "preprocessing" / "segmentation"
if str(SEG_ROOT) not in sys.path:
    sys.path.insert(0, str(SEG_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from keyframe_selection import (
    _assign_final_selection_roles,
    _force_fill_entries,
    _build_rejected_rescue_set,
    _entries_respect_similarity,
    _final_polish_selected_set,
    _temporal_guard_entries,
    optimize_keyframe_set,
    _assign_view_families,
    _quality_swap_preserves_diversity,
)
from utils.io_utils import _quality_global_candidates, _temporal_front_guard


def make_entry(idx, base, coverage, focus, descriptor, context=1.0, family_descriptor=None):
    coverage = np.asarray(coverage, dtype=np.float32)
    focus = np.asarray(focus, dtype=np.float32)
    descriptor = np.asarray(descriptor, dtype=np.float32)
    descriptor = descriptor / max(float(np.linalg.norm(descriptor)), 1e-6)
    if family_descriptor is None:
        family_descriptor = descriptor
    family_descriptor = np.asarray(family_descriptor, dtype=np.float32)
    family_descriptor = family_descriptor / max(float(np.linalg.norm(family_descriptor)), 1e-6)
    return {
        "idx": int(idx),
        "global_base_score": float(base),
        "stats": {
            "context_score": float(context),
            "coverage_vec": coverage,
            "focus_vec": focus,
            "grid_vec": (coverage > 0.1).astype(np.float32),
            "descriptor": descriptor,
            "family_descriptor": family_descriptor,
            "useful_masks": int((coverage > 0.1).sum()),
            "grid_cells": int((coverage > 0.1).sum()),
            "frame_quality": 1.0,
            "utility_score": float(base),
            "useful_union_ratio": float(coverage.mean()),
            "texture_penalty": 0.0,
            "overlap_penalty": 0.0,
            "concentration_penalty": 0.0,
        },
    }


class KeyframeSelectionTests(unittest.TestCase):
    def test_hard_gap_is_respected(self):
        entries = [
            make_entry(0, 0.95, [1, 0, 0, 0], [1, 0, 0], [1, 0, 0, 0], context=1.0),
            make_entry(20, 0.96, [1, 0, 0, 0], [1, 0, 0], [0.99, 0.01, 0, 0], context=1.0),
            make_entry(100, 0.80, [0, 1, 0, 0], [0, 1, 0], [0, 1, 0, 0], context=0.9),
            make_entry(180, 0.78, [0, 0, 1, 0], [0, 0, 1], [0, 0, 1, 0], context=0.85),
        ]
        selected, diag = optimize_keyframe_set(
            all_entries=entries,
            target_count=3,
            total_frames=600,
            min_frame_gap=56,
            diversity_weight=0.95,
            similarity_cap=0.90,
        )
        selected_ids = [entry["idx"] for entry in selected]
        self.assertEqual(sum(idx in selected_ids for idx in [0, 20]), 1)
        self.assertGreaterEqual(min(np.diff(sorted(selected_ids))), 56)
        self.assertGreaterEqual(diag["active_gap"], 40)

    def test_diverse_coverage_beats_redundant_high_score_frames(self):
        entries = [
            make_entry(0, 0.98, [1, 1, 0, 0], [1, 0, 0], [1.0, 0.0, 0.0, 0.0], context=1.0),
            make_entry(70, 0.97, [1, 1, 0, 0], [1, 0, 0], [0.995, 0.005, 0.0, 0.0], context=1.0),
            make_entry(150, 0.78, [0, 0, 1, 1], [0, 1, 0], [0.0, 1.0, 0.0, 0.0], context=0.95),
            make_entry(230, 0.76, [0, 1, 1, 0], [0, 0, 1], [0.0, 0.0, 1.0, 0.0], context=0.92),
        ]
        selected, _ = optimize_keyframe_set(
            all_entries=entries,
            target_count=3,
            total_frames=600,
            min_frame_gap=56,
            diversity_weight=0.95,
            similarity_cap=0.90,
        )
        selected_ids = sorted(entry["idx"] for entry in selected)
        self.assertIn(0, selected_ids)
        self.assertNotIn(70, selected_ids)
        self.assertIn(150, selected_ids)
        self.assertIn(230, selected_ids)

    def test_stage1_candidate_pool_keeps_multiple_descriptor_clusters(self):
        scores = np.array([
            0.95, 0.94, 0.93, 0.92,
            0.70, 0.69, 0.68, 0.67,
            0.66, 0.65, 0.64, 0.63,
        ], dtype=np.float32)
        descriptors = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.99, 0.01, 0.0, 0.0],
            [0.98, 0.02, 0.0, 0.0],
            [0.97, 0.03, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.99, 0.01, 0.0],
            [0.0, 0.98, 0.02, 0.0],
            [0.0, 0.97, 0.03, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.01, 0.0, 0.99, 0.0],
            [0.02, 0.0, 0.98, 0.0],
            [0.03, 0.0, 0.97, 0.0],
        ], dtype=np.float32)
        descriptors = descriptors / np.linalg.norm(descriptors, axis=1, keepdims=True)
        pool = _quality_global_candidates(scores, descriptors, n_keyframes=4, top_k=2)
        self.assertTrue(any(idx < 4 for idx in pool))
        self.assertTrue(any(4 <= idx < 8 for idx in pool))
        self.assertTrue(any(8 <= idx < 12 for idx in pool))

    def test_stage1_front_of_pool_is_diverse_not_same_cluster(self):
        scores = np.array([
            0.99, 0.98, 0.97, 0.96, 0.95, 0.94,
            0.80, 0.79, 0.78, 0.77,
            0.76, 0.75, 0.74, 0.73,
        ], dtype=np.float32)
        descriptors = np.array([
            [1.00, 0.00, 0.00, 0.00],
            [0.99, 0.01, 0.00, 0.00],
            [0.98, 0.02, 0.00, 0.00],
            [0.97, 0.03, 0.00, 0.00],
            [0.96, 0.04, 0.00, 0.00],
            [0.95, 0.05, 0.00, 0.00],
            [0.00, 1.00, 0.00, 0.00],
            [0.00, 0.99, 0.01, 0.00],
            [0.00, 0.98, 0.02, 0.00],
            [0.00, 0.97, 0.03, 0.00],
            [0.00, 0.00, 1.00, 0.00],
            [0.01, 0.00, 0.99, 0.00],
            [0.02, 0.00, 0.98, 0.00],
            [0.03, 0.00, 0.97, 0.00],
        ], dtype=np.float32)
        descriptors = descriptors / np.linalg.norm(descriptors, axis=1, keepdims=True)
        pool = _quality_global_candidates(scores, descriptors, n_keyframes=6, top_k=4)
        front = pool[:6]
        front3 = pool[:3]
        self.assertTrue(any(idx < 6 for idx in front))
        self.assertTrue(any(6 <= idx < 10 for idx in front))
        self.assertTrue(any(10 <= idx < 14 for idx in front))
        self.assertTrue(any(idx < 6 for idx in front3))
        self.assertTrue(any(6 <= idx < 10 for idx in front3))
        self.assertTrue(any(10 <= idx < 14 for idx in front3))

    def test_temporal_front_guard_spreads_close_indices(self):
        ordered = [87, 88, 89, 120, 121, 150, 151, 180, 181, 210]
        guarded = _temporal_front_guard(ordered, front_target=5, min_gap=20)
        front = guarded[:5]
        self.assertEqual(front, [87, 120, 150, 180, 210])

    def test_temporal_guard_entries_spreads_rejected_front(self):
        entries = [
            make_entry(87, 0.99, [1, 0, 0, 0], [1, 0, 0], [1, 0, 0, 0], context=11.2),
            make_entry(88, 0.98, [0, 1, 0, 0], [0, 1, 0], [0, 1, 0, 0], context=11.1),
            make_entry(92, 0.97, [0, 0, 1, 0], [0, 0, 1], [0, 0, 1, 0], context=11.0),
            make_entry(140, 0.96, [0, 0, 0, 1], [0, 0, 0.5], [0, 0, 0, 1], context=10.9),
            make_entry(200, 0.95, [1, 1, 0, 0], [1, 0, 0], [0.7, 0.7, 0, 0], context=10.8),
        ]
        for entry in entries:
            entry["support_score"] = entry["global_base_score"]
            entry["overview_score"] = entry["global_base_score"]
        guarded = _temporal_guard_entries(entries, front_target=3, min_gap=20, score_key="support_score")
        self.assertEqual([entry["idx"] for entry in guarded[:3]], [87, 140, 200])

    def test_view_family_assignment_splits_transitive_chain(self):
        entries = [
            make_entry(0, 0.95, [1, 1, 0, 0], [1, 0, 0], [1, 0, 0, 0], context=1.0, family_descriptor=[1.0, 0.0, 0.0, 0.0]),
            make_entry(60, 0.94, [1, 0.8, 0.2, 0.0], [1, 0, 0], [0.9, 0.3, 0.0, 0.0], context=0.98, family_descriptor=[0.93, 0.37, 0.0, 0.0]),
            make_entry(120, 0.93, [0.8, 0.2, 0.7, 0.0], [0.6, 0.4, 0.0], [0.8, 0.5, 0.0, 0.0], context=0.96, family_descriptor=[0.78, 0.63, 0.0, 0.0]),
            make_entry(240, 0.92, [0, 0, 1, 1], [0, 1, 0], [0.0, 1.0, 0.0, 0.0], context=0.95, family_descriptor=[0.0, 1.0, 0.0, 0.0]),
        ]
        for entry in entries:
            entry["overview_score"] = entry["global_base_score"]
            entry["support_score"] = entry["global_base_score"]
        families = _assign_view_families(entries, family_similarity_threshold=0.84)
        self.assertGreaterEqual(len(families), 3)
        family_by_idx = {entry["idx"]: entry["view_family"] for entry in entries}
        self.assertEqual(family_by_idx[0], family_by_idx[60])
        self.assertNotEqual(family_by_idx[0], family_by_idx[120])
        self.assertNotEqual(family_by_idx[120], family_by_idx[240])

    def test_cross_family_similarity_is_relaxed_for_different_families(self):
        entries = [
            make_entry(0, 0.99, [1, 1, 0, 0], [1, 0, 0], [1.0, 0.0, 0.0, 0.0], context=11.8, family_descriptor=[1.0, 0.0, 0.0, 0.0]),
            make_entry(70, 0.98, [1, 0.7, 0.3, 0.0], [1, 0, 0], [0.98, 0.20, 0.0, 0.0], context=11.5, family_descriptor=[0.0, 1.0, 0.0, 0.0]),
            make_entry(140, 0.97, [0.0, 1, 1, 0.0], [0, 1, 0], [0.97, 0.24, 0.0, 0.0], context=11.3, family_descriptor=[0.0, 0.0, 1.0, 0.0]),
            make_entry(210, 0.96, [0.0, 0.0, 1, 1], [0, 0, 1], [0.96, 0.28, 0.0, 0.0], context=11.1, family_descriptor=[0.0, 0.0, 0.0, 1.0]),
        ]
        for entry in entries:
            entry["overview_score"] = entry["global_base_score"]
            entry["support_score"] = entry["global_base_score"]
        _assign_view_families(entries, family_similarity_threshold=0.88)
        self.assertTrue(_entries_respect_similarity(entries, similarity_cap=0.90))

    def test_stage1_does_not_promote_low_quality_novel_frames_into_front(self):
        scores = np.array([0.98, 0.97, 0.96, 0.95, 0.15, 0.14, 0.13, 0.12], dtype=np.float32)
        descriptors = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        pool = _quality_global_candidates(scores, descriptors, n_keyframes=4, top_k=2)
        self.assertEqual(set(pool[:4]), {0, 1, 2, 3})

    def test_low_quality_complement_does_not_replace_good_overview(self):
        entries = [
            make_entry(0, 0.95, [1, 1, 0, 0], [1, 0, 0], [1, 0, 0, 0], context=11.7),
            make_entry(90, 0.93, [0, 1, 1, 0], [0, 1, 0], [0, 1, 0, 0], context=11.3),
            make_entry(180, 0.92, [0, 0, 1, 1], [0, 0, 1], [0, 0, 1, 0], context=11.1),
            make_entry(270, 0.91, [1, 0, 0, 1], [0, 0, 0.5], [0, 0, 0, 1], context=10.9),
            make_entry(135, 0.75, [0, 0, 0, 1], [0, 0, 0.2], [0.7, 0.7, 0, 0], context=8.4),
        ]
        weak = entries[-1]
        weak["stats"]["grid_cells"] = 2
        weak["stats"]["useful_union_ratio"] = 0.08
        weak["stats"]["texture_penalty"] = 0.12
        selected, _ = optimize_keyframe_set(
            all_entries=entries,
            target_count=4,
            total_frames=600,
            min_frame_gap=56,
            diversity_weight=0.95,
            similarity_cap=0.90,
        )
        selected_ids = sorted(entry["idx"] for entry in selected)
        self.assertNotIn(135, selected_ids)
        self.assertGreaterEqual(len(selected_ids), 3)
        self.assertTrue(set(selected_ids).issubset({0, 90, 180, 270}))

    def test_view_family_cap_blocks_same_corner_overselection(self):
        entries = [
            make_entry(0, 0.98, [1, 1, 0, 0], [1, 0, 0], [1.0, 0.0, 0.0, 0.0], context=1.00),
            make_entry(70, 0.97, [1, 1, 0, 0], [1, 0, 0], [0.99, 0.01, 0.0, 0.0], context=0.99),
            make_entry(140, 0.96, [1, 0.9, 0, 0], [1, 0, 0], [0.98, 0.02, 0.0, 0.0], context=0.98),
            make_entry(210, 0.95, [0, 1, 1, 0], [0, 1, 0], [0.0, 1.0, 0.0, 0.0], context=0.97),
            make_entry(280, 0.94, [0, 0, 1, 1], [0, 0, 1], [0.0, 0.0, 1.0, 0.0], context=0.96),
            make_entry(350, 0.93, [1, 0, 0, 1], [0, 0, 0.5], [0.0, 0.0, 0.0, 1.0], context=0.95),
            make_entry(420, 0.92, [0.5, 0, 0.8, 0.2], [0, 0.5, 0.5], [0.0, 0.7, 0.7, 0.0], context=0.94),
            make_entry(490, 0.91, [0.2, 0.8, 0, 0.5], [0.5, 0.5, 0], [0.7, 0.0, 0.7, 0.0], context=0.93),
        ]
        selected, diag = optimize_keyframe_set(
            all_entries=entries,
            target_count=6,
            total_frames=600,
            min_frame_gap=56,
            diversity_weight=0.95,
            similarity_cap=0.90,
        )
        selected_ids = sorted(entry["idx"] for entry in selected)
        self.assertLessEqual(sum(idx in selected_ids for idx in [0, 70, 140]), 2)
        self.assertGreaterEqual(diag["view_family_count"], 4)
        family_counts = {}
        for entry in selected:
            family_id = entry["view_family"]
            family_counts[family_id] = family_counts.get(family_id, 0) + 1
        self.assertLessEqual(max(family_counts.values()), 2)

    def test_support_prefers_new_family_before_second_same_family(self):
        entries = [
            make_entry(0, 0.99, [1, 1, 0, 0], [1, 0, 0], [1.0, 0.0, 0.0, 0.0], context=1.00),
            make_entry(90, 0.98, [1, 1, 0, 0], [1, 0, 0], [0.99, 0.01, 0.0, 0.0], context=0.99),
            make_entry(180, 0.90, [0, 1, 1, 0], [0, 1, 0], [0.0, 1.0, 0.0, 0.0], context=0.95),
            make_entry(270, 0.89, [0, 0, 1, 1], [0, 0, 1], [0.0, 0.0, 1.0, 0.0], context=0.94),
            make_entry(360, 0.88, [1, 0, 0, 1], [0.2, 0.2, 0.6], [0.0, 0.0, 0.0, 1.0], context=0.93),
            make_entry(450, 0.87, [0.4, 0.0, 0.8, 0.2], [0.0, 0.6, 0.4], [0.0, 0.7, 0.7, 0.0], context=0.92),
        ]
        selected, _ = optimize_keyframe_set(
            all_entries=entries,
            target_count=4,
            total_frames=600,
            min_frame_gap=56,
            diversity_weight=0.95,
            similarity_cap=0.90,
        )
        selected_ids = sorted(entry["idx"] for entry in selected)
        self.assertIn(0, selected_ids)
        self.assertIn(180, selected_ids)
        self.assertIn(270, selected_ids)
        self.assertNotIn(90, selected_ids)

    def test_low_quality_new_family_does_not_force_support_pick(self):
        entries = [
            make_entry(0, 0.99, [1, 1, 0, 0], [1, 0, 0], [1.0, 0.0, 0.0, 0.0], context=11.8),
            make_entry(90, 0.98, [1, 0.8, 0.2, 0.0], [1, 0, 0], [0.99, 0.01, 0.0, 0.0], context=11.6),
            make_entry(180, 0.94, [0, 1, 1, 0], [0, 1, 0], [0.0, 1.0, 0.0, 0.0], context=11.3),
            make_entry(270, 0.93, [0, 0, 1, 1], [0, 0, 1], [0.0, 0.0, 1.0, 0.0], context=11.1),
            make_entry(360, 0.92, [1, 0, 0, 1], [0.5, 0.0, 0.5], [0.0, 0.0, 0.0, 1.0], context=10.9),
            make_entry(450, 0.80, [0.08, 0.02, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.2, 0.7, 0.65], context=8.2),
        ]
        weak = entries[-1]
        weak["stats"]["grid_cells"] = 3
        weak["stats"]["useful_union_ratio"] = 0.10
        weak["stats"]["texture_penalty"] = 0.08
        selected, _ = optimize_keyframe_set(
            all_entries=entries,
            target_count=4,
            total_frames=600,
            min_frame_gap=56,
            diversity_weight=0.95,
            similarity_cap=0.90,
        )
        selected_ids = sorted(entry["idx"] for entry in selected)
        self.assertNotIn(450, selected_ids)

    def test_top_rejected_fallback_can_recover_strong_same_family_frame(self):
        entries = [
            make_entry(0, 0.99, [1, 1, 0, 0], [1, 0, 0], [1.0, 0.0, 0.0, 0.0], context=11.8),
            make_entry(90, 0.98, [1, 0.9, 0.1, 0.0], [1, 0, 0], [0.93, 0.37, 0.0, 0.0], context=11.6),
            make_entry(180, 0.95, [0, 1, 1, 0], [0, 1, 0], [0.0, 1.0, 0.0, 0.0], context=11.4),
            make_entry(270, 0.94, [0, 0, 1, 1], [0, 0, 1], [0.0, 0.0, 1.0, 0.0], context=11.2),
            make_entry(360, 0.82, [0.10, 0.05, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], context=8.5),
        ]
        weak = entries[-1]
        weak["stats"]["grid_cells"] = 3
        weak["stats"]["useful_union_ratio"] = 0.12
        weak["stats"]["texture_penalty"] = 0.08

        selected, diag = optimize_keyframe_set(
            all_entries=entries,
            target_count=4,
            total_frames=600,
            min_frame_gap=56,
            diversity_weight=0.95,
            similarity_cap=0.90,
        )
        selected_ids = sorted(entry["idx"] for entry in selected)
        self.assertIn(90, selected_ids)
        self.assertNotIn(360, selected_ids)
        self.assertTrue(any("fallback from top rejected" in note for note in diag["fallback_notes"]))

    def test_relaxed_support_quality_can_add_additional_family(self):
        entries = [
            make_entry(0, 0.99, [1, 1, 0, 0], [1, 0, 0], [1.0, 0.0, 0.0, 0.0], context=12.0, family_descriptor=[1.0, 0.0, 0.0, 0.0]),
            make_entry(80, 0.98, [0, 1, 1, 0], [0, 1, 0], [0.0, 1.0, 0.0, 0.0], context=11.8, family_descriptor=[0.0, 1.0, 0.0, 0.0]),
            make_entry(160, 0.97, [0, 0, 1, 1], [0, 0, 1], [0.0, 0.0, 1.0, 0.0], context=11.6, family_descriptor=[0.0, 0.0, 1.0, 0.0]),
            make_entry(240, 0.95, [1, 0, 0, 1], [0.2, 0.3, 0.5], [0.0, 0.0, 0.0, 1.0], context=10.8, family_descriptor=[0.0, 0.0, 0.0, 1.0]),
            make_entry(320, 0.93, [0.7, 0.2, 0.0, 0.8], [0.2, 0.4, 0.4], [0.5, 0.5, 0.0, 0.5], context=9.6, family_descriptor=[0.5, 0.5, 0.5, 0.5]),
        ]
        selected, diag = optimize_keyframe_set(
            all_entries=entries,
            target_count=5,
            total_frames=600,
            min_frame_gap=56,
            diversity_weight=0.95,
            similarity_cap=0.90,
        )
        selected_ids = sorted(entry["idx"] for entry in selected)
        self.assertIn(240, selected_ids)
        self.assertIn(320, selected_ids)

    def test_rejected_set_rescue_can_form_fuller_alternative(self):
        entries = [
            make_entry(0, 0.99, [1, 1, 0, 0], [1, 0, 0], [1.0, 0.0, 0.0, 0.0], context=12.0, family_descriptor=[1.0, 0.0, 0.0, 0.0]),
            make_entry(80, 0.98, [0, 1, 1, 0], [0, 1, 0], [0.0, 1.0, 0.0, 0.0], context=11.8, family_descriptor=[0.0, 1.0, 0.0, 0.0]),
            make_entry(160, 0.97, [0, 0, 1, 1], [0, 0, 1], [0.0, 0.0, 1.0, 0.0], context=11.6, family_descriptor=[0.0, 0.0, 1.0, 0.0]),
            make_entry(240, 0.96, [1, 0, 0, 1], [0.4, 0.2, 0.4], [0.0, 0.0, 0.0, 1.0], context=11.4, family_descriptor=[0.0, 0.0, 0.0, 1.0]),
        ]
        for entry in entries:
            entry["overview_score"] = entry["global_base_score"]
            entry["support_score"] = entry["global_base_score"]
            entry["is_overview_candidate"] = True
            entry["is_support_candidate"] = True
        _assign_view_families(entries, family_similarity_threshold=0.88)
        rescue = _build_rejected_rescue_set(
            rejected_entries=entries,
            target_count=4,
            total_frames=600,
            diversity_weight=0.95,
            min_frame_gap=44,
            similarity_cap=0.92,
            anchor_target=2,
            support_target=2,
        )
        self.assertEqual([entry["idx"] for entry in rescue], [0, 80, 160, 240])

    def test_force_fill_entries_reaches_target_count(self):
        entries = []
        for idx in [0, 40, 80, 120, 160, 200]:
            entry = make_entry(
                idx,
                0.95 - 0.01 * (idx // 40),
                [1, 0, 0, 1],
                [0.3, 0.3, 0.4],
                [1.0, 0.0, 0.0, 0.0],
                context=11.0 - 0.1 * (idx // 40),
                family_descriptor=[1.0, 0.0, 0.0, 0.0],
            )
            entry["overview_score"] = entry["global_base_score"]
            entry["support_score"] = entry["global_base_score"]
            entries.append(entry)
        for idx in [240, 280, 320, 360]:
            entry = make_entry(
                idx,
                0.70 - 0.01 * ((idx - 240) // 40),
                [0, 1, 1, 0],
                [0.2, 0.5, 0.3],
                [0.0, 1.0, 0.0, 0.0],
                context=8.8,
                family_descriptor=[0.0, 1.0, 0.0, 0.0],
            )
            entry["overview_score"] = entry["global_base_score"]
            entry["support_score"] = entry["global_base_score"]
            entries.append(entry)
        selected = entries[:6]
        filled, notes = _force_fill_entries(
            selected_entries=selected,
            all_entries=entries,
            target_count=10,
            total_frames=600,
            diversity_weight=0.95,
            support_gap=44,
            support_similarity_cap=0.92,
        )
        self.assertEqual(len(filled), 10)
        self.assertTrue(notes)

    def test_force_fill_entries_uses_quality_floor_for_tail_slots(self):
        selected = []
        for idx in [0, 40, 80, 120, 160, 200, 240, 280]:
            entry = make_entry(
                idx,
                0.92,
                [1, 0, 0, 1],
                [0.3, 0.3, 0.4],
                [1.0, 0.0, 0.0, 0.0],
                context=11.0,
                family_descriptor=[1.0, 0.0, 0.0, 0.0],
            )
            entry["overview_score"] = 0.82
            entry["support_score"] = 0.82
            selected.append(entry)

        strong_fill = []
        for idx in [320, 360]:
            entry = make_entry(
                idx,
                0.90,
                [0, 1, 1, 0],
                [0.2, 0.5, 0.3],
                [0.0, 1.0, 0.0, 0.0],
                context=10.9,
                family_descriptor=[0.0, 1.0, 0.0, 0.0],
            )
            entry["overview_score"] = 0.78
            entry["support_score"] = 0.79
            strong_fill.append(entry)

        weak_fill = []
        for idx in [400, 440]:
            entry = make_entry(
                idx,
                0.60,
                [0, 0, 1, 0],
                [0.1, 0.1, 0.8],
                [0.0, 0.0, 1.0, 0.0],
                context=8.6,
                family_descriptor=[0.0, 0.0, 1.0, 0.0],
            )
            entry["overview_score"] = 0.10
            entry["support_score"] = 0.10
            weak_fill.append(entry)

        all_entries = selected + strong_fill + weak_fill
        filled, _ = _force_fill_entries(
            selected_entries=selected,
            all_entries=all_entries,
            target_count=10,
            total_frames=600,
            diversity_weight=0.95,
            support_gap=44,
            support_similarity_cap=0.92,
        )
        filled_ids = [entry["idx"] for entry in filled]
        self.assertIn(320, filled_ids)
        self.assertIn(360, filled_ids)
        self.assertNotIn(400, filled_ids)
        self.assertNotIn(440, filled_ids)

    def test_force_fill_entries_avoids_blurry_tail_slots_when_sharper_exist(self):
        selected = []
        for idx in [0, 40, 80, 120, 160, 200, 240, 280]:
            entry = make_entry(
                idx,
                0.92,
                [1, 0, 0, 1],
                [0.3, 0.3, 0.4],
                [1.0, 0.0, 0.0, 0.0],
                context=11.0,
                family_descriptor=[1.0, 0.0, 0.0, 0.0],
            )
            entry["overview_score"] = 0.82
            entry["support_score"] = 0.82
            entry["stats"]["frame_quality"] = 8.8
            selected.append(entry)

        sharp_fill = []
        for idx in [320, 360]:
            entry = make_entry(
                idx,
                0.90,
                [0, 1, 1, 0],
                [0.2, 0.5, 0.3],
                [0.0, 1.0, 0.0, 0.0],
                context=10.9,
                family_descriptor=[0.0, 1.0, 0.0, 0.0],
            )
            entry["overview_score"] = 0.78
            entry["support_score"] = 0.79
            entry["stats"]["frame_quality"] = 8.4
            sharp_fill.append(entry)

        blurry_fill = []
        for idx in [400, 440]:
            entry = make_entry(
                idx,
                0.90,
                [0, 1, 1, 0],
                [0.2, 0.5, 0.3],
                [0.0, 0.0, 1.0, 0.0],
                context=10.9,
                family_descriptor=[0.0, 0.0, 1.0, 0.0],
            )
            entry["overview_score"] = 0.78
            entry["support_score"] = 0.79
            entry["stats"]["frame_quality"] = 7.2
            blurry_fill.append(entry)

        all_entries = selected + sharp_fill + blurry_fill
        filled, _ = _force_fill_entries(
            selected_entries=selected,
            all_entries=all_entries,
            target_count=10,
            total_frames=600,
            diversity_weight=0.95,
            support_gap=44,
            support_similarity_cap=0.92,
        )
        filled_ids = [entry["idx"] for entry in filled]
        self.assertIn(320, filled_ids)
        self.assertIn(360, filled_ids)
        self.assertNotIn(400, filled_ids)
        self.assertNotIn(440, filled_ids)

    def test_final_role_assignment_preserves_instruction_split(self):
        entries = []
        for idx, ov, disc, rep in [
            (0, 0.98, 0.10, 0.12),
            (40, 0.95, 0.14, 0.16),
            (80, 0.93, 0.61, 0.22),
            (120, 0.91, 0.58, 0.20),
            (160, 0.90, 0.56, 0.18),
            (200, 0.89, 0.18, 0.67),
            (240, 0.88, 0.20, 0.64),
            (280, 0.87, 0.16, 0.62),
            (320, 0.86, 0.12, 0.60),
            (360, 0.85, 0.28, 0.26),
            (400, 0.84, 0.24, 0.24),
            (440, 0.83, 0.22, 0.22),
        ]:
            entry = make_entry(
                idx,
                ov,
                [1, 0, 0, 1],
                [0.3, 0.3, 0.4],
                [1.0, 0.0, 0.0, 0.0],
                context=11.0,
                family_descriptor=[1.0, 0.0, 0.0, 0.0],
            )
            entry["overview_score"] = ov
            entry["support_score"] = max(ov, disc, rep)
            entry["discovery_score"] = disc
            entry["repair_score"] = rep
            entry["discovery_novelty"] = disc * 0.7
            entry["mask_rise"] = disc * 0.4
            entry["coverage_rise"] = disc * 0.2
            entry["appearance_jump"] = rep * 0.5
            entry["transition_jump"] = rep * 0.4
            entry["mask_drop"] = rep * 0.3
            entry["coverage_drop"] = rep * 0.2
            entries.append(entry)

        assigned = _assign_final_selection_roles(entries, overview_target=5, discovery_target=3, repair_target=4)
        counts = {}
        for entry in assigned:
            counts[entry["selection_role"]] = counts.get(entry["selection_role"], 0) + 1
        self.assertEqual(counts.get("overview", 0), 5)
        self.assertEqual(counts.get("discovery", 0), 3)
        self.assertEqual(counts.get("repair", 0), 4)

    def test_final_polish_prefers_new_family_over_weak_duplicate(self):
        selected = []
        family_basis = np.eye(12, dtype=np.float32)
        family_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 3, 10]
        for idx, fam in zip([0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 440, 480], family_ids):
            entry = make_entry(
                idx,
                0.92,
                [1, 0, 0, 1],
                [0.3, 0.3, 0.4],
                [1.0, 0.0, 0.0, 0.0],
                context=11.0,
                family_descriptor=family_basis[fam],
            )
            entry["overview_score"] = 0.80
            entry["support_score"] = 0.80
            entry["view_family"] = fam
            selected.append(entry)
        selected[10]["support_score"] = 0.62
        selected[10]["overview_score"] = 0.55
        selected[10]["stats"]["context_score"] = 9.8
        selected[10]["stats"]["useful_union_ratio"] = 0.18

        better_new_family = make_entry(
            500,
            0.94,
            [0, 1, 1, 0],
            [0.2, 0.5, 0.3],
            [0.0, 1.0, 0.0, 0.0],
            context=11.4,
            family_descriptor=[0.0] * 11 + [1.0],
        )
        better_new_family["overview_score"] = 0.83
        better_new_family["support_score"] = 0.86
        better_new_family["view_family"] = 11

        low_gain_same_family = make_entry(
            520,
            0.93,
            [0, 1, 0, 1],
            [0.3, 0.3, 0.4],
            [0.0, 1.0, 0.0, 0.0],
            context=10.9,
            family_descriptor=family_basis[3],
        )
        low_gain_same_family["overview_score"] = 0.82
        low_gain_same_family["support_score"] = 0.84
        low_gain_same_family["view_family"] = 3

        polished, notes = _final_polish_selected_set(
            selected_entries=selected,
            all_entries=selected + [better_new_family, low_gain_same_family],
            target_count=12,
            total_frames=600,
            diversity_weight=0.95,
            min_frame_gap=44,
            similarity_cap=0.92,
        )
        polished_ids = [entry["idx"] for entry in polished]
        self.assertIn(500, polished_ids)
        self.assertNotIn(440, polished_ids)
        self.assertNotIn(520, polished_ids)
        self.assertTrue(any("final polish swap" in note for note in notes))

    def test_final_polish_rejects_blurry_negative_gain_swap(self):
        selected = []
        family_basis = np.eye(12, dtype=np.float32)
        family_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        for idx, fam in zip([0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440], family_ids):
            entry = make_entry(
                idx,
                0.92,
                [1, 0, 0, 1],
                [0.3, 0.3, 0.4],
                [1.0, 0.0, 0.0, 0.0],
                context=11.0,
                family_descriptor=family_basis[fam],
            )
            entry["overview_score"] = 0.80
            entry["support_score"] = 0.80
            entry["stats"]["frame_quality"] = 8.8
            entry["view_family"] = fam
            selected.append(entry)

        victim = selected[-1]
        victim["support_score"] = 0.82
        victim["overview_score"] = 0.83
        victim["stats"]["frame_quality"] = 8.3
        victim["view_family"] = 11

        blurry_candidate = make_entry(
            500,
            0.90,
            [0, 1, 1, 0],
            [0.2, 0.5, 0.3],
            [0.0, 1.0, 0.0, 0.0],
            context=10.5,
            family_descriptor=family_basis[11],
        )
        blurry_candidate["overview_score"] = 0.79
        blurry_candidate["support_score"] = 0.78
        blurry_candidate["stats"]["frame_quality"] = 7.5
        blurry_candidate["view_family"] = 11

        polished, notes = _final_polish_selected_set(
            selected_entries=selected,
            all_entries=selected + [blurry_candidate],
            target_count=12,
            total_frames=600,
            diversity_weight=0.95,
            min_frame_gap=44,
            similarity_cap=0.92,
        )
        polished_ids = [entry["idx"] for entry in polished]
        self.assertNotIn(500, polished_ids)
        self.assertFalse(any("500" in note for note in notes))

    def test_quality_swap_does_not_reduce_family_diversity(self):
        current = [
            {"idx": 0, "view_family": 0},
            {"idx": 101, "view_family": 1},
            {"idx": 166, "view_family": 2},
            {"idx": 529, "view_family": 9},
        ]
        trial = [
            {"idx": 0, "view_family": 0},
            {"idx": 101, "view_family": 1},
            {"idx": 166, "view_family": 2},
            {"idx": 505, "view_family": 0},
        ]
        allowed, trial_stats = _quality_swap_preserves_diversity(
            current,
            trial,
            quality_family_cap=4,
        )
        self.assertFalse(allowed)
        self.assertLess(trial_stats["unique_families"], 4)


if __name__ == "__main__":
    unittest.main()
