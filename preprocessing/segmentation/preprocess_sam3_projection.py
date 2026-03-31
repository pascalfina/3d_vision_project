# preprocess_sam3_projection.py
"""
command: 
python preprocessing/segmentation/preprocess_sam3_projection.py --scan-id e61b0e04-bada-2f31-82d6-72831a602ba7 --data-root /cluster/project/cvg/data/3RScan

python preprocessing/segmentation/preprocess_sam3_projection.py --save-debug-npz \
  --data-root  \
  --scan-id SCAN_ID \
  --overwrite \
  --save-debug-npz
"""
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm.auto import tqdm

import numpy as np
from PIL import Image
import torch
from transformers import Sam3VideoModel, Sam3VideoProcessor

from utils import common


DEFAULT_EXCLUDED_LABELS = {
    #"wall",
    #"floor",
    #"ceiling",
}


def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def normalise_label(label: str) -> str:
    return " ".join(label.strip().lower().split())


def slugify_prompt(prompt: str) -> str:
    slug = prompt.strip().lower().replace(" ", "_").replace("/", "_")
    return "".join(c for c in slug if c.isalnum() or c in {"_", "-"})


def extract_frame_id(path: Path) -> str:
    stem = path.stem
    match = re.match(r"frame-(.+?)\.color$", stem)
    if match:
        return match.group(1)
    match = re.match(r"frame-(.+?)\.depth$", stem)
    if match:
        return match.group(1)
    match = re.match(r"frame-(.+)$", stem)
    if match:
        return match.group(1)
    return stem


def load_frames(frames_dir: Path) -> Tuple[List[Image.Image], List[Path], List[str]]:
    frame_paths = sorted(
        [p for p in frames_dir.iterdir() if p.name.lower().endswith(".color.jpg")],
        key=natural_key,
    )
    if not frame_paths:
        raise FileNotFoundError(f"No image frames found in {frames_dir}")

    frames = [Image.open(p).convert("RGB") for p in frame_paths]
    frame_ids = [extract_frame_id(p) for p in frame_paths]
    return frames, frame_paths, frame_ids


def resolve_default_frames_dir(data_root: Path, scan_id: str) -> Path:
    return data_root / "scenes" / scan_id / "sequence"


def build_prompt_specs_from_objects_json(
    objects_json_path: str,
    scan_id: str,
    exclude_labels: Optional[set[str]] = None,
    include_stuff: bool = False,
) -> List[Dict[str, Any]]:
    if exclude_labels is None:
        exclude_labels = set()

    with open(objects_json_path, "r") as f:
        data = json.load(f)

    target_scan = None
    for scan_entry in data.get("scans", []):
        if scan_entry.get("scan") == scan_id:
            target_scan = scan_entry
            break

    if target_scan is None:
        raise ValueError(f"scan_id '{scan_id}' not found in {objects_json_path}")

    grouped: Dict[str, List[int]] = {}
    for obj in target_scan.get("objects", []):
        label = normalise_label(obj.get("label", ""))
        if not label:
            continue
        if not include_stuff and label in exclude_labels:
            continue
        grouped.setdefault(label, []).append(int(obj["id"]))

    prompt_specs = [
        {
            "prompt": label,
            "source_obj_ids": sorted(obj_ids),
        }
        for label, obj_ids in grouped.items()
    ]
    prompt_specs.sort(key=lambda item: item["prompt"])

    if not prompt_specs:
        raise ValueError(
            f"No prompts found for scan_id '{scan_id}' after filtering in {objects_json_path}"
        )
    return prompt_specs


def save_prompt_frame_outputs(
    out_dir: Path,
    prompt_slug: str,
    frame_name: str,
    object_ids: np.ndarray,
    scores: np.ndarray,
    boxes: np.ndarray,
    masks: np.ndarray,
) -> None:
    frame_out_dir = out_dir / "per_prompt" / prompt_slug
    frame_out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "object_ids": object_ids.astype(np.int32),
        "scores": scores.astype(np.float32),
        "boxes": boxes.astype(np.float32),
        "masks": masks.astype(np.float32),
    }
    np.savez_compressed(frame_out_dir / f"{frame_name}.npz", **payload)


def run_sam3_video_for_prompt(
    model: Sam3VideoModel,
    processor: Sam3VideoProcessor,
    frames: List[Image.Image],
    frame_paths: List[Path],
    frame_ids: List[str],
    prompt: str,
    device: str,
    processing_device: str,
    video_storage_device: str,
    dtype: torch.dtype,
) -> Dict[str, Any]:
    print(f'[INFO] Initialising SAM3 Video session for prompt: "{prompt}"')

    inference_session = processor.init_video_session(
        video=frames,
        inference_device=device,
        processing_device=processing_device,
        video_storage_device=video_storage_device,
        dtype=dtype,
    )
    inference_session = processor.add_text_prompt(
        inference_session=inference_session,
        text=prompt,
    )

    results_by_frame: Dict[str, Dict[str, Any]] = {}
    with torch.no_grad():
        pbar = tqdm(
            model.propagate_in_video_iterator(inference_session),
            total=len(frame_paths),
            desc=f'SAM3 "{prompt}"',
            dynamic_ncols=True,
            leave=False,
        )
        for model_outputs in pbar:
            processed_outputs = processor.postprocess_outputs(
                inference_session=inference_session,
                model_outputs=model_outputs,
            )

            frame_idx = int(model_outputs.frame_idx)
            frame_path = frame_paths[frame_idx]
            frame_name = frame_path.stem
            frame_id = frame_ids[frame_idx]
            num_objects = len(processed_outputs["object_ids"])
            
            results_by_frame[frame_id] = {
                "frame_idx": frame_idx,
                "frame_id": frame_id,
                "frame_name": frame_name,
                "prompt": prompt,
                "object_ids": processed_outputs["object_ids"].detach().cpu().numpy(),
                "scores": processed_outputs["scores"].detach().cpu().numpy(),
                "boxes": processed_outputs["boxes"].detach().cpu().numpy(),
                "masks": processed_outputs["masks"].detach().cpu().numpy(),
            }

            pbar.set_postfix(
                frame=frame_id,
                objects=num_objects,
            )
            #print(
            #    f'[INFO] prompt="{prompt}" frame={frame_idx:04d} '
            #    f"id={frame_id} objects={len(results_by_frame[frame_id]['object_ids'])}"
            #)

    return results_by_frame


def summarize_prompt_tracks(
    prompt: str,
    per_frame_results: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    tracks: Dict[int, Dict[str, Any]] = {}
    for frame_id, frame_result in per_frame_results.items():
        frame_idx = int(frame_result["frame_idx"])
        object_ids = np.asarray(frame_result["object_ids"])
        scores = np.asarray(frame_result["scores"])
        masks = np.asarray(frame_result["masks"])

        if masks.ndim != 3 or masks.shape[0] == 0:
            continue

        for idx in range(len(object_ids)):
            local_id = int(object_ids[idx])
            mask = masks[idx] > 0.5
            area = int(mask.sum())
            if area <= 0:
                continue

            track = tracks.setdefault(
                local_id,
                {
                    "prompt": prompt,
                    "local_object_id": local_id,
                    "num_frames_visible": 0,
                    "total_area": 0.0,
                    "total_score": 0.0,
                    "first_frame_idx": frame_idx,
                    "first_frame_id": frame_id,
                    "last_frame_idx": frame_idx,
                },
            )
            track["num_frames_visible"] += 1
            track["total_area"] += float(area)
            track["total_score"] += float(scores[idx])
            track["first_frame_idx"] = min(track["first_frame_idx"], frame_idx)
            track["last_frame_idx"] = max(track["last_frame_idx"], frame_idx)

    summaries = []
    for track in tracks.values():
        num_frames = max(1, int(track["num_frames_visible"]))
        summaries.append(
            {
                **track,
                "mean_area": float(track["total_area"]) / float(num_frames),
                "mean_score": float(track["total_score"]) / float(num_frames),
            }
        )
    return summaries


def select_tracks_for_prompt(
    track_summaries: List[Dict[str, Any]],
    min_mean_area: float,
    min_mean_score: float,
) -> List[Dict[str, Any]]:
    filtered = [
        item
        for item in track_summaries
        if float(item["mean_area"]) >= float(min_mean_area)
        and float(item["mean_score"]) >= float(min_mean_score)
    ]
    return sorted(
        filtered,
        key=lambda item: (
            -float(item["mean_area"]),
            -float(item["mean_score"]),
            int(item["first_frame_idx"]),
            int(item["local_object_id"]),
        ),
    )


def build_objects_sam_entry(
    scan_id: str,
    selected_tracks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    objects = []
    for track in selected_tracks:
        objects.append(
            {
                "id": int(track["global_object_id"]),
                "label": track["prompt"],
                "sam_local_object_id": int(track["local_object_id"]),
                "num_frames_visible": int(track["num_frames_visible"]),
                "mean_area": float(track["mean_area"]),
                "mean_score": float(track["mean_score"]),
                "first_frame_id": str(track["first_frame_id"]),
            }
        )
    return {"scan": scan_id, "objects": objects}


def build_frame_instances(
    frame_result: Optional[Dict[str, Any]],
    prompt: str,
    local_to_global_id: Dict[int, int],
    mask_threshold: float,
) -> List[Dict[str, Any]]:
    if frame_result is None:
        return []

    object_ids = np.asarray(frame_result["object_ids"])
    scores = np.asarray(frame_result["scores"])
    masks = np.asarray(frame_result["masks"])
    if masks.ndim != 3 or masks.shape[0] == 0:
        return []

    instances = []
    for idx in range(len(object_ids)):
        local_id = int(object_ids[idx])
        if local_id not in local_to_global_id:
            continue
        mask = masks[idx] > mask_threshold
        area = int(mask.sum())
        if area <= 0:
            continue
        instances.append(
            {
                "prompt": prompt,
                "local_object_id": local_id,
                "global_object_id": int(local_to_global_id[local_id]),
                "score": float(scores[idx]),
                "area": area,
                "mask": mask,
            }
        )
    return instances


def merge_frame_instances(
    frame_shape: Tuple[int, int],
    instances: List[Dict[str, Any]],
) -> np.ndarray:
    merged = np.zeros(frame_shape, dtype=np.int32)
    ordered = sorted(
        instances,
        key=lambda item: (
            float(item["score"]),
            float(item["area"]),
            int(item["global_object_id"]),
        ),
    )
    for inst in ordered:
        merged[inst["mask"]] = int(inst["global_object_id"])
    return merged


def save_scan_projection(
    *,
    output_dir: Path,
    scan_id: str,
    merged_maps: Dict[str, np.ndarray],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = output_dir / f"{scan_id}.pkl"
    serializable = {
        str(frame_id): np.asarray(mask_map, dtype=np.int32)
        for frame_id, mask_map in merged_maps.items()
    }
    common.write_pkl_data(serializable, str(pkl_path))
    return pkl_path


def save_objects_sam(
    *,
    objects_file: Path,
    scan_entry: Dict[str, Any],
) -> Path:
    if objects_file.exists():
        payload = common.load_json(str(objects_file))
    else:
        payload = {"scans": []}

    scans = [entry for entry in payload.get("scans", []) if entry.get("scan") != scan_entry["scan"]]
    scans.append(scan_entry)
    scans.sort(key=lambda item: item["scan"])
    payload["scans"] = scans

    common.ensure_dir(str(objects_file.parent))
    common.write_json(payload, str(objects_file))
    return objects_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True, default='/cluster/project/cvg/data/3RScan')
    parser.add_argument("--scan-id", type=str, required=True, default='e61b0e04-bada-2f31-82d6-72831a602ba7')
    parser.add_argument("--frames-dir", type=str, default=None)
    parser.add_argument("--output-source", type=str, default="sam3_projection")
    parser.add_argument("--objects-input-file", type=str, default=None)
    parser.add_argument("--objects-output-file", type=str, default="objects_sam.json")
    parser.add_argument("--debug-output-dir", type=str, default=None)

    parser.add_argument("--include-stuff", action="store_true")
    parser.add_argument("--exclude-labels", nargs="*", default=None)

    parser.add_argument("--model-id", type=str, default="models/pretrained/sam3")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--processing-device", type=str, default="cpu")
    parser.add_argument("--video-storage-device", type=str, default="cpu")

    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--min-mean-area", type=float, default=256.0)
    parser.add_argument("--min-mean-score", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save-debug-npz", action="store_true")
    parser.add_argument("--visualize", action="store_true")

    args = parser.parse_args()

    data_root = Path(args.data_root)
    scan_id = args.scan_id
    frames_dir = (
        Path(args.frames_dir)
        if args.frames_dir is not None
        else resolve_default_frames_dir(data_root, scan_id)
    )
    objects_input_file = (
        Path(args.objects_input_file)
        if args.objects_input_file is not None
        else data_root / "files" / "objects.json"
    )
    objects_output_file = (
        Path(args.objects_output_file)
        if Path(args.objects_output_file).is_absolute()
        else data_root / "files" / args.objects_output_file
    )
    debug_output_dir = (
        Path(args.debug_output_dir)
        if args.debug_output_dir is not None
        else data_root / "files" / args.output_source / "debug" / scan_id
    )
    visualize_output_dir = debug_output_dir / "visualizations"
    output_dir = data_root / "files" / args.output_source / "obj_id_pkl"
    output_pkl = output_dir / f"{scan_id}.pkl"

    if output_pkl.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists at {output_pkl}; pass --overwrite to replace it."
        )

    exclude_labels = set(DEFAULT_EXCLUDED_LABELS)
    if args.exclude_labels is not None:
        exclude_labels = {normalise_label(x) for x in args.exclude_labels}
    prompt_specs = build_prompt_specs_from_objects_json(
        objects_json_path=str(objects_input_file),
        scan_id=scan_id,
        exclude_labels=exclude_labels,
        include_stuff=args.include_stuff,
    )

    print(f"[INFO] Number of prompts: {len(prompt_specs)}")
    print(f"[INFO] Prompts: {[item['prompt'] for item in prompt_specs]}")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print(f"[INFO] Loading frames from: {frames_dir}")
    frames, frame_paths, frame_ids = load_frames(frames_dir)
    print(f"[INFO] Loaded {len(frames)} frames")

    print(f"[INFO] Loading model: {args.model_id}")
    processor = Sam3VideoProcessor.from_pretrained(args.model_id)
    model = Sam3VideoModel.from_pretrained(args.model_id).to(args.device, dtype=dtype)
    model.eval()

    all_prompt_results: Dict[str, Dict[str, Any]] = {}
    all_track_summaries: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {
        "scan_id": scan_id,
        "model_id": args.model_id,
        "output_source": args.output_source,
        "objects_input_file": str(objects_input_file),
        "objects_output_file": str(objects_output_file),
        "num_frames": len(frames),
        "prompts": prompt_specs,
        "selected_tracks": [],
        "frames": {},
    }

    for prompt_spec in prompt_specs:
        prompt = prompt_spec["prompt"]
        prompt_slug = slugify_prompt(prompt)
        per_frame_results = run_sam3_video_for_prompt(
            model=model,
            processor=processor,
            frames=frames,
            frame_paths=frame_paths,
            frame_ids=frame_ids,
            prompt=prompt,
            device=args.device,
            processing_device=args.processing_device,
            video_storage_device=args.video_storage_device,
            dtype=dtype,
        )
        all_prompt_results[prompt] = per_frame_results

        track_summaries = summarize_prompt_tracks(prompt, per_frame_results)
        selected_tracks = select_tracks_for_prompt(
            track_summaries,
            min_mean_area=args.min_mean_area,
            min_mean_score=args.min_mean_score,
        )
        for track in selected_tracks:
            all_track_summaries.append(track)

        if args.save_debug_npz:
            for _, frame_result in per_frame_results.items():
                save_prompt_frame_outputs(
                    out_dir=debug_output_dir,
                    prompt_slug=prompt_slug,
                    frame_name=frame_result["frame_name"],
                    object_ids=frame_result["object_ids"],
                    scores=frame_result["scores"],
                    boxes=frame_result["boxes"],
                    masks=frame_result["masks"],
                )

    all_track_summaries.sort(
        key=lambda item: (
            item["prompt"],
            int(item["first_frame_idx"]),
            -float(item["mean_area"]),
            int(item["local_object_id"]),
        )
    )
    for global_id, track in enumerate(all_track_summaries, start=1):
        track["global_object_id"] = global_id

    local_to_global_by_prompt: Dict[str, Dict[int, int]] = {}
    for track in all_track_summaries:
        local_to_global_by_prompt.setdefault(track["prompt"], {})[
            int(track["local_object_id"])
        ] = int(track["global_object_id"])

    objects_sam_entry = build_objects_sam_entry(scan_id, all_track_summaries)
    written_objects = save_objects_sam(
        objects_file=objects_output_file,
        scan_entry=objects_sam_entry,
    )

    merged_maps: Dict[str, np.ndarray] = {}
    frame_shape = (frames[0].size[1], frames[0].size[0])
    for frame_id, frame_path in zip(frame_ids, frame_paths):
        instances = []
        for prompt_spec in prompt_specs:
            prompt = prompt_spec["prompt"]
            prompt_result = all_prompt_results[prompt].get(frame_id)
            instances.extend(
                build_frame_instances(
                    frame_result=prompt_result,
                    prompt=prompt,
                    local_to_global_id=local_to_global_by_prompt.get(prompt, {}),
                    mask_threshold=args.mask_threshold,
                )
            )

        merged_obj_id_map = merge_frame_instances(frame_shape, instances)
        merged_maps[frame_id] = merged_obj_id_map
        summary["frames"][frame_id] = {
            "frame_name": frame_path.name,
            "num_instances": int(len(instances)),
            "instances": [
                {
                    "prompt": inst["prompt"],
                    "local_object_id": int(inst["local_object_id"]),
                    "global_object_id": int(inst["global_object_id"]),
                    "score": float(inst["score"]),
                    "area": int(inst["area"]),
                }
                for inst in instances
            ],
        }

        if args.save_debug_npz:
            common.ensure_dir(str(debug_output_dir))
            np.save(debug_output_dir / f"{frame_id}_merged_obj_id_map.npy", merged_obj_id_map)

    written_pkl = save_scan_projection(
        output_dir=output_dir,
        scan_id=scan_id,
        merged_maps=merged_maps,
    )

    if args.visualize:
        common.ensure_dir(str(visualize_output_dir))
        for frame_id, frame_path in zip(frame_ids, frame_paths):
            image = Image.open(frame_path).convert("RGB")
            obj_id_map = merged_maps[frame_id]

            overlay = np.array(image).copy()
            for obj_id in np.unique(obj_id_map):
                if obj_id == 0:
                    continue
                mask = obj_id_map == obj_id
                color = np.array(
                    [
                        (int(obj_id) * 67) % 256,
                        (int(obj_id) * 131) % 256,
                        (int(obj_id) * 197) % 256,
                    ],
                    dtype=np.uint8,
                )
                overlay[mask] = (
                    0.65 * overlay[mask] + 0.35 * color
                ).astype(np.uint8) 

            out_path = visualize_output_dir / f"{frame_path.stem}.png"
            Image.fromarray(overlay).save(out_path)
        print(f"[DONE] Wrote visualization overlays to: {visualize_output_dir}")


    summary["selected_tracks"] = [
        {
            "prompt": track["prompt"],
            "local_object_id": int(track["local_object_id"]),
            "global_object_id": int(track["global_object_id"]),
            "num_frames_visible": int(track["num_frames_visible"]),
            "mean_area": float(track["mean_area"]),
            "mean_score": float(track["mean_score"]),
            "first_frame_id": str(track["first_frame_id"]),
        }
        for track in all_track_summaries
    ]
    common.ensure_dir(str(debug_output_dir))
    with open(debug_output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[DONE] Wrote scan projection to: {written_pkl}")
    print(f"[DONE] Wrote SAM objects file to: {written_objects}")


if __name__ == "__main__":
    main()
