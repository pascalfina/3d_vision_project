import os
import shutil
import numpy as np
from PIL import Image

opj = os.path.join
ol = os.listdir

DATASET = os.environ.get("DATASET", "3RScan")
DATA_ROOT_DIR = os.environ.get("DATA_ROOT_DIR", "/cluster/scratch/ealegret/sam2object")
SCAN_IDS_ENV = os.environ.get("SCAN_IDS", "")
SCAN_IDS = [s.strip() for s in SCAN_IDS_ENV.split(",") if s.strip()]

path_list = [opj(DATA_ROOT_DIR, "scenes")]
save_path = opj(DATA_ROOT_DIR, "posed_images")


def parse_3rscan_intrinsics(info_path, key):
    """
    Reads intrinsics from 3RScan sequence/_info.txt.

    Expected keys are usually:
      m_calibrationColorIntrinsic
      m_calibrationDepthIntrinsic
    """
    with open(info_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith(key):
            values = line.split("=")[1].strip().split()
            values = [float(v) for v in values]

            # Sometimes stored as 4x4 = 16 values
            if len(values) == 16:
                mat = np.array(values, dtype=np.float32).reshape(4, 4)

            # Sometimes stored as 3x3 = 9 values
            elif len(values) == 9:
                mat3 = np.array(values, dtype=np.float32).reshape(3, 3)
                mat = np.eye(4, dtype=np.float32)
                mat[:3, :3] = mat3

            else:
                raise ValueError(f"Unexpected intrinsic size for {key}: {len(values)} values")

            return mat

    raise FileNotFoundError(f"Could not find {key} in {info_path}")


def parse_3rscan_dimensions(info_path):
    color_w = color_h = depth_w = depth_h = None
    with open(info_path, "r") as f:
        for line in f:
            if line.startswith("m_colorWidth"):
                color_w = int(float(line.split("=")[1]))
            elif line.startswith("m_colorHeight"):
                color_h = int(float(line.split("=")[1]))
            elif line.startswith("m_depthWidth"):
                depth_w = int(float(line.split("=")[1]))
            elif line.startswith("m_depthHeight"):
                depth_h = int(float(line.split("=")[1]))
    return color_w, color_h, depth_w, depth_h


def scale_intrinsics(mat, src_w, src_h, dst_w, dst_h):
    scaled = mat.copy().astype(np.float32)
    scaled[0, 0] *= float(dst_w) / float(src_w)
    scaled[0, 2] *= float(dst_w) / float(src_w)
    scaled[1, 1] *= float(dst_h) / float(src_h)
    scaled[1, 2] *= float(dst_h) / float(src_h)
    return scaled


for path in path_list:
    if DATASET == 'ScanNet':
        for scene_dir in ol(path):
            os.makedirs(opj(save_path, scene_dir), exist_ok=True)
            shutil.copyfile(opj(path, scene_dir, 'intrinsics', 'intrinsic_color.txt'), opj(save_path, scene_dir, 'intrinsics_color.txt'))
            shutil.copyfile(opj(path, scene_dir, 'intrinsics', 'intrinsic_depth.txt'), opj(save_path, scene_dir, 'intrinsics_depth.txt'))
            for img in ol(opj(path, scene_dir, 'color')):
                shutil.copyfile(opj(path, scene_dir, 'color', img), opj(save_path, scene_dir, img))

            for img in ol(opj(path, scene_dir, 'depth')):
                shutil.copyfile(opj(path, scene_dir, 'depth', img), opj(save_path, scene_dir, img))

            for img in ol(opj(path, scene_dir, 'pose')):
                shutil.copyfile(opj(path, scene_dir, 'pose', img), opj(save_path, scene_dir, img))
    elif DATASET == '3RScan':
        for scene_dir in SCAN_IDS: # ol(path):
            scene_path = opj(path, scene_dir)
            seq_path = opj(scene_path, "sequence")

            if not os.path.isdir(seq_path):
                print(f"[SKIP] No sequence folder found for {scene_dir}")
                continue

            out_scene_path = opj(save_path, scene_dir)
            os.makedirs(out_scene_path, exist_ok=True)

            info_path = opj(seq_path, "_info.txt")

            if os.path.exists(info_path):
                K_color = parse_3rscan_intrinsics(
                    info_path,
                    "m_calibrationColorIntrinsic",
                )
                K_depth = parse_3rscan_intrinsics(
                    info_path,
                    "m_calibrationDepthIntrinsic",
                )
                color_w, color_h, depth_w, depth_h = parse_3rscan_dimensions(info_path)

                K_depth_out = K_depth

                files = ol(seq_path)
                color_files = sorted([f for f in files if f.endswith(".color.jpg")])
                if color_files and color_w and color_h:
                    first_frame_id = (
                        color_files[0].replace("frame-", "").replace(".color.jpg", "")
                    )
                    first_depth = opj(seq_path, f"frame-{first_frame_id}.depth.pgm")
                    if os.path.exists(first_depth):
                        with Image.open(first_depth) as depth_img:
                            actual_depth_w, actual_depth_h = depth_img.size
                        if (
                            depth_w is not None
                            and depth_h is not None
                            and (actual_depth_w != depth_w or actual_depth_h != depth_h)
                        ):
                            # Predicted Pi3X/MUSt3R depths live on a resized color-image grid
                            # rather than the original 3RScan depth sensor grid. Use the
                            # color intrinsics scaled to the actual stored depth resolution
                            # so visibility checks in graph clustering stay geometrically
                            # consistent with the predicted depth map.
                            K_depth_out = scale_intrinsics(
                                K_color,
                                color_w,
                                color_h,
                                actual_depth_w,
                                actual_depth_h,
                            )
                            print(
                                f"[INFO] Depth size mismatch for {scene_dir}: "
                                f"_info depth={depth_w}x{depth_h}, actual={actual_depth_w}x{actual_depth_h}. "
                                "Using scaled color intrinsics for depth."
                            )

                np.savetxt(opj(out_scene_path, "intrinsic_color.txt"), K_color)
                np.savetxt(opj(out_scene_path, "intrinsics_color.txt"), K_color)
                np.savetxt(opj(out_scene_path, "intrinsic_depth.txt"), K_depth_out)
                np.savetxt(opj(out_scene_path, "intrinsics_depth.txt"), K_depth_out)
            else:
                print(f"[WARNING] No _info.txt found for {scene_dir}")

            # Copy RGB, depth, and pose files.
            # We rename them to match ScanNet-style format:
            #   000000.jpg
            #   000000.png
            #   000000.txt
            files = ol(seq_path)
            color_files = sorted([f for f in files if f.endswith(".color.jpg")])

            for color_file in color_files:
                # frame-000000.color.jpg -> 000000
                frame_id = color_file.replace("frame-", "").replace(".color.jpg", "")

                src_color = opj(seq_path, f"frame-{frame_id}.color.jpg")
                src_depth = opj(seq_path, f"frame-{frame_id}.depth.pgm")
                src_pose = opj(seq_path, f"frame-{frame_id}.pose.txt")

                dst_color = opj(out_scene_path, f"{frame_id}.jpg")
                dst_depth = opj(out_scene_path, f"{frame_id}.png")
                dst_pose = opj(out_scene_path, f"{frame_id}.txt")

                if os.path.exists(src_color):
                    shutil.copyfile(src_color, dst_color)

                if os.path.exists(src_depth):
                    # Convert 3RScan .pgm depth to .png
                    depth = Image.open(src_depth)
                    depth.save(dst_depth)

                if os.path.exists(src_pose):
                    shutil.copyfile(src_pose, dst_pose)

            print(f"[OK] Processed {scene_dir}: {len(color_files)} frames")
