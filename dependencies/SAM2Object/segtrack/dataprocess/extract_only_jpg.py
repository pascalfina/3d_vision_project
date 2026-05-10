import os
import shutil

opj = os.path.join
ol = os.listdir

opj = os.path.join
ol = os.listdir

DATASET = os.environ.get("DATASET", "3RScan")
DATA_ROOT_DIR = os.environ.get("DATA_ROOT_DIR", "/cluster/scratch/ealegret/sam2object")
SCAN_IDS_ENV = os.environ.get("SCAN_IDS", "")
SCAN_IDS = [s.strip() for s in SCAN_IDS_ENV.split(",") if s.strip()]

path_list = [opj(DATA_ROOT_DIR, "scenes")]
save_path = opj(DATA_ROOT_DIR, "color_images_cluster")

os.makedirs(save_path,exist_ok=True)

frame_step = 1
for path in path_list:
    if DATASET == 'ScanNet':
        for scene_dir in ol(path):
            if os.path.exists(opj(save_path, scene_dir)):
                continue
            
            os.makedirs(opj(save_path, scene_dir), exist_ok=True)
            for img in ol(opj(path, scene_dir, 'color')):
                file_prefix = int(img.split('.')[0])
                if file_prefix % frame_step != 0:
                    continue
                shutil.copyfile(opj(path, scene_dir, 'color', img), opj(save_path, scene_dir, str(int(file_prefix / frame_step))+'.jpg'))
    elif DATASET == '3RScan':
        for scene_dir in SCAN_IDS: 
            scene_path = opj(path, scene_dir)
            seq_path = opj(scene_path, "sequence")

            if not os.path.isdir(seq_path):
                print(f"[SKIP] No sequence folder found for {scene_dir}")
                continue

            out_scene_path = opj(save_path, scene_dir)
            os.makedirs(out_scene_path, exist_ok=True)

            color_files = sorted([
                f for f in ol(seq_path)
                if f.endswith(".color.jpg")
            ])

            copied = 0

            for color_file in color_files:
                # frame-000000.color.jpg -> 000000
                frame_id = color_file.replace("frame-", "").replace(".color.jpg", "")
                frame_idx = int(frame_id)

                if frame_idx % frame_step != 0:
                    continue

                src_img = opj(seq_path, color_file)

                # Option A: keep original frame numbering
                # Output: 000000.jpg, 000005.jpg, 000010.jpg, ...
                dst_img = opj(out_scene_path, f"{frame_id}.jpg")

                shutil.copyfile(src_img, dst_img)
                copied += 1

            print(f"[OK] Processed 3RScan scene {scene_dir}: copied {copied} images")
