import os
import shutil

split            = os.environ.get("SPLIT", "val")  # or 'train' or test
DATASET          = os.environ.get("DATASET", "3RScan")
DATA_PATH        = os.environ.get("DATA_ROOT_DIR", "/cluster/scratch/ealegret/sam2object/scenes")
destination_root = DATA_PATH
base_dir         = os.environ.get("SAM2OBJECT_DIR", "/cluster/home/ealegret/3d_vision_project/dependencies/SAM2Object/segtrack/outputs")


VIEW_FREQ        = 1
def copy_png_files(source_dir, destination_dir):
    os.makedirs(destination_dir, exist_ok=True)

    for filename in os.listdir(source_dir):
        if filename.endswith('.png'):
            idx = int(filename.split('.')[0].split('_')[-1]) * VIEW_FREQ
            source_file = os.path.join(source_dir, filename)
            destination_file = os.path.join(destination_dir, f"maskraw_{idx:06d}.png")
            shutil.copy2(source_file, destination_file)
            print(f'Copied: {source_file} to {destination_file}')


# Select scans from SCAN_IDS (set by the batch script) for both datasets.
# Scenes whose mask_data_merge dir is absent are skipped in the loop below.
SCAN_IDS = [s.strip() for s in os.environ.get("SCAN_IDS", "").split(",") if s.strip()]
if not SCAN_IDS:
    raise ValueError("SCAN_IDS env is empty; set it (the batch script exports it)")
video_dir_scene_ids = SCAN_IDS


for scene in video_dir_scene_ids:
    source_directory = os.path.join(base_dir, scene, 'mask_data_merge')
    if not os.path.exists(source_directory):
        continue
    
    destination_directory = os.path.join(f'{destination_root}/2D_masks', scene, 'semantic-sam')
    #if not os.path.exists(destination_directory):
    #    print(f"Processing scene:")
    copy_png_files(source_directory, destination_directory)
