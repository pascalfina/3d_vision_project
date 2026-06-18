import os
import cv2
import numpy as np

SCENE_ID = "5341b7e3-8a66-2cdd-8709-66a2159f0017"
FRAME_ID = 5

rgb_path = f"/cluster/scratch/ealegret/3Rscan/color_images_cluster/{SCENE_ID}/{FRAME_ID:06d}.jpg"
mask_path = f"/cluster/scratch/ealegret/3Rscan/2D_masks/{SCENE_ID}/semantic-sam/maskraw_{FRAME_ID:06d}.png"

out_dir = f"/cluster/scratch/ealegret/debug_2d_masks/{SCENE_ID}"
os.makedirs(out_dir, exist_ok=True)

rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
if rgb is None:
    raise FileNotFoundError(f"Could not read RGB: {rgb_path}")

mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
if mask is None:
    raise FileNotFoundError(f"Could not read mask: {mask_path}")

print("RGB path:", rgb_path)
print("Mask path:", mask_path)
print("RGB shape:", rgb.shape, rgb.dtype)
print("Mask shape:", mask.shape, mask.dtype)
print("Mask min/max:", mask.min(), mask.max())

ids = np.unique(mask)
valid_ids = ids[ids > 0]
print("Number of mask IDs:", len(valid_ids))
print("First IDs:", valid_ids[:30])

# Create random but deterministic colours for each ID
rng = np.random.default_rng(12345)
colour_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)

for obj_id in valid_ids:
    colour = rng.integers(40, 255, size=3, dtype=np.uint8)
    colour_mask[mask == obj_id] = colour

# Overlay
alpha = 0.55
overlay = rgb.copy()
fg = mask > 0
overlay[fg] = cv2.addWeighted(
    rgb[fg],
    1 - alpha,
    colour_mask[fg],
    alpha,
    0,
)

# Save results
cv2.imwrite(os.path.join(out_dir, f"rgb_{FRAME_ID:06d}.jpg"), rgb)
cv2.imwrite(os.path.join(out_dir, f"mask_coloured_{FRAME_ID:06d}.png"), colour_mask)
cv2.imwrite(os.path.join(out_dir, f"overlay_{FRAME_ID:06d}.jpg"), overlay)

print("Saved:")
print(os.path.join(out_dir, f"rgb_{FRAME_ID:06d}.jpg"))
print(os.path.join(out_dir, f"mask_coloured_{FRAME_ID:06d}.png"))
print(os.path.join(out_dir, f"overlay_{FRAME_ID:06d}.jpg"))