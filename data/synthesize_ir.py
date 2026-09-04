import os
import glob
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

def generate_synthetic_ir(
    rgb_image_path: str,
    save_ir_path: str,
    clahe_clip: float = 2.0,
    clahe_grid: tuple = (8, 8),
    colormap_code: int = cv2.COLORMAP_INFERNO
) -> bool:
    """
    Generates a synthetic Infrared (IR) counterpart for an RGB image.
    Process: Grayscale conversion -> CLAHE contrast enhancement -> Thermal colormap remapping.
    """
    img = cv2.imread(rgb_image_path)
    if img is None:
        return False
    
    # Step 1: Grayscale conversion
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step 2: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    gray_clahe = clahe.apply(gray)
    
    # Step 3: Thermal colormap remapping (e.g. INFERNO)
    thermal_img = cv2.applyColorMap(gray_clahe, colormap_code)
    
    # Ensure directory exists and write image
    os.makedirs(os.path.dirname(save_ir_path), exist_ok=True)
    return cv2.imwrite(save_ir_path, thermal_img)


def synthesize_dataset(
    market1501_dir: str,
    output_synthetic_ir_dir: str,
    clahe_clip: float = 2.0,
    clahe_grid: tuple = (8, 8)
) -> dict:
    """
    Processes all RGB images in Market-1501 split directories (bounding_box_train, bounding_box_test, query)
    and generates paired synthetic IR images with identical naming conventions.
    """
    splits = ["bounding_box_train", "bounding_box_test", "query"]
    summary = {}
    
    print("\n[Synthetic IR Generator] Starting RGB -> Synthetic IR dataset conversion...")
    for split in splits:
        src_split_dir = os.path.join(market1501_dir, split)
        dst_split_dir = os.path.join(output_synthetic_ir_dir, split)
        
        if not os.path.exists(src_split_dir):
            print(f"[Synthetic IR Generator] Warning: Source split directory not found: {src_split_dir}")
            continue
            
        os.makedirs(dst_split_dir, exist_ok=True)
        image_files = glob.glob(os.path.join(src_split_dir, "*.jpg")) + glob.glob(os.path.join(src_split_dir, "*.png"))
        
        print(f"[Synthetic IR Generator] Processing split '{split}' ({len(image_files)} images)...")
        count = 0
        for img_path in tqdm(image_files, desc=f"Synthesizing {split}"):
            filename = os.path.basename(img_path)
            dst_path = os.path.join(dst_split_dir, filename)
            
            # Skip if already synthesized
            if not os.path.exists(dst_path):
                success = generate_synthetic_ir(
                    rgb_image_path=img_path,
                    save_ir_path=dst_path,
                    clahe_clip=clahe_clip,
                    clahe_grid=clahe_grid
                )
                if success:
                    count += 1
            else:
                count += 1
                
        summary[split] = count
        print(f"[Synthetic IR Generator] Completed '{split}': {count} synthetic IR images ready.")
        
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic IR dataset from RGB images")
    parser.add_argument("--data-dir", type=str, default="./data_store/market1501/Market-1501-v15.09.15", help="Path to Market-1501 root")
    parser.add_argument("--out-dir", type=str, default="./data_store/market1501/synthetic_ir", help="Output path for synthetic IR dataset")
    args = parser.parse_args()
    
    if os.path.exists(args.data_dir):
        synthesize_dataset(args.data_dir, args.out_dir)
    else:
        print(f"Data directory {args.data_dir} does not exist. Please run training or dataset manager first.")
