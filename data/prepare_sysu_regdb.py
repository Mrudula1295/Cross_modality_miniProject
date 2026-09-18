import os
import re
import glob
import random
from pathlib import Path


class SYSUPrepare:
    """
    Parser for real SYSU-MM01 RGB-Infrared Cross-Modality Dataset.
    Camera Mapping:
      - Visible (RGB) Cameras: cam1, cam2, cam4, cam5
      - Infrared (IR) Cameras : cam3, cam6
    """
    VISIBLE_CAMS = {"cam1": 0, "cam2": 1, "cam4": 3, "cam5": 4}
    INFRARED_CAMS = {"cam3": 2, "cam6": 5}
    
    def __init__(self, sysu_dir: str, subset_size: int = 2000, seed: int = 42):
        self.sysu_dir = os.path.abspath(sysu_dir)
        self.subset_size = subset_size
        self.seed = seed
        
    def validate_dataset_path(self):
        """Validates that sysu_dir exists and contains at least one camera folder."""
        if not os.path.exists(self.sysu_dir):
            raise FileNotFoundError(
                f"\n[ERROR] SYSU-MM01 dataset path not found at: '{self.sysu_dir}'\n"
                "Please check configs/default.yaml or pass a valid dataset root directory.\n"
                "Do NOT fall back to synthetic data when sysu_mm01 is configured."
            )
            
        found_cams = [c for c in list(self.VISIBLE_CAMS.keys()) + list(self.INFRARED_CAMS.keys())
                      if os.path.exists(os.path.join(self.sysu_dir, c))]
                      
        if not found_cams:
            raise FileNotFoundError(
                f"\n[ERROR] Invalid SYSU-MM01 directory at '{self.sysu_dir}'.\n"
                "Expected camera subdirectories (cam1, cam2, cam3, cam4, cam5, cam6) were not found."
            )
            
    def _parse_id_file(self, file_path: str) -> list:
        """Parses comma-separated or newline-separated PID integers from exp/ files."""
        if not os.path.exists(file_path):
            return []
        pids = []
        with open(file_path, "r") as f:
            content = f.read().replace(",", " ")
            for token in content.split():
                token = token.strip()
                if token.isdigit():
                    pids.append(int(token))
        return sorted(list(set(pids)))

    def parse_dataset(self) -> dict:
        """
        Parses SYSU-MM01 directory structure, reads train/test splits from exp/ files,
        and creates training pairs and evaluation sets.
        """
        self.validate_dataset_path()
        exp_dir = os.path.join(self.sysu_dir, "exp")
        
        # Read official train and test PIDs
        train_pids = self._parse_id_file(os.path.join(exp_dir, "train_id.txt"))
        val_pids = self._parse_id_file(os.path.join(exp_dir, "val_id.txt"))
        test_pids = self._parse_id_file(os.path.join(exp_dir, "test_id.txt"))
        
        # Scan image files per camera
        cam_files = {}
        all_pids_scanned = set()
        total_rgb_count = 0
        total_ir_count = 0

        for cam_name, cam_idx in {**self.VISIBLE_CAMS, **self.INFRARED_CAMS}.items():
            cam_path = os.path.join(self.sysu_dir, cam_name)
            if not os.path.exists(cam_path):
                continue
                
            # Scan files inside subfolders (camX/0001/0001.jpg) or flat files (camX/0001_0001.jpg)
            image_files = glob.glob(os.path.join(cam_path, "**", "*.jpg"), recursive=True) + \
                          glob.glob(os.path.join(cam_path, "**", "*.png"), recursive=True)
                          
            cam_files[cam_name] = []
            for fpath in image_files:
                # Extract PID from folder name or filename
                parts = os.path.normpath(fpath).split(os.sep)
                parent_dir = parts[-2]
                filename = parts[-1]
                
                pid = -1
                if parent_dir.isdigit():
                    pid = int(parent_dir)
                else:
                    match = re.search(r"(\d{4})", filename)
                    if match:
                        pid = int(match.group(1))
                        
                if pid > 0:
                    all_pids_scanned.add(pid)
                    cam_files[cam_name].append({
                        "path": fpath,
                        "pid": pid,
                        "camid": cam_idx,
                        "modality": "rgb" if cam_name in self.VISIBLE_CAMS else "ir"
                    })
                    if cam_name in self.VISIBLE_CAMS:
                        total_rgb_count += 1
                    else:
                        total_ir_count += 1

        # If train_pids or test_pids were not specified in exp/, derive them deterministically
        all_sorted_pids = sorted(list(all_pids_scanned))
        if not train_pids or not test_pids:
            random.seed(self.seed)
            shuffled = list(all_sorted_pids)
            random.shuffle(shuffled)
            split_idx = int(len(shuffled) * 0.8)
            train_pids = sorted(shuffled[:split_idx])
            test_pids = sorted(shuffled[split_idx:])

        # Build Training Data: Group training images by PID and modality
        train_pids_set = set(train_pids) | set(val_pids)
        train_rgb_by_pid = {}
        train_ir_by_pid = {}

        for cam_name, files in cam_files.items():
            is_rgb = cam_name in self.VISIBLE_CAMS
            for item in files:
                pid = item["pid"]
                if pid in train_pids_set:
                    if is_rgb:
                        train_rgb_by_pid.setdefault(pid, []).append(item)
                    else:
                        train_ir_by_pid.setdefault(pid, []).append(item)

        # Pair RGB and IR images per identity
        active_train_pids = sorted(list(set(train_rgb_by_pid.keys()) & set(train_ir_by_pid.keys())))
        all_train_pairs = []

        for pid in active_train_pids:
            rgb_list = train_rgb_by_pid[pid]
            ir_list = train_ir_by_pid[pid]
            
            # Create paired tuples (rgb_path, ir_path, pid, camid)
            for i, rgb_item in enumerate(rgb_list):
                ir_item = ir_list[i % len(ir_list)]  # Cyclically pair
                all_train_pairs.append((rgb_item["path"], ir_item["path"], pid, rgb_item["camid"]))

        # Identity-Aware Reproducible Subset Sampling
        if self.subset_size and self.subset_size > 0 and len(all_train_pairs) > self.subset_size:
            random.seed(self.seed)
            selected_pairs = []
            pairs_per_pid = max(1, self.subset_size // len(active_train_pids))
            
            # Sample balanced pairs per identity
            for pid in active_train_pids:
                pid_pairs = [p for p in all_train_pairs if p[2] == pid]
                random.shuffle(pid_pairs)
                selected_pairs.extend(pid_pairs[:pairs_per_pid])
                
            # If slightly short of target, fill up to target subset_size
            if len(selected_pairs) < self.subset_size:
                remaining = [p for p in all_train_pairs if p not in set(selected_pairs)]
                random.shuffle(remaining)
                selected_pairs.extend(remaining[: self.subset_size - len(selected_pairs)])
                
            train_pairs = selected_pairs[: self.subset_size]
        else:
            train_pairs = all_train_pairs

        # Build Test Data (Query: IR, Gallery: RGB for test PIDs)
        test_pids_set = set(test_pids)
        query_ir = []
        gallery_rgb = []

        for cam_name, files in cam_files.items():
            is_rgb = cam_name in self.VISIBLE_CAMS
            for item in files:
                if item["pid"] in test_pids_set:
                    if is_rgb:
                        gallery_rgb.append(item)
                    else:
                        query_ir.append(item)

        # Print Dataset Summary Banner
        banner = f"""
================================================================================
[DATASET SUMMARY] REAL SYSU-MM01 RGB-Infrared Dataset Loaded
--------------------------------------------------------------------------------
Dataset Root Path   : {self.sysu_dir}
Total RGB Images    : {total_rgb_count:,}
Total IR Images     : {total_ir_count:,}
Total Identities    : {len(all_sorted_pids)}
Train Identities    : {len(active_train_pids)}
Test Identities     : {len(test_pids_set)}
Selected Train Pairs: {len(train_pairs):,} (Identity-Balanced Reproducible Subset)
================================================================================
        """
        print(banner)

        return {
            "train_pairs": train_pairs,
            "query_ir": query_ir,
            "gallery_rgb": gallery_rgb,
            "num_train_classes": len(active_train_pids),
            "summary": {
                "sysu_dir": self.sysu_dir,
                "rgb_count": total_rgb_count,
                "ir_count": total_ir_count,
                "total_pids": len(all_sorted_pids),
                "train_pids": len(active_train_pids),
                "test_pids": len(test_pids_set),
                "train_pairs": len(train_pairs)
            }
        }


class RegDBPrepare:
    """
    Parser for RegDB dataset (Dual-Camera RGB-Infrared Dataset).
    Contains 412 identities, each having 10 RGB and 10 Thermal images.
    """
    def __init__(self, regdb_dir: str, trial: int = 1):
        self.regdb_dir = regdb_dir
        self.trial = trial
        
    def parse_split(self):
        """Parses RegDB trial split files."""
        if not os.path.exists(self.regdb_dir):
            print(f"[RegDBPrepare] Directory {self.regdb_dir} does not exist. Skipping.")
            return [], [], [], []
            
        print(f"[RegDBPrepare] Parsing RegDB trial {self.trial} structure...")
        rgb_paths, ir_paths, pids, camids = [], [], [], []
        train_rgb_list = os.path.join(self.regdb_dir, "idx", f"train_visible_{self.trial}.txt")
        train_ir_list = os.path.join(self.regdb_dir, "idx", f"train_thermal_{self.trial}.txt")
        
        if os.path.exists(train_rgb_list) and os.path.exists(train_ir_list):
            with open(train_rgb_list, "r") as f_rgb, open(train_ir_list, "r") as f_ir:
                for line_rgb, line_ir in zip(f_rgb, f_ir):
                    r_path, pid_r = line_rgb.strip().split()
                    i_path, pid_i = line_ir.strip().split()
                    rgb_paths.append(os.path.join(self.regdb_dir, r_path))
                    ir_paths.append(os.path.join(self.regdb_dir, i_path))
                    pids.append(int(pid_r))
                    camids.append(0)
                    
        return rgb_paths, ir_paths, pids, camids


if __name__ == "__main__":
    print("SYSU-MM01 / RegDB Preparation Utility Ready.")
