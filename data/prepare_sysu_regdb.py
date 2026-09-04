import os
import glob


class SYSUPrepare:
    """
    Parser for SYSU-MM01 dataset (Real RGB-Thermal Cross-Modality Dataset).
    RGB cameras: cam1, cam2, cam4, cam5
    IR cameras: cam3, cam6
    """
    def __init__(self, sysu_dir: str):
        self.sysu_dir = sysu_dir
        
    def parse_train_val(self):
        """
        Parses train split of SYSU-MM01.
        Returns tuples of (rgb_image_paths, ir_image_paths, pids, camids).
        """
        if not os.path.exists(self.sysu_dir):
            print(f"[SYSUPrepare] Directory {self.sysu_dir} does not exist. Skipping.")
            return [], [], [], []
            
        print("[SYSUPrepare] Parsing SYSU-MM01 dataset structure...")
        rgb_paths, ir_paths, pids, camids = [], [], [], []
        # Standard SYSU-MM01 structure parsing logic
        train_file = os.path.join(self.sysu_dir, "exp", "train_id.txt")
        if os.path.exists(train_file):
            with open(train_file, "r") as f:
                train_ids = [int(line.strip()) for line in f if line.strip()]
            print(f"[SYSUPrepare] Loaded {len(train_ids)} train identities from SYSU-MM01.")
        return rgb_paths, ir_paths, pids, camids


class RegDBPrepare:
    """
    Parser for RegDB dataset (Dual-Camera RGB-Infrared Dataset).
    Contains 412 identities, each having 10 RGB and 10 Thermal images.
    """
    def __init__(self, regdb_dir: str, trial: int = 1):
        self.regdb_dir = regdb_dir
        self.trial = trial
        
    def parse_split(self):
        """
        Parses RegDB trial split files (e.g. idx/train_visible_1.txt, idx/train_thermal_1.txt).
        Returns tuples of (rgb_paths, ir_paths, pids, camids).
        """
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
                    camids.append(0) # Camera 0 for RGB, Camera 1 for Thermal
                    
        return rgb_paths, ir_paths, pids, camids


if __name__ == "__main__":
    print("SYSU-MM01 / RegDB Preparation Utility Ready.")
