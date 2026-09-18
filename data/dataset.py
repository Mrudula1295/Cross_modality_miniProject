import os
import re
import glob
import zipfile
import requests
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .synthesize_ir import synthesize_dataset, generate_synthetic_ir
from .prepare_sysu_regdb import SYSUPrepare

# Fallback download URLs & Google Drive IDs for Market-1501 baseline
GDRIVE_FILE_IDS = [
    "0B8-rUzbwVRk0c054eEozWG9COHM",
    "0B8-BflA1ovbwd0hway13Y2EzZzA"
]

MARKET1501_FALLBACK_URLS = [
    "http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip",
    "https://openmmlab.oss-cn-hangzhou.aliyuncs.com/datasets/Market-1501-v15.09.15.zip"
]


class Market1501Manager:
    """
    Automated dataset manager for Market-1501 baseline. Downloads the RGB dataset via torchreid, gdown,
    or direct mirrors, and automatically triggers synthetic IR generation.
    """
    def __init__(self, root_dir: str = "./data_store"):
        self.root_dir = os.path.abspath(root_dir)
        self.market_dir = os.path.join(self.root_dir, "market1501")
        self.rgb_dataset_dir = os.path.join(self.market_dir, "Market-1501-v15.09.15")
        self.synthetic_ir_dir = os.path.join(self.market_dir, "synthetic_ir")
        
    def acquire_dataset(self) -> tuple:
        """
        Ensures both Market-1501 RGB and paired Synthetic-IR datasets are ready.
        Returns tuple of (rgb_dataset_dir, synthetic_ir_dir).
        """
        os.makedirs(self.market_dir, exist_ok=True)
        
        # Check if RGB dataset exists
        train_rgb_path = os.path.join(self.rgb_dataset_dir, "bounding_box_train")
        if not os.path.exists(train_rgb_path) or len(glob.glob(os.path.join(train_rgb_path, "*.jpg"))) == 0:
            print("[Dataset Manager] Market-1501 RGB dataset not found locally. Initiating automated download...")
            self._download_market1501()
        else:
            print(f"[Dataset Manager] Verified Market-1501 RGB dataset at: {self.rgb_dataset_dir}")
            
        # Check if Synthetic IR dataset exists
        train_ir_path = os.path.join(self.synthetic_ir_dir, "bounding_box_train")
        if not os.path.exists(train_ir_path) or len(glob.glob(os.path.join(train_ir_path, "*.jpg"))) == 0:
            print("[Dataset Manager] Synthetic IR dataset missing or incomplete. Synthesizing IR pairs now...")
            synthesize_dataset(self.rgb_dataset_dir, self.synthetic_ir_dir)
        else:
            print(f"[Dataset Manager] Verified Synthetic-IR dataset at: {self.synthetic_ir_dir}")
            
        return self.rgb_dataset_dir, self.synthetic_ir_dir
        
    def _download_market1501(self):
        """Attempts torchreid download first, gdown Google Drive second, direct mirrors third."""
        download_success = False
        
        try:
            import torchreid
            print("[Dataset Manager] Attempting download via torchreid...")
            torchreid.data.datasets.image.Market1501(root=self.market_dir)
            if os.path.exists(os.path.join(self.rgb_dataset_dir, "bounding_box_train")):
                download_success = True
        except Exception as e:
            print(f"[Dataset Manager] torchreid download notice: {e}")
            
        if not download_success:
            try:
                import gdown
                zip_path = os.path.join(self.market_dir, "Market-1501-v15.09.15.zip")
                for gid in GDRIVE_FILE_IDS:
                    print(f"[Dataset Manager] Attempting Google Drive download via gdown (ID: {gid})...")
                    gdown.download(id=gid, output=zip_path, quiet=False)
                    if os.path.exists(zip_path) and os.path.getsize(zip_path) > 1000000:
                        print("[Dataset Manager] Download successful. Extracting archive...")
                        with zipfile.ZipFile(zip_path, "r") as zip_ref:
                            zip_ref.extractall(self.market_dir)
                        download_success = True
                        break
            except Exception as e:
                print(f"[Dataset Manager] gdown attempt notice: {e}")

        if not download_success:
            zip_path = os.path.join(self.market_dir, "Market-1501-v15.09.15.zip")
            for url in MARKET1501_FALLBACK_URLS:
                try:
                    print(f"[Dataset Manager] Attempting HTTP download from: {url}")
                    response = requests.get(url, stream=True, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
                    if response.status_code == 200:
                        with open(zip_path, "wb") as f:
                            for chunk in response.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    f.write(chunk)
                        print("[Dataset Manager] Download complete. Extracting dataset archive...")
                        with zipfile.ZipFile(zip_path, "r") as zip_ref:
                            zip_ref.extractall(self.market_dir)
                        download_success = True
                        break
                except Exception as ex:
                    print(f"[Dataset Manager] Mirror download attempt failed ({url}): {ex}")
                    
        if not download_success or not os.path.exists(os.path.join(self.rgb_dataset_dir, "bounding_box_train")):
            print("\n[Dataset Manager] Generating synthetic Market-1501 RGB dataset locally for fallback...")
            self._create_synthetic_placeholder_rgb_dataset()

    def _create_synthetic_placeholder_rgb_dataset(self, num_identities: int = 20, images_per_id: int = 4):
        os.makedirs(self.rgb_dataset_dir, exist_ok=True)
        splits = ["bounding_box_train", "bounding_box_test", "query"]
        for s in splits:
            os.makedirs(os.path.join(self.rgb_dataset_dir, s), exist_ok=True)
            
        for pid in range(1, num_identities + 1):
            for img_idx in range(images_per_id):
                camid = (img_idx % 6) + 1
                filename = f"{pid:04d}_c{camid}s1_{img_idx:06d}_00.jpg"
                img_np = np.zeros((128, 64, 3), dtype=np.uint8)
                color = [int(c) for c in np.random.randint(50, 255, 3)]
                img_np[:, :] = color
                img_pil = Image.fromarray(img_np)
                
                if pid <= num_identities // 2:
                    img_pil.save(os.path.join(self.rgb_dataset_dir, "bounding_box_train", filename))
                else:
                    if img_idx % 2 == 0:
                        img_pil.save(os.path.join(self.rgb_dataset_dir, "query", filename))
                    else:
                        img_pil.save(os.path.join(self.rgb_dataset_dir, "bounding_box_test", filename))


def parse_market1501_filename(filepath: str) -> tuple:
    filename = os.path.basename(filepath)
    pattern = re.compile(r"([-\d]+)_c(\d)")
    match = pattern.search(filename)
    if not match:
        return -1, -1
    pid, camid = map(int, match.groups())
    return pid, camid - 1


class BaseCrossModalDataset(Dataset):
    """
    Unified Cross-Modality PyTorch Dataset for RGB and Infrared image pairs.
    """
    def __init__(
        self,
        rgb_image_paths: list,
        ir_image_paths: list,
        pids: list,
        camids: list,
        transform=None,
        is_train: bool = True
    ):
        self.rgb_paths = rgb_image_paths
        self.ir_paths = ir_image_paths
        self.raw_pids = pids
        self.camids = camids
        self.transform = transform
        self.is_train = is_train
        
        # Relabel PIDs to contiguous range 0..N-1 for classification
        self.unique_pids = sorted(list(set(pids)))
        self.pid_map = {pid: i for i, pid in enumerate(self.unique_pids)}
        self.pids = [self.pid_map[pid] for pid in pids]
        self.num_classes = len(self.unique_pids)
        
    def __len__(self):
        return len(self.rgb_paths)
        
    def __getitem__(self, index):
        rgb_path = self.rgb_paths[index]
        ir_path = self.ir_paths[index]
        pid = self.pids[index]
        camid = self.camids[index]
        
        rgb_img = Image.open(rgb_path).convert("RGB")
        ir_img = Image.open(ir_path).convert("RGB")
        
        if self.transform is not None:
            rgb_img = self.transform(rgb_img)
            ir_img = self.transform(ir_img)
            
        return {
            "rgb": rgb_img,
            "ir": ir_img,
            "pid": torch.tensor(pid, dtype=torch.long),
            "camid": torch.tensor(camid, dtype=torch.long),
            "raw_pid": torch.tensor(self.raw_pids[index], dtype=torch.long)
        }


class EvaluationModalDataset(Dataset):
    """
    Dataset for single modality evaluation (Query or Gallery set).
    Returns (img_tensor, pid, camid, modality_str).
    """
    def __init__(self, image_paths: list, pids: list, camids: list, modality: str, transform=None):
        self.image_paths = image_paths
        self.pids = pids
        self.camids = camids
        self.modality = modality
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, index):
        img_path = self.image_paths[index]
        pid = self.pids[index]
        camid = self.camids[index]
        
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
            
        return {
            "img": img,
            "pid": torch.tensor(pid, dtype=torch.long),
            "camid": torch.tensor(camid, dtype=torch.long),
            "modality": self.modality,
            "path": img_path
        }


def get_transforms(img_size=(224, 224)):
    """Creates training and testing data augmentations."""
    if isinstance(img_size, list):
        img_size = tuple(img_size)
        
    train_transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def get_cross_modal_dataloaders(
    root_dir: str = None,
    batch_size: int = 16,
    img_size=(224, 224),
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    dataset_name: str = "sysu_mm01",
    subset_size: int = 2000
) -> tuple:
    """
    Main loader interface initializing PyTorch DataLoaders for training and evaluation.
    Supports REAL SYSU-MM01 dataset ("sysu_mm01") and Market-1501 synthetic baseline ("market1501_synthetic_ir").
    """
    train_transform, val_transform = get_transforms(img_size)

    if dataset_name == "sysu_mm01":
        # Resolve SYSU-MM01 root directory (Explicit root_dir -> Colab Drive -> Colab local -> local fallback)
        target_sysu_dir = None
        if root_dir and os.path.exists(root_dir):
            if any(os.path.exists(os.path.join(root_dir, c)) for c in ["cam1", "cam2", "cam3", "cam4", "cam5", "cam6"]):
                target_sysu_dir = root_dir
        
        if target_sysu_dir is None and not root_dir:
            for p in ["/content/drive/MyDrive/SYSU-MM01", "/content/SYSU-MM01", "./data_store/sysu_mm01"]:
                if os.path.exists(p) and any(os.path.exists(os.path.join(p, c)) for c in ["cam1", "cam2", "cam3", "cam4", "cam5", "cam6"]):
                    target_sysu_dir = p
                    break

        if target_sysu_dir is None:
            sysu_path_err = root_dir if root_dir else "./data_store/sysu_mm01"
            raise FileNotFoundError(
                f"\n[ERROR] Real SYSU-MM01 dataset not found at: '{sysu_path_err}'\n"
                "Please place SYSU-MM01 (containing cam1..cam6 & exp/) at './data_store/sysu_mm01' "
                "or specify the correct path in configs/default.yaml under dataset.root_dir.\n"
                "Refusing silent fallback to synthetic dataset when 'sysu_mm01' is configured."
            )

        # Parse Real SYSU-MM01 Dataset
        sysu_prep = SYSUPrepare(sysu_dir=target_sysu_dir, subset_size=subset_size)
        sysu_data = sysu_prep.parse_dataset()
        
        train_pairs = sysu_data["train_pairs"]
        rgb_paths = [p[0] for p in train_pairs]
        ir_paths = [p[1] for p in train_pairs]
        pids = [p[2] for p in train_pairs]
        camids = [p[3] for p in train_pairs]

        train_dataset = BaseCrossModalDataset(
            rgb_image_paths=rgb_paths,
            ir_image_paths=ir_paths,
            pids=pids,
            camids=camids,
            transform=train_transform,
            is_train=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            drop_last=True
        )

        # Test Data: Query IR (cam3,6) and Gallery RGB (cam1,2,4,5)
        query_ir = sysu_data["query_ir"]
        gallery_rgb = sysu_data["gallery_rgb"]

        query_ir_paths = [q["path"] for q in query_ir]
        query_ir_pids = [q["pid"] for q in query_ir]
        query_ir_camids = [q["camid"] for q in query_ir]

        gallery_rgb_paths = [g["path"] for g in gallery_rgb]
        gallery_rgb_pids = [g["pid"] for g in gallery_rgb]
        gallery_rgb_camids = [g["camid"] for g in gallery_rgb]

        query_ir_loader = DataLoader(
            EvaluationModalDataset(query_ir_paths, query_ir_pids, query_ir_camids, "ir", val_transform),
            batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )
        gallery_rgb_loader = DataLoader(
            EvaluationModalDataset(gallery_rgb_paths, gallery_rgb_pids, gallery_rgb_camids, "rgb", val_transform),
            batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )

        eval_loaders = {
            "ir_to_rgb": {"query": query_ir_loader, "gallery": gallery_rgb_loader},
            "rgb_to_ir": {"query": gallery_rgb_loader, "gallery": query_ir_loader}
        }

        return train_loader, eval_loaders, train_dataset.num_classes

    elif dataset_name == "market1501_synthetic_ir":
        # Baseline Market-1501 + Synthetic IR loader
        target_root = root_dir if root_dir else "./data_store"
        manager = Market1501Manager(root_dir=target_root)
        rgb_dir, ir_dir = manager.acquire_dataset()
        
        train_rgb_files = sorted(glob.glob(os.path.join(rgb_dir, "bounding_box_train", "*.jpg")))
        rgb_paths, ir_paths, pids, camids = [], [], [], []
        
        for rgb_path in train_rgb_files:
            filename = os.path.basename(rgb_path)
            pid, camid = parse_market1501_filename(filename)
            if pid <= 0:
                continue
            ir_path = os.path.join(ir_dir, "bounding_box_train", filename)
            if os.path.exists(ir_path):
                rgb_paths.append(rgb_path)
                ir_paths.append(ir_path)
                pids.append(pid)
                camids.append(camid)
                
        train_dataset = BaseCrossModalDataset(
            rgb_image_paths=rgb_paths,
            ir_image_paths=ir_paths,
            pids=pids,
            camids=camids,
            transform=train_transform,
            is_train=True
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            drop_last=True
        )
        
        def parse_eval_split(split_name: str, base_dir: str):
            split_files = sorted(glob.glob(os.path.join(base_dir, split_name, "*.jpg")))
            paths, e_pids, e_cids = [], [], []
            for p in split_files:
                pid, camid = parse_market1501_filename(p)
                if pid <= 0:
                    continue
                paths.append(p)
                e_pids.append(pid)
                e_cids.append(camid)
            return paths, e_pids, e_cids

        query_rgb_paths, query_pids, query_camids = parse_eval_split("query", rgb_dir)
        gallery_ir_paths, gallery_pids, gallery_camids = parse_eval_split("bounding_box_test", ir_dir)
        query_ir_paths, query_ir_pids, query_ir_camids = parse_eval_split("query", ir_dir)
        gallery_rgb_paths, gallery_rgb_pids, gallery_rgb_camids = parse_eval_split("bounding_box_test", rgb_dir)

        query_rgb_loader = DataLoader(
            EvaluationModalDataset(query_rgb_paths, query_pids, query_camids, "rgb", val_transform),
            batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )
        gallery_ir_loader = DataLoader(
            EvaluationModalDataset(gallery_ir_paths, gallery_pids, gallery_camids, "ir", val_transform),
            batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )
        query_ir_loader = DataLoader(
            EvaluationModalDataset(query_ir_paths, query_ir_pids, query_ir_camids, "ir", val_transform),
            batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )
        gallery_rgb_loader = DataLoader(
            EvaluationModalDataset(gallery_rgb_paths, gallery_rgb_pids, gallery_rgb_camids, "rgb", val_transform),
            batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        )

        eval_loaders = {
            "rgb_to_ir": {"query": query_rgb_loader, "gallery": gallery_ir_loader},
            "ir_to_rgb": {"query": query_ir_loader, "gallery": gallery_rgb_loader}
        }
        
        return train_loader, eval_loaders, train_dataset.num_classes

    else:
        raise ValueError(f"Unknown dataset name '{dataset_name}'. Supported options: 'sysu_mm01', 'market1501_synthetic_ir'.")
