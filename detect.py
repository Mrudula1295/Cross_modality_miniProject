import os
import argparse
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from models.vit_reid import ViTCrossModalReID


class PersonDetectorReID:
    """
    YOLOv8 Person Detector and ViT Re-ID Feature Extractor Pipeline.
    Detects person bounding boxes from raw images/videos, crops them,
    and extracts cross-modality feature embeddings.
    """
    def __init__(self, weights_path: str = None, config_path: str = "./configs/default.yaml", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Initialize YOLOv8 Detector
        if YOLO is None:
            raise ImportError("ultralytics package is required for detect.py. Please run 'pip install ultralytics'.")
        print("[Detector] Loading YOLOv8 person detection model (yolov8n.pt)...")
        self.yolo = YOLO("yolov8n.pt")
        
        # Transforms for Re-ID Backbone
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load Re-ID Backbone if weights path provided
        self.reid_model = None
        if weights_path and os.path.exists(weights_path):
            print(f"[Re-ID Model] Loading trained weights from {weights_path}...")
            ckpt = torch.load(weights_path, map_location=self.device)
            self.reid_model = ViTCrossModalReID(
                num_classes=ckpt.get("num_classes", 751),
                grad_checkpointing=False
            ).to(self.device)
            self.reid_model.load_state_dict(ckpt["model_state_dict"])
            self.reid_model.eval()

    def detect_and_extract(self, image_path: str, save_crop_dir: str = "./runs/crops") -> list:
        """
        Detects person boxes in raw image, crops persons, and extracts Re-ID features.
        Returns list of dicts containing bbox, crop image, and feature vector.
        """
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"[Error] Could not load image: {image_path}")
            return []
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Run YOLOv8 detection
        results = self.yolo(img_rgb, verbose=False)[0]
        person_crops = []
        
        os.makedirs(save_crop_dir, exist_ok=True)
        crop_count = 0
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Filter class 0 (Person) with confidence threshold >= 0.5
            if cls_id == 0 and conf >= 0.5:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop_bgr = img_bgr[y1:y2, x1:x2]
                crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
                
                # Save crop image
                crop_filename = f"person_{crop_count:03d}_conf{conf:.2f}.jpg"
                crop_path = os.path.join(save_crop_dir, crop_filename)
                cv2.imwrite(crop_path, crop_bgr)
                
                # Extract Re-ID embedding
                feature = None
                if self.reid_model is not None:
                    tensor = self.transform(crop_pil).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        feature = self.reid_model.extract_features(tensor).cpu().numpy()[0]
                        
                person_crops.append({
                    "crop_id": crop_count,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf,
                    "crop_path": crop_path,
                    "feature": feature
                })
                crop_count += 1
                
        print(f"[Detector] Found {crop_count} person(s) in {image_path}. Saved crops to {save_crop_dir}")
        return person_crops


def main():
    parser = argparse.ArgumentParser(description="YOLOv8 Person Detection and Re-ID Feature Extraction")
    parser.add_argument("--image", type=str, required=True, help="Path to input raw image")
    parser.add_argument("--weights", type=str, default=None, help="Path to trained Re-ID model weights")
    parser.add_argument("--out-dir", type=str, default="./runs/crops", help="Output directory for cropped bounding boxes")
    args = parser.parse_args()

    detector = PersonDetectorReID(weights_path=args.weights)
    crops = detector.detect_and_extract(args.image, save_crop_dir=args.out_dir)
    for c in crops:
        print(f"Crop #{c['crop_id']} | BBox: {c['bbox']} | Conf: {c['confidence']:.2f} | Feature Shape: {c['feature'].shape if c['feature'] is not None else 'N/A'}")


if __name__ == "__main__":
    main()
