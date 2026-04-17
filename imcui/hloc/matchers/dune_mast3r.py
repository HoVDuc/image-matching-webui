import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .. import DEVICE, MODEL_REPO_ID, logger
from ..utils.base_model import BaseModel

mast3r_path = Path(__file__).parent / "../../third_party/mast3r"
sys.path.append(str(mast3r_path))

dust3r_path = Path(__file__).parent / "../../third_party/dust3r"
sys.path.append(str(dust3r_path))

from dust3r.inference import inference
from dust3r.utils.image import load_images
from mast3r.fast_nn import fast_reciprocal_NNs
from mast3r.model import load_dune_mast3r_model

from .duster import Duster

NAVER_CHECKPOINT_URL = (
    "https://download.europe.naverlabs.com/dune/{model_name}"
)


class DuneMast3r(Duster):
    default_conf = {
        "name": "dune_mast3r",
        "model_name": "dunemast3r_cvpr25_vitbase.pth",
        "max_keypoints": 2000,
        "image_size": 518,
        "border_margin": 3,
    }

    required_inputs = ["image0", "image1"]

    def _init(self, conf):
        model_name = self.conf["model_name"]
        try:
            model_path = self._download_model(
                repo_id=MODEL_REPO_ID,
                filename="dune_mast3r/{}".format(model_name),
            )
        except Exception:
            # Fallback: download trực tiếp từ NAVER Labs
            import torch.hub

            cache_dir = Path.home() / ".cache" / "dune_mast3r"
            cache_dir.mkdir(parents=True, exist_ok=True)
            model_path = cache_dir / model_name
            if not model_path.exists():
                url = NAVER_CHECKPOINT_URL.format(model_name=model_name)
                logger.info(f"Downloading DUNE+MASt3R checkpoint from {url}")
                torch.hub.download_url_to_file(url, str(model_path), progress=True)
            model_path = str(model_path)

        logger.info("Loading DUNE+MASt3R model")
        self.net = load_dune_mast3r_model(model_path, DEVICE)
        self.patch_size = self.net.patch_size
        logger.info("Loaded DUNE+MASt3R model")

    def _tensor_to_pil(self, img_tensor: torch.Tensor) -> Image.Image:
        """Convert (1, C, H, W) float tensor in [0,1] to PIL RGB image."""
        arr = img_tensor.squeeze(0).cpu().numpy()  # (C, H, W)
        arr = (arr * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
        return Image.fromarray(arr)

    def _forward(self, data):
        img0_tensor = data["image0"]  # (1, C, H, W) in [0, 1]
        img1_tensor = data["image1"]

        # load_images() nhận file path — lưu ảnh tạm trước khi gọi
        with (
            tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f0,
            tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f1,
        ):
            img0_path = f0.name
            img1_path = f1.name

        self._tensor_to_pil(img0_tensor).save(img0_path)
        self._tensor_to_pil(img1_tensor).save(img1_path)

        images = load_images(
            [img0_path, img1_path],
            size=self.conf["image_size"],
            patch_size=self.patch_size,
            square_ok=True,
            verbose=False,
        )

        output = inference(
            [tuple(images)], self.net, DEVICE, batch_size=1, verbose=False
        )

        view1, pred1 = output["view1"], output["pred1"]
        view2, pred2 = output["view2"], output["pred2"]

        desc1 = pred1["desc"].squeeze(0).detach()
        desc2 = pred2["desc"].squeeze(0).detach()

        matches_im0, matches_im1 = fast_reciprocal_NNs(
            desc1,
            desc2,
            subsample_or_initxy1=8,
            device=DEVICE,
            dist="dot",
            block_size=2**13,
        )

        # Filter border pixels
        m = self.conf["border_margin"]
        H0, W0 = view1["true_shape"][0]
        H1, W1 = view2["true_shape"][0]

        valid0 = (
            (matches_im0[:, 0] >= m)
            & (matches_im0[:, 0] < int(W0) - m)
            & (matches_im0[:, 1] >= m)
            & (matches_im0[:, 1] < int(H0) - m)
        )
        valid1 = (
            (matches_im1[:, 0] >= m)
            & (matches_im1[:, 0] < int(W1) - m)
            & (matches_im1[:, 1] >= m)
            & (matches_im1[:, 1] < int(H1) - m)
        )
        valid = valid0 & valid1
        mkpts0 = matches_im0[valid]
        mkpts1 = matches_im1[valid]

        if len(mkpts0) == 0:
            logger.warning("DUNE+MASt3R: no matches found")
            return {
                "keypoints0": torch.zeros([0, 2]),
                "keypoints1": torch.zeros([0, 2]),
            }

        top_k = self.conf["max_keypoints"]
        if top_k is not None and len(mkpts0) > top_k:
            keep = np.round(np.linspace(0, len(mkpts0) - 1, top_k)).astype(int)
            mkpts0 = mkpts0[keep]
            mkpts1 = mkpts1[keep]

        logger.info(f"DUNE+MASt3R: matched {len(mkpts0)} keypoints")
        return {
            "keypoints0": torch.from_numpy(mkpts0),
            "keypoints1": torch.from_numpy(mkpts1),
        }
