import os
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from net import MultiModalFusionUNet

# Enable cuDNN auto-tuner for potential speedups on fixed-size inputs
torch.backends.cudnn.benchmark = True

strtmp = 'CRMANet16_val'

class Config:
    # Device selection and general hyperparameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4
    lr = 1e-4
    final_lr = 8e-5
    warmup_epochs = 10
    epochs = 100
    img_size = 256

    # Dataset directories (same as your original)
    train_hazy = "data/RNH-he/train/hazy"
    train_clean = "data/RNH-he/train/vis"
    train_nir = "data/RNH-he/train/nir"
    val_hazy = "data/RNH-he/val/hazy"
    val_clean = "data/RNH-he/val/vis"
    val_nir = "data/RNH-he/val/nir"
    test_hazy = "data/RNH-he/test/hazy"
    test_clean = "data/RNH-he/test/vis"
    test_nir = "data/RNH-he/test/nir"

    # Output and model path - NO trained_models or log folder dependency
    save_dir = "results"
    # Expect model file directly at this filename in current working dir (or specify full path)
    model_filename = strtmp + ".pth"

    # Model channel base
    channels = 16

class DehazeDataset(Dataset):
    """
    Dataset that returns a triplet: (hazy_rgb, clean_rgb, nir_gray).
    Filenames are taken from the hazy directory and assumed to exist with the same
    names in the clean and nir directories.
    """
    def __init__(self, hazy_dir, clean_dir, nir_dir, is_train=True):
        self.is_train = is_train
        self.hazy_dir = hazy_dir
        self.clean_dir = clean_dir
        self.nir_dir = nir_dir

        # List image files from the hazy directory
        # accept common extensions
        self.hazy_filenames = sorted([f for f in os.listdir(hazy_dir) if f.lower().endswith(('.png', '.tiff', '.tif', '.jpg', '.jpeg'))])

        # Transform for RGB images (hazy and clean)
        self.transform_rgb = transforms.Compose([
            transforms.Resize((Config.img_size, Config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # maps [0,1] -> [-1,1]
        ])

        # Transform for single-channel NIR images
        self.transform_nir = transforms.Compose([
            transforms.Resize((Config.img_size, Config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # maps [0,1] -> [-1,1]
        ])

    def __len__(self):
        return len(self.hazy_filenames)

    def __getitem__(self, idx):
        # Load hazy RGB, clean RGB, and NIR images using the same filename
        hazy_filename = self.hazy_filenames[idx]
        hazy_path = os.path.join(self.hazy_dir, hazy_filename)
        clean_path = os.path.join(self.clean_dir, hazy_filename)
        nir_path = os.path.join(self.nir_dir, hazy_filename)

        hazy = Image.open(hazy_path).convert('RGB')
        clean = Image.open(clean_path).convert('RGB')
        nir = Image.open(nir_path).convert('L')

        # Apply synchronous random flips only during training
        if self.is_train:
            if torch.rand(1).item() > 0.5:
                hazy = hazy.transpose(Image.FLIP_LEFT_RIGHT)
                clean = clean.transpose(Image.FLIP_LEFT_RIGHT)
                nir = nir.transpose(Image.FLIP_LEFT_RIGHT)
            if torch.rand(1).item() > 0.5:
                hazy = hazy.transpose(Image.FLIP_TOP_BOTTOM)
                clean = clean.transpose(Image.FLIP_TOP_BOTTOM)
                nir = nir.transpose(Image.FLIP_TOP_BOTTOM)

        # Return transformed tensors
        return (
            self.transform_rgb(hazy),
            self.transform_rgb(clean),
            self.transform_nir(nir)
        )

def test(cfg, model):
    """
    Run inference on the test set using a saved model (expects model file at cfg.model_filename).
    Computes PSNR and SSIM for all test images and prints overall metrics.
    """
    model_path_to_load = cfg.model_filename  # directly use filename (no trained_models folder)

    if not os.path.exists(model_path_to_load):
        print(f"Error: Model file not found at '{model_path_to_load}'. Please place the trained model file there.")
        return

    # Load model weights and set model to eval mode
    model.load_state_dict(torch.load(model_path_to_load, map_location=cfg.device))
    model.eval()

    # Prepare dataloader for testing
    test_set = DehazeDataset(cfg.test_hazy, cfg.test_clean, cfg.test_nir, is_train=False)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    os.makedirs(cfg.save_dir, exist_ok=True)

    total_psnr = 0.0
    total_ssim = 0.0

    use_cuda_amp = (cfg.device.startswith("cuda") and torch.cuda.is_available())

    # Inference loop
    with torch.no_grad():
        for batch_idx, (hazy, clean, nir) in enumerate(test_loader):
            hazy = hazy.to(cfg.device, non_blocking=True)
            clean = clean.to(cfg.device, non_blocking=True)
            nir = nir.to(cfg.device, non_blocking=True)

            if use_cuda_amp:
                # mixed precision on CUDA
                from torch.cuda.amp import autocast
                with autocast():
                    output = model(hazy, nir)
            else:
                output = model(hazy, nir)

            # Convert normalized outputs from [-1, 1] to [0, 255] uint8 images
            output = (output.clamp(-1, 1) + 1) / 2.0 * 255.0
            clean = (clean + 1) / 2.0 * 255.0

            output_np = output.cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)
            clean_np = clean.cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)

            for i in range(output_np.shape[0]):
                pred = output_np[i]
                gt = clean_np[i]

                current_psnr = psnr(gt, pred, data_range=255)
                # use channel_axis for skimage's ssim
                current_ssim = ssim(gt, pred, data_range=255, channel_axis=2)
                total_psnr += float(current_psnr)
                total_ssim += float(current_ssim)

    # Compute and print overall metrics (divide by number of images)
    num_images = len(test_set)
    if num_images == 0:
        print("No test images found. Please check the test dataset path.")
        return

    overall_avg_psnr = total_psnr / num_images
    overall_avg_ssim = total_ssim / num_images
    log_info = f"{strtmp} Overall PSNR: {overall_avg_psnr:.2f} dB Overall SSIM: {overall_avg_ssim:.4f}"
    print(log_info)

if __name__ == "__main__":
    cfg = Config()

    # Instantiate the multimodal fusion U-Net model and move it to device
    model = MultiModalFusionUNet(
        dim=cfg.channels,
        dim_mults=(1, 2, 4, 8),
        num_blocks_encoder=[2, 2, 2, 2],
        num_blocks_decoder=[2, 2, 2, 2]
    ).to(cfg.device)

    print("Starting testing...")
    test(cfg, model)
    print("Testing finished.")
