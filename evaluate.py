import argparse
import os
import cv2
import numpy as np
import torch
from models.network_swinir import SwinIR  # Adjust import based on your SwinIR setup


def parse_args():
    parser = argparse.ArgumentParser(description="Semiconductor Image Restoration Inference")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing test images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save restored outputs")
    parser.add_argument("--weights", type=str, default="weights/swinir_model.pth", help="Path to model weights")
    return parser.parse_args()


def load_model(weights_path, device):
    # Initialize SwinIR architecture (matching your model config: scale 4, medium model)
    model = SwinIR(
        upscale=4,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="nearest+conv",
        resi_connection="1conv",
    )
    pretrained = torch.load(weights_path, map_location=device)
    param_key = "params_ema" if "params_ema" in pretrained else "params"
    model.load_state_dict(pretrained[param_key] if param_key in pretrained else pretrained, strict=True)
    model.eval()
    model = model.to(device)
    return model


def process_and_restore(input_path, model, device):
    # Load .npy or standard image formats
    if input_path.endswith(".npy"):
        img_raw = np.load(input_path)
        img_norm = cv2.normalize(img_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        if len(img_norm.shape) == 2:
            img = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2RGB)
        else:
            img = img_norm
    else:
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to Tensor (0.0 to 1.0)
    img_t = torch.from_numpy(np.transpose(img, (2, 0, 1))).float() / 255.0
    img_t = img_t.unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        output_t = model(img_t)

    # Convert back to image
    output_np = output_t.squeeze(0).float().detach().cpu().clamp_(0, 1).numpy()
    output_np = np.transpose(output_np, (1, 2, 0)) * 255.0
    output_np = output_np.round().astype(np.uint8)
    output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
    return output_bgr


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.weights, device)
    valid_extensions = (".npy", ".png", ".jpg", ".jpeg", ".tif", ".bmp")
    file_list = [f for f in os.listdir(args.input_dir) if f.lower().endswith(valid_extensions)]

    for file_name in file_list:
        input_path = os.path.join(args.input_dir, file_name)
        restored_img = process_and_restore(input_path, model, device)
        if restored_img is not None:
            base_name = os.path.splitext(file_name)[0]
            output_path = os.path.join(args.output_dir, f"{base_name}.png")
            cv2.imwrite(output_path, restored_img)

    print(f"Restoration complete! {len(file_list)} images written to {args.output_dir}")


if __name__ == "__main__":
    main()