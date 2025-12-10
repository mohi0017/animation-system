import os, math, json, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import make_grid, save_image
from torchvision import models
from skimage.metrics import structural_similarity as ssim
import numpy as np

from utils.preprocess_utils import get_dataloaders
from utils.io_utils import ensure_dir

# ----------------------------
# Config (edit as needed)
# ----------------------------
CFG = dict(
    train_csv="E:\dataScience\Fiver orders\Order 15\AI Animation system\AI anim\stages\stage1_cleanup\stages\stage1_cleanup\manifest_train.csv",
    val_csv="E:\dataScience\Fiver orders\Order 15\AI Animation system\AI anim\stages\stage1_cleanup\stages\stage1_cleanup\manifest_val.csv",
    test_csv="E:\dataScience\Fiver orders\Order 15\AI Animation system\AI anim\stages\stage1_cleanup\stages\stage1_cleanup\manifest_test.csv",
    out_dir="stages/stage1/outputs",
    img_size=512,
    batch_size=4,            # T4/V100 16GB OK @512px
    num_workers=2,
    max_epochs=40,
    lr=2e-4,
    betas=(0.5, 0.999),
    gan_lambda=0.02,         # low weight to avoid ringing:contentReference[oaicite:3]{index=3}
    l1_lambda=1.0,
    ssim_lambda=0.2,
    alpha_lambda=0.05,
    eval_every=1,
    fid_every=5,             # set to 0 to disable
    val_subset=128,          # speed
    seed=42,
    phase_labels=["rough","tiedown","line","clean","color","skeleton"]
)

# ----------------------------
# Utils
# ----------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def rgb_alpha_split(x):
    # x: (B,4,H,W) normalized: RGB [-1,1], A [0,1]
    rgb = x[:, :3]
    alpha = x[:, 3:4]
    return rgb, alpha

def ssim_batch(x, y):
    # x,y in [-1,1], shape Bx3xHxW; compute per-image SSIM on CPU (fast subset)
    x = x.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    B = x.shape[0]
    scores = []
    for i in range(B):
        xi = ((x[i].transpose(1,2,0)+1)*127.5).astype(np.uint8)
        yi = ((y[i].transpose(1,2,0)+1)*127.5).astype(np.uint8)
        s = ssim(xi, yi, channel_axis=2, data_range=255)
        scores.append(s)
    return float(np.mean(scores))

# Optional LPIPS (if installed)
def try_lpips():
    try:
        import lpips
        return lpips.LPIPS(net='alex').eval().cuda()
    except Exception:
        return None

# Optional FID using torch-fidelity or pytorch-fid (subset only)
def compute_fid_stub():
    # Place-holder; integrate torch-fidelity if you want:
    # from torch_fidelity import calculate_metrics
    # metrics = calculate_metrics(input1=real_dir, input2=fake_dir, ...)
    return None

# ----------------------------
# Conditioning: phase embeddings
# ----------------------------
class PhaseEmbedder(nn.Module):
    def __init__(self, labels, embed_dim=16):
        super().__init__()
        self.label_to_idx = {l:i for i,l in enumerate(labels)}
        self.embed = nn.Embedding(len(labels), embed_dim)

    def forward(self, input_phase, target_phase, B, H, W, device):
        # Convert strings to indices, then to embeddings, then tile to HxW and concat → (B, 2*embed_dim, H, W)
        inp_idx = torch.tensor([self.label_to_idx[p] for p in input_phase], device=device)
        tgt_idx = torch.tensor([self.label_to_idx[p] for p in target_phase], device=device)
        inp_e = self.embed(inp_idx)   # (B, E)
        tgt_e = self.embed(tgt_idx)   # (B, E)
        cond = torch.cat([inp_e, tgt_e], dim=1)  # (B, 2E)
        cond = cond.unsqueeze(-1).unsqueeze(-1).expand(B, cond.shape[1], H, W)
        return cond

# ----------------------------
# UNet Generator (light)
# ----------------------------
class UNetBlock(nn.Module):
    def __init__(self, in_c, out_c, down=True, use_dropout=False):
        super().__init__()
        if down:
            self.block = nn.Sequential(
                nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5) if use_dropout else nn.Identity()
    def forward(self, x):
        x = self.block(x)
        return self.dropout(x)

class UNetGenerator(nn.Module):
    # Input channels = 4 (RGBA) + 2*E (phase cond)
    def __init__(self, in_ch, out_ch=4):
        super().__init__()
        # down: 512→256→128→64→32→16→8→4 (at 512px, depth 7 is fine)
        self.d1 = UNetBlock(in_ch, 64,  down=True)    # 256
        self.d2 = UNetBlock(64,   128, down=True)     # 128
        self.d3 = UNetBlock(128,  256, down=True)     # 64
        self.d4 = UNetBlock(256,  512, down=True)     # 32
        self.d5 = UNetBlock(512,  512, down=True)     # 16
        self.d6 = UNetBlock(512,  512, down=True)     # 8
        self.d7 = UNetBlock(512,  512, down=True)     # 4

        self.u1 = UNetBlock(512,  512, down=False, use_dropout=True)   # 8
        self.u2 = UNetBlock(1024, 512, down=False, use_dropout=True)   # 16
        self.u3 = UNetBlock(1024, 512, down=False, use_dropout=True)   # 32
        self.u4 = UNetBlock(1024, 256, down=False)                     # 64
        self.u5 = UNetBlock(512,  128, down=False)                     # 128
        self.u6 = UNetBlock(256,  64,  down=False)                     # 256
        self.u7 = nn.ConvTranspose2d(128, out_ch, 4, 2, 1)             # 512
        self.tanh = nn.Tanh()

    def forward(self, x):
        d1 = self.d1(x); d2 = self.d2(d1); d3 = self.d3(d2); d4 = self.d4(d3)
        d5 = self.d5(d4); d6 = self.d6(d5); d7 = self.d7(d6)
        u1 = self.u1(d7)
        u2 = self.u2(torch.cat([u1, d6], dim=1))
        u3 = self.u3(torch.cat([u2, d5], dim=1))
        u4 = self.u4(torch.cat([u3, d4], dim=1))
        u5 = self.u5(torch.cat([u4, d3], dim=1))
        u6 = self.u6(torch.cat([u5, d2], dim=1))
        u7 = self.u7(torch.cat([u6, d1], dim=1))
        return self.tanh(u7)  # RGB in [-1,1], Alpha will be coerced via alpha loss

# ----------------------------
# PatchGAN Discriminator (conditional: concat input & output)
# ----------------------------
class NLayerDiscriminator(nn.Module):
    def __init__(self, in_ch=8, ndf=64, n_layers=3):
        super().__init__()
        # in_ch = 4 (input) + 4 (pred/target)
        kw = 4; padw = 1
        sequence = [
            nn.Conv2d(in_ch, ndf, kw, 2, padw), nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kw, 2, padw, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kw, 1, padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)

# ----------------------------
# Training
# ----------------------------
def train():
    set_seed(CFG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    train_loader, val_loader, _ = get_dataloaders(
        CFG["train_csv"], CFG["val_csv"], CFG["test_csv"],
        batch_size=CFG["batch_size"], num_workers=CFG["num_workers"],
        augment=True, size=CFG["img_size"]
    )

    # Conditioning
    E = 16
    embedder = PhaseEmbedder(CFG["phase_labels"], embed_dim=E)
    G = UNetGenerator(in_ch=4 + 2*E, out_ch=4)
    D = NLayerDiscriminator(in_ch=8)

    embedder = embedder.to(device)
    G = G.to(device)
    D = D.to(device)

    # Optims & scaler
    opt_G = torch.optim.AdamW(G.parameters(), lr=CFG["lr"], betas=CFG["betas"])
    opt_D = torch.optim.AdamW(D.parameters(), lr=CFG["lr"], betas=CFG["betas"])
    scaler = GradScaler()

    bce = nn.BCEWithLogitsLoss()

    # Optional LPIPS
    lpips_model = try_lpips()

    ensure_dir(CFG["out_dir"])
    ckpt_dir = ensure_dir(os.path.join(CFG["out_dir"], "ckpts"))
    sample_dir = ensure_dir(os.path.join(CFG["out_dir"], "samples"))
    log_path = os.path.join(CFG["out_dir"], "val_report_stage1.json")
    hist = []

    global_step = 0
    best_ssim = -1

    for epoch in range(1, CFG["max_epochs"]+1):
        G.train(); D.train()
        t0 = time.time()
        for batch in train_loader:
            global_step += 1
            x = batch["input"].to(device)     # (B,4,H,W)
            y = batch["target"].to(device)    # (B,4,H,W)
            inp_phase = batch["input_phase"]
            tgt_phase = batch["target_phase"]

            B, _, H, W = x.shape
            cond = embedder(inp_phase, tgt_phase, B, H, W, device)
            x_cond = torch.cat([x, cond], dim=1)  # (B, 4+2E, H, W)

            # ---- Train D ----
            with autocast():
                fake = G(x_cond).detach()
                D_real = D(torch.cat([x, y], dim=1))
                D_fake = D(torch.cat([x, fake], dim=1))
                real_loss = bce(D_real, torch.ones_like(D_real))
                fake_loss = bce(D_fake, torch.zeros_like(D_fake))
                d_loss = 0.5 * (real_loss + fake_loss)

            opt_D.zero_grad(set_to_none=True)
            scaler.scale(d_loss).backward()
            scaler.step(opt_D)

            # ---- Train G ----
            with autocast():
                fake = G(x_cond)
                D_fake = D(torch.cat([x, fake], dim=1))
                gan_loss = bce(D_fake, torch.ones_like(D_fake)) * CFG["gan_lambda"]

                # L1 (RGB) + alpha L1 + SSIM
                fake_rgb, fake_a = rgb_alpha_split(fake)
                y_rgb, y_a       = rgb_alpha_split(y)

                l1 = F.l1_loss(fake_rgb, y_rgb) * CFG["l1_lambda"]
                alpha_l1 = F.l1_loss(fake_a, y_a) * CFG["alpha_lambda"]

                # SSIM (on subset of batch for speed)
                ssim_w = CFG["ssim_lambda"]
                if ssim_w > 0:
                    # detach to CPU for skimage; use 2 images to keep it cheap
                    k = min(B, 2)
                    ssim_val = ssim_batch(fake_rgb[:k], y_rgb[:k])
                    ssim_loss = (1.0 - ssim_val) * ssim_w
                else:
                    ssim_loss = 0.0

                g_loss = gan_loss + l1 + alpha_l1 + (ssim_loss if isinstance(ssim_loss, float) else ssim_loss)

            opt_G.zero_grad(set_to_none=True)
            scaler.scale(g_loss).backward()
            scaler.step(opt_G)
            scaler.update()

        # ---------------- Eval per epoch ----------------
        if epoch % CFG["eval_every"] == 0:
            G.eval()
            with torch.no_grad():
                # SSIM on small val subset
                val_ssim_scores = []
                val_lpips_scores = []
                seen = 0
                for batch in val_loader:
                    x = batch["input"].to(device)
                    y = batch["target"].to(device)
                    inp_phase = batch["input_phase"]; tgt_phase = batch["target_phase"]
                    B, _, H, W = x.shape
                    cond = embedder(inp_phase, tgt_phase, B, H, W, device)
                    pred = G(torch.cat([x, cond], dim=1))
                    # SSIM
                    s = ssim_batch(pred[:, :3], y[:, :3])
                    val_ssim_scores.append(s)
                    # LPIPS optional
                    if lpips_model is not None:
                        lp = lpips_model(pred[:, :3], y[:, :3]).mean().item()
                        val_lpips_scores.append(lp)
                    seen += x.size(0)
                    if seen >= CFG["val_subset"]:
                        break

                mean_ssim = float(np.mean(val_ssim_scores)) if val_ssim_scores else 0.0
                mean_lpips = float(np.mean(val_lpips_scores)) if val_lpips_scores else None

                # Save sample grid (first batch of val)
                grid_path = os.path.join(sample_dir, f"epoch_{epoch:03d}.png")
                # Show input RGB, pred RGB, target RGB (alpha over checkerboard is for later)
                inp_rgb = (x[:, :3].clamp(-1,1)+1)/2
                pred_rgb = (pred[:, :3].clamp(-1,1)+1)/2
                tgt_rgb = (y[:, :3].clamp(-1,1)+1)/2
                grid = make_grid(torch.cat([inp_rgb, pred_rgb, tgt_rgb], dim=0), nrow=x.size(0))
                save_image(grid, grid_path)

                record = dict(epoch=epoch, ssim=mean_ssim, lpips=mean_lpips, samples=grid_path)
                hist.append(record)
                with open(log_path, "w") as f:
                    json.dump(hist, f, indent=2)

                print(f"[Epoch {epoch}] SSIM={mean_ssim:.4f}" + (f", LPIPS={mean_lpips:.4f}" if mean_lpips is not None else ""))

                # Save ckpts
                torch.save({"G": G.state_dict(), "D": D.state_dict(), "cfg": CFG}, os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pth"))
                if mean_ssim > best_ssim:
                    best_ssim = mean_ssim
                    torch.save({"G": G.state_dict(), "cfg": CFG}, os.path.join(ckpt_dir, "stage1_multiphase_cleaner.pth"))
                    # export encoder backbone (first half of UNet down path)
                    enc = {
                        "d1": G.d1.state_dict(), "d2": G.d2.state_dict(), "d3": G.d3.state_dict(),
                        "d4": G.d4.state_dict(), "d5": G.d5.state_dict(), "d6": G.d6.state_dict(), "d7": G.d7.state_dict(),
                    }
                    torch.save({"encoder": enc, "cfg": CFG}, os.path.join(ckpt_dir, "stage1_encoder_backbone.pth"))

        # (Optional) FID every N epochs (stubbed to avoid heavy compute)
        if CFG["fid_every"] and (epoch % CFG["fid_every"] == 0):
            pass  # integrate torch-fidelity here if desired

        print(f"Epoch {epoch}/{CFG['max_epochs']} done in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    os.makedirs(CFG["out_dir"], exist_ok=True)
    train()