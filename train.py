import os
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import vgg16, VGG16_Weights
from diffusers.optimization import get_cosine_schedule_with_warmup
from torchvision.utils import save_image, make_grid
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import autocast, GradScaler
from contextlib import nullcontext

# ===========================
# Dataset
# ===========================
class UnpairedDataset(Dataset):
    def __init__(self, domain_A_dir, domain_B_dir, transform_A=None, transform_B=None):
        self.A_paths = sorted([p for p in os.listdir(domain_A_dir) if not p.startswith('.')])
        self.B_paths = sorted([p for p in os.listdir(domain_B_dir) if not p.startswith('.')])
        self.A_dir = domain_A_dir
        self.B_dir = domain_B_dir
        self.transform_A = transform_A
        self.transform_B = transform_B
        self.A_len = len(self.A_paths)
        self.B_len = len(self.B_paths)
        self.max_len = max(self.A_len, self.B_len)

    def __len__(self):
        return self.max_len

    def __getitem__(self, idx):
        A_img = Image.open(os.path.join(self.A_dir, self.A_paths[idx % self.A_len])).convert('RGB')
        B_img = Image.open(os.path.join(self.B_dir, self.B_paths[idx % self.B_len])).convert('L')  # 1채널 NIR
        if self.transform_A:
            A_img = self.transform_A(A_img)
        if self.transform_B:
            B_img = self.transform_B(B_img)
        return A_img, B_img

# ===========================
# Generator (ResNet)
# ===========================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, norm_layer=nn.InstanceNorm2d, act_layer=nn.ReLU):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
            norm_layer(out_ch),
            act_layer(True) if act_layer else nn.Identity()
        )
    def forward(self, x):
        return self.block(x)

class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer=nn.InstanceNorm2d, use_dropout=False):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, bias=False),
            norm_layer(dim),
            nn.ReLU(True),
            nn.Dropout(0.5) if use_dropout else nn.Identity(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, padding=0, bias=False),
            norm_layer(dim)
        )
    def forward(self, x):
        return x + self.conv_block(x)

class ResNetGenerator(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, num_res_blocks=9):  # out_ch=1 for NIR
        super().__init__()
        self.num_res_blocks = num_res_blocks
        self.enc1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch,64,7,1,0,bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True)
        )
        self.enc2 = ConvBlock(64,128,3,2,1)
        self.enc3 = ConvBlock(128,256,3,2,1)
        self.res_blocks = nn.Sequential(*[ResnetBlock(256) for _ in range(num_res_blocks)])
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256,128,3,2,1,output_padding=1,bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128,64,3,2,1,output_padding=1,bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(64,out_ch,7,padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.res_blocks(x3)
        out = self.dec(x4)
        return out

    def forward_features(self, x, layers=[1,2,3,4]):
        feats = []
        x1 = self.enc1(x)
        if 1 in layers: feats.append(x1)
        x2 = self.enc2(x1)
        if 2 in layers: feats.append(x2)
        x3 = self.enc3(x2)
        if 3 in layers: feats.append(x3)
        x4 = self.res_blocks(x3)
        if 4 in layers: feats.append(x4)
        return feats

# ===========================
# Discriminator (PatchGAN)
# ===========================
class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=1):  # NIR 1채널
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_ch,64,4,2,1),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(64,128,4,2,1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(128,256,4,2,1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2,True),
            nn.Conv2d(256,1,4,1,1)
        )
    def forward(self,x):
        return self.model(x)

# ===========================
# Perceptual Loss
# ===========================
class PerceptualLossFixed(nn.Module):
    def __init__(self, feature_layer=16):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:feature_layer]
        vgg.eval()
        for p in vgg.parameters(): p.requires_grad=False
        self.vgg = vgg
        self.normalize = transforms.Normalize(mean=[0.485,0.456,0.406],
                                              std=[0.229,0.224,0.225])
    def forward(self,x,y):
        x_scaled = self.normalize((x + 1)/2)
        y_scaled = self.normalize((y + 1)/2)
        x_feat = self.vgg(x_scaled)
        with torch.no_grad():
            y_feat = self.vgg(y_scaled)
        return F.l1_loss(x_feat, y_feat)

# ===========================
# PatchNCE Loss
# ===========================
class PatchNCELoss(nn.Module):
    def __init__(self, n_patches=256, temperature=0.07):
        super().__init__()
        self.n_patches = n_patches
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
    def forward(self, feat_q_list, feat_k_list):
        device = feat_q_list[0].device
        total_loss = 0.0
        for q_map, k_map in zip(feat_q_list, feat_k_list):
            B,C,H,W = q_map.shape
            HW = H*W
            q_flat = q_map.permute(0,2,3,1).reshape(B, HW, C)
            k_flat = k_map.permute(0,2,3,1).reshape(B, HW, C)
            n_samples = min(self.n_patches, HW)
            idx = torch.randint(0, HW, (B, n_samples), device=device)
            q_sampled = torch.stack([q_flat[b, idx[b]] for b in range(B)], dim=0).reshape(B*n_samples, C)
            q_sampled = F.normalize(q_sampled, dim=1)
            k_all = F.normalize(k_flat.reshape(B*HW,C), dim=1)
            labels = (torch.arange(B, device=device).unsqueeze(1) * HW + idx).reshape(-1)
            total_loss += self.cross_entropy(torch.matmul(q_sampled, k_all.T)/self.temperature, labels)
        return total_loss / len(feat_q_list)


# ===========================
# Training loop
# ===========================
def train_cut_resnet_finetune(chck_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 200
    batch_size = 8
    image_size = 256

    lr_G, lr_D = 2e-4, 1e-4
    lambda_patchnce, lambda_percep, lambda_gan,lambda_d = 1, 0.4, 1, 1.5
    n_patches = 256
    layers_for_nce = [2,3,4]
    num_res_blocks = 9

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    transform_A = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    transform_B = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])

    dataset = UnpairedDataset("dataset/train/rgb", "dataset/train/nir", transform_A, transform_B)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=8, pin_memory=True, persistent_workers=True, drop_last=True)

    G = ResNetGenerator(in_ch=3,num_res_blocks=num_res_blocks).to(device)
    D = PatchDiscriminator().to(device)

    optimizer_G = optim.Adam(G.parameters(), lr=lr_G, betas=(0.5,0.999))
    optimizer_D = optim.Adam(D.parameters(), lr=lr_D, betas=(0.5,0.999))

    total_steps = len(loader)*epochs
    warmup_steps = max(100, total_steps//20)

    scheduler_G = get_cosine_schedule_with_warmup(optimizer_G, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scheduler_D = get_cosine_schedule_with_warmup(optimizer_D, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    criterion_GAN = nn.MSELoss()
    criterion_patchnce = PatchNCELoss(n_patches=n_patches).to(device)
    criterion_percep = PerceptualLossFixed().to(device)

    scaler = GradScaler()
    autocast_context = torch.cuda.amp.autocast if device=="cuda" else nullcontext
    
    
    start_epoch=0
    if chck_path is not None:
        # ===========================
        # 체크포인트 로드 (Fine-tune용)
        # ===========================
        checkpoint = torch.load(chck_path, map_location=device)
        start_epoch = checkpoint.get('epoch', 0)
        global_step = checkpoint.get('global_step', start_epoch*len(loader))
        

        # 3.2. 모델 로드 
        G.load_state_dict(checkpoint['G_state_dict']) 
        D.load_state_dict(checkpoint['D_state_dict']) 
        # 3.3. 옵티마이저 로드 (학습 상태를 복원) 
        optimizer_G.load_state_dict(checkpoint['optimizer_G']) 
        optimizer_D.load_state_dict(checkpoint['optimizer_D'])

        scaler.load_state_dict(checkpoint['scaler'])

        scheduler_G.last_epoch = global_step - 1 
        scheduler_D.last_epoch = global_step - 1 
        for pg in optimizer_G.param_groups: pg['lr'] = lr_G
        for pg in optimizer_D.param_groups: pg['lr'] = lr_D
        scheduler_G.base_lrs = [lr_G for _ in optimizer_G.param_groups] 
        scheduler_D.base_lrs = [lr_D for _ in optimizer_D.param_groups]
        
        
        print(f"Loaded fine-tune checkpoint from {chck_path}, starting epoch {start_epoch}")

    print(f"Starting fine-tune training on {device}...")

    for epoch in range(start_epoch,epochs):
        for i, (real_A, real_B) in enumerate(loader):
            real_A, real_B = real_A.to(device), real_B.to(device)

            # ================= Generator step =================
            optimizer_G.zero_grad()
            with autocast_context():
                fake_B = G(real_A)

                feat_k = G.forward_features(real_A, layers_for_nce)
                fake_B_3ch = torch.cat([fake_B] * 3, dim=1)
                feat_q = G.forward_features(fake_B_3ch, layers_for_nce)
                loss_PatchNCE = criterion_patchnce(feat_q, feat_k) * lambda_patchnce

                loss_Percep = criterion_percep(fake_B_3ch, real_A) * lambda_percep

                pred_fake = D(fake_B)
                loss_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake)*0.9)*lambda_gan

                loss_G = loss_GAN + loss_PatchNCE + loss_Percep

            scaler.scale(loss_G).backward()
            scaler.step(optimizer_G)
            scheduler_G.step()

            # ================= Discriminator step =================
            optimizer_D.zero_grad()
            with autocast_context():
                loss_D_real = criterion_GAN(D(real_B), torch.ones_like(D(real_B))*0.9)
                loss_D_fake = criterion_GAN(D(fake_B.detach()), torch.zeros_like(D(fake_B)))
                loss_D = 0.5*(loss_D_real + loss_D_fake)*lambda_d

            scaler.scale(loss_D).backward()
            scaler.step(optimizer_D)
            scheduler_D.step()

            scaler.update()

            if i % 100 == 0:
                current_lr_G = optimizer_G.param_groups[0]['lr']
                current_lr_D = optimizer_D.param_groups[0]['lr']
                print(f"[{epoch+1}/{epochs}] Step [{i}/{len(loader)}], "
                      f"Loss_G: {loss_G.item():.4f}, Loss_D: {loss_D.item():.4f}, "
                      f"Loss_GAN: {loss_GAN.item():.4f}, Loss_NCE: {loss_PatchNCE.item():.4f}, "
                      f"Loss_Percep: {loss_Percep.item():.4f}, "
                     f"LR_G: {current_lr_G:.6f}, LR_D: {current_lr_D:.6f}")

            if i % 500 == 0:
                with torch.no_grad():
                    sample_A = real_A[:4]
                    sample_B = real_B[:4]
                    sample_fake = fake_B[:4]
                    sample_B_3ch = sample_B.repeat(1, 3, 1, 1)
                    sample_fake_3ch = sample_fake.repeat(1, 3, 1, 1)
                    combined = torch.cat((sample_A, sample_B_3ch, sample_fake_3ch), dim=0)
                    grid = make_grid(combined, nrow=4, normalize=True)
                    save_image(grid, f"logs/sample_e{epoch+1}_s{i}.png")

        # ================= Checkpoint =================
        if (epoch+1) % 10 == 0:
            torch.save({
                'epoch': epoch+1,
                'G_state_dict': G.state_dict(),
                'D_state_dict': D.state_dict(),
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
                'scheduler_G': scheduler_G.state_dict(),
                'scheduler_D': scheduler_D.state_dict(),
                'scaler': scaler.state_dict()
            }, f"checkpoints/cut_resnet_finetune_epoch_{epoch+1}.pth")
            print(f"[Epoch {epoch+1}] Saved checkpoint.")


def save_checkpoint_as_torchscript(
    checkpoint_path: str, 
    output_script_path: str, 
    num_res_blocks: int = 9, 
    input_size: tuple = (1, 3, 256, 256)
):
    """
    체크포인트 파일에서 Generator 가중치를 로드하여 TorchScript 파일로 저장합니다.
    
    Args:
        checkpoint_path (str): 로드할 전체 체크포인트 파일 경로 (예: 'checkpoints/latest_checkpoint.pth').
        output_script_path (str): 저장할 TorchScript 파일 경로 (예: 'deploy_netG.pt').
        num_res_blocks (int): ResNet Generator 생성 시 사용된 블록 수.
        input_size (tuple): 모델 추적(Tracing) 시 사용할 입력 텐서의 크기 (Batch, C, H, W).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 원본 Generator 모델 인스턴스 생성
    # 학습 시 사용한 파라미터와 정확히 일치해야 합니다. (in_ch=3, out_ch=1)
    G = ResNetGenerator(in_ch=3, out_ch=1, num_res_blocks=num_res_blocks).to(device)
    G.eval()

    # 2. 체크포인트에서 가중치 로드
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 'G_state_dict' 키를 사용하여 Generator의 가중치 로드
        if 'G_state_dict' in checkpoint:
            G.load_state_dict(checkpoint['G_state_dict'])
            print(f"✅ 체크포인트 '{checkpoint_path}'에서 Generator 가중치 로드 완료.")
        else:
            # G.state_dict()만 저장된 추론용 파일일 경우 대비
            G.load_state_dict(checkpoint)
            print(f"✅ 추론용 가중치 파일 '{checkpoint_path}' 로드 완료.")
            
    except Exception as e:
        print(f"❌ 체크포인트 로드 실패: {e}")
        return

    # 3. 모델 추적(Tracing) 및 TorchScript로 저장
    try:
        # Tracing을 위한 더미 입력 생성
        dummy_input = torch.randn(input_size, device=device)
        
        # torch.jit.trace를 사용하여 모델 구조와 가중치를 하나의 그래프로 기록
        traced_script_module = torch.jit.trace(G, dummy_input)
        
        # TorchScript 파일로 저장
        traced_script_module.save(output_script_path)
        
        print(f"🎉 TorchScript 모델이 성공적으로 저장되었습니다: {output_script_path}")
        print("이 파일은 Python 클래스 정의 없이도 추론할 수 있습니다.")

    except Exception as e:
        print(f"❌ TorchScript 변환 실패 (모델 구조에 동적 요소가 있을 수 있음): {e}")

if __name__ == '__main__':
    # train_cut_resnet_finetune("checkpoints/cut_resnet_finetune_epoch_160.pth")
    save_checkpoint_as_torchscript(checkpoint_path="checkpoints/cut_resnet_finetune_epoch_200.pth",output_script_path="models/rgb_to_nir_cut/model.pt")
