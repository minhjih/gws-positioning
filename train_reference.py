import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np, matplotlib.pyplot as plt
from typing import Tuple

# ---------------------------------------------------------------------------
# 1. Building blocks
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, 8, 3, 1, 1); self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, 1, 1); self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, cout, 3, 1, 1); self.bn3 = nn.BatchNorm2d(cout)
        self.short = nn.Sequential()
        if cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1), nn.BatchNorm2d(cout))
        self.act = nn.LeakyReLU(0.2)
    def forward(self,x):
        h = self.act(self.bn1(self.conv1(x)))
        h = self.act(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))
        return self.act(h + self.short(x))

class LocationEncoder(nn.Module):
    def __init__(self, n_pos=24, dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 4, 2, 1)   # 30→15
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 4, 2, 1)  # 15→7
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32,64, 3, 2, 1)  # 7→4
        self.bn3 = nn.BatchNorm2d(64)
        self.res1 = ResidualBlock(64,64)
        self.res2 = ResidualBlock(64,64)
        self.res3 = ResidualBlock(64,64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1  = nn.Linear(64, dim)
        self.bn1_l = nn.BatchNorm1d(dim)
        self.fc2  = nn.Linear(dim, dim)
        self.bn2_l = nn.BatchNorm1d(dim)
        self.cls  = nn.Linear(dim, n_pos)

        self.act, self.drop = nn.LeakyReLU(0.2), nn.Dropout(0.5)
    def forward(self,x):
        x = (x + 1) / 2
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        x = self.res1(x); x = self.res2(x); x = self.res3(x)
        x = self.pool(x).view(x.size(0),-1)
        x = self.act(self.bn1_l(self.fc1(x)))
        feat = self.bn2_l(self.fc2(x))
        return self.cls(self.act(feat)), feat

class SpatialEncoder(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,4,2,1)
        self.conv2 = nn.Conv2d(16,32,4,2,1)
        self.conv3 = nn.Conv2d(32,64,3,2,1)
        self.pool  = nn.AdaptiveAvgPool2d(1)
        self.fc    = nn.Linear(64,dim)
        self.act   = nn.LeakyReLU(0.2)
        self.bn = nn.BatchNorm1d(dim)
    def forward(self,x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        return self.bn(self.fc(self.pool(x).view(x.size(0),-1)))

class MLP(nn.Module):
    def __init__(self, sdim, adain_dim):
        super().__init__()
        self.fc1 = nn.Linear(sdim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, adain_dim*2)  # scale, bias
        self.bn3 = nn.BatchNorm1d(adain_dim*2)
        self.act = nn.LeakyReLU(0.2)
    def forward(self, s):
        x = self.act(self.bn1(self.fc1(s)))
        x = self.act(self.bn2(self.fc2(x)))
        x = self.bn3(self.fc3(x))
        scale, bias = x.chunk(2, dim=1)
        return scale, bias

def adain(x, scale, bias, eps=1e-5):
    mean = x.mean(dim=[2,3], keepdim=True)
    std = x.std(dim=[2,3], keepdim=True) + eps
    scale = scale.unsqueeze(-1).unsqueeze(-1)
    bias = bias.unsqueeze(-1).unsqueeze(-1)
    return scale * (x - mean) / std + bias

def adain_linear(x, scale, bias, eps=1e-5):
    """
    선형(latent) 벡터에 대한 AdaIN 함수.
    x: (batch, feature)
    scale, bias: (batch, feature)
    """
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True) + eps
    return scale * (x - mean) / std + bias

class AdaINResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2)
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
        self.alpha =0.3

    def forward(self, x, scale, bias):
        identity = self.skip(x)
        out = self.act(self.conv1(x))
        #out = self.alpha*adain(out, scale, bias)+(1-self.alpha)*out
        out = self.act(self.conv2(out))
        out = self.alpha*adain(out, scale, bias)+(1-self.alpha)*out
        return self.act(out + identity)

class Generator(nn.Module):
    def __init__(self, ldim=64, sdim=64):
        super().__init__()
        self.fc1 = nn.Linear(ldim, 32*4*4)
        #self.fc2 = nn.Linear(64, 64)
        #self.fc3 = nn.Linear(64, 64*4*4)
        self.res1 = AdaINResidualBlock(32, 32)
        self.res2 = AdaINResidualBlock(32, 32)
        self.up1  = nn.ConvTranspose2d(32, 32, 4, 2, 1)  # 4x4 -> 8x8
        self.up2  = nn.ConvTranspose2d(32, 16, 4, 2, 1)  # 8x8 -> 16x16
        self.up3  = nn.ConvTranspose2d(16, 8, 3, 2, 1, 1)  # 16x16 -> 30x30
        self.out  = nn.Conv2d(8, 3, 3, 1, 1)  # 30x30 -> 30x30
        self.crop = nn.AdaptiveAvgPool2d(30)  # 30x30 -> 30x30
        self.act  = nn.LeakyReLU(0.2)
        self.mlp = MLP(sdim, 32)
    def forward(self, l, s):
        x = self.act(self.fc1(l))
        #x = self.act(self.fc2(x))
        #x = self.act(self.fc3(x))
        x = x.view(x.size(0), 32, 4, 4)
        scale, bias = self.mlp(s)
        x = self.res1(x, scale, bias)
        x = self.res2(x, scale, bias)  # 필요시 활성화
        x = self.act(self.up1(x))
        x = self.act(self.up2(x))
        x = self.act(self.up3(x))
        return torch.tanh(self.crop(self.out(x)))

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,4,2,1)
        self.conv2 = nn.Conv2d(32,64,4,2,1)
        self.conv3 = nn.Conv2d(64,128,3,2,1)
        self.conv4 = nn.Conv2d(128,256,4,1,0)
        self.fc = nn.Linear(256,1)
        self.act, self.drop = nn.LeakyReLU(0.2), nn.Dropout(0.3)
    def forward(self,x):
        x = self.drop(self.act(self.conv1(x)))
        x = self.drop(self.act(self.conv2(x)))
        x = self.drop(self.act(self.conv3(x)))
        x = self.act(self.conv4(x)).view(x.size(0),-1)
        
        return torch.sigmoid(self.fc(x))

# ---------------------------------------------------------------------------
# 2. DEGN-LIC wrapper
# ---------------------------------------------------------------------------

class DEGN_LIC:
    def __init__(self, n_pos=24, device='cuda'):
        self.device = device
        self.E_L  = LocationEncoder(n_pos).to(device)
        self.E_SI = SpatialEncoder().to(device)
        self.E_SC = SpatialEncoder().to(device)
        self.G_I  = Generator().to(device)
        self.G_C  = Generator().to(device)
        self.D_I  = Discriminator().to(device)
        self.D_C  = Discriminator().to(device)
        self.best_loss = 40
        self.ce, self.l1, self.gan = nn.CrossEntropyLoss(), nn.L1Loss(), nn.BCEWithLogitsLoss()
        self.lam1, self.lam2, self.lam3 = 10, 2, 2

    # ------------------------ Step-1 ---------------------------------------
    def step1_train_location_encoder(self,x0,x1,labels,epochs=500,lr=1e-3,batch=16, pass_=False):
        if pass_ == False:
            print("\n[Step-1] Train E_L ...")
            data = torch.cat([x0,x1]); y = torch.cat([labels,labels])
            n = len(data); n_tr=int(0.8*n) # num train
            idx = torch.randperm(n)
            tr,va = TensorDataset(data[idx[:n_tr]],y[idx[:n_tr]]), TensorDataset(data[idx[n_tr:]],y[idx[n_tr:]])
            tr_ld,va_ld = DataLoader(tr,batch,True), DataLoader(va,batch)
            opt = optim.Adam(self.E_L.parameters(),lr=lr); sched=optim.lr_scheduler.StepLR(opt,30,0.5)
            best=0
            for ep in range(epochs):
                self.E_L.train(); tot=acc=loss_sum=0
                for x,yb in tr_ld:
                    x,yb=x.to(self.device), yb.to(self.device)
                    opt.zero_grad()
                    logit,_ = self.E_L(x); loss=self.ce(logit,yb); loss.backward(); opt.step()
                    loss_sum+=loss.item(); acc+=(logit.argmax(1)==yb).sum().item(); tot+=yb.numel()
                self.E_L.eval(); vtot= vacc= vloss=0
                with torch.no_grad():
                    for x,yb in va_ld:
                        x,yb=x.to(self.device), yb.to(self.device)
                        logit,_=self.E_L(x); vloss+=self.ce(logit,yb).item()
                        vacc+=(logit.argmax(1)==yb).sum().item(); vtot+=yb.numel()
                print(f" Ep {ep+1}/{epochs}  train_acc {100*acc/tot:5.1f}%  val_acc {100*vacc/vtot:5.1f}%")
                if vacc>=best: best=vacc; torch.save(self.E_L.state_dict(),"ref_models/best_E_L.pth")
                sched.step()
            self.E_L.load_state_dict(torch.load("ref_models/best_E_L.pth"))
            self.E_L.eval();  [p.requires_grad_(False) for p in self.E_L.parameters()]
            print(f" Best Val Acc = {100*best/vtot:.2f}%")
        else:
            self.E_L.load_state_dict(torch.load("ref_models/best_E_L.pth"))
            self.E_L.eval();  [p.requires_grad_(False) for p in self.E_L.parameters()]
    # ------------------------ Step-2  (버그 fix 포함) ------------------------
    def step2_train_main_network(self,x0,x1,labels, epochs=300,batch=8,lr_g=1e-5,lr_d=2e-5, pass_=False):
        if(pass_ == False):
            print("\n[Step-2] Train full DEGN-LIC ...")
            loader = DataLoader(TensorDataset(x0,x1,labels),batch,True)
            opt_G = optim.Adam(
                list(self.E_SI.parameters())+list(self.E_SC.parameters())+
                list(self.G_I.parameters()) +list(self.G_C.parameters()),
                lr=lr_g, betas=(0.5,0.999))
            opt_D = optim.Adam(
                list(self.D_I.parameters())+list(self.D_C.parameters()),
                lr=lr_d, betas=(0.5,0.999))

            for ep in range(epochs):
                D_sum=G_sum=loss_recon_sum=loss_l_sum=loss_s_sum=0.0
                for b,(xI,xC,_) in enumerate(loader):
                    xI,xC = xI.to(self.device), xC.to(self.device)
                    bs = xI.size(0)
                    real,fake = torch.ones(bs,1,device=self.device), torch.zeros(bs,1,device=self.device)

                    # ------------------- Discriminator -------------------
                    opt_D.zero_grad()
                    with torch.no_grad():
                        _,lI = self.E_L(xI); _,lC = self.E_L(xC) #cls 넣기 전을 feature vector로 넣는것이 맞는지 확인필요.
                    sI = self.E_SI(xI); sC = self.E_SC(xC)
                    xI_hat_det = self.G_I(lI,sI).detach()
                    xC_hat_det = self.G_C(lC,sC).detach()
                    sC_r, sI_r = torch.randn_like(sC), torch.randn_like(sI)
                    xI2C_det = self.G_C(lI,sC_r).detach()
                    xC2I_det = self.G_I(lC,sI_r).detach()

                    loss_D = (
                        self.gan(self.D_I(xI),real)+ self.gan(self.D_I(xI_hat_det),fake)+ self.gan(self.D_I(xC2I_det),fake)+ # xI GAN
                        self.gan(self.D_C(xC),real)+ self.gan(self.D_C(xC_hat_det),fake)+ self.gan(self.D_C(xI2C_det),fake)  # xC GAN
                    )/6
                    loss_D.backward(); opt_D.step()

                    # ------------------- Generator (+E_S) -----------------
                    opt_G.zero_grad()
                    # fresh forward (새 그래프)
                    sI = self.E_SI(xI); sC = self.E_SC(xC)
                    xI_hat = self.G_I(lI,sI)
                    xC_hat = self.G_C(lC,sC)
                    xI2C = self.G_C(lI,sC_r)
                    xC2I = self.G_I(lC,sI_r)

                    loss_GAN = (
                        self.gan(self.D_I(xI_hat),real)+ self.gan(self.D_C(xC_hat),real)+
                        self.gan(self.D_I(xC2I),real)  + self.gan(self.D_C(xI2C),real)
                    )/4
                    loss_recon = self.l1(xI_hat,xI)+self.l1(xC_hat,xC)
                    with torch.no_grad():
                        _,lC2I = self.E_L(xC2I); _,lI2C = self.E_L(xI2C)
                    sC2I = self.E_SI(xC2I); sI2C = self.E_SC(xI2C)
                    loss_l = (self.l1(lC2I,lC)+self.l1(lI2C,lI))/2
                    loss_s = (self.l1(sC2I,sI_r)+self.l1(sI2C,sC_r))/2

                    loss_G = loss_GAN + self.lam1*loss_recon + self.lam2*loss_l + self.lam3*loss_s
                    loss_G.backward(); opt_G.step()

                    D_sum += loss_D.item(); G_sum += loss_G.item()
                    loss_recon_sum += loss_recon.item()
                    loss_l_sum += loss_l.item()
                    loss_s_sum += loss_s.item()
                print(f" >> Ep {ep+1:03d}  mean D {D_sum/len(loader):.3f}  G {G_sum/len(loader):.3f}, loss_recon {loss_recon_sum/len(loader):.3f}, loss_l {loss_l_sum/len(loader):.3f}, loss_s {loss_s_sum/len(loader):.3f}")
                if G_sum/len(loader) < self.best_loss:
                        self.best_loss = G_sum/len(loader)
                        torch.save(self.G_I.state_dict(),"ref_models/best_G_I.pth")
                        torch.save(self.G_C.state_dict(),"ref_models/best_G_C.pth")
                        torch.save(self.E_SI.state_dict(),"ref_models/best_E_SI.pth")
                        torch.save(self.E_SC.state_dict(),"ref_models/best_E_SC.pth")
                        torch.save(self.D_I.state_dict(),"ref_models/best_D_I.pth")
                        torch.save(self.D_C.state_dict(),"ref_models/best_D_C.pth")
                        print("Updated best model.", G_sum/len(loader))
            print("Step-2 finished.")


    # ------------------------ Step-3 ---------------------------------------
    def augment_data(self,x0,x1_pair,labels,ratio=2.0)->Tuple[torch.Tensor,torch.Tensor]:
        self.G_C.load_state_dict(torch.load("ref_models/best_G_C.pth"))
        self.E_SI.load_state_dict(torch.load("ref_models/best_E_SI.pth"))
        self.E_SC.load_state_dict(torch.load("ref_models/best_E_SC.pth"))
        self.E_L.load_state_dict(torch.load("ref_models/best_E_L.pth"))
        self.G_I.load_state_dict(torch.load("ref_models/best_G_I.pth"))
        self.D_I.load_state_dict(torch.load("ref_models/best_D_I.pth"))
        self.D_C.load_state_dict(torch.load("ref_models/best_D_C.pth"))
        self.E_L.eval(); self.G_C.eval(); self.E_SI.eval(); self.E_SC.eval(); self.G_I.eval(); self.D_I.eval(); self.D_C.eval()
        print(f"\n[Step-3] Augment ×{ratio} ...")
        self.E_L.eval(); self.G_C.eval()
        aug_x,aug_y = [],[]
        with torch.no_grad():
            for xi,x1,lab in zip(x0,x1_pair,labels):
                xi = xi.unsqueeze(0).to(self.device)
                x1 = x1.unsqueeze(0).to(self.device)
                _,loc = self.E_L(x1)
                spa = self.E_SC(x1) # 테스트
                for _ in range(int(ratio)):
                    #spa = torch.randn(1, 64,device=self.device)
                    aug = self.G_I(loc, spa).cpu()
                    aug_x.append(aug); aug_y.append(lab.unsqueeze(0))
        return torch.cat(aug_x), torch.cat(aug_y)

    def save_model(self,path="ref_models/degn_lic_scene0_to_scene1.pth"):
        torch.save({k:v.state_dict() for k,v in self.__dict__.items() if isinstance(v,nn.Module)},path)
        print(f"Saved model → {path}")

# ---------------------------------------------------------------------------
# 3. 데이터 로드 & 시각화
# ---------------------------------------------------------------------------
def normalize_img(img):
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        return (img - img_min) / (img_max - img_min)
    else:
        return img * 0  # 값이 모두 같을 때는 0으로

def load_real_data()->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    print("\nLoad scene0/scene1 ...")
    x0 = np.load('data/scene0_cir.npy'); x1=np.load('data/scene1_cir.npy')
    if np.iscomplexobj(x0): x0=np.abs(x0)
    if np.iscomplexobj(x1): x1=np.abs(x1)
    x0 = torch.from_numpy(x0).float()                # 289
    x1_pair = torch.from_numpy(x1[:x0.shape[0]]).float()  # 앞 289

    # x0와 x1_pair를 -1~1 범위로 정규화
    x0_min, x0_max = x0.min(), x0.max()
    x1_min, x1_max = x1_pair.min(), x1_pair.max()
    x0 = 2 * (x0 - x0_min) / (x0_max - x0_min) - 1
    x1_pair = 2 * (x1_pair - x1_min) / (x1_max - x1_min) - 1

    num_pos=24; per=x0.shape[0]//num_pos
    lab=[rp for rp in range(num_pos) for _ in range(per)]
    labels=torch.tensor(lab,dtype=torch.long)
    print(" scene0",x0.shape," scene1-pair",x1_pair.shape, " labels", labels.shape)
    return x0,x1_pair,labels

def visualize(org,org_1,aug,n=5):
    org = np.transpose(org, (0, 2, 3, 1))
    org_1 = np.transpose(org_1, (0, 2, 3, 1))
    aug = np.transpose(aug, (0, 2, 3, 1))
    fig,ax=plt.subplots(3,n,figsize=(14,6))
    for i in range(n):
        ax[0,i].imshow(normalize_img(org[i]),cmap='inferno'); ax[0,i].set_title(f"O0{i}"); ax[0,i].axis('off')
        ax[1,i].imshow(normalize_img(org_1[i]),cmap='inferno'); ax[1,i].set_title(f"O1{i}"); ax[1,i].axis('off')
        ax[2,i].imshow(normalize_img(aug[i]),cmap='inferno'); ax[2,i].set_title(f"A{i}"); ax[2,i].axis('off')
    plt.tight_layout(); plt.savefig('ref_results/augmentation_results.png'); plt.show()

# ---------------------------------------------------------------------------
# 4. main
# ---------------------------------------------------------------------------

def main():
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print("Device :",device)
    x0,x1_pair,labels = load_real_data()

    model = DEGN_LIC(n_pos=len(torch.unique(labels)),device=device)

    model.step1_train_location_encoder(x0.to(device),x1_pair.to(device),
                                       labels.to(device))
    model.step2_train_main_network(x0.to(device),x1_pair.to(device),labels.to(device))
    aug_x,aug_y = model.augment_data(x0,x1_pair,labels,ratio=1.0)
    np.save('ref_results/augmented_scene1_style.npy',aug_x.numpy())
    np.save('ref_results/augmented_labels.npy',aug_y.numpy())
    model.save_model()
    visualize(x1_pair[60:511:90],x0[60:511:90],aug_x[60:511:90])
    print("\nFinished. Files generated:\n  • ref_models/best_E_L.pth\n  • ref_models/degn_lic_scene0_to_scene1.pth"
          "\n  • ref_results/augmented_scene1_style.npy\n  • ref_results/augmented_labels.npy\n  • ref_results/augmentation_results.png")

if __name__=="__main__":
    main()
