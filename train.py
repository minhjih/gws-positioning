

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
        self.conv2 = nn.Conv2d(16, 32, 4, 2, 1)  # 15→7
        self.conv3 = nn.Conv2d(32,64, 3, 2, 1)  # 7→4
        self.res1 = ResidualBlock(64,64)
        self.res2 = ResidualBlock(64,64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1  = nn.Linear(64, dim)
        self.fc2  = nn.Linear(dim, dim)
        self.cls  = nn.Linear(dim, n_pos)
        self.act, self.drop = nn.LeakyReLU(0.2), nn.Dropout(0.2)
    def forward(self,x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.res1(x); x = self.res2(x)
        x = self.pool(x).view(x.size(0),-1)
        feat = self.fc2(self.drop(self.act(self.fc1(x))))
        return self.cls(feat), feat

class SpatialEncoder(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(3,32,4,2,1)
        self.conv2 = nn.Conv2d(32,64,4,2,1)
        self.conv3 = nn.Conv2d(64,128,3,2,1)
        self.pool  = nn.AdaptiveAvgPool2d(1)
        self.fc    = nn.Linear(128,dim)
        self.act   = nn.LeakyReLU(0.2)
    def forward(self,x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        return self.fc(self.pool(x).view(x.size(0),-1))

class Generator(nn.Module):
    def __init__(self, ldim=64, sdim=64): #cls 넣기 전을 feature vector로 넣는것이 맞는지 확인필요.
        super().__init__()
        tdim = ldim+sdim
        self.fc1 = nn.Linear(tdim,256)
        self.fc2 = nn.Linear(256,512)
        self.fc3 = nn.Linear(512,128*4*4)
        self.res1 = ResidualBlock(128,128)
        self.res2 = ResidualBlock(128,128)
        self.up1  = nn.ConvTranspose2d(128,64,4,2,1)     # 4→8
        self.up2  = nn.ConvTranspose2d(64,32,4,2,1)      # 8→16
        self.up3  = nn.ConvTranspose2d(32,16,3,2,1,1)    # 16→32
        self.out  = nn.Conv2d(16,3,3,1,1)
        self.crop = nn.AdaptiveAvgPool2d(30)
        self.act  = nn.LeakyReLU(0.2)
    def forward(self,l,s):
        x = torch.cat([l,s],1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = x.view(x.size(0),128,4,4)
        x = self.res1(x); x = self.res2(x)
        x = self.act(self.up1(x)); x = self.act(self.up2(x)); x = self.act(self.up3(x))
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
        return self.fc(x)

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

        self.ce, self.l1, self.gan = nn.CrossEntropyLoss(), nn.L1Loss(), nn.BCEWithLogitsLoss()
        self.lam1, self.lam2, self.lam3 = 10.0, 1.0, 1.0

    # ------------------------ Step-1 ---------------------------------------
    def step1_train_location_encoder(self,x0,x1,labels,epochs=100,lr=0.8e-2,batch=16):
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
            if vacc>=best: best=vacc; torch.save(self.E_L.state_dict(),"best_E_L.pth")
            sched.step()
        self.E_L.load_state_dict(torch.load("best_E_L.pth"))
        self.E_L.eval();  [p.requires_grad_(False) for p in self.E_L.parameters()]
        print(f" Best Val Acc = {100*best/vtot:.2f}%")

    # ------------------------ Step-2  (버그 fix 포함) ------------------------
    def step2_train_main_network(self,x0,x1,labels,epochs=100,batch=8,lr_g=2e-4,lr_d=2e-4):
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
            D_sum=G_sum=0.0
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
                if b%50==0:
                    print(f" Ep {ep+1:03d}/{epochs}  B{b:03d}  D {loss_D.item():.3f}  G {loss_G.item():.3f}")
            if ep%10==0:
                print(f" >> Ep {ep+1:03d}  mean D {D_sum/len(loader):.3f}  G {G_sum/len(loader):.3f}")
        print("Step-2 finished.")

    # ------------------------ Step-3 ---------------------------------------
    def augment_data(self,x0,labels,ratio=2.0)->Tuple[torch.Tensor,torch.Tensor]:
        print(f"\n[Step-3] Augment ×{ratio} ...")
        self.E_L.eval(); self.G_C.eval()
        aug_x,aug_y = [],[]
        with torch.no_grad():
            for xi,lab in zip(x0,labels):
                xi = xi.unsqueeze(0).to(self.device)
                _,loc = self.E_L(xi)
                for _ in range(int(ratio)):
                    spa = torch.randn(1,64,device=self.device)
                    aug = self.G_C(loc,spa).cpu()
                    aug_x.append(aug); aug_y.append(lab.unsqueeze(0))
        return torch.cat(aug_x), torch.cat(aug_y)

    def save_model(self,path="degn_lic_scene0_to_scene1.pth"):
        torch.save({k:v.state_dict() for k,v in self.__dict__.items() if isinstance(v,nn.Module)},path)
        print(f"Saved model → {path}")

# ---------------------------------------------------------------------------
# 3. 데이터 로드 & 시각화
# ---------------------------------------------------------------------------

def load_real_data()->Tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    print("\nLoad scene0/scene1 ...")
    x0 = np.load('scene0_cir.npy'); x1=np.load('scene1_cir.npy')
    if np.iscomplexobj(x0): x0=np.abs(x0)
    if np.iscomplexobj(x1): x1=np.abs(x1)
    x0 = torch.from_numpy(x0).float()                # 289
    x1_pair = torch.from_numpy(x1[:x0.shape[0]]).float()  # 앞 289
    num_pos=24; per=x0.shape[0]//num_pos
    lab=[rp for rp in range(num_pos) for _ in range(per)]
    print(lab)
    labels=torch.tensor(lab,dtype=torch.long)
    print(" scene0",x0.shape," scene1-pair",x1_pair.shape, " labels", labels.shape)
    return x0,x1_pair,labels

def visualize(org,aug,n=5):
    fig,ax=plt.subplots(2,n,figsize=(14,6))
    for i in range(n):
        ax[0,i].imshow(org[i,0],cmap='viridis'); ax[0,i].set_title(f"O{i}"); ax[0,i].axis('off')
        ax[1,i].imshow(aug[i,0],cmap='viridis'); ax[1,i].set_title(f"A{i}"); ax[1,i].axis('off')
    plt.tight_layout(); plt.savefig('augmentation_results.png'); plt.show()

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
    aug_x,aug_y = model.augment_data(x0,labels,ratio=2.0)
    np.save('augmented_scene1_style.npy',aug_x.numpy())
    np.save('augmented_labels.npy',aug_y.numpy())
    model.save_model()
    visualize(x0[:5],aug_x[:5])
    print("\nFinished. Files generated:\n  • best_E_L.pth\n  • degn_lic_scene0_to_scene1.pth"
          "\n  • augmented_scene1_style.npy\n  • augmented_labels.npy\n  • augmentation_results.png")

if __name__=="__main__":
    main()
