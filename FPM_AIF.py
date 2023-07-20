import time
import mat73
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import load_model_with_required_grad
from network import FullModel,FullModel_v2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the Gaussian blur kernel
def gaussian_kernel(kernel_size, sigma, truncate=4.0):
    kernel = torch.Tensor([[np.exp(-(x**2 + y**2)/(2*sigma**2)) 
                            for x in range(-kernel_size//2+1, kernel_size//2+1)] 
                           for y in range(-kernel_size//2+1, kernel_size//2+1)])
    kernel = kernel / kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    return kernel

# Define lightfield-based all-in-focus module
class LightField_AIF(nn.Module):
    def __init__(self, sigma1, sigma2, truncate=4.0):
        super(LightField_AIF, self).__init__()
        kernel_size1 = int(2 * truncate * sigma1 - 1)
        kernel_size2 = int(2 * truncate * sigma2 - 1)
        self.padding1 = kernel_size1 // 2 
        self.padding2 = kernel_size2 // 2 
        self.kernel_1 = gaussian_kernel(kernel_size1, sigma1)
        self.kernel_2 = gaussian_kernel(kernel_size2, sigma2)
        
    def forward(self, img, depth):
        img_pad = F.pad(img, (self.padding1, self.padding1, self.padding1, self.padding1), mode='replicate')
        img_blur = F.conv2d(img_pad,self.kernel_1.to(device), padding=0, groups=1)
        img_hf = img - img_blur
        img_pad_hf = F.pad(img_hf, (self.padding2, self.padding2, self.padding2, self.padding2), mode='replicate')
        w_sharp = F.conv2d(img_pad_hf**2, self.kernel_2.to(device), padding=0, groups=1)
        im_AIF = torch.sum(w_sharp * img_hf, dim=0) / torch.sum(w_sharp, dim=0)
        im_depth = torch.sum(w_sharp * depth.unsqueeze(1).unsqueeze(1).unsqueeze(1), dim=0) / torch.sum(w_sharp, dim=0)
        return im_AIF, im_depth

# Define balance map for normal variance method
def CreateBalanceMap(m, patchsize, patchpace):
    Iternum = int((m - patchsize) / patchpace + 1)
    balancemap = np.zeros((m, m))
    
    for i in range(Iternum):
        for j in range(Iternum):
            start_ud = i * patchpace
            end_ud = start_ud + patchsize
            start_lr = j * patchpace
            end_lr = start_lr + patchsize
            
            balancemap[start_ud:end_ud, start_lr:end_lr] += np.ones((patchsize, patchsize))
    
    balancemap = 1 / balancemap
    
    return Iternum, balancemap

# AIF Normal Variance method
def AIF_NV(imgs_gt):
    m, n, framenum = imgs_gt.shape
    
    o_sum = np.mean(imgs_gt, axis=2)
    
    intensity, edges = np.histogram(np.abs(o_sum), bins='auto')
    pos_i = np.argmax(intensity)
    background = edges[pos_i]
    
    o_fusion = np.zeros((m, n))
    
    patchsize = 64
    patchpace = 16
    Iternum, balancemap = CreateBalanceMap(m, patchsize, patchpace)
    
    for i in range(Iternum):
        for j in range(Iternum):
            NV = []
            for k in range(framenum):
                start_ud = i * patchpace 
                end_ud = start_ud + patchsize
                start_lr = j * patchpace 
                end_lr = start_lr + patchsize
                
                o_cropped = imgs_gt[start_ud:end_ud, start_lr:end_lr, k]
                
                # Rule out FPM artifacts (seems not necessary for fluorescence imaging)
                if background > 0.1:
                    o_cropped[o_cropped > background] = background
                
                imgs_gt[start_ud:end_ud, start_lr:end_lr, k] = o_cropped
                
                mu = np.mean(o_cropped)
                NV.append(np.sum((o_cropped - mu) ** 2) / (patchsize ** 2 * mu))
            
            loc_info = np.argmax(NV)
            o_fusion[start_ud:end_ud, start_lr:end_lr] += imgs_gt[start_ud:end_ud, start_lr:end_lr, loc_info]
    
    o_fusion_balanced = o_fusion * balancemap
    return o_fusion_balanced

if __name__ == "__main__":

    #  ~ 6 minutes to run the whole script

    # Parameters
    num_feats = 32
    M = 800 # image size
    N = 800
    # LED central wavelength
    wavelength = 0.5226  # um
    # free-space k-vector
    k0 = 2 * np.pi / wavelength
    # Objective lens magnification
    mag = 20
    # Camera pixel pitch (unit: um)
    pixel_size = 5.5
    # pixel size at image plane (unit: um)
    D_pixel = pixel_size / mag
    # Objective lens NA
    NA = 0.4
    # Maximum k-value
    kmax = NA * k0
    # Upsampliing ratio
    MAGimg = 2 
    # Number of LEDs
    ID_len = 145
    # Upsampled pixel count 
    MM = int(M * MAGimg)
    NN = int(N * MAGimg)
    
    # Define Pupil support        
    Fx1, Fy1 = np.meshgrid(np.arange(-N / 2, N / 2), np.arange(-M / 2, M / 2))
    Fx2 = (Fx1 / (N * D_pixel) * (2 * np.pi)) ** 2
    Fy2 = (Fy1 / (M * D_pixel) * (2 * np.pi)) ** 2
    Fxy2 = Fx2 + Fy2
    Pupil0 = np.zeros((M, N))
    Pupil0[Fxy2 <= (kmax ** 2)] = 1

    Pupil0 = torch.from_numpy(Pupil0).view(1, 1, Pupil0.shape[0], Pupil0.shape[1]).to(device)

    # Define depth of field of brightfield microscope for determine selected z-plane
    DOF = 0.5 / NA**2 + pixel_size / mag / NA     
    # z-slice separation (emphirically set)
    delta_z = 1.2 * DOF
    # z-range
    z_max = 20.0
    z_min = -10.0
    # number of selected z-slices
    num_z = int( np.ceil( (z_max - z_min) / delta_z ) )

    # Define LED Batch size
    led_batch_size = 1

    model = FullModel_v2(
        w=MM,
        h=MM,
        num_feats=num_feats,
        Pupil0=Pupil0,
        n_views=ID_len,
        z_min=z_min,
        z_max=z_max
    ).to(device)

    # Load trained model
    load_model_with_required_grad(model, 'trained_model.pth')
    # model inference
    dz = torch.linspace(z_min,z_max,121).to(device).view(121)
    with torch.no_grad():
        out = []
        start_time = time.time()
        for z in torch.chunk(dz, 32):
            img_ampli, img_phase = model(z)
            _img_complex = img_ampli * torch.exp(1j*img_phase)
            out.append(_img_complex)
        img_complex = torch.cat(out, dim=0)
        end_time = time.time()
    _imgs = img_complex.abs()

    print('Model inference time elpased: ', np.round(end_time - start_time,2),'s')
    # Save amplitude 
    imgs = (_imgs - _imgs.min()) / (_imgs.max() - _imgs.min())

    ## Light field method

    # All-in-focus Section
    sigma1 = 64                      ################### These two are important and need to adjust
    sigma2 = 10
    
    # load AIF model
    AIF_LF = LightField_AIF(sigma1=sigma1, sigma2=sigma2).to(device)

    # AIF FPM-INR
    start_time = time.time()
    img_AIF,img_Depth = AIF_LF(imgs.unsqueeze(1),dz)
    end_time = time.time()
    print('FPM-INR AIF_LF time elpased: ', np.round(end_time - start_time,2),'s')
    
    # Load ground truth
    o_set  = np.abs(mat73.loadmat('data/20190113_Thyroid_NB11796113_pap_5_b_stack_0219.mat')['o_set']).astype('float32')
    _imgs_gt = torch.from_numpy(np.moveaxis(o_set,-1,0)).to(device)
    imgs_gt = (_imgs_gt - _imgs_gt.min()) / (_imgs_gt.max() - _imgs_gt.min())

    # All-in-focus FPM
    start_time = time.time()
    img_AIF_gt,img_Depth_gt = AIF_LF(imgs_gt.unsqueeze(1),dz)
    end_time = time.time()
    print('FPM AIF_LF Time elpased: ', np.round(end_time - start_time,2),'s')
    
    # Calculate PSNR
    mse_loss = F.mse_loss(img_AIF_gt, img_AIF)
    AIF_psnr = 10 * -torch.log10(mse_loss).item()
    abs_error = F.l1_loss(img_Depth_gt, img_Depth)
    Depth_error = abs_error.item()
    
    print('AIF PSNR: ', np.round(AIF_psnr,2), 'dB')
    print('Depth Error: ', np.round(Depth_error,2), 'um with brightfield microscope DOF: ', np.round(DOF,2), 'um')

    # Plot
    fig, ax = plt.subplots(1, 2)
    fig.set_dpi(600)
    cmap = 'gray'
    ax[0].imshow(img_AIF_gt[0].detach().cpu().numpy(), cmap=cmap)
    ax[0].set_title('FPM AIF_LF Image')
    ax[0].axis('off')
    ax[1].imshow(img_AIF[0].detach().cpu().numpy(), cmap=cmap)
    ax[1].set_title('FPM-INR AIF_LF Image')
    ax[1].axis('off')
    
    
    cmap = 'jet'
    fig, ax = plt.subplots(1, 2)
    fig.set_dpi(600)
    ax[0].imshow(img_Depth_gt[0].detach().cpu().numpy(), cmap=cmap)
    ax[0].set_title('FPM LF Depth Map')
    ax[0].axis('off')
    ax[1].imshow(img_Depth[0].detach().cpu().numpy(), cmap=cmap)
    ax[1].set_title('FPM-INR LF Depth Map')
    ax[1].axis('off')
    plt.show()


    ## Normal Variance AIF method
    start_time = time.time()
    imgs_gt = (o_set - np.min(o_set)) / (np.max(o_set) - np.min(o_set))
    img_AIFNV_gt = AIF_NV(imgs_gt)
    end_time = time.time()
    print('FPM-INR AIF_NV time elpased: ', np.round(end_time - start_time,2),'s')
    
    start_time = time.time()
    imgs_AIF_INR = np.moveaxis(imgs.detach().cpu().numpy(),0,-1)
    img_AIFNV = AIF_NV(imgs_AIF_INR)
    end_time = time.time()
    print('FPM-INR AIF_NV time elpased: ', np.round(end_time - start_time,2),'s')

    fig, ax = plt.subplots(1, 2)
    fig.set_dpi(400)
    cmap = 'gray'
    ax[0].imshow(img_AIFNV_gt, cmap=cmap)
    ax[0].set_title('FPM AIF_NV Image')
    ax[0].axis('off')
    ax[1].imshow(img_AIFNV, cmap=cmap)
    ax[1].set_title('FPM-INR AIF_NV Image')
    ax[1].axis('off')
    

    mse_loss = F.mse_loss(torch.from_numpy(img_AIFNV_gt).to(device), torch.from_numpy(img_AIFNV).to(device))
    psnr = 10 * -torch.log10(mse_loss).item()
    
    print('MSE: ',np.round(mse_loss.item(),5))
    print('PSNR: ', np.round(psnr,2),'dB')