# Main script for FPM-INR reconstruction
# Written by Haowen Zhou and Brandon Y. Feng
# Last modified on 10/26/2023
# Contact: Haowen Zhou (hzhou7@caltech.edu) 


import os
import tqdm
import mat73
import scipy.io as sio
import imageio
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import torch.nn.functional as F

from network import FullModel
from utils import save_model_with_required_grad

torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_sub_spectrum(img_complex, led_num, x_0, y_0, x_1, y_1, spectrum_mask, mag):
    O = torch.fft.fftshift(torch.fft.fft2(img_complex))
    to_pad_x = (spectrum_mask.shape[-2] * mag - O.shape[-2]) // 2
    to_pad_y = (spectrum_mask.shape[-1] * mag - O.shape[-1]) // 2
    O = F.pad(O, (to_pad_x, to_pad_x, to_pad_y, to_pad_y, 0, 0), "constant", 0)

    O_sub = torch.stack(
        [O[:, x_0[i] : x_1[i], y_0[i] : y_1[i]] for i in range(len(led_num))], dim=1
    )
    O_sub = O_sub * spectrum_mask
    o_sub = torch.fft.ifft2(torch.fft.ifftshift(O_sub))
    oI_sub = torch.abs(o_sub)

    return oI_sub


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", default=15, type=int) 
    parser.add_argument("--lr_decay_step", default=6, type=int) 
    parser.add_argument("--num_feats", default=32, type=int)
    parser.add_argument("--num_modes", default=512, type=int)
    parser.add_argument("--c2f", default=False, action="store_true")
    parser.add_argument("--fit_3D", default=True, action="store_true")
    parser.add_argument("--layer_norm", default=False, action="store_true")
    parser.add_argument("--amp", default=True, action="store_true")
    parser.add_argument("--sample", default="Thyroid", type=str)
    parser.add_argument("--color", default="g", type=str)
    parser.add_argument("--is_system", default="Linux", type=str) # "Windows". "Linux"

    args = parser.parse_args()

    fit_3D = args.fit_3D
    num_epochs = args.num_epochs
    num_feats = args.num_feats
    num_modes = args.num_modes
    lr_decay_step = args.lr_decay_step
    use_c2f = args.c2f
    use_layernorm = args.layer_norm
    use_amp = args.amp
    
    sample = args.sample
    color = args.color
    is_os = args.is_system
    
    sample_list = ["Thyroid"]
    color_list = ['r', 'g', 'b']
    if sample not in sample_list: 
        print("Error message: sample name is wrong.")
        print("Avaliable sample names: ['Thyroid'] ")
    if color not in color_list:
        print("Error message: color name is wrong.")
        print("Avaliable color names: ['r', 'g', 'b']")

    vis_dir = f"./vis/feat{num_feats}"

    
    if fit_3D:
        vis_dir += "_3D"
        os.makedirs(f"{vis_dir}/vid", exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # Load data
    I = sio.loadmat(os.path.join('data','Thyroid','Thyroid_'+color+'_1024.mat'))['I'].astype('float32') / 255

    # Raw data central frame preview
    plt.figure(dpi=200)
    plt.imshow(I[:, :, 113], cmap='gray')
    plt.title('Raw data central frame preview')
    plt.savefig(f'{vis_dir}/raw_data.png')
    
    # Select ROI
    # I = I[0:512,0:512,:]
    
    # Raw measurement sidelength
    M = I.shape[0]
    N = I.shape[1]
    # number of LEDs along each axis
    ledM = 15 
    ledN = 15  
    
    # Distance between two adjacent LEDs (unit: um)
    D_led = 4000
    # Distance from central LED to sample (unit: um)
    h = 66000
    # LED central wavelength
    if color == 'r':
        wavelength = 0.617  # um
    elif  color == 'g':
        wavelength = 0.5226  # um
    elif color =='b':
        wavelength = 0.465 # um
        
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
    # [x,y] image patch center shift with respect to the optical center (unit: pixel)
    x = 0 # 450 
    y = 0 # -50
    # shift in um
    objdx = x * D_pixel
    objdy = y * D_pixel
    
    # Check Nyquist Sampling (Not needed here)
    # Rcam = wavelength / NA * mag / 2 / pixel_size 
    # RLED = NA * np.sqrt(D_led ** 2 + h ** 2) / D_led
    # Roverlap = 1 / np.pi * (2 * np.arccos(1 / 2 / RLED) - 1 / RLED * np.sqrt(1 - (1 / 2 / RLED) ** 2))
    
    # Calculate upsampliing ratio
    MAGimg = 2 
    # Upsampled pixel count 
    MM = int(M * MAGimg)
    NN = int(N * MAGimg)

    # Define spatial frequency coordinates
    Fxx1, Fyy1 = np.meshgrid(np.arange(-NN / 2, NN / 2), np.arange(-MM / 2, MM / 2))
    Fxx1 = Fxx1[0, :] / (N * D_pixel) * (2 * np.pi)
    Fyy1 = Fyy1[:, 0] / (M * D_pixel) * (2 * np.pi)
    # Center LED coordinate
    lit_cenv = (ledM - 1) / 2
    lit_cenh = (ledN - 1) / 2
    # LED coordinate system
    vled = np.arange(0, 2 * lit_cenv + 1) - lit_cenv
    hled = np.arange(0, 2 * lit_cenh + 1) - lit_cenh
    hhled, vvled = np.meshgrid(hled, vled)
    
    # Calculate illumination NA
    u = (hhled * D_led + objdx) / np.sqrt((hhled * D_led + objdx) ** 2 + (vvled * D_led + objdy) ** 2 + h ** 2)
    v = (vvled * D_led + objdy) / np.sqrt((hhled * D_led + objdx) ** 2 + (vvled * D_led + objdy) ** 2 + h ** 2)
    NAillu = np.sqrt(u**2 + v**2)
    
    # Define LEDs to use
    # IdxUseMask = NAillu <= NA # Synthetic_NA = NA + NA
    IdxUseMask = np.sqrt((hhled)**2 + (vvled)**2)<=7 # 145 LEDs used
    ID_len = np.sum(IdxUseMask)

    u_use, v_use = np.zeros((ID_len)), np.zeros((ID_len))
    hhled_use, vvled_use = np.zeros((ID_len)), np.zeros((ID_len))
    I_idx_use = np.zeros((ID_len)).astype('int')
    Isum = np.zeros((M,N,ID_len))
    count = 0
    for i in range(ledM):
        for j in range(ledN):
            if IdxUseMask[i,j] != 0:
                u_use[count], v_use[count] = u[i,j], v[i,j] 
                hhled_use[count], vvled_use[count] = hhled[i,j], vvled[i,j]
                I_idx_use[count] = int(j + i * ledN)
                count += 1
    Isum = I[:,:,I_idx_use]
    
    # NA shift in pixel from different LED illuminations
    ledpos_true = np.zeros((ID_len, 2), dtype=int)
    count = 0
    for idx in range(ID_len):
        Fx1_temp = np.abs(Fxx1 - k0*u_use[idx])
        ledpos_true[count,0] = np.argmin(Fx1_temp)
        Fy1_temp = np.abs(Fyy1 - k0*v_use[idx])
        ledpos_true[count,1] = np.argmin(Fy1_temp)
        count += 1

    # Raw measurements
    # Isum = I[:, :, order] / np.max(I)

    # Define angular spectrum
    kxx, kyy = np.meshgrid(Fxx1[:M], Fxx1[:N])
    kxx, kyy = kxx - np.mean(kxx), kyy - np.mean(kyy)
    krr = np.sqrt(kxx**2 + kyy**2)
    mask_k = k0**2 - krr**2 > 0
    kzz_ampli = mask_k * np.abs(
        np.sqrt((k0**2 - krr.astype("complex64") ** 2))
    )  
    kzz_phase = np.angle(np.sqrt((k0**2 - krr.astype("complex64") ** 2)))
    kzz = kzz_ampli * np.exp(1j * kzz_phase)

    # Define Pupil support
    Fx1, Fy1 = np.meshgrid(np.arange(-N / 2, N / 2), np.arange(-M / 2, M / 2))
    Fx2 = (Fx1 / (N * D_pixel) * (2 * np.pi)) ** 2
    Fy2 = (Fy1 / (M * D_pixel) * (2 * np.pi)) ** 2
    Fxy2 = Fx2 + Fy2
    Pupil0 = np.zeros((M, N))
    Pupil0[Fxy2 <= (kmax**2)] = 1

    Pupil0 = (
        torch.from_numpy(Pupil0).view(1, 1, Pupil0.shape[0], Pupil0.shape[1]).to(device)
    )
    kzz = torch.from_numpy(kzz).to(device).unsqueeze(0)
    Isum = torch.from_numpy(Isum).to(device)

    # Define depth of field of brightfield microscope for determine selected z-plane
    DOF = (
        0.5 / NA**2 #+ pixel_size / mag / NA
    )  # wavelength is emphrically set as 0.5 um
    # z-slice separation (emphirically set)
    delta_z = 1.6 * DOF
    # z-range
    z_max = 20.0
    z_min = -10.0
    # number of selected z-slices
    num_z = int(np.ceil((z_max - z_min) / delta_z))
    

    # Define LED Batch size
    led_batch_size = 1
    cur_ds = 1
    if use_c2f:
        c2f_sche = (
            [4] * (num_epochs // 5)
            + [2] * (num_epochs // 5)
            + [1] * (num_epochs // 5)
        )
        cur_ds = c2f_sche[0]

    model = FullModel(
        w=MM,
        h=MM,
        num_feats=num_feats,
        x_mode=num_modes,
        y_mode=num_modes,
        z_min=z_min,
        z_max=z_max,
        ds_factor=cur_ds,
        use_layernorm=use_layernorm,
    ).to(device)

    optimizer = torch.optim.Adam(
        lr=1e-3,
        params=filter(lambda p: p.requires_grad, model.parameters()),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_decay_step, gamma=0.1
    )

    t = tqdm.trange(num_epochs)
    for epoch in t:
        led_idices = list(np.arange(ID_len))  # list(np.random.permutation(ID_len)) #
        # _fill = len(led_idices) - (len(led_idices) % led_batch_size)
        # led_idices = led_idices + list(np.random.choice(led_idices, _fill, replace=False))
        if fit_3D:
            dzs = (
                (torch.randperm(num_z - 1)[: num_z // 2] + torch.rand(num_z // 2))
                * ((z_max - z_min) // (num_z - 1))
            ).to(device) + z_min
            if epoch % 2 == 0:
                dzs = torch.linspace(z_min, z_max, num_z).to(device)
        else:
            dzs = torch.FloatTensor([5.0]).to(device)

        if use_c2f and c2f_sche[epoch] < model.ds_factor:
            model.init_scale_grids(ds_factor=c2f_sche[epoch])
            print(f"ds_factor changed to {c2f_sche[epoch]}")
            model_fn = torch.jit.trace(model, dzs[0:1])

        if epoch == 0:
            if is_os == "Windows":
                model_fn = torch.jit.trace(model, dzs[0:1])
            elif is_os == "Linux":
                model_fn = torch.compile(model, backend="inductor")
            else:
                raise NotImplementedError


        for dz in dzs:
            dz = dz.unsqueeze(0)

            for it in range(ID_len // led_batch_size):  # + 1
                model.zero_grad()
                dfmask = torch.exp(
                    1j
                    * kzz.repeat(dz.shape[0], 1, 1)
                    * dz[:, None, None].repeat(1, kzz.shape[1], kzz.shape[2])
                )
                led_num = led_idices[it * led_batch_size : (it + 1) * led_batch_size]
                dfmask = dfmask.unsqueeze(1).repeat(1, len(led_num), 1, 1)
                spectrum_mask_ampli = Pupil0.repeat(
                    len(dz), len(led_num), 1, 1
                ) * torch.abs(dfmask)
                spectrum_mask_phase = Pupil0.repeat(len(dz), len(led_num), 1, 1) * (
                    torch.angle(dfmask) + 0
                )  # 0 represent Pupil0 Phase
                spectrum_mask = spectrum_mask_ampli * torch.exp(
                    1j * spectrum_mask_phase
                )

                with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.bfloat16):
                    img_ampli, img_phase = model_fn(dz)
                    img_complex = img_ampli * torch.exp(1j * img_phase)
                    uo, vo = ledpos_true[led_num, 0], ledpos_true[led_num, 1]
                    x_0, x_1 = vo - M // 2, vo + M // 2
                    y_0, y_1 = uo - N // 2, uo + N // 2

                    oI_cap = torch.sqrt(Isum[:, :, led_num])
                    oI_cap = (
                        oI_cap.permute(2, 0, 1).unsqueeze(0).repeat(len(dz), 1, 1, 1)
                    )

                    oI_sub = get_sub_spectrum(
                        img_complex, led_num, x_0, y_0, x_1, y_1, spectrum_mask, MAGimg
                    )

                    l1_loss = F.smooth_l1_loss(oI_cap, oI_sub)
                    loss = l1_loss
                    mse_loss = F.mse_loss(oI_cap, oI_sub)

                loss.backward()

                psnr = 10 * -torch.log10(mse_loss).item()
                t.set_postfix(Loss=f"{loss.item():.4e}", PSNR=f"{psnr:.2f}")
                optimizer.step()

        scheduler.step()
        

        if (epoch+1) % 10 == 0 or ( epoch % 2 == 0 and epoch < 20) or epoch == num_epochs:

            amplitude = (img_ampli[0].float()).cpu().detach().numpy() 
            phase = (img_phase[0].float()).cpu().detach().numpy()

            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

            im = axs[0].imshow(amplitude, cmap="gray")
            axs[0].axis("image")
            axs[0].set_title("Reconstructed amplitude")
            divider = make_axes_locatable(axs[0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical")

            im = axs[1].imshow(phase , cmap="gray") # - phase.mean()
            axs[1].axis("image")
            axs[1].set_title("Reconstructed phase")
            divider = make_axes_locatable(axs[1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical")

            plt.savefig(f"{vis_dir}/e_{epoch}.png")

        if fit_3D and (epoch % 5 == 0 or epoch == num_epochs) and epoch > 0:
            dz = torch.linspace(z_min, z_max, 61).to(device).view(61)
            with torch.no_grad():
                out = []
                for z in torch.chunk(dz, 32):
                    img_ampli, img_phase = model(z)
                    _img_complex = img_ampli * torch.exp(1j * img_phase)
                    out.append(_img_complex)
                img_complex = torch.cat(out, dim=0)
            _imgs = img_complex.abs().cpu().detach().numpy()
            # Save amplitude
            imgs = (_imgs - _imgs.min()) / (_imgs.max() - _imgs.min())
            imageio.mimsave(
                f"{vis_dir}/vid/{epoch}.mp4", np.uint8(imgs * 255), fps=5, quality=8
            )

    save_path = os.path.join('trained_models', sample +'_'+ color + '.pth')
    save_model_with_required_grad(model, save_path)
