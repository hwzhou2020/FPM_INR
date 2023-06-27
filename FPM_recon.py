import os
import tqdm
import mat73
import imageio
import argparse
import scipy.io as sio
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
import torch.nn.functional as F

from network import FullModel
from utils import newcmp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_sub_spectrum(img_complex, led_num, x_0, y_0, x_1, y_1, spectrum_mask,epoch):
    O = torch.fft.fftshift(torch.fft.fft2(img_complex))
    O_sub = torch.stack([
        O[:, x_0[i]:x_1[i], y_0[i]:y_1[i]] for i in range(len(led_num))
    ], dim = 1)
    O_sub = O_sub * spectrum_mask
    o_sub = torch.fft.ifft2(torch.fft.ifftshift(O_sub))
    oI_sub = torch.abs(o_sub) ** 2

    # # For debug purpose: Sanity check spectrum mask
    
    # if epoch % 10 == 0:
    #     for idx in range((oI_sub.size())[1]):
    #         fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
        
    #         im = axs[0,0].imshow(np.log(np.abs(O_sub[0,idx,:,:].detach().cpu().numpy())+1), cmap='gray')
    #         axs[0,0].axis('image')
    #         axs[0,0].set_title('Reconstructed spectrum')
    #         divider = make_axes_locatable(axs[0,0])
    #         cax = divider.append_axes('right', size='5%', pad=0.05)
    #         fig.colorbar(im, cax=cax, orientation='vertical')
        
    #         im = axs[0,1].imshow(oI_sub[0,idx,:,:].detach().cpu().numpy(), cmap="gray")
    #         axs[0,1].axis('image')
    #         axs[0,1].set_title('Guessed capture')
    #         divider = make_axes_locatable(axs[0,1])
    #         cax = divider.append_axes('right', size='5%', pad=0.05)
    #         fig.colorbar(im, cax=cax, orientation='vertical')
             
    #         im = axs[1,0].imshow(np.abs(spectrum_mask[0,idx,:,:].detach().cpu().numpy()), cmap='gray')
    #         axs[1,0].axis('image')
    #         axs[1,0].set_title('Sub aperture amplitude')
    #         divider = make_axes_locatable(axs[1,0])
    #         cax = divider.append_axes('right', size='5%', pad=0.05)
    #         fig.colorbar(im, cax=cax, orientation='vertical')
        
    #         im = axs[1,1].imshow(np.angle(spectrum_mask[0,idx,:,:].detach().cpu().numpy()), cmap="gray")
    #         axs[1,1].axis('image')
    #         axs[1,1].set_title('Sub aperture phase')
    #         divider = make_axes_locatable(axs[1,1])
    #         cax = divider.append_axes('right', size='5%', pad=0.05)
    #         fig.colorbar(im, cax=cax, orientation='vertical')
            
    #         plt.show()
    
    return oI_sub


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--lr_decay_step', default=30, type=int)
    parser.add_argument('--num_feats', default=32, type=int)
    parser.add_argument('--fit_3D',default=True, action='store_true')
    args = parser.parse_args()

    fit_3D = args.fit_3D
    num_epochs = args.num_epochs
    num_feats = args.num_feats
    lr_decay_step = args.lr_decay_step

    vis_dir = f'./vis/feat{num_feats}'
    if fit_3D:
        vis_dir += '_3D'
        os.makedirs(f'{vis_dir}/vid', exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    if fit_3D:
        # LED Matrix setup
        # Load data
        I = np.load('data/I_800_800_225.npy')

        # ROI selection 
        Isum = I

        # Raw data central frame preview
        plt.figure(dpi=200)
        plt.imshow(I[:, :, 113], cmap='gray')
        plt.title('Raw data central frame preview')
        plt.savefig(f'{vis_dir}/raw_data.png')

        plt.figure(dpi=200)
        plt.imshow(Isum[:, :, 113], cmap='gray')
        plt.title('Raw data cropped preview')
        plt.savefig(f'{vis_dir}/raw_data_cropped.png')

        # Raw measurement sidelength
        M = Isum.shape[0]
        N = Isum.shape[1]
        # number of LEDs along each axis
        ledM = 15 
        ledN = 15  
        
        # Distance between two adjacent LEDs (unit: um)
        D_led = 4000
        # Distance from central LED to sample (unit: um)
        h = 66000
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
        # [x,y] image patch center shift with respect to the optical center (unit: pixel)
        x = 450 
        y = -50
        # shift in um
        objdx = x * D_pixel
        objdy = y * D_pixel
        
        # Check Nyquist Sampling (Not needed here)
        # Rcam = wavelength / NA * mag / 2 / pixel_size 
        # RLED = NA * np.sqrt(D_led ** 2 + h ** 2) / D_led
        # Roverlap = 1 / np.pi * (2 * np.arccos(1 / 2 / RLED) - 1 / RLED * np.sqrt(1 - (1 / 2 / RLED) ** 2))
        
        # Calculate upsampliing ratio
        MAGimg = np.ceil(1 + 2 * D_pixel * (ledM - 1) / 2 * D_led / np.sqrt(((ledM - 1) / 2 * D_led) ** 2 + h ** 2) / wavelength)
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
        
        
        # Only use brightfield LEDs        
        NAillu1D = NAillu.reshape(ledM * ledN)
        u1D = u.reshape(ledM * ledN)
        v1D = v.reshape(ledM * ledN)
        hhled1D = hhled.reshape(ledM * ledN)
        vvled1D = vvled.reshape(ledM * ledN)
        
        NAsort = np.sort(NAillu1D)
        order = np.argsort(NAillu1D)
        
        NAuse = NAsort[NAsort<=NA-0.05] # -0.05 is a bias (121 LEDs used)    
        ID_len = len(NAuse)

        u_use = u1D[order[:ID_len]]
        v_use = v1D[order[:ID_len]]
        hhled_use = hhled1D[order[:ID_len]]
        vvled_use = vvled1D[order[:ID_len]]
        
        I_idx_use = ( (hhled_use + ledM//2) * (ledM) + 
                     (vvled_use + ledN//2 + 1) ).astype('int')

        Isum = Isum[:,:,I_idx_use]
        
        # NA shift in pixel from the LED oblique illuminations
        ledpos_true = np.zeros((ID_len, 2), dtype=int)
        count = 0
        for idx in range(ID_len):
                Fx1_temp = np.abs(Fxx1 - k0*u_use[idx])
                ledpos_true[count,0] = np.argmin(Fx1_temp)
                Fy1_temp = np.abs(Fyy1 - k0*v_use[idx])
                ledpos_true[count,1] = np.argmin(Fy1_temp)
                count += 1
        
    else:
        # LED Dome setup for high NA FPM
        
        # Load data and some parameters
        data = mat73.loadmat('data/202105051634_H5095_2170103_C11440_b.mat') # b stands for blue
        imlow = data['imlow']
        darkframe = data['darkframe']
        exposure_set = data['exposure_set']
        ID = data['ID']
        NA_lower = data['NA_lower']
        NA_upper = data['NA_upper']
        # ROI size
        M = 256  # cropped image size
        N = 256   # cropped image size
        # LED central wavelength (unit: um)
        wavelength = 0.465
        # free space k-vector
        k0 = 2 * np.pi / wavelength
        # Camera pixel pitch (unit: um)
        pixel_size = 6.5
        # Objective lens magnification
        mag = 50
        # Pixel size at image plane (unit: um)
        D_pixel = 6.5 / 50  # camera pixel_size/mag
        # Objective lens numerical aperture
        NA = 0.95
        # Maximum k-value
        kmax = NA * k0
        # NA pre-calibration data
        NA_shift_corrected = sio.loadmat('data/illu_NA_20210503_H5095.mat')['NA_shift_corrected']
        NA_illu_max = np.max(np.sqrt(NA_shift_corrected[:, 0]**2 + NA_shift_corrected[:,1]**2));
        
        # Calculate upsampliing ratio
        MAGimg = 2 #np.ceil(1+2*D_pixel*NA_illu_max/lambda); %upsampling rate
        MM = int(M*MAGimg)
        NN = int(N*MAGimg)  
        # Central of ROI
        cx = 1040
        cy = 1050
        # upsampling frequency domain
        Fxx1, Fyy1 = np.meshgrid(np.arange(-NN / 2, NN / 2), np.arange(-MM / 2, MM / 2))
        Fxx1 = Fxx1[0, :] / (N * D_pixel) * (2 * np.pi)  ## [1,1600]
        Fyy1 = Fyy1[:, 0] / (M * D_pixel) * (2 * np.pi)  ## [1600,1]
        Fxx2 = Fxx1 * Fxx1
        Fyy2 = Fyy1 * Fyy1
        Frr2 = Fxx2 + Fyy2
        # ROI central shift with respect to the optical axis center
        offset_cy = -(cx-1024) * D_pixel
        offset_cx = (cy-1024) * D_pixel
        # Number of LEDs
        ID_len = 39
        # NA shift in pixel from the LED oblique illuminations
        ledpos_true = np.zeros((ID_len, 2), dtype=int)    
        for i in range(ID_len):
            Fx1_temp = np.abs(Fxx1 - k0 * NA_shift_corrected[i, 0])
            ledpos_true[i, 0] = np.argmin(Fx1_temp)
            Fy1_temp = np.abs(Fyy1 - k0 * NA_shift_corrected[i, 1])
            ledpos_true[i, 1] = np.argmin(Fy1_temp)
        # Raw data for use
        Isum = imlow[int(cy-N/2): int(cy+N/2), int(cx-M/2):int(cx+M/2), :]

    # Raw measurements
    Isum = Isum.astype('float64')
    Isum = Isum / np.max(Isum)

    # Define angular spectrum
    kxx, kyy = np.meshgrid(Fxx1[:M], Fxx1[:N])
    kxx = kxx - np.mean(kxx)
    kyy = kyy - np.mean(kyy)
    krr = np.sqrt(kxx ** 2 + kyy ** 2)
    mask_k = k0 ** 2 - krr ** 2 > 0
    kzz_ampli = mask_k * np.abs(np.sqrt((k0 ** 2 - krr.astype('complex64') ** 2)) )  ########## fixed bug here 
    kzz_phase = np.angle(np.sqrt((k0 ** 2 - krr.astype('complex64') ** 2)) )
    kzz = kzz_ampli * np.exp(1j*kzz_phase)

    # Define Pupil support        
    Fx1, Fy1 = np.meshgrid(np.arange(-N / 2, N / 2), np.arange(-M / 2, M / 2))
    Fx1 = Fx1 / (N * D_pixel) * (2 * np.pi)
    Fy1 = Fy1 / (M * D_pixel) * (2 * np.pi)
    Fx2 = Fx1 ** 2
    Fy2 = Fy1 ** 2
    Fxy2 = Fx2 + Fy2
    Pupil0 = np.zeros((M, N))
    Pupil0[Fxy2 <= (kmax ** 2)] = 1

    Pupil0 = torch.from_numpy(Pupil0).view(1, 1, Pupil0.shape[0], Pupil0.shape[1]).to(device)
    kzz = torch.from_numpy(kzz).to(device).unsqueeze(0)
    Isum = torch.from_numpy(Isum).to(device)

    led_batch_size = 64

    model = FullModel(
        w=MM,
        h=MM,
        num_feats=num_feats,
        Pupil0=Pupil0,
        n_views=ID_len,
        z_min=-10.0,
        z_max=20.0
    ).to(device)

    optimizer = torch.optim.Adam(lr=1e-2, params=model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=0.1)

    t = tqdm.trange(num_epochs + 1)
    for epoch in t:
        led_idices = list(np.arange(ID_len)) # list(np.random.permutation(ID_len))
        _fill = len(led_idices) - (len(led_idices) % led_batch_size)
        led_idices = led_idices + list(np.random.choice(led_idices, _fill, replace=False))

        if fit_3D:
            # dzs = torch.rand(6).to(device) * 25 - 5
            # if epoch % 2 == 0:
            #     dzs = torch.FloatTensor([-1.2500, 1.7500, 4.7500, 7.7500, 10.7500, 13.7500, 16.7500, 19.7500]).to(device)
            
            
            ############################# Modified to single plane 
            ############################# Not working at large defocus distance (single plane)
            dzs = torch.FloatTensor([20.0]).to(device)
        
        else:
            dzs = torch.FloatTensor([0.0]).to(device)

        if epoch == 0:
            model_fn = torch.jit.trace(model, dzs[0:1])

        for dz in dzs:
            dz = dz.unsqueeze(0)
   
            artists = []
            
            for it in range(ID_len // led_batch_size + 1): 
                model.zero_grad()
                dfmask = torch.exp(1j * kzz.repeat(dz.shape[0], 1, 1) * dz[:, None, None].repeat(1, kzz.shape[1], kzz.shape[2]))
                led_num = led_idices[it * led_batch_size: (it + 1) * led_batch_size]
                dfmask = dfmask.unsqueeze(1).repeat(1, len(led_num), 1, 1)
                spectrum_mask_ampli = Pupil0.repeat(len(dz), len(led_num), 1, 1) * torch.abs(dfmask) / (MAGimg**2)
                spectrum_mask_phase = Pupil0.repeat(len(dz), len(led_num), 1, 1) * torch.angle(dfmask) / (MAGimg**2)
                spectrum_mask = spectrum_mask_ampli * torch.exp(1j*spectrum_mask_phase)
                
                img_ampli, img_phase = model_fn(dz)
                img_complex = img_ampli * torch.exp(1j*img_phase)
                uo, vo = ledpos_true[led_num, 0], ledpos_true[led_num, 1]
                x_0, x_1 = vo - M // 2, vo + M // 2
                y_0, y_1 = uo - N // 2, uo + N // 2

                oI_sub = get_sub_spectrum(img_complex, led_num, x_0, y_0, x_1, y_1, spectrum_mask, it)
                
                oI_cap = Isum[:, :, led_num]
                oI_cap = oI_cap.permute(2, 0, 1).unsqueeze(0).repeat(len(dz), 1, 1, 1)

                    
                l1_loss = F.smooth_l1_loss(oI_cap, oI_sub)
                loss = l1_loss
                mse_loss = F.mse_loss(oI_cap, oI_sub)

                loss.backward()

                psnr = 10 * -torch.log10(mse_loss).item()
                t.set_postfix(Loss = f'{loss.item():.4e}', PSNR = f'{psnr:.2f}')
                optimizer.step()
                
                # # For debug purpose: Sanity check oI_cap and oI_sub
                # if epoch == 100:
                #     for idx in range((oI_cap.size())[1]):
                #         i = it*led_batch_size + idx
                #         fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    
                #         im = axs[0].imshow(oI_cap[0,idx,:,:].detach().cpu().numpy(), cmap='gray')
                #         axs[0].axis('image')
                #         axs[0].set_title('Raw captures' + str(i))
                #         divider = make_axes_locatable(axs[0])
                #         cax = divider.append_axes('right', size='5%', pad=0.05)
                #         fig.colorbar(im, cax=cax, orientation='vertical')
    
                #         im = axs[1].imshow(oI_sub[0,idx,:,:].detach().cpu().numpy(), cmap="gray")
                #         axs[1].axis('image')
                #         axs[1].set_title('Guessed images' + str(i))
                #         divider = make_axes_locatable(axs[1])
                #         cax = divider.append_axes('right', size='5%', pad=0.05)
                #         fig.colorbar(im, cax=cax, orientation='vertical')
                        
                #         # plt.savefig('./vis/' + 'Compare ' + str(i) + '.png')
                #         plt.show()

        scheduler.step()
    
        if epoch % 100 == 0 or (epoch % 10 == 0 and epoch < 50) or epoch == num_epochs:
            # img_complex = img_complex[0]
            amplitude = (img_ampli[0]).cpu().detach().numpy() ** 2
            phase = (img_phase[0]).cpu().detach().numpy()

            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

            im = axs[0].imshow(amplitude, cmap='gray')
            axs[0].axis('image')
            axs[0].set_title('Reconstructed amplitude')
            divider = make_axes_locatable(axs[0])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')

            im = axs[1].imshow(phase - phase.mean(), cmap="magma")
            axs[1].axis('image')
            axs[1].set_title('Reconstructed phase')
            divider = make_axes_locatable(axs[1])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')

            plt.savefig(f'{vis_dir}/e_{epoch}.png')

        # if fit_3D and (epoch % 5 == 0 or epoch == num_epochs) and epoch > 0:
        #     # Render a focal sweep from 0 to 1
        #     dz = (torch.arange(60) / 59.0).to(device).view(60)
        #     dz = dz * 2.5 - 0.5
        #     dz = dz * 10
        #     with torch.no_grad():
        #         out = []
        #         for z in torch.chunk(dz, 32):
        #             img_real, img_imag = model(z)
        #             _img_complex = torch.complex(img_real, img_imag)
        #             out.append(_img_complex)
        #         img_complex = torch.cat(out, dim=0)
        #     _imgs = img_complex.abs().cpu().detach().numpy() ** 2
        #     imgs = (_imgs - _imgs.min()) / (_imgs.max() - _imgs.min())
        #     imageio.mimsave(f'{vis_dir}/vid/{epoch}.mp4', np.uint8(imgs * 255), fps=15, quality=8)


    # # For save debug video
    # writer = imageio.get_writer('./vis/Compare.mp4', fps=3)
    
    # for ids in range(64):
    #     writer.append_data(imageio.imread('./vis/' + 'Compare ' + str(ids) + '.png'))
    # writer.close()