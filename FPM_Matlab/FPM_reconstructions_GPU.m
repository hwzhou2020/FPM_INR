%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% FPM reconstruction with first-order gradient descent method and pupil 
% correction functionality

% The code comes with GPU processing
% Matlab parallel computing toolbox is required for using GPU

% This code is realted to the paper:
% FPM-INR: Fourier ptychographic microscopy image stack reconstruction using implicit neural representations

% GitHub:
% https://github.com/hwzhou2020/FPM_INR

% Written by Haowen Zhou and Mingshu Liang @ Caltech Biophotonics Lab 
% Last modified on 10/26/2023 
% Contact:
% hzhou7@caltech.edu (Haowen Zhou) 
% mlliang@caltech.edu (Mingshu Liang)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Code starts here
clear all; clc;close all;

addpath(genpath('../FPM_INR'))
%% Operator
F  = @(x) fftshift(fft2(ifftshift(x)));
Ft = @(x) ifftshift(ifft2(fftshift(x)));
logamp = @(x) log10(abs(x)+1);

%%
use_GPU = true;
if_3D = true;
use_PupilCorrection = false;
color = 'r';

if if_3D
    sample_name = 'Siemens'; % 'sheepblood', 'WU1005'
    dz_set = 0; % defocus distance
else
    sample_name = 'BloodSmearTilt';
    dz_set = 0;%-20:0.25:20;
end

filename = [sample_name,'_',color,'.mat'];

% load raw data
fileName = ['./data/',sample_name,'/', filename];
load(fileName);

%% Set all necessary parameters (Unit: um)
NA = na_cal;            % objective NA
mag = mag;              % objective magnification
pixel_size = 3.45;      % camera pixel pitch
D_led = 4*1000;         % adjacent LED distance
D_pixel = pixel_size/mag;

NAshift_x = -na_calib(:,1)'; 
NAshift_y = -na_calib(:,2)';
% NAshift_x = -na_design(:,1)';
% NAshift_y = -na_design(:,2)';

switch color
    case 'g'
        lambda = 0.5162; % wavelength
    case 'r'
        lambda = 0.632;
    case 'b'
        lambda = 0.471;
end

% Compute k-vector
k0 = 2*pi/lambda;
kmax = NA*k0; % maximum spatial frequency 

%%
I = double(I_low);

clear('I_low');
[M,N,ID_len] = size(I);

MAGimg = 2; % upsampling rate
MM=M*MAGimg;NN=N*MAGimg;

Niter = 25;

x=0;y=0; % field of view center shift

objdx=x*D_pixel;
objdy=y*D_pixel;

%% generate pupil function
[Fx1,Fy1]=meshgrid(-(N/2):(N/2-1),-(M/2):(M/2-1));
Fx1=Fx1./(N*D_pixel).*(2*pi);
Fy1=Fy1./(M*D_pixel).*(2*pi);
Fx2=Fx1.*Fx1;
Fy2=Fy1.*Fy1;
Fxy2=Fx2+Fy2;
Pupil0=zeros(M,N);
Pupil0(Fxy2<=(kmax^2))=1;%
[Fxx1,Fyy1]=meshgrid(-(NN/2):(NN/2-1),-(MM/2):(MM/2-1));
Fxx1=Fxx1(1,:)./(N*D_pixel).*(2*pi);%
Fyy1=Fyy1(:,1)./(M*D_pixel).*(2*pi);%


% corresponding angles for each LEDs
u = NAshift_x;%na_calib(:,1);
v = NAshift_y;%na_calib(:,2);
NAillu = sqrt(u.^2+v.^2);
[NAuse, order] = sort(NAillu);
I = I(:,:,order);
u = u(order);
v = v(order);

ledpos_true = zeros(ID_len,2);
for i=1:ID_len
    Fx1_temp=abs(Fxx1-k0*u(i));
    ledpos_true(i,1)=find(Fx1_temp==min(Fx1_temp));
    Fy1_temp=abs(Fyy1-k0*v(i));
    ledpos_true(i,2)=find(Fy1_temp==min(Fy1_temp));
end

%%  load images and divide into pieces & specify background region coordinates
Isum=double(I);
Isum=Isum./max(max(max(Isum)));
clear('I');


%% 
tic;
if use_GPU
    o_set = gpuArray(zeros(M*MAGimg,N*MAGimg,length(dz_set)));
else
    o_set = zeros(M*MAGimg,N*MAGimg,length(dz_set));
end

for dz = 1:length(dz_set)

    if use_GPU
        cmask = gpuArray(Pupil0);
    else
        cmask = Pupil0;
    end    
    
    [kxx, kyy] = meshgrid(Fxx1(1:M),Fxx1(1:N));
    kxx = kxx - mean2(kxx);
    kyy = kyy - mean2(kyy);
    krr = sqrt(kxx.^2 + kyy.^2);
    
    kzz = sqrt(k0.^2 - krr.^2);
    % figure(1001), imagesc(angle(cmask.*exp(1i*kzz*dz)));
    if use_GPU
        dfmask = cmask.* gpuArray(exp(1i*kzz*dz_set(dz)));
        oI=gpuArray(imresize(Isum(:,:,1),MAGimg,'bilinear'));
    else
        dfmask = cmask.* exp(1i*kzz*dz_set(dz));
        oI=imresize(Isum(:,:,1),MAGimg,'bilinear');
    end
    
    
    o=sqrt(oI);
    O=fftshift(fft2(o));
    Pupil=Pupil0;
    PupilSUM=O.*0;
    
    alpha=0.5;beta=0.1;
    
    % tic;
    for iter=1:Niter
        error_now=0;
        
        count = 0;
        for led_num=1:ID_len
    
            uo=ledpos_true(led_num,1);
            vo=ledpos_true(led_num,2);
            
            OP_bef=O((vo-M/2):(vo-1+M/2),(uo-N/2):(uo-1+N/2))./(MAGimg^2).*Pupil.*dfmask;
            o_bef=ifft2(fftshift(OP_bef));
            oI_bef=abs(o_bef).^2;
            
            if use_GPU
                oI_cap=gpuArray(Isum(:,:,led_num));
            else
                oI_cap=Isum(:,:,led_num);
            end
            if ((mean2(oI_cap)>0.1) && (mean2(oI_bef)>0.1)) || ((mean2(oI_cap)<0.1) && (mean2(oI_bef)<0.1))
                o_aft=sqrt(oI_cap)./sqrt(oI_bef).*o_bef;
            else
                o_aft=o_bef;
            end
    
            OP_aft=fftshift(fft2(o_aft));
            
            OP_diff=(OP_aft-OP_bef);
            
            temp=O((vo-M/2):(vo-1+M/2),(uo-N/2):(uo-1+N/2));
            O((vo-M/2):(vo-1+M/2),(uo-N/2):(uo-1+N/2))=temp+...
                alpha*abs(Pupil).*conj(Pupil).*OP_diff./max(max(abs(Pupil))) ...
                ./(abs(Pupil).^2+1).*conj(dfmask);
    
            if use_PupilCorrection
                Pupil=Pupil+...
                    beta*(abs(OP_bef).*conj(OP_bef)).*OP_diff./max(max(abs(temp)))./(abs(OP_bef).^2+1000).*Pupil0;
            else
                Pupil=Pupil0.*exp(1i.*angle(Pupil));
            end
            
            if (iter==1)
                PupilSUM((vo-M/2):(vo-1+M/2),(uo-N/2):(uo-1+N/2))=Pupil0+...
                    PupilSUM((vo-M/2):(vo-1+M/2),(uo-N/2):(uo-1+N/2)).*(ones(M,N)-Pupil0);
            else
                O=O.*PupilSUM;
            end
            error_now=error_now+sum(sum((abs(o_bef)-sqrt(oI_cap)).^2));
        end
    
        if  iter>1 && (error_bef-error_now)/error_bef<0.01 
            
            % Reduce the stepsize when no sufficient progress is made
            alpha = alpha/2;
            beta=beta/2;
            % Stop the iteration when Alpha is less than 0.001(convergenced)
            if(alpha<0.00001)
                alpha = 0;
            end
            
        end
        
        error_bef = error_now;
        
    
        spectrum=log(abs(O)+1);
        o=ifft2(fftshift(O));
        oI=abs(o).^2;
    
        % subplot(1,2,1)
        % imshow(spectrum,[]);
        % title('Fourier spectrum');
        % subplot(1,2,2)
        % imshow(oI,[]);
        % title(['Iteration No. = ',int2str(iter), '  \alpha = ',num2str(alpha)]);
        % pause(0.1)
        
        if(alpha == 0)
            break; 
        end
    end
    
    time = toc
    o_set(:,:,dz) = o;
    oP=angle(o);
    
    %%
    figure(1000);
    subplot(1,2,1), imagesc(oI/max(oI(:))), axis image, colormap gray; axis off; title('Amplitude');colorbar;
    subplot(1,2,2), imagesc(oP), axis image, colormap gray; axis off; caxis([-pi,pi]);title('Phase');colorbar;
    % saveas(1000+2*dz,['img_',num2str(dz),'.tif']);
end
times = toc;
o_set = gather(o_set);
save([sample_name,'_',color,'_FPM.mat'],'o_set','dz_set','-v7.3')
