clear;close all;clc;
addpath(genpath('subfuncions'));

%% Operator
F  = @(x) fftshift(fft2(ifftshift(x)));
Ft = @(x) ifftshift(ifft2(fftshift(x)));
logamp = @(x) log10(abs(x)+1);

%% Specify input data

dz_set = 0%-20:0.5:20;
o_set = zeros(2048,2048,length(dz_set));
for idz = 1:length(dz_set)

defocus = dz_set(idz); % unit: um

sample_name = 'WU1005_2';
color = 'r';
filename = [sample_name,'_',color,'_FPM.mat'];
% load raw data
fileName = ['FPM_Data/',sample_name,'/Preprocessed_', filename];
load(fileName);

%% Specify output folder
out_dir = 'Results_Haowen';
mkdir(out_dir);

%% Set all necessary parameters (Unit: um)
NA_ape = na_cal; % subaperture NA, objective NA in FPMA
spsize = dpix_c/mag; % pixel size on sample plane
NAshift_x = na_calib(:,1)';
NAshift_y = na_calib(:,2)';
% NAshift_x = na_design(:,1)';
% NAshift_y = na_design(:,2)';

%% Prepare input for AlterMin
% prepare I
I = double(I_low);
[Nmy,Nmx,Nimg] = size(I);

Np = Nmy; % ASSUME Nmy = Nmx
% sampling size at Fourier plane set by the image size (FoV) = 1/FoV, FoV = Np*psize;
if mod(Np,2) == 1
    du = 1/(spsize*(Np-1));
else
    du = 1/(Np*spsize);
end

% prepare Ns (pixel index of subaperture center in f space)
NAillu = sqrt(NAshift_x.^2+NAshift_y.^2);
% spatial freq index for each plane wave relative to the center
idx_u = round((-NAshift_x/lambda)/du);
idx_v = round((-NAshift_y/lambda)/du);

for m = 1:Nimg
    Nsh_lit(:,m) = idx_u(m);
    Nsv_lit(:,m) = idx_v(m);
end
Ns = [];
Ns(:,:,1) = -Nsv_lit;
Ns(:,:,2) = -Nsh_lit; % [list of spatial freq values,m = 1:Nimg,v or h]

% prepare No (size of reconstructed image)
NA_max = max(NAillu(:))+NA_ape; % maximum synthesized NA 
um_p = NA_max/lambda;
% N_obj = round(2*um_p/du)*2;
N_obj = 2*Np; % 2 %ceil(N_obj/Np)* 
if N_obj== Np
    N_obj = Np*2;
end
No = [N_obj,N_obj];

% generate w_NA (support of subaperture) & support of raw image spectrum
m = 1:Np;
[mm,nn] = meshgrid(m-round((Np+1)/2));
ridx = sqrt(mm.^2+nn.^2);
um_m = NA_ape/lambda;
um_idx = um_m/du;
w_NA = double(ridx<um_idx);
% figure;imshow(w_NA,[]);
rawimage_NA = double(ridx<(um_idx*2)); % lpf for low-res raw images
% figure;imshow(rawimage_NA,[]);

% generate H0 (pupil phase term for known defocus abberation)
k0 = 2*pi/lambda;
kmax = pi/(spsize);
dkx = kmax/(Np/2);
dky = kmax/(Np/2);
[kx, ky] = meshgrid(linspace(-kmax,kmax-dkx,Np),linspace(-kmax,kmax-dky,Np));
mask = (kx.^2+ky.^2<=(k0*NA_ape)^2).*ones(Np,Np);
H0 = mask.*(exp(1i*real(sqrt(k0^2-kx.^2-ky.^2))*defocus));

% support of reconstructed sample spectrum
% for FPMB, it is objective NA; for FPMA, it is the total subaperture coverage
m = 1:N_obj;
[mm,nn] = meshgrid(m-round((N_obj+1)/2));
ridx = sqrt(mm.^2+nn.^2);
um_idx = um_p/du;
s_NA = double(ridx<um_idx);
s_NA = zeros(No); % updated during the iteration
% figure;imshow(s_NA,[]);

%% Preprocess before algorithm starts
% lpf low-res raw images with 2NA_obj
lpf_rawimage = 0;
if lpf_rawimage == 1
    for i = 1:Nimg
        temp = I(:,:,i);
        I(:,:,i) = abs(Ft(F(temp).*rawimage_NA)).^2;
    end
end

%sort I and Ns according to the NAillu
reorder = 1;
if reorder == 1
    [NAillu_reorder,order] = sort(NAillu);
    unsorted = 1:length(NAillu);
    newInd(order) = unsorted;
    I_reorder = I(:,:,order);
    Ns_reorder = Ns(:,order,:);
else
    I_reorder = I;
    Ns_reorder = Ns;
end

%choose I and Ns to be used (for example, only Bright Field)
BF_only = 0;
if BF_only == 1
    Nused = sum(squeeze(NAillu_reorder <= NA_ape));
else
    Nused = Nimg;
end
idx_used = 1:Nused;
disp(['The number of used images is ' num2str(Nused)]);
I_used = I_reorder(:,:,idx_used);
Ns_used = Ns_reorder(:,idx_used,:);
opts.scale = ones(length(idx_used),1); % LED brightness map, but never used

%show used low-res raw images before reconstruction
last_check = 0;
if last_check == 1
    % create the video writer with 1 fps
    [~,name,~] = fileparts(filename);
    writerObj = VideoWriter([name '.avi']);
    writerObj.FrameRate = 5;
    % set the seconds per image
    % open the video writer
    open(writerObj);
    for i = 1:Nused
        figure(13),imshow(I_used(:,:,i),[]);title(num2str(i));
        if NAillu_reorder(i)>NA
            title(['Dark field, NA = ' num2str(NAillu_reorder(i))]);
        else
            title(['Bright field, NA = ' num2str(NAillu_reorder(i))]);
        end
        set(gca,'FontName','Times New Roman','FontSize',15);
        writeVideo(writerObj, getframe(gcf));
        drawnow;
    end
    close(writerObj);
end

%% reconstruction algorithm options: opts
opts.tol = 1e-8;
opts.maxIter = 11;
opts.minIter = 10;
opts.monotone = 1;

opts.display = 0;
% 'full', display every subroutine
% 'iter', display only results from outer loop
% 0, no display
opts.mode = 'real'; % 'fourier'/'real'
opts.saveIterResult = 0;
opts.out_dir = out_dir;

upsamp = @(x) padarray(x,double([(N_obj-Np)/2,(N_obj-Np)/2]));
opts.O0 = F(sqrt(I_used(:,:,1)));
opts.O0 = upsamp(opts.O0);
opts.P0 = w_NA;
opts.Os = s_NA; % support constraint for O0
opts.Ps = w_NA; % support constraint for P0
opts.AS = 1;
opts.eta = 0.001; % threshold to stop adaptive stepsize
opts.OP_alpha = 0.5;
opts.OP_beta = 0.1;
opts.H0 = H0; % known portion of the aberration function

% LED intensity correction
opts.flag_inten_corr = 1;
opts.iter_inten_corr = 3; % when to start intensity correction

% LED position correction
opts.poscalibrate = '0'; % '0','sa','ga','ps'
opts.calibratetol = 1e-1; % only for 'sa'
opts.sp = 5; % search range radius, unit: pixel
opts.iter_pos_corr = 10; % when to start intensity correction
opts.costFunc = 'ssim'; % 'mse'

% sample priori
opts.is_pure_phase = 0;
opts.iter_pure_phase = 15; % when to start intensity correction

opts.F = F;
opts.Ft = Ft;
opts.logamp = logamp;

%% real start
tic;
[O,P,err_pc,Ns_cal] = AlterMin(I_used,No,round(Ns_used),opts);
toc;
fprintf('processing completes\n');

%% display results
I_kohler = sum(I_used,3);
figure;imshow(sqrt(I_kohler),[]);title('Kohler illumination');

figure;
subplot(221);
imshow(abs(O), []);title('Sample amplitude');colorbar;
subplot(222);
imshow(angle(O), []);title('Sample phase');caxis([-pi,pi]);colorbar;
subplot(223);
imshow(abs(P), []);title('Probe amplitude');colorbar;
subplot(224);
imshow(angle(P),[]);title('Probe phase');colorbar;

figure(1213);
imshow(abs(O), []);title('Sample amplitude');%colorbar;

% figure;
% plot(1:length(err_pc),err_pc,'r-','LineWidth',1);
% set(gca,'YScale','log');
% axis square;xlabel('Iterations'),ylabel('Error');

% figure;
% plot(Ns_used(1,:,1),Ns_used(1,:,2),'r*',Ns_cal(1,:,1),Ns_cal(1,:,2),'b*','LineWidth',1);
% axis equal,axis tight;legend('Input','Output');

% %% compare Ns_cal and freqXY_calib
% xc = floor(Nmx/2+1);
% yc = floor(Nmy/2+1);
% Ns_cal_origin_order = fliplr(squeeze(Ns_cal(:,newInd,:)));
% freqXY_calib_opt = [(Ns_cal_origin_order(:,1)+xc)';(Ns_cal_origin_order(:,2)+yc)']';
% figure;
% plot(freqXY_calib(:,1),freqXY_calib(:,2),'r*',freqXY_calib_opt(:,1),freqXY_calib_opt(:,2),'b*','LineWidth',1);
% axis equal,axis tight;legend('Cali','Opt Cali');
o_set(:,:,idz) = O;
end
%% save reconstruction
flag_save = 0;
if flag_save
    SaveName = ['FPM_Data/',sample_name,'/Reconstruct_', filename];
    save([sample_name,'_',color,'_stack.mat'],'o_set','dz_set');
end


