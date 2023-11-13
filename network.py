import torch
import torch.nn as nn


class G_Renderer(nn.Module):
    def __init__(
        self, in_dim=32, hidden_dim=32, num_layers=2, out_dim=1, use_layernorm=False
    ):
        super().__init__()
        act_fn = nn.ReLU()
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_dim, elementwise_affine=False))
        layers.append(act_fn)
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim, elementwise_affine=False))
            layers.append(act_fn)

        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out


class G_FeatureTensor(nn.Module):
    def __init__(self, x_dim, y_dim, num_feats=32, ds_factor=1):
        super().__init__()
        self.x_dim, self.y_dim = x_dim, y_dim
        x_mode, y_mode = x_dim // ds_factor, y_dim // ds_factor
        self.num_feats = num_feats

        self.data = nn.Parameter(
            2e-4 * torch.rand((x_mode, y_mode, num_feats)) - 1e-4, requires_grad=True
        )

        half_dx, half_dy = 0.5 / x_dim, 0.5 / y_dim
        xs = torch.linspace(half_dx, 1 - half_dx, x_dim)
        ys = torch.linspace(half_dx, 1 - half_dy, y_dim)
        xv, yv = torch.meshgrid([xs, ys], indexing="ij")
        xy = torch.stack((yv.flatten(), xv.flatten())).t()

        xs = xy * torch.tensor([x_mode, y_mode], device=xs.device).float()
        indices = xs.long()
        self.lerp_weights = nn.Parameter(xs - indices.float(), requires_grad=False)

        self.x0 = nn.Parameter(
            indices[:, 0].clamp(min=0, max=x_mode - 1), requires_grad=False
        )
        self.y0 = nn.Parameter(
            indices[:, 1].clamp(min=0, max=y_mode - 1), requires_grad=False
        )
        self.x1 = nn.Parameter((self.x0 + 1).clamp(max=x_mode - 1), requires_grad=False)
        self.y1 = nn.Parameter((self.y0 + 1).clamp(max=y_mode - 1), requires_grad=False)

    def sample(self):
        return (
            self.data[self.y0, self.x0]
            * (1.0 - self.lerp_weights[:, 0:1])
            * (1.0 - self.lerp_weights[:, 1:2])
            + self.data[self.y0, self.x1]
            * self.lerp_weights[:, 0:1]
            * (1.0 - self.lerp_weights[:, 1:2])
            + self.data[self.y1, self.x0]
            * (1.0 - self.lerp_weights[:, 0:1])
            * self.lerp_weights[:, 1:2]
            + self.data[self.y1, self.x1]
            * self.lerp_weights[:, 0:1]
            * self.lerp_weights[:, 1:2]
        )

    def forward(self):
        return self.sample()


class G_Tensor(G_FeatureTensor):
    def __init__(self, im_size, num_feats=32, ds_factor=1):
        super().__init__(im_size, im_size, num_feats=num_feats, ds_factor=ds_factor)
        self.renderer = G_Renderer(in_dim=num_feats)

    def forward(self):
        feats = self.sample()
        return self.renderer(feats)


class G_Tensor3D(nn.Module):
    def __init__(
        self, x_mode, y_mode, z_dim, z_min, z_max, num_feats=32, use_layernorm=False
    ):
        super().__init__()
        self.x_mode, self.y_mode, self.num_feats = x_mode, y_mode, num_feats
        self.data = nn.Parameter(
            2e-4 * torch.randn((self.x_mode, self.y_mode, self.num_feats)),
            requires_grad=True,
        )
        self.renderer = G_Renderer(in_dim=self.num_feats, use_layernorm=use_layernorm)
        self.x0 = None

        self.z_mode = z_dim
        self.z_data = nn.Parameter(
            torch.randn((self.z_mode, self.num_feats)), requires_grad=True
        )
        self.z_min = z_min
        self.z_max = z_max
        self.z_dim = z_dim

    def create_coords(self, x_dim, y_dim, x_max, y_max):
        half_dx, half_dy = 0.5 / x_dim, 0.5 / y_dim
        xs = torch.linspace(half_dx, 1 - half_dx, x_dim)
        ys = torch.linspace(half_dx, 1 - half_dy, y_dim)
        xv, yv = torch.meshgrid([xs, ys], indexing="ij")
        xy = torch.stack((yv.flatten(), xv.flatten())).t()
        xs = xy * torch.tensor([x_max, y_max], device=xs.device).float()
        indices = xs.long()
        self.x_dim, self.y_dim = x_dim, y_dim
        self.xy_coords = nn.Parameter(
            xy[None],
            requires_grad=False,
        )

        if self.x0 is not None:
            device = self.x0.device
            self.x0.data = (indices[:, 0].clamp(min=0, max=x_max - 1)).to(device)
            self.y0.data = indices[:, 1].clamp(min=0, max=y_max - 1).to(device)
            self.x1.data = (self.x0 + 1).clamp(max=x_max - 1).to(device)
            self.y1.data = (self.y0 + 1).clamp(max=y_max - 1).to(device)
            self.lerp_weights.data = (xs - indices.float()).to(device)
        else:
            self.x0 = nn.Parameter(
                indices[:, 0].clamp(min=0, max=x_max - 1),
                requires_grad=False,
            )
            self.y0 = nn.Parameter(
                indices[:, 1].clamp(min=0, max=y_max - 1),
                requires_grad=False,
            )
            self.x1 = nn.Parameter(
                (self.x0 + 1).clamp(max=x_max - 1), requires_grad=False
            )
            self.y1 = nn.Parameter(
                (self.y0 + 1).clamp(max=y_max - 1), requires_grad=False
            )
            self.lerp_weights = nn.Parameter(xs - indices.float(), requires_grad=False)

    def normalize_z(self, z):
        return (self.z_dim - 1) * (z - self.z_min) / (self.z_max - self.z_min)

    def sample(self, z):
        z = self.normalize_z(z)
        z0 = z.long().clamp(min=0, max=self.z_dim - 1)
        z1 = (z0 + 1).clamp(max=self.z_dim - 1)
        zlerp_weights = (z - z.long().float())[:, None]

        xy_feat = (
            self.data[self.y0, self.x0]
            * (1.0 - self.lerp_weights[:, 0:1])
            * (1.0 - self.lerp_weights[:, 1:2])
            + self.data[self.y0, self.x1]
            * self.lerp_weights[:, 0:1]
            * (1.0 - self.lerp_weights[:, 1:2])
            + self.data[self.y1, self.x0]
            * (1.0 - self.lerp_weights[:, 0:1])
            * self.lerp_weights[:, 1:2]
            + self.data[self.y1, self.x1]
            * self.lerp_weights[:, 0:1]
            * self.lerp_weights[:, 1:2]
        )
        z_feat = (
            self.z_data[z0] * (1.0 - zlerp_weights) + self.z_data[z1] * zlerp_weights
        )
        z_feat = z_feat[:, None].repeat(1, xy_feat.shape[0], 1)

        feat = xy_feat[None].repeat(z.shape[0], 1, 1) * z_feat
        # feat = torch.cat([feat, self.xy_coords], -1)

        return feat

    def forward(self, z):
        feat = self.sample(z)

        out = self.renderer(feat)
        b = z.shape[0]
        w, h = self.x_dim, self.y_dim
        out = out.view(b, 1, w, h)

        return out


class FullModel(nn.Module):
    def __init__(
        self, w, h, num_feats, x_mode, y_mode, z_min, z_max, ds_factor, use_layernorm
    ):
        super().__init__()
        self.img_real = G_Tensor3D(
            x_mode=x_mode,
            y_mode=y_mode,
            z_dim=5,
            z_min=z_min,
            z_max=z_max,
            num_feats=num_feats,
            use_layernorm=use_layernorm,
        )
        self.img_imag = G_Tensor3D(
            x_mode=x_mode,
            y_mode=y_mode,
            z_dim=5,
            z_min=z_min,
            z_max=z_max,
            num_feats=num_feats,
            use_layernorm=use_layernorm,
        )
        self.w, self.h = w, h
        self.init_scale_grids(ds_factor=ds_factor)

        # In case we want to learn the pupil function
        # self.Pupil0 = nn.Parameter(Pupil0, requires_grad=True)

        # In case we want to learn the shift of each LED
        # xF, yF = torch.meshgrid(torch.arange(-w // 2, w // 2), torch.arange(-h // 2, h // 2), indexing="ij")
        # self.xF = nn.Parameter(xF, requires_grad=False)
        # self.yF = nn.Parameter(yF, requires_grad=False)
        # self.shift = nn.Parameter(
        #    torch.zeros(n_views, 2), requires_grad=True
        # )

    def init_scale_grids(self, ds_factor):
        self.img_real.create_coords(
            x_dim=self.w // ds_factor,
            y_dim=self.h // ds_factor,
            x_max=self.img_real.x_mode,
            y_max=self.img_real.y_mode,
        )
        self.img_imag.create_coords(
            x_dim=self.w // ds_factor,
            y_dim=self.h // ds_factor,
            x_max=self.img_imag.x_mode,
            y_max=self.img_imag.y_mode,
        )
        self.ds_factor = ds_factor
        self.us_module = nn.Upsample(scale_factor=ds_factor, mode="bilinear")

    def get_shift_grid(self, led_num):
        w = self.w
        shift = self.shift[led_num].unsqueeze(1).unsqueeze(1)
        grid = torch.exp(
            -1j
            * 2
            * torch.pi
            * (self.xF[None] * shift[..., 0] + self.yF[None] * shift[..., 1])
            / w
        )
        return grid

    def forward(self, dz):
        img_real = self.img_real(dz)
        img_imag = self.img_imag(dz)
        img_real = self.us_module(img_real).squeeze(1)
        img_imag = self.us_module(img_imag).squeeze(1)

        return img_real, img_imag
