import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
import math


class FeatureProcessingBlock(nn.Module):
    """Simple residual feature processing: Conv -> GroupNorm -> SiLU, with skip connection."""
    def __init__(self, dim):
        super().__init__()
        groups = 8 if dim >= 8 else 1
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(groups, dim),
            nn.SiLU()
        )

    def forward(self, x):
        return self.block(x) + x


class LocalRegionalFusionBlock(nn.Module):
    """
    Local + regional fusion block.
    - Local branch: two standard conv layers with normalization and nonlinearity.
    - Regional branch: multiple dilated depthwise convolutions plus a pointwise projection.
    - Fuse local and regional features with a 1x1 conv, apply residual connection.
    """
    def __init__(self, in_dim, out_dim, reduction=4, dilations=(1, 2, 4)):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.reduction = reduction
        self.dilations = dilations

        # optional 1x1 projection to match channels for residual
        if in_dim != out_dim:
            self.proj = nn.Conv2d(in_dim, out_dim, 1)
        else:
            self.proj = None

        # choose a GroupNorm group count that divides channels or fallback to 1
        def _valid_groups(ch):
            g = 8
            while g > 1 and ch % g != 0:
                g //= 2
            if ch % g != 0:
                g = 1
            return g

        gn_groups = _valid_groups(out_dim)

        # local branch: two conv layers with GroupNorm and SiLU
        self.local = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
            nn.GroupNorm(gn_groups, out_dim),
            nn.SiLU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1),
            nn.GroupNorm(gn_groups, out_dim),
            nn.SiLU()
        )

        # regional branch: several dilated depthwise convs (one per dilation)
        self.reg_depthwise = nn.ModuleList()
        for d in dilations:
            pad = d
            # depthwise convolution with dilation
            self.reg_depthwise.append(
                nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=pad, dilation=d, groups=out_dim, bias=False)
            )

        # after concatenating original + dilated outputs, reduce channels via 1x1 conv
        self.reg_pw = nn.Sequential(
            nn.Conv2d(out_dim * (len(self.reg_depthwise) + 1), out_dim, 1, bias=False),
            nn.GroupNorm(gn_groups, out_dim),
            nn.SiLU()
        )

        # fuse local and regional features then normalize and activate
        self.fuse = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim, 1, bias=False),
            nn.GroupNorm(gn_groups, out_dim),
            nn.SiLU()
        )

    def forward(self, x):
        # project input channels if needed
        if self.proj is not None:
            x0 = self.proj(x)
        else:
            x0 = x

        # local features
        local_feat = self.local(x0)

        # regional features: include original and multiple dilated depthwise conv outputs
        regs = [x0]
        for dw in self.reg_depthwise:
            r = dw(x0)
            r = F.silu(r)
            regs.append(r)
        reg_cat = torch.cat(regs, dim=1)
        reg_feat = self.reg_pw(reg_cat)

        # fuse local and regional branches and add residual
        fused = torch.cat([local_feat, reg_feat], dim=1)
        out = self.fuse(fused)
        out = out + x0
        return out


class Encoder(nn.Module):
    """Encoder stack: apply a sequence of LocalRegionalFusionBlock layers and downsample between stages."""
    def __init__(self, dim, dim_mults, num_blocks=1):
        super().__init__()
        self.dims = [dim * m for m in dim_mults]
        in_out = list(zip([dim] + self.dims[:-1], self.dims))
        if isinstance(num_blocks, int):
            num_blocks = [num_blocks] * len(in_out)

        self.blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for (in_dim, out_dim), n in zip(in_out, num_blocks):
            # create n sequential fusion blocks for this stage
            layers = [LocalRegionalFusionBlock(in_dim if i == 0 else out_dim, out_dim) for i in range(n)]
            self.blocks.append(nn.Sequential(*layers))
            # downsample by a stride-2 conv
            self.downsamples.append(nn.Conv2d(out_dim, out_dim, 3, stride=2, padding=1))

    def forward(self, x):
        skips = []
        for block, downsample in zip(self.blocks, self.downsamples):
            x = block(x)
            skips.append(x)
            x = downsample(x)
        return skips, x


class MultiSkipDecoder(nn.Module):
    """
    Decoder that aggregates multiple encoder skips for each stage.
    - Projects all encoder skips to the current stage channels.
    - Computes per-skip scores via a small network and a structural variance term.
    - Softmax-normalizes scores to produce weights, aggregates skips, applies a spatial gate,
      concatenates aggregated skip with upsampled feature and processes via LocalRegionalFusionBlock.
    """
    def __init__(self, dim, dim_mults, num_blocks=1):
        super().__init__()
        encoder_dims = [dim * m for m in dim_mults]
        reversed_dims = list(reversed(encoder_dims))
        reversed_dims.append(dim)
        in_out = list(zip(reversed_dims, reversed_dims[1:]))
        if isinstance(num_blocks, int):
            num_blocks = [num_blocks] * len(in_out)

        self.upsamples = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.feature_processing_blocks = nn.ModuleList()

        # per-stage score nets, gate nets and projectors for all encoder skips
        self.score_nets = nn.ModuleList()
        self.gate_nets = nn.ModuleList()
        self.skip_projectors = nn.ModuleList()

        for idx, ((in_dim, out_dim), n) in enumerate(zip(in_out, num_blocks)):
            # upsample module: nearest upsample + conv
            self.upsamples.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_dim, in_dim, 3, padding=1)
            ))

            # build stage blocks; first block expects concat(x, skip) channels
            layers = [LocalRegionalFusionBlock(in_dim + in_dim if i == 0 else out_dim, out_dim) for i in range(n)]
            self.blocks.append(nn.Sequential(*layers))
            self.feature_processing_blocks.append(FeatureProcessingBlock(out_dim))

            # lightweight scoring network: outputs a single scalar (after avg pool) per sample
            gn_groups = 8 if in_dim >= 8 and in_dim % 8 == 0 else 1
            score_net = nn.Sequential(
                nn.Conv2d(in_dim * 2, in_dim, kernel_size=3, padding=1),
                nn.GroupNorm(gn_groups, in_dim),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_dim, 1, kernel_size=1)
            )
            self.score_nets.append(score_net)

            # spatial gating network: produces a per-pixel [0,1] mask for aggregated skip
            gate_net = nn.Sequential(
                nn.Conv2d(in_dim * 2, in_dim, kernel_size=1, bias=False),
                nn.GroupNorm(gn_groups, in_dim),
                nn.Sigmoid()
            )
            self.gate_nets.append(gate_net)

            # project each encoder skip to this stage's in_dim using 1x1 convs
            proj_list = nn.ModuleList()
            for skip_ch in encoder_dims:
                proj_list.append(nn.Conv2d(skip_ch, in_dim, kernel_size=1, bias=False))
            self.skip_projectors.append(proj_list)

        # learnable weight to balance score_net output and variance term
        self.beta = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.eps = 1e-6

    def forward(self, x, fused_skips):
        """
        x: current decoder input (bottleneck or previous stage output)
        fused_skips: list of encoder skip features (one per encoder stage)
        """
        B = x.shape[0]
        num_skips = len(fused_skips)
        for idx, (upsample, block, feature_block, score_net, gate_net, proj_list) in enumerate(
                zip(self.upsamples, self.blocks, self.feature_processing_blocks,
                    self.score_nets, self.gate_nets, self.skip_projectors)):

            # upsample the decoder feature to next spatial resolution
            x = upsample(x)
            H, W = x.shape[2], x.shape[3]

            # resize and project all encoder skips to current spatial size and channel count
            projected_skips = []
            for j, s in enumerate(fused_skips):
                s_resized = F.interpolate(s, size=(H, W), mode='bilinear', align_corners=False)
                s_proj = proj_list[j](s_resized)
                projected_skips.append(s_proj)

            # compute a score for each projected skip: network output + beta * variance term
            scores = []
            for s_proj in projected_skips:
                inp = torch.cat([x, s_proj], dim=1)
                score_map = score_net(inp)
                var_s = s_proj.var(dim=[2, 3], keepdim=True).mean(dim=1, keepdim=True)
                final_score = score_map + self.beta * var_s
                scores.append(final_score)

            # normalize scores across skips with softmax to get weights per-skip
            scores_cat = torch.cat(scores, dim=1)
            scores_flat = scores_cat.view(B, num_skips)
            weights = F.softmax(scores_flat, dim=1)

            # weighted aggregation of projected skips
            aggregated = projected_skips[0].new_zeros(B, projected_skips[0].shape[1], H, W)
            for j, s_proj in enumerate(projected_skips):
                w = weights[:, j].view(B, 1, 1, 1)
                aggregated = aggregated + w * s_proj

            # spatial gate modulates aggregated skip per-pixel
            gate = gate_net(torch.cat([x, aggregated], dim=1))
            aggregated = aggregated * gate

            # concatenate aggregated skip with upsampled x and process through block
            x = torch.cat([x, aggregated], dim=1)
            x = block(x)
            x = feature_block(x)

        return x


class CrossModalFusion(nn.Module):
    """Gated fusion for two modalities using a learned attention weight and feature interaction."""
    def __init__(self, dim):
        super().__init__()
        reduced_dim = max(4, dim // 4)

        # attention network produces per-channel gating weights in [0,1]
        self.attention = nn.Sequential(
            nn.Conv2d(2 * dim, reduced_dim, kernel_size=1),
            nn.GroupNorm(2, reduced_dim),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, dim, kernel_size=1),
            nn.Sigmoid()
        )

        # interaction module to mix modalities after gating
        self.interaction = nn.Sequential(
            nn.Conv2d(2 * dim, dim, kernel_size=1),
            nn.GroupNorm(8 if dim >= 8 else 1, dim),
            nn.SiLU()
        )

        # residual 1x1 conv to guarantee matching channels
        self.residual = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, rgb_feat, nir_feat):
        combined = torch.cat([rgb_feat, nir_feat], dim=1)
        attn_weight = self.attention(combined)

        # weighted combination of modalities
        weighted_rgb = rgb_feat * attn_weight
        weighted_nir = nir_feat * (1 - attn_weight)

        interacted = self.interaction(combined)
        fused = weighted_rgb + weighted_nir + interacted + self.residual(rgb_feat)
        return fused


class StatisticalRecalibrationModule(nn.Module):
    """
    Adaptive statistical recalibration between two modalities.
    - Computes channel-wise mean and variance for both inputs.
    - Predicts scale (gamma) and shift (beta) parameters to recalibrate each modality.
    - Applies a spatial gate to apply corrections selectively and does light post-processing.
    """
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        groups = 8 if dim % 8 == 0 else 1

        calib_net_channels = dim * 4
        hidden_dim = max(dim, 16)

        # networks that predict gamma and beta for each modality
        self.calib_net_b_on_a = nn.Sequential(
            nn.Conv2d(calib_net_channels, hidden_dim, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, dim * 2, 1)
        )
        self.calib_net_a_on_b = nn.Sequential(
            nn.Conv2d(calib_net_channels, hidden_dim, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, dim * 2, 1)
        )

        # spatial gates to control where recalibration is applied
        self.spatial_gate_b_on_a = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, padding=1),
            nn.GroupNorm(groups, dim),
            nn.Sigmoid()
        )
        self.spatial_gate_a_on_b = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, padding=1),
            nn.GroupNorm(groups, dim),
            nn.Sigmoid()
        )

        # small post-processing blocks
        self.post_process_a = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.GroupNorm(groups, dim), nn.SiLU())
        self.post_process_b = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.GroupNorm(groups, dim), nn.SiLU())

    def forward(self, a, b):
        # compute channel-wise mean and std for both modalities
        N, C, H, W = a.shape
        mu_a = a.mean(dim=[2, 3], keepdim=True)
        sigma_a = a.var(dim=[2, 3], keepdim=True).sqrt() + self.eps
        mu_b = b.mean(dim=[2, 3], keepdim=True)
        sigma_b = b.var(dim=[2, 3], keepdim=True).sqrt() + self.eps

        # predict calibration params for a using statistics of both a and b
        combined_stats = torch.cat([mu_a, sigma_a, mu_b, sigma_b], dim=1)
        calib_params_a = self.calib_net_b_on_a(combined_stats)
        gamma_a, beta_a = torch.chunk(calib_params_a, 2, dim=1)

        # normalize a and apply predicted scale and shift
        a_norm = (a - mu_a) / sigma_a
        a_recalibrated = gamma_a * a_norm + beta_a

        # spatial gate to apply recalibration selectively
        combined_features = torch.cat([a, b], dim=1)
        spatial_mask_a = self.spatial_gate_b_on_a(combined_features)
        a_corrected = a + spatial_mask_a * a_recalibrated

        # symmetric operations for b using predicted params from combined stats
        calib_params_b = self.calib_net_a_on_b(combined_stats)
        gamma_b, beta_b = torch.chunk(calib_params_b, 2, dim=1)

        b_norm = (b - mu_b) / sigma_b
        b_recalibrated = gamma_b * b_norm + beta_b

        spatial_mask_b = self.spatial_gate_a_on_b(combined_features)
        b_corrected = b + spatial_mask_b * b_recalibrated

        # post-process and add residual
        a_out = self.post_process_a(a_corrected) + a_corrected
        b_out = self.post_process_b(b_corrected) + b_corrected

        return a_out, b_out


class MultiModalFusionUNet(nn.Module):
    """Multimodal UNet that fuses RGB and NIR features at multiple scales for restoration."""
    def __init__(self, dim=32, dim_mults=(1, 2, 4, 8), num_blocks_encoder=1, num_blocks_decoder=1):
        super().__init__()
        # RGB encoder branch
        self.init_conv_rgb = nn.Conv2d(3, dim, 7, padding=3)
        self.encoder_rgb = Encoder(dim, dim_mults, num_blocks=num_blocks_encoder)

        # NIR encoder branch
        self.init_conv_nir = nn.Conv2d(1, dim, 7, padding=3)
        self.encoder_nir = Encoder(dim, dim_mults, num_blocks=num_blocks_encoder)

        # fusion modules for each skip level
        encoder_dims = [dim * m for m in dim_mults]
        self.skip_fusers = nn.ModuleList([CrossModalFusion(dim) for dim in encoder_dims])

        # bottleneck alignment and fusion
        mid_dim = encoder_dims[-1]
        self.mid_alignment = StatisticalRecalibrationModule(mid_dim)
        self.mid_fuser = CrossModalFusion(mid_dim)

        # bottleneck processing
        self.mid_block = nn.Sequential(
            LocalRegionalFusionBlock(mid_dim, mid_dim),
            LocalRegionalFusionBlock(mid_dim, mid_dim)
        )

        # per-skip statistical alignment modules
        self.skip_alignment = nn.ModuleList([
            StatisticalRecalibrationModule(dim * mult) for mult in dim_mults
        ])

        # decoder that aggregates multi-scale skips
        self.decoder = MultiSkipDecoder(dim, dim_mults, num_blocks=num_blocks_decoder)

        # final conv to produce RGB output
        self.final_conv = nn.Conv2d(dim, 3, 3, padding=1)

    def forward(self, rgb, nir):
        # encode RGB
        x_rgb = self.init_conv_rgb(rgb)
        skips_rgb, x_rgb = self.encoder_rgb(x_rgb)

        # encode NIR
        x_nir = self.init_conv_nir(nir)
        skips_nir, x_nir = self.encoder_nir(x_nir)

        # align bottleneck statistics across modalities and fuse
        x_rgb_aligned_mid, x_nir_aligned_mid = self.mid_alignment(x_rgb, x_nir)
        x = self.mid_fuser(x_rgb_aligned_mid, x_nir_aligned_mid)
        x = self.mid_block(x)

        # multi-scale skip fusion: align and fuse each encoder skip pair
        fused_skips = []
        for i in range(len(skips_rgb)):
            s_rgb = skips_rgb[i]
            s_nir = skips_nir[i]
            alignment_module = self.skip_alignment[i]
            s_rgb_aligned, s_nir_aligned = alignment_module(s_rgb, s_nir)
            fused = self.skip_fusers[i](s_rgb_aligned, s_nir_aligned)
            fused_skips.append(fused)

        # decode using aggregated skips
        x = self.decoder(x, fused_skips)

        # produce final RGB output
        out = self.final_conv(x)
        return out


def calculate_flops_params():
    """Utility to run a forward pass and estimate FLOPs and parameter count using thop."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    model = MultiModalFusionUNet(dim=16, dim_mults=(1, 2, 4, 8),
                           num_blocks_encoder=[2, 2, 2, 2],
                           num_blocks_decoder=[2, 2, 2, 2]).to(device)

    rgb_input = torch.randn(1, 3, 256, 256).to(device)
    nir_input = torch.randn(1, 1, 256, 256).to(device)

    model.eval()
    with torch.no_grad():
        start = time.time()
        result = model(rgb_input, nir_input)
        end = time.time()
        print(f"Forward pass time: {(end - start) * 1000:.2f} ms")
        print("Output shape:", tuple(result.shape))

    # run thop profiling on CPU for stability
    model.to('cpu')
    rgb_input = rgb_input.to('cpu')
    nir_input = nir_input.to('cpu')

    try:
        flops, params = profile(model, inputs=(rgb_input, nir_input), verbose=False)
        gflops = flops / 1e9
        params_m = params / 1e6
        print(f"FLOPs: {gflops:.2f} G")
        print(f"Params: {params_m:.2f} M")
    except Exception as e:
        print(f"Could not calculate FLOPs and Params: {e}")
        print("Ensure thop is installed (`pip install thop`) and inputs/model are on CPU for profile.")


if __name__ == "__main__":
    calculate_flops_params()
