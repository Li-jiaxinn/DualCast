# ============================================================================
# 工具函数
# ============================================================================

def LayerNorm(x):
    mean = mean(x, axis=-1, keepdims=True)
    std = std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + eps)

def Dropout(x, rate):
    mask = random_mask(x.shape, p=rate)
    return x * mask

def SiLU(x):
    return x * sigmoid(x)


# ============================================================================
# 核心运算: Selective Scan (SSM)
# ============================================================================

def selective_scan(x, dt, dA, dB, C, D, z):
    """
    状态空间模型选择性扫描
    """
    h = zeros(dA.shape[0], dA.shape[1])
    outputs = []

    for t in range(x.shape[-1]):
        h = dA * h + dB[:,:,t] * x[:,:,t]
        y = einsum('bn,bn->b', h, C[:,:,t]) + D * x[:,:,t]
        outputs.append(y * sigmoid(z[:,:,t]))

    return stack(outputs, dim=-1)


def causal_conv(x, conv):
    """因果卷积"""
    return conv(x)[:,:,:x.shape[-1]]


# ============================================================================
# 4方向扫描Mamba (单输入, 2D空间网格)
# ============================================================================

class BiMamba4D:
    """
    4个方向:
        方向1: 左→右  (逐行扫描W个像素)
        方向2: 右→左  (逐行反向扫描)
        方向3: 上→下  (逐列扫描H个像素)
        方向4: 下→上  (逐列反向扫描)
    """
    def __init__(self, d_model, H, W, d_state=16, d_conv=4, expand=2):
        self.H = H
        self.W = W
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.dt_rank = ceil(d_model / 16)
        self.d_state = d_state

        # 4套独立参数 (每方向一组)
        for direction in ['LR', 'RL', 'TB', 'BT']:
            prefix = f'{direction}'
            setattr(self, f'{prefix}_in_proj', Linear(d_model, self.d_inner * 2))
            setattr(self, f'{prefix}_conv1d', Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner))
            setattr(self, f'{prefix}_x_proj', Linear(self.d_inner, self.dt_rank + d_state * 2))
            setattr(self, f'{prefix}_dt_proj', Linear(self.dt_rank, self.d_inner))
            setattr(self, f'{prefix}_out_proj', Linear(self.d_inner, d_model))
            setattr(self, f'{prefix}_A', Parameter(log(arange(1, d_state+1)).repeat(self.d_inner, 1)))
            setattr(self, f'{prefix}_D', Parameter(ones(self.d_inner)))

    def _scan_1d(self, x, fwd_dir, bwd_dir):
        """
        对1D序列做双向扫描
        """
        # ============ 前向 (fwd_dir权重) ============
        xz_fwd = getattr(self, f'{fwd_dir}_in_proj')(x)
        x_fwd, z_fwd = split(xz_fwd, 2, dim=1)
        x_conv = causal_conv(x_fwd, getattr(self, f'{fwd_dir}_conv1d'))
        dt, B_f, C_f = split(getattr(self, f'{fwd_dir}_x_proj')(x_conv),
                             [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = softplus(getattr(self, f'{fwd_dir}_dt_proj')(dt))
        A = exp(-exp(getattr(self, f'{fwd_dir}_A')))
        y_fwd = selective_scan(x_conv, dt, exp(dt*A), dt*B_f, C_f,
                               getattr(self, f'{fwd_dir}_D'), z_fwd)

        # ============ 后向 (bwd_dir权重, flip后扫描再flip回来) ============
        x_b = flip(x, dim=-1)
        xz_bwd = getattr(self, f'{bwd_dir}_in_proj')(x_b)
        x_b_1d, z_bwd = split(xz_bwd, 2, dim=1)
        x_b_conv = causal_conv(x_b_1d, getattr(self, f'{bwd_dir}_conv1d'))
        dt_b, B_b, C_b = split(getattr(self, f'{bwd_dir}_x_proj')(x_b_conv),
                               [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_b = softplus(getattr(self, f'{bwd_dir}_dt_proj')(dt_b))
        A_b = exp(-exp(getattr(self, f'{bwd_dir}_A')))
        y_bwd = flip(selective_scan(x_b_conv, dt_b, exp(dt_b*A_b), dt_b*B_b, C_b,
                                    getattr(self, f'{bwd_dir}_D'), z_bwd), dim=-1)

        y = (y_fwd + y_bwd) / 2
        return getattr(self, f'{fwd_dir}_out_proj')(y)

    def forward(self, x_2d):
        """
        Args:
            x_2d: [B, H, W, C] - 2D空间网格
        Returns:
            out: [B, H, W, C]
        """
        B, H, W, C = x_2d.shape

        # ============ 方向1: 左→右 (前向LR, 后向RL) ============
        out_LR = zeros(B, H, W, self.d_model)
        for h in range(H):
            out_LR[:, h, :, :] = self._scan_1d(x_2d[:, h, :, :], 'LR', 'RL')

        # ============ 方向2: 右→左 (前向RL, 后向LR) ============
        out_RL = zeros(B, H, W, self.d_model)
        for h in range(H):
            out_RL[:, h, :, :] = self._scan_1d(x_2d[:, h, :, :], 'RL', 'LR')
        out_RL = flip(out_RL, dim=2)

        # ============ 方向3: 上→下 (前向TB, 后向BT) ============
        out_TB = zeros(B, H, W, self.d_model)
        for w in range(W):
            out_TB[:, :, w, :] = self._scan_1d(x_2d[:, :, w, :], 'TB', 'BT')

        # ============ 方向4: 下→上 (前向BT, 后向TB) ============
        out_BT = zeros(B, H, W, self.d_model)
        for w in range(W):
            out_BT[:, :, w, :] = self._scan_1d(x_2d[:, :, w, :], 'BT', 'TB')
        out_BT = flip(out_BT, dim=1)

        # ============ 4方向平均 ============
        out = (out_LR + out_RL + out_TB + out_BT) / 4
        return out


# ============================================================================
# 4方向扫描Mamba (双输入多模态, 2D空间网格)
# ============================================================================

class MMBiMamba4D:
    """
    4方向双向多模态Mamba
    """
    def __init__(self, d_model, H, W, d_state=16, d_conv=4, expand=2):
        self.H = H
        self.W = W
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.dt_rank = ceil(d_model / 16)
        self.d_state = d_state

        # 4方向 × 2模态 (雷达用'a'前缀, 卫星用'v'前缀)
        for direction in ['LR', 'RL', 'TB', 'BT']:
            for mod in ['a', 'v']:
                p = f'{direction}_{mod}'
                setattr(self, f'{p}_in_proj', Linear(d_model, self.d_inner * 2))
                setattr(self, f'{p}_conv1d', Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner))
                setattr(self, f'{p}_x_proj', Linear(self.d_inner, self.dt_rank + d_state * 2))
                setattr(self, f'{p}_dt_proj', Linear(self.dt_rank, self.d_inner))
                setattr(self, f'{p}_out_proj', Linear(self.d_inner, d_model))
                setattr(self, f'{p}_A', Parameter(log(arange(1, d_state+1)).repeat(self.d_inner, 1)))
                setattr(self, f'{p}_D', Parameter(ones(self.d_inner)))

    def _scan_1d_pair(self, r_x, s_x, fwd_dir, bwd_dir):
        """
        对雷达和卫星分别做1D双向扫描
        """
        # ============ 雷达 - 前向 (fwd_dir权重) ============
        r_xz_fwd = getattr(self, f'{fwd_dir}_a_in_proj')(r_x)
        r_x_fwd, r_z_fwd = split(r_xz_fwd, 2, dim=1)
        r_x_conv = causal_conv(r_x_fwd, getattr(self, f'{fwd_dir}_a_conv1d'))
        r_dt, r_B_f, r_C_f = split(getattr(self, f'{fwd_dir}_a_x_proj')(r_x_conv),
                                   [self.dt_rank, self.d_state, self.d_state], dim=-1)
        r_dt = softplus(getattr(self, f'{fwd_dir}_a_dt_proj')(r_dt))
        r_A = exp(-exp(getattr(self, f'{fwd_dir}_a_A')))
        r_y_fwd = selective_scan(r_x_conv, r_dt, exp(r_dt*r_A), r_dt*r_B_f, r_C_f,
                                 getattr(self, f'{fwd_dir}_a_D'), r_z_fwd)

        # ============ 雷达 - 后向 (bwd_dir权重) ============
        r_x_b = flip(r_x, dim=-1)
        r_xz_bwd = getattr(self, f'{bwd_dir}_a_in_proj')(r_x_b)
        r_x_b_1d, r_z_bwd = split(r_xz_bwd, 2, dim=1)
        r_x_b_conv = causal_conv(r_x_b_1d, getattr(self, f'{bwd_dir}_a_conv1d'))
        r_dt_b, r_B_b, r_C_b = split(getattr(self, f'{bwd_dir}_a_x_proj')(r_x_b_conv),
                                     [self.dt_rank, self.d_state, self.d_state], dim=-1)
        r_dt_b = softplus(getattr(self, f'{bwd_dir}_a_dt_proj')(r_dt_b))
        r_A_b = exp(-exp(getattr(self, f'{bwd_dir}_a_A')))
        r_y_bwd = flip(selective_scan(r_x_b_conv, r_dt_b, exp(r_dt_b*r_A_b), r_dt_b*r_B_b, r_C_b,
                                     getattr(self, f'{bwd_dir}_a_D'), r_z_bwd), dim=-1)
        out_r = getattr(self, f'{fwd_dir}_a_out_proj')((r_y_fwd + r_y_bwd) / 2)

        # ============ 卫星 - 前向 (fwd_dir权重) ============
        s_xz_fwd = getattr(self, f'{fwd_dir}_v_in_proj')(s_x)
        s_x_fwd, s_z_fwd = split(s_xz_fwd, 2, dim=1)
        s_x_conv = causal_conv(s_x_fwd, getattr(self, f'{fwd_dir}_v_conv1d'))
        s_dt, s_B_f, s_C_f = split(getattr(self, f'{fwd_dir}_v_x_proj')(s_x_conv),
                                   [self.dt_rank, self.d_state, self.d_state], dim=-1)
        s_dt = softplus(getattr(self, f'{fwd_dir}_v_dt_proj')(s_dt))
        s_A = exp(-exp(getattr(self, f'{fwd_dir}_v_A')))
        s_y_fwd = selective_scan(s_x_conv, s_dt, exp(s_dt*s_A), s_dt*s_B_f, s_C_f,
                                 getattr(self, f'{fwd_dir}_v_D'), s_z_fwd)

        # ============ 卫星 - 后向 (bwd_dir权重) ============
        s_x_b = flip(s_x, dim=-1)
        s_xz_bwd = getattr(self, f'{bwd_dir}_v_in_proj')(s_x_b)
        s_x_b_1d, s_z_bwd = split(s_xz_bwd, 2, dim=1)
        s_x_b_conv = causal_conv(s_x_b_1d, getattr(self, f'{bwd_dir}_v_conv1d'))
        s_dt_b, s_B_b, s_C_b = split(getattr(self, f'{bwd_dir}_v_x_proj')(s_x_b_conv),
                                     [self.dt_rank, self.d_state, self.d_state], dim=-1)
        s_dt_b = softplus(getattr(self, f'{bwd_dir}_v_dt_proj')(s_dt_b))
        s_A_b = exp(-exp(getattr(self, f'{bwd_dir}_v_A')))
        s_y_bwd = flip(selective_scan(s_x_b_conv, s_dt_b, exp(s_dt_b*s_A_b), s_dt_b*s_B_b, s_C_b,
                                     getattr(self, f'{bwd_dir}_v_D'), s_z_bwd), dim=-1)
        out_s = getattr(self, f'{fwd_dir}_v_out_proj')((s_y_fwd + s_y_bwd) / 2)

        return out_r, out_s

    def forward(self, r_x_2d, s_x_2d):
        B, H, W, C = r_x_2d.shape

        # 方向1: 左→右
        r_LR, s_LR = zeros(B,H,W,self.d_model), zeros(B,H,W,self.d_model)
        for h in range(H):
            r_out, s_out = self._scan_1d_pair(r_x_2d[:,h,:,:], s_x_2d[:,h,:,:], 'LR', 'RL')
            r_LR[:,h,:,:], s_LR[:,h,:,:] = r_out, s_out

        # 方向2: 右→左
        r_RL, s_RL = zeros(B,H,W,self.d_model), zeros(B,H,W,self.d_model)
        for h in range(H):
            r_out, s_out = self._scan_1d_pair(r_x_2d[:,h,:,:], s_x_2d[:,h,:,:], 'RL', 'LR')
            r_RL[:,h,:,:], s_RL[:,h,:,:] = flip(r_out,1), flip(s_out,1)

        # 方向3: 上→下
        r_TB, s_TB = zeros(B,H,W,self.d_model), zeros(B,H,W,self.d_model)
        for w in range(W):
            r_out, s_out = self._scan_1d_pair(r_x_2d[:,:,w,:], s_x_2d[:,:,w,:], 'TB', 'BT')
            r_TB[:,:,w,:], s_TB[:,:,w,:] = r_out, s_out

        # 方向4: 下→上
        r_BT, s_BT = zeros(B,H,W,self.d_model), zeros(B,H,W,self.d_model)
        for w in range(W):
            r_out, s_out = self._scan_1d_pair(r_x_2d[:,:,w,:], s_x_2d[:,:,w,:], 'BT', 'TB')
            r_BT[:,:,w,:], s_BT[:,:,w,:] = flip(r_out,1), flip(s_out,1)

        r_out = (r_LR + r_RL + r_TB + r_BT) / 4
        s_out = (s_LR + s_RL + s_TB + s_BT) / 4
        return r_out, s_out


# ============================================================================
# 1D Mamba (时间维度单向因果扫描)
# ============================================================================

class BiMamba1D:
    """单向1D Mamba (用于时间维度, 因果)"""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        self.d_inner = expand * d_model
        self.dt_rank = ceil(d_model / 16)
        self.d_state = d_state

        self.in_proj = Linear(d_model, self.d_inner * 2)
        self.conv1d = Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner)
        self.x_proj = Linear(self.d_inner, self.dt_rank + d_state * 2)
        self.dt_proj = Linear(self.dt_rank, self.d_inner)
        self.out_proj = Linear(self.d_inner, d_model)
        self.A_log = Parameter(log(arange(1, d_state+1)).repeat(self.d_inner, 1))
        self.D = Parameter(ones(self.d_inner))

    def forward(self, x):
        """
        Args:
            x: [B, L, C] - 时间序列 (L=时间步)
        Returns:
            out: [B, L, C]
        """
        xz = self.in_proj(x)
        x_1d, z = split(xz, 2, dim=1)
        x_conv = causal_conv(x_1d, self.conv1d)
        dt, B_param, C_param = split(self.x_proj(x_conv),
                                     [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = softplus(self.dt_proj(dt))
        A = exp(-exp(self.A_log))
        y = selective_scan(x_conv, dt, exp(dt*A), dt*B_param, C_param, self.D, z)
        return self.out_proj(y)


class UniMamba:
    """单向Mamba (时间维度, 因果)"""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        self.inner = BiMamba1D(d_model, d_state, d_conv, expand)
    def forward(self, x):
        return self.inner.forward(x)


class MMUniMamba:
    """双输入单向Mamba (时间维度多模态, 因果)"""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        self.inner_r = BiMamba1D(d_model, d_state, d_conv, expand)
        self.inner_s = BiMamba1D(d_model, d_state, d_conv, expand)
    def forward(self, r_x, s_x):
        return self.inner_r.forward(r_x), self.inner_s.forward(s_x)


# ============================================================================
# 单模态空间-时间分离Mamba层
# ============================================================================

class SpatialTemporalMambaLayer:

    def __init__(self, d_model, T, H, W):
        self.T = T
        self.H = H
        self.W = W
        self.d_model = d_model

        # 空间: 4方向双向Mamba
        self.spatial_mamba = BiMamba4D(d_model, H, W)
        self.spatial_scale = 0.5

        # 时间: 单向因果Mamba
        self.temporal_mamba = UniMamba(d_model)
        self.temporal_scale = 0.5

        self.norm_spatial = LayerNorm
        self.norm_temporal = LayerNorm

    def forward(self, x):
        """
        Args:
            x: [B, T, C, H, W]
        Returns:
            out: [B, T, C, H, W]
        """
        B, T, C, H, W = x.shape

        # ============ 1. 空间: 逐帧做4方向扫描 ============
        # x: [B, T, C, H, W]
        x_spatial = zeros(B, T, self.spatial_mamba.d_model, H, W)

        for t in range(T):
            # 每帧独立做空间4方向扫描
            x_frame = x[:, t, :, :, :]  # [B, C, H, W]
            x_frame = permute(x_frame, [0, 2, 3, 1])  # [B, H, W, C]

            # 4方向扫描 + Pre-Norm + Residual
            x_frame_out = self.spatial_mamba(self.norm_spatial(x_frame))
            x_frame = permute(x_frame, [0, 3, 1, 2])  # 恢复 [B, C, H, W]
            x_frame_out = permute(x_frame_out, [0, 3, 1, 2])  # [B, C, H, W]

            x_spatial[:, t, :, :, :] = x_frame + self.spatial_scale * Dropout(x_frame_out)

        # ============ 2. 时间: 逐像素位置做单向扫描 ============
        # x_spatial: [B, T, C, H, W]
        # 变换: [B,H,W,C,T] → [B*H*W, C, T]
        x_temporal = permute(x_spatial, [0, 3, 4, 2, 1])  # [B, H, W, C, T]
        x_temporal = reshape(x_temporal, B*self.H*self.W, self.spatial_mamba.d_model, self.T)
        # 沿T维度扫描: 输入 [B*H*W, C, T] → 输出 [B*H*W, C, T]
        x_temporal = x_temporal + self.temporal_scale * Dropout(
            self.temporal_mamba(self.norm_temporal(x_temporal))
        )
        # 恢复: [B*H*W, C, T] → [B, H, W, C, T] → [B, T, C, H, W]
        x_temporal = reshape(x_temporal, B, self.H, self.W, self.spatial_mamba.d_model, self.T)
        x_temporal = permute(x_temporal, [0, 4, 3, 1, 2])
        out = x_temporal

        return out


# ============================================================================
# 多模态空间-时间分离Mamba层
# ============================================================================

class MMSpatialTemporalMambaLayer:

    def __init__(self, d_model, T, H, W):
        self.T = T
        self.H = H
        self.W = W
        self.d_model = d_model

        # 空间: 4方向双向多模态Mamba
        self.spatial_mamba = MMBiMamba4D(d_model, H, W)
        self.spatial_scale = 0.5

        # 时间: 单向多模态Mamba
        self.temporal_mamba = MMUniMamba(d_model)
        self.temporal_scale = 0.5

        self.norm_a_spatial = LayerNorm
        self.norm_v_spatial = LayerNorm
        self.norm_a_temporal = LayerNorm
        self.norm_v_temporal = LayerNorm

    def forward(self, r_x, s_x):
        """
        Args:
            r_x: [B, T, C, H, W] - 雷达特征 (1通道, VIL)
            s_x: [B, T, C, H, W] - 卫星特征 (3通道, ir069/ir107/lght)
        Returns:
            r_out, s_out: [B, T, C, H, W]
        """
        B, T, C, H, W = r_x.shape

        # ============ 1. 空间: 逐帧做4方向扫描 (双模态联合) ============
        r_spatial = zeros(B, T, self.spatial_mamba.d_model, H, W)
        s_spatial = zeros(B, T, self.spatial_mamba.d_model, H, W)

        for t in range(T):
            r_frame = r_x[:, t, :, :, :]  # [B, C, H, W]
            s_frame = s_x[:, t, :, :, :]

            r_frame = permute(r_frame, [0, 2, 3, 1])  # [B, H, W, C]
            s_frame = permute(s_frame, [0, 2, 3, 1])

            # 4方向双模态联合扫描
            r_out_frame, s_out_frame = self.spatial_mamba(
                self.norm_a_spatial(r_frame),
                self.norm_v_spatial(s_frame)
            )

            r_frame = permute(r_frame, [0, 3, 1, 2])
            s_frame = permute(s_frame, [0, 3, 1, 2])
            r_out_frame = permute(r_out_frame, [0, 3, 1, 2])
            s_out_frame = permute(s_out_frame, [0, 3, 1, 2])

            r_spatial[:, t, :, :, :] = r_frame + self.spatial_scale * Dropout(r_out_frame)
            s_spatial[:, t, :, :, :] = s_frame + self.spatial_scale * Dropout(s_out_frame)

        # ============ 2. 时间: 逐像素位置做单向扫描 (双模态独立) ============
        r_temporal = permute(r_spatial, [0, 3, 4, 2, 1])  # [B,H,W,C,T]
        s_temporal = permute(s_spatial, [0, 3, 4, 2, 1])

        r_temporal = reshape(r_temporal, B*self.H*self.W, self.spatial_mamba.d_model, self.T)
        s_temporal = reshape(s_temporal, B*self.H*self.W, self.spatial_mamba.d_model, self.T)

        r_temporal_out, s_temporal_out = self.temporal_mamba(
            self.norm_a_temporal(r_temporal),
            self.norm_v_temporal(s_temporal)
        )

        r_temporal = r_temporal + self.temporal_scale * Dropout(r_temporal_out)
        s_temporal = s_temporal + self.temporal_scale * Dropout(s_temporal_out)

        # 恢复形状: [B*H*W, C, T] → [B, H, W, C, T] → [B, T, C, H, W]
        r_out = permute(reshape(r_temporal, B, self.H, self.W, self.spatial_mamba.d_model, self.T), [0, 4, 3, 1, 2])
        s_out = permute(reshape(s_temporal, B, self.H, self.W, self.spatial_mamba.d_model, self.T), [0, 4, 3, 1, 2])

        return r_out, s_out


# ============================================================================
# HAMamba (多模态时空编码器)
# ============================================================================

class HAMamba:
    """
    HAMamba (多模态时空编码器)
    输入: 雷达特征 + 卫星特征 [B, T, C, H, W]
    输出: 各自特征 [B, T, C, H, W]
    """
    def __init__(self, input_size, output_sizes, num_layers, T, H, W):
        self.T = T
        self.H = H
        self.W = W

        self.cnn_layers = []
        self.mamba_layers = []

        for i in range(len(output_sizes)):
            # CNN: 通道变换 (仅C维度)
            self.cnn_layers.append(CNNLayer(
                in_ch = input_size if i < 1 else output_sizes[i-1],
                out_ch = output_sizes[i]
            ))

            # MMSpatialTemporalMamba层
            self.mamba_layers.append(MMSpatialTemporalMambaLayer(
                d_model = output_sizes[i],
                T = T, H = H, W = W
            ))

    def forward(self, r_x, s_x):
        """
        Args:
            r_x: [B, T, C_in, H, W] - 雷达特征
            s_x: [B, T, C_in, H, W] - 卫星特征
        Returns:
            r_out, s_out: [B, T, C_last, H, W]
        """
        r_out, s_out = r_x, s_x

        for cnn_layer, mamba_layer in zip(self.cnn_layers, self.mamba_layers):
            # CNN: 通道变换 (不改变T/H/W维度)
            # [B,T,C,H,W] → permute → [B,C,T,H,W] → CNN → [B,C,T,H,W] → permute → [B,T,C_out,H,W]
            r_out = permute(r_out, [0, 1, 3, 4, 2])  # [B,T,H,W,C]
            s_out = permute(s_out, [0, 1, 3, 4, 2])

            # CNN 2D作用于H×W空间: [B,T,C,H,W] → [B,T,C,H,W]
            r_out, s_out = cnn_layer(r_out, s_out)

            r_out = permute(r_out, [0, 1, 3, 4, 2])  # [B,T,H,W,C] → [B,T,C,H,W]
            s_out = permute(s_out, [0, 1, 3, 4, 2])

            # 时空Mamba (空间4方向 + 时间单向)
            r_out, s_out = mamba_layer(r_out, s_out)

        return r_out, s_out


# ============================================================================
# DCMamba (单模态时空编码器)
# ============================================================================

class DCMamba:
    """
    DCMamba (单模态时空编码器)
    """
    def __init__(self, input_size, output_sizes, num_layers, T, H, W):
        self.T = T
        self.H = H
        self.W = W

        self.cnn_layers = []
        self.mamba_layers = []

        for i in range(len(output_sizes)):
            self.cnn_layers.append(CNNLayer(
                in_ch = input_size if i < 1 else output_sizes[i-1],
                out_ch = output_sizes[i]
            ))

            self.mamba_layers.append(SpatialTemporalMambaLayer(
                d_model = output_sizes[i],
                T = T, H = H, W = W
            ))

    def forward(self, x):
        """
        Args:
            x: [B, T, C_in, H, W]
        Returns:
            out: [B, T, C_last, H, W]
        """
        out = x

        for cnn_layer, mamba_layer in zip(self.cnn_layers, self.mamba_layers):
            # CNN: 通道变换
            out = permute(out, [0, 1, 3, 4, 2])
            out = cnn_layer(out)
            out = permute(out, [0, 1, 3, 4, 2])

            # 时空Mamba (空间4方向 + 时间单向)
            out = mamba_layer(out)

        return out


# ============================================================================
# CNN层 (2D卷积, 仅改变C维度, T/H/W不变)
# ============================================================================

class CNNLayer:
    """2D卷积 + BN + SiLU + Dropout + 跳跃连接 (不改变H/W维度)"""
    def forward(self, x):
        out = SiLU(BatchNorm(Conv2d(x, out_ch, 3, padding=1)))
        out = Dropout(out, p=0.1)
        if self.has_skip:
            x = Conv2d(x, out_ch, 1)
        out = out + x
        return out


# ============================================================================
# 完整模型 (PFM)
# ============================================================================

class PFM:
    """
    完整模型架构:
    1. HAMamba: 雷达+卫星双模态时空编码 (空间4方向 + 时间单向)
    2. 拼接: [r_out, s_out] 沿C维度
    3. DCMamba: 融合特征时空编码 (空间4方向 + 时间单向)
    """
    def __init__(self, T=13, H=12, W=12):
        self.T = T
        self.H = H
        self.W = W

        # 多模态编码 (通道: 雷达1→128, 卫星3→128)
        self.ha_mamba = HAMamba(128, [256, 64], num_layers=8, T=T, H=H, W=W)

        # 单模态编码
        self.dc_mamba = DCMamba(128, [128], num_layers=8, T=T, H=H, W=W)

    def forward(self, r_x, s_x):
        """
        Args:
            r_x: [B, T, C_r, H, W]  雷达特征 (1通道, VIL)
            s_x: [B, T, C_s, H, W]  卫星特征 (3通道, ir069/ir107/lght)
        Returns:
            x: [B, T, C, H, W]  融合特征, 后续处理由其他模块负责
        """
        r_out, s_out = self.ha_mamba(r_x, s_x)
        x = concat([r_out, s_out], dim=2)
        x = self.dc_mamba(x)
        return x
