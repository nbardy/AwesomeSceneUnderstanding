# Deep Technical Review - Video Understanding & Self-Supervised Learning

## V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video

### Summary
V-JEPA introduces a feature prediction objective for self-supervised video pre-training that operates entirely in latent space without pixel reconstruction. The method learns representations by predicting missing spatiotemporal features from visible context, achieving 81.9% on Kinetics-400 with frozen features. The key insight is that predicting abstract features rather than pixels enables better semantic understanding while avoiding low-level detail reconstruction.

### Key Improvements
1. **Kinetics-400 Accuracy**: Baseline supervised → 81.9% (frozen backbone)
2. **Something-Something-v2**: Previous SOTA → 72.2% (temporal reasoning tasks)
3. **Training Efficiency**: No pretrained encoders required (trained from scratch)
4. **Data Requirements**: 2M videos sufficient (vs 10M+ for other methods)
5. **ImageNet Transfer**: Zero-shot → 77.9% (without image pre-training)

### How It Works

```python
def vjepa_forward(video_frames, encoder, predictor, mask_strategy):
    """
    V-JEPA core algorithm - predict masked features in latent space
    
    Mathematical formulation:
    L = ||sg(f_θ(x_masked)) - g_φ(f_θ(x_visible))||²
    where sg() is stop-gradient, f_θ is encoder, g_φ is predictor
    """
    # 1. Patchify video: [B, T, H, W, C] -> [B, N, D]
    # Patches: 2x16x16 (temporal x spatial)
    patches = patchify_video(video_frames, patch_size=(2, 16, 16))
    
    # 2. Apply masking strategy
    visible_patches, mask_indices = mask_strategy.apply(patches)
    # Typical masking: 80-90% of patches masked
    
    # 3. Encode visible patches only (efficiency)
    visible_features = encoder(visible_patches)  # [B, N_vis, D]
    
    # 4. Predict masked patch features from visible context
    predicted_features = predictor(
        visible_features, 
        mask_indices,
        positional_embeddings
    )  # [B, N_mask, D]
    
    # 5. Encode full video with stop-gradient for targets
    with torch.no_grad():
        all_features = encoder(patches)  # [B, N, D]
        target_features = all_features[mask_indices]  # [B, N_mask, D]
    
    # 6. Feature prediction loss (no pixel reconstruction)
    loss = F.smooth_l1_loss(predicted_features, target_features)
    
    return loss, predicted_features
```

**Mathematical Foundation**:
- Feature prediction objective: minimize ||f̂_masked - f_masked||²
- No reconstruction loss, contrastive loss, or adversarial loss
- Predictor capacity limited to prevent shortcut learning

**Critical Implementation Details**:
- Encoder: Standard ViT with minimal modifications
- Predictor: Lightweight transformer (6-12 layers)
- Stop-gradient on target features prevents collapse
- Smooth L1 loss more stable than L2 for features

### Algorithm Steps
1. **Video Sampling**: Sample 16 frames at 2-4 fps from 2-10 second clips
2. **Patch Embedding**: 2x16x16 patches → 1024/1280-dim embeddings
3. **Masking**: Block-wise masking with 80-90% ratio
4. **Feature Encoding**: ViT-L/16 or ViT-H/16 encoder
5. **Feature Prediction**: 6-layer transformer predictor
6. **Loss Computation**: Smooth L1 on masked features only

### Implementation Details
- Architecture: ViT-L/16 (307M) and ViT-H/16 (632M params)
- Batch Size: 2400-3072 (distributed across GPUs)
- Learning Rate: 1e-3 with cosine schedule
- Training: 90k iterations on VideoMix2M dataset
- Hardware: 64 V100 GPUs for ~1 week
- No augmentations beyond temporal sampling
- EMA of encoder for stability (τ=0.999)

### Integration Notes
```python
# Modifications to standard video transformer:
# In models/vision_transformer.py:

class VideoViT(nn.Module):
    def __init__(self):
        # Standard ViT with 3D position embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches_time * num_patches_space, embed_dim)
        )
        
    def forward(self, x, mask=None):
        # Only process visible patches during training
        if mask is not None:
            x = x[~mask]  # Skip masked patches
            pos_embed = self.pos_embed[~mask]
        
        # Standard transformer forward
        x = x + pos_embed
        x = self.transformer(x)
        return x

# In training loop:
optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.05)
scheduler = CosineAnnealingLR(optimizer, T_max=90000)

for batch in dataloader:
    # Apply spatiotemporal masking
    mask = generate_spacetime_mask(
        batch.shape, 
        mask_ratio=0.9,
        mask_type='blockwise'
    )
    
    loss = vjepa_forward(batch, encoder, predictor, mask)
    loss.backward()
```

### Speed/Memory Tradeoffs
- Training: ~1 week on 64 V100s (168 GPU-days)
- Inference: 89ms per 16-frame clip on V100 (ViT-H/16)
- Memory: 32GB GPU for batch size 48 per GPU
- Feature extraction: 5x faster than pixel reconstruction methods
- Linear probe training: 10x faster convergence than from scratch

---

## V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning

### Summary
V-JEPA 2 extends V-JEPA to 1M+ hours of video data and introduces action-conditioned variants (V-JEPA 2-AC) for robotic planning. The method achieves 77.3% on Something-Something v2 (motion understanding) and 39.7 recall@5 on Epic-Kitchens-100 (action anticipation), while enabling zero-shot robotic manipulation. The key innovation is learning world models that support both passive understanding and active planning through conditional feature prediction.

### Key Improvements
1. **Something-Something v2**: 72.2% → 77.3% (motion understanding)
2. **Epic-Kitchens-100 Anticipation**: Previous SOTA → 39.7 recall@5
3. **Scale**: 2M → 1M+ hours of video
4. **PerceptionTest VQA**: New benchmark → 84.0%
5. **Zero-shot Robot Deployment**: 0 → successful pick-and-place

### How It Works

```python
def vjepa2_forward(video, encoder, predictor, action=None):
    """
    V-JEPA 2 with optional action conditioning
    
    Key difference: Conditional feature prediction
    L = ||f(x_t+k) - g(f(x_≤t), a_t:t+k)||² if action else ||f(x_t+k) - g(f(x_≤t))||²
    """
    # 1. Enhanced temporal modeling with longer sequences
    patches = patchify_video(video, patch_size=(2, 16, 16))
    T, H, W = video.shape[1:4]
    
    # 2. Hierarchical masking strategy
    # Mask future frames more aggressively for prediction
    if self.training:
        mask = generate_predictive_mask(
            shape=patches.shape,
            past_ratio=0.3,    # Keep 70% of past
            future_ratio=0.9   # Mask 90% of future
        )
    
    # 3. Encode with temporal positional encoding
    features = encoder(patches, mask)  # [B, T, N_patches, D]
    
    # 4. Action-conditioned prediction (V-JEPA 2-AC)
    if action is not None:
        # Project actions to feature space
        action_features = self.action_encoder(action)  # [B, T_action, D_action]
        
        # Cross-attention between video and action
        predicted = predictor(
            video_features=features,
            action_features=action_features,
            predict_horizon=self.predict_horizon
        )
    else:
        # Standard future prediction
        predicted = predictor(features, predict_horizon=self.predict_horizon)
    
    # 5. Compute prediction loss at multiple horizons
    losses = []
    for k in [1, 2, 4, 8, 16]:  # Multi-scale temporal prediction
        target = encode_future_frames(video, t + k)
        loss_k = F.smooth_l1_loss(predicted[k], target.detach())
        losses.append(loss_k)
    
    return sum(losses) / len(losses)

class ActionConditionedPredictor(nn.Module):
    """Predicts future features conditioned on actions"""
    def __init__(self, d_model=1024, n_heads=16, n_layers=12):
        super().__init__()
        self.cross_attention = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads) 
            for _ in range(n_layers)
        ])
        self.temporal_model = nn.LSTM(d_model, d_model, 2)
        
    def forward(self, video_features, action_features, predict_horizon):
        # Fuse video and action information
        for cross_attn in self.cross_attention:
            video_features = cross_attn(
                query=video_features,
                key=action_features,
                value=action_features
            )[0] + video_features
        
        # Predict future features autoregressively
        predictions = {}
        hidden = None
        current = video_features[:, -1]  # Last visible frame
        
        for k in range(1, predict_horizon + 1):
            current, hidden = self.temporal_model(current, hidden)
            predictions[k] = current
            
        return predictions
```

### Algorithm Steps
1. **Data Loading**: 1M+ hours from YouTube, Ego4D, Something-Something
2. **Multi-Scale Sampling**: 1-4 fps, 32-128 frames per clip
3. **Hierarchical Masking**: Past (30% masked) vs Future (90% masked)
4. **Conditional Encoding**: Optional action conditioning for planning
5. **Multi-Horizon Prediction**: Predict 1, 2, 4, 8, 16 frames ahead
6. **Joint Training**: Video understanding + action prediction objectives

### Implementation Details
- Architecture: ViT-L/14 and ViT-H/14 at 8B parameter scale
- Training: 200k iterations on 512 A100 GPUs
- Batch Size: 8192 globally (16 per GPU)
- Learning Rate: 1.5e-4 with 10k warmup steps
- Action Encoder: 3-layer MLP for robot actions
- Prediction Horizons: Up to 5 seconds (150 frames)
- Memory: 80GB A100s required for 8B model

### Integration Notes
```python
# For robotic deployment:
# In robot_planning.py:

class VJepa2Planner:
    def __init__(self, checkpoint_path):
        self.model = load_vjepa2_ac(checkpoint_path)
        self.model.eval()
        
    def plan(self, current_obs, goal_image, max_steps=50):
        """Zero-shot planning with image goals"""
        # Encode current and goal
        current_features = self.model.encode(current_obs)
        goal_features = self.model.encode(goal_image)
        
        # Search over action sequences
        best_plan = None
        best_score = -inf
        
        for _ in range(100):  # Random shooting
            action_sequence = sample_action_sequence(max_steps)
            
            # Predict future features given actions
            predicted = self.model.predict_with_actions(
                current_features, 
                action_sequence
            )
            
            # Score: similarity to goal
            score = F.cosine_similarity(
                predicted[-1], 
                goal_features
            )
            
            if score > best_score:
                best_score = score
                best_plan = action_sequence
                
        return best_plan

# Deployment on real robot:
planner = VJepa2Planner("vjepa2_ac_8b.pth")
while True:
    obs = get_robot_observation()
    action = planner.plan(obs, goal_image)[0]
    execute_action(action)
```

### Speed/Memory Tradeoffs
- Training: 2 weeks on 512 A100s (7168 GPU-days)
- Inference: 230ms per planning step (8B model)
- Memory: 80GB for training, 40GB for inference
- Planning: 50 steps in ~10 seconds
- Feature caching reduces repeated encoding by 5x

---

## VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training

### Summary
VideoMAE demonstrates that extremely high masking ratios (90-95%) enable data-efficient self-supervised video pre-training, achieving 87.4% on Kinetics-400 with only 3.5k videos. The method adapts image MAE to videos using tube masking that preserves temporal coherence while the temporal redundancy in videos allows more aggressive masking than images. This enables training on small domain-specific datasets rather than massive general video collections.

### Key Improvements
1. **Masking Ratio**: Image MAE 75% → Video 90-95% (higher is better)
2. **Data Efficiency**: 1M videos → 3.5k videos for strong performance  
3. **Kinetics-400**: Previous SOTA 85.0% → 87.4%
4. **Something-Something V2**: 70.6% → 75.4%
5. **UCF101 (small dataset)**: 85.1% → 91.3%

### How It Works

```python
def videomae_forward(video, masking_ratio=0.9):
    """
    VideoMAE with tube masking strategy
    
    Key insight: Mask spatiotemporal tubes, not random patches
    Reconstruction target: Normalized pixel values per patch
    """
    # 1. Tubelet embedding: Joint space-time tokenization
    # Each tube: 2×16×16 (time×height×width)
    B, C, T, H, W = video.shape
    tubelets = rearrange(
        video, 
        'b c (t pt) (h ph) (w pw) -> b (t h w) (pt ph pw c)',
        pt=2, ph=16, pw=16
    )  # [B, N_tubes, D_tube]
    
    # 2. Tube masking - key difference from image MAE
    num_tubes = tubelets.shape[1]
    num_keep = int(num_tubes * (1 - masking_ratio))
    
    # Random sampling without replacement
    noise = torch.rand(B, num_tubes)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_keep = ids_shuffle[:, :num_keep]
    
    # Important: Keep tubes aligned temporally
    masked_tubelets = torch.gather(
        tubelets, 
        dim=1, 
        index=ids_keep.unsqueeze(-1).expand(-1, -1, tubelets.shape[-1])
    )
    
    # 3. Encoder: Process only visible tubes (10% of video!)
    x = self.tube_embed(masked_tubelets)  # Linear projection
    x = x + self.pos_embed[:, ids_keep]   # Learned 3D positions
    
    # Vanilla ViT encoder
    for block in self.encoder_blocks:
        x = block(x)  # [B, N_keep, D]
    
    # 4. Decoder: Reconstruct all tubes
    # Add mask tokens for missing tubes
    mask_tokens = self.mask_token.expand(B, num_tubes - num_keep, -1)
    x = torch.cat([x, mask_tokens], dim=1)
    
    # Unshuffle to original positions
    x = torch.gather(
        x, dim=1,
        index=ids_shuffle.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    )
    
    # Add full positional embeddings
    x = x + self.decoder_pos_embed
    
    # Lightweight decoder (fewer blocks than encoder)
    for block in self.decoder_blocks:
        x = block(x)
    
    # 5. Pixel reconstruction head
    x = self.decoder_pred(x)  # [B, N_tubes, tube_size]
    x = rearrange(
        x, 
        'b (t h w) (pt ph pw c) -> b c (t pt) (h ph) (w pw)',
        t=T//2, h=H//16, w=W//16, pt=2, ph=16, pw=16, c=3
    )
    
    # 6. Normalized pixel MSE loss (important detail)
    target = patchify(video, self.patch_size)
    # Normalize by patch mean and variance
    mean = target.mean(dim=-1, keepdim=True)
    var = target.var(dim=-1, keepdim=True)
    target = (target - mean) / (var + 1e-6) ** 0.5
    
    loss = F.mse_loss(x, target, reduction='none')
    loss = loss.mean(dim=-1)  # Mean per patch
    loss = (loss * mask).sum() / mask.sum()  # Only on masked
    
    return loss

class TubeMasking:
    """Cube/tube masking for temporal consistency"""
    def __init__(self, input_size, patch_size, mask_ratio=0.9):
        self.frames, self.height, self.width = input_size
        self.p_t, self.p_h, self.p_w = patch_size
        
        # Tube grid dimensions
        self.T = self.frames // self.p_t
        self.H = self.height // self.p_h  
        self.W = self.width // self.p_w
        
    def __call__(self, batch_size):
        # Can mask whole frames (extreme but works)
        if random.random() < 0.5:
            # Temporal tube masking
            num_masked_frames = int(self.T * 0.5)
            frame_ids = random.sample(range(self.T), num_masked_frames)
            mask = torch.zeros(batch_size, self.T, self.H, self.W)
            mask[:, frame_ids, :, :] = 1
        else:
            # Spatial tube masking (through time)
            num_masked_spatial = int(self.H * self.W * self.mask_ratio)
            for b in range(batch_size):
                spatial_ids = random.sample(
                    range(self.H * self.W), 
                    num_masked_spatial
                )
                spatial_mask = torch.zeros(self.H * self.W)
                spatial_mask[spatial_ids] = 1
                spatial_mask = spatial_mask.reshape(self.H, self.W)
                # Apply same spatial mask to all frames
                mask[b, :] = spatial_mask.unsqueeze(0).expand(self.T, -1, -1)
                
        return mask.flatten(1).bool()
```

### Algorithm Steps
1. **Video Sampling**: 16 frames uniformly from 2-10 second clips
2. **Tube Tokenization**: 2×16×16 patches → 768/1024-dim tokens  
3. **Aggressive Masking**: 90-95% tubes masked (only 5-10% visible!)
4. **Efficient Encoding**: Process only visible tubes
5. **Full Reconstruction**: Decoder predicts all tubes
6. **Normalized Loss**: Per-patch normalized MSE (not raw pixels)

### Implementation Details
- Architecture: ViT-B/L/H with asymmetric encoder-decoder
- Encoder Depth: 12/24/32 blocks for B/L/H
- Decoder Depth: 4 blocks (much smaller)
- Embedding Dim: 768/1024/1280 for B/L/H
- Training: 800/1600/2400 epochs on small datasets
- Batch Size: 256-512 (data-efficient)
- Hardware: 8 V100s sufficient for ViT-B
- Key trick: Joint space-time position embeddings

### Integration Notes
```python
# Key modifications for video:
# In videomae/modeling_videomae.py:

class VideoMAE(nn.Module):
    def __init__(self, ...):
        # Critical: 3D position embeddings
        self.pos_embed = get_3d_sincos_pos_embed(
            embed_dim=768,
            grid_size=(T//2, H//16, W//16)
        )
        
        # Separate embeddings for decoder (full resolution)
        self.decoder_pos_embed = get_3d_sincos_pos_embed(
            embed_dim=512,
            grid_size=(T//2, H//16, W//16)
        )
        
    def forward_encoder(self, x, mask):
        # Only process unmasked tubes - huge speedup
        x = self.patch_embed(x)
        
        # Remove masked positions before encoder
        B, N, D = x.shape
        x_vis = x[~mask].reshape(B, -1, D)
        
        # Add position only to visible
        pos_vis = self.pos_embed.expand(B, -1, -1)[~mask].reshape(B, -1, D)
        x_vis = x_vis + pos_vis
        
        # Efficient encoding of 5-10% of video
        for blk in self.blocks:
            x_vis = blk(x_vis)
            
        return x_vis

# Training recipe for small datasets:
def train_videomae_small_data(dataset, num_videos=3500):
    # Aggressive augmentation for small data
    transform = Compose([
        RandomResizedCrop(224, scale=(0.5, 1.0)),
        RandomHorizontalFlip(),
        ColorJitter(0.4, 0.4, 0.4),
        GrayScale(p=0.2),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])
    
    # Longer training for small datasets
    if num_videos < 10000:
        num_epochs = 2400  # 3x longer
        mask_ratio = 0.95  # More aggressive
    else:
        num_epochs = 800
        mask_ratio = 0.90
```

### Speed/Memory Tradeoffs
- Training: 3 days on 8 V100s for ViT-B (24 GPU-days)
- Inference: Encoder only - 45ms per 16-frame clip
- Memory: 90% masking reduces memory by ~8x vs full video
- Data loading: 10x faster than methods needing millions of videos
- Fine-tuning: Converges in 50 epochs vs 300 for from-scratch

---

## VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking

### Summary
VideoMAE V2 introduces dual masking to scale video masked autoencoders to billion-parameter ViT-g models trained on million-scale datasets. The key innovation is masking different sets of tubes for the encoder and decoder, enabling efficient training of massive models. This achieves 90.0% on Kinetics-400 and 68.7% on Something-Something V1, setting new state-of-the-art records while remaining computationally tractable.

### Key Improvements
1. **Model Scale**: ViT-L (307M) → ViT-g (1B+ params)
2. **Kinetics-400**: 87.4% → 90.0% (new SOTA)
3. **Training Efficiency**: 1.4x faster than VideoMAE v1
4. **Memory Usage**: 40% reduction with dual masking
5. **Something-Something V1**: 65.4% → 68.7%

### How It Works

```python
def videomae_v2_dual_masking(video, encoder_ratio=0.5, decoder_ratio=0.75):
    """
    Dual masking: Different masks for encoder and decoder
    
    Key innovation: Encoder sees 50%, decoder reconstructs different 25%
    This breaks the information bottleneck of single masking
    """
    B, C, T, H, W = video.shape
    
    # 1. Patchify into tubes
    tubes = patchify_3d(video, patch_size=(2, 16, 16))
    N = tubes.shape[1]  # Total number of tubes
    
    # 2. Dual masking strategy - the key innovation
    # Encoder mask: Keep 50% of tubes
    encoder_keep = int(N * (1 - encoder_ratio))
    encoder_ids = torch.randperm(N)[:encoder_keep]
    
    # Decoder mask: Different 25% of tubes  
    # Critical: Decoder targets are disjoint from encoder input
    remaining_ids = torch.randperm(N)[encoder_keep:]
    decoder_targets = int(N * decoder_ratio)
    decoder_ids = remaining_ids[:decoder_targets]
    
    # 3. Encoder forward - sees 50% of video
    encoder_input = tubes[:, encoder_ids]
    encoder_pos = self.pos_embed[:, encoder_ids]
    
    x = self.patch_embed(encoder_input)
    x = x + encoder_pos
    
    # Scaled ViT-g blocks with efficient attention
    for block in self.encoder_blocks:
        x = block(x)  # Using window attention for efficiency
    
    encoder_output = self.encoder_norm(x)
    
    # 4. Decoder forward - predicts different 25%
    # Key: Decoder cannot see target patches
    memory_tokens = encoder_output
    
    # Initialize query tokens for targets
    query_pos = self.decoder_pos_embed[:, decoder_ids]
    queries = self.mask_token.expand(B, len(decoder_ids), -1)
    queries = queries + query_pos
    
    # Cross-attention decoder
    for block in self.decoder_blocks:
        queries = block(
            queries, 
            memory=memory_tokens,
            memory_pos=encoder_pos
        )
    
    # 5. Predict target tube pixels
    predictions = self.decoder_head(queries)
    predictions = unpatchify_3d(predictions, decoder_ids)
    
    # 6. Loss only on decoder target tubes
    targets = tubes[:, decoder_ids]
    targets = normalize_targets(targets)  # Per-patch normalization
    
    loss = F.mse_loss(predictions, targets)
    
    return loss, {
        'encoder_load': encoder_keep / N,
        'decoder_load': decoder_targets / N,
        'total_coverage': (encoder_keep + decoder_targets) / N
    }

class EfficientVideoAttention(nn.Module):
    """Window attention for billion-scale video models"""
    def __init__(self, dim, num_heads, window_size=(8, 7, 7)):
        super().__init__()
        self.window_size = window_size  # (T, H, W)
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        # Factorized attention for efficiency
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        
        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * 
                       (2 * window_size[1] - 1) * 
                       (2 * window_size[2] - 1), num_heads)
        )
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Window partition for efficient attention
        x = rearrange(x, 'b (t h w) c -> b t h w c',
                     t=self.T, h=self.H, w=self.W)
        
        # Partition into windows
        x_windows = window_partition_3d(x, self.window_size)
        B_win = x_windows.shape[0]
        
        # Windowed self-attention
        qkv = self.qkv(x_windows).reshape(
            B_win, -1, 3, self.num_heads, C // self.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn + self._get_rel_pos_bias()
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B_win, -1, C)
        x = self.proj(x)
        
        # Merge windows back
        x = window_unpartition_3d(x, self.window_size, (self.T, self.H, self.W))
        return rearrange(x, 'b t h w c -> b (t h w) c')
```

### Algorithm Steps
1. **Initialization**: Random init for ViT-g scale model
2. **Encoder Masking**: Keep 50% of tubes (vs 10% in V1)
3. **Decoder Targeting**: Reconstruct different 25% of tubes
4. **Efficient Attention**: Window attention for billion-scale
5. **Multi-scale Training**: Progressive resolution increase
6. **Distributed Training**: Model parallel for ViT-g

### Implementation Details
- Architecture: ViT-B/L/H/g with 14×14 patches
- Model Sizes: 86M → 307M → 632M → 1.1B parameters
- Window Size: 8×7×7 for efficient attention
- Training: UnlabeledHybrid dataset (1.35M videos)
- Batch Size: 4096 global (8 per GPU × 512 GPUs)
- Learning Rate: 1.5e-4 with layer-wise LR decay
- Hardware: 512 A100 80GB GPUs
- Training Time: 5 days for ViT-g

### Integration Notes
```python
# Scaling tricks for billion-parameter training:
# In train_videomae_v2.py:

def create_model_v2(model_type='vit_g_patch14_224'):
    model = VideoMAEv2(
        patch_size=(2, 14, 14),
        encoder_embed_dim=1408,
        encoder_depth=40,
        encoder_num_heads=16,
        decoder_embed_dim=512,  # Smaller decoder
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.843,  # Specific for ViT-g
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        # Dual masking ratios
        encoder_mask_ratio=0.5,
        decoder_mask_ratio=0.75
    )
    
    # Initialize with truncated normal
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    model.apply(_init_weights)
    
    # Model parallelism for ViT-g
    if model_type == 'vit_g_patch14_224':
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,
            static_graph=True  # Memory optimization
        )
    
    return model

# Memory-efficient data loading:
class VideoMAEv2Dataset:
    def __init__(self, ...):
        # On-the-fly decoding saves memory
        self.decoder_backend = 'pyav'  # Faster than opencv
        
    def __getitem__(self, idx):
        # Decode only required frames
        video_path = self.samples[idx]
        
        # Efficient frame sampling
        total_frames = get_video_duration(video_path)
        frame_ids = self.sample_frames(total_frames, self.num_frames)
        
        # Parallel frame decoding
        frames = decode_frames_parallel(
            video_path, 
            frame_ids,
            num_threads=4
        )
        
        return self.transform(frames)

# Distributed training optimizations:
def train_one_epoch_v2(model, data_loader, optimizer, epoch):
    # Gradient accumulation for large models
    accumulation_steps = 8
    
    for step, batch in enumerate(data_loader):
        # Mixed precision is crucial
        with torch.cuda.amp.autocast():
            loss = model(batch)
            loss = loss / accumulation_steps
            
        scaler.scale(loss).backward()
        
        if (step + 1) % accumulation_steps == 0:
            # Gradient clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
```

### Speed/Memory Tradeoffs
- Training: 5 days on 512 A100s for ViT-g (2560 GPU-days)
- Throughput: 1024 videos/second globally
- Memory: 70GB per GPU with dual masking (vs 100GB+ single mask)
- Inference: 124ms per 16-frame clip (ViT-g)
- Dual masking speedup: 1.4x training, 1.2x inference

---

## OmniMAE: Single Model Masked Pretraining on Images and Videos

### Summary
OmniMAE demonstrates that a single Vision Transformer can be trained on both images and videos using masked autoencoding, achieving 86.6% on ImageNet and 75.5% on Something-Something-v2. The method uses extremely high masking ratios (90% for images, 95% for videos) and shows that unified architectures can match or exceed specialized models. The key insight is that spatiotemporal transformers naturally handle both 2D and 3D inputs without architectural modifications.

### Key Improvements
1. **Unified Model**: Separate models → Single model for both modalities
2. **ImageNet-1K**: Specialized MAE 85.9% → OmniMAE 86.6%
3. **Something-Something-v2**: Video-only 74.2% → 75.5%
4. **Training Efficiency**: 2x fewer parameters than separate models
5. **Masking Ratios**: 90% images, 95% videos (domain-adaptive)

### How It Works

```python
def omnimae_forward(input_data, modality='auto'):
    """
    Unified masked autoencoding for images and videos
    
    Key: Same architecture processes both with different tokenization
    Images: [B, 3, H, W] → [B, N_patches, D]
    Videos: [B, 3, T, H, W] → [B, N_tubes, D]
    """
    # 1. Modality-aware tokenization
    if modality == 'auto':
        modality = 'video' if input_data.dim() == 5 else 'image'
    
    if modality == 'image':
        # Standard 2D patchification
        B, C, H, W = input_data.shape
        patches = rearrange(
            input_data,
            'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
            p1=16, p2=16
        )
        mask_ratio = 0.9  # 90% for images
        pos_embed = self.pos_embed_2d
        
    else:  # video
        # 3D tube tokenization
        B, C, T, H, W = input_data.shape
        patches = rearrange(
            input_data,
            'b c (t pt) (h p1) (w p2) -> b (t h w) (pt p1 p2 c)',
            pt=2, p1=16, p2=16
        )
        mask_ratio = 0.95  # 95% for videos - more redundancy
        pos_embed = self.pos_embed_3d
    
    # 2. Unified masking strategy
    N = patches.shape[1]
    num_masked = int(N * mask_ratio)
    
    # Random masking for both modalities
    noise = torch.rand(B, N, device=patches.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    
    # Keep first subset
    ids_keep = ids_shuffle[:, :N-num_masked]
    x_masked = torch.gather(
        patches, dim=1,
        index=ids_keep.unsqueeze(-1).expand(-1, -1, patches.size(-1))
    )
    
    # 3. Shared transformer encoder
    x = self.patch_embed(x_masked)  # Same projection for both
    
    # Add modality-specific position embeddings
    x = x + torch.gather(
        pos_embed.expand(B, -1, -1), dim=1,
        index=ids_keep.unsqueeze(-1).expand(-1, -1, pos_embed.size(-1))
    )
    
    # Standard ViT processing
    for blk in self.blocks:
        x = blk(x)
    x = self.norm(x)
    
    # 4. Lightweight shared decoder
    # Append mask tokens
    mask_tokens = self.mask_token.repeat(B, num_masked, 1)
    x_ = torch.cat([x, mask_tokens], dim=1)
    
    # Unshuffle
    x_ = torch.gather(
        x_, dim=1,
        index=ids_restore.unsqueeze(-1).expand(-1, -1, x.size(-1))
    )
    
    # Add full position embeddings
    x_ = x_ + pos_embed.expand(B, -1, -1)
    
    # Decode
    for blk in self.decoder_blocks:
        x_ = blk(x_)
    
    # 5. Modality-specific reconstruction
    if modality == 'image':
        pred = self.decoder_pred_2d(x_)
        pred = rearrange(
            pred, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
            h=H//16, w=W//16, p1=16, p2=16, c=3
        )
    else:
        pred = self.decoder_pred_3d(x_)
        pred = rearrange(
            pred, 'b (t h w) (pt p1 p2 c) -> b c (t pt) (h p1) (w p2)',
            t=T//2, h=H//16, w=W//16, pt=2, p1=16, p2=16, c=3
        )
    
    # 6. Compute reconstruction loss
    target = patchify(input_data, self.patch_size[modality])
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # Mean per patch
    
    # Only compute loss on masked patches
    if modality == 'image':
        mask = torch.ones([B, H//16 * W//16])
    else:
        mask = torch.ones([B, T//2 * H//16 * W//16])
    mask[:, ids_keep] = 0
    mask = mask.bool()
    
    loss = (loss * mask).sum() / mask.sum()
    
    return loss

class UnifiedPositionEmbedding(nn.Module):
    """Shared position embedding that works for 2D and 3D"""
    def __init__(self, num_patches_2d, num_patches_3d, embed_dim):
        super().__init__()
        # Separate embeddings for each modality
        self.pos_embed_2d = nn.Parameter(
            torch.zeros(1, num_patches_2d, embed_dim)
        )
        self.pos_embed_3d = nn.Parameter(
            torch.zeros(1, num_patches_3d, embed_dim)
        )
        
        # Initialize with sin-cos embedding
        pos_embed_2d = get_2d_sincos_pos_embed(
            embed_dim, int(num_patches_2d**0.5)
        )
        pos_embed_3d = get_3d_sincos_pos_embed(
            embed_dim, 
            (num_patches_3d // (14*14), 14, 14)  # (T, H, W)
        )
        
        self.pos_embed_2d.data.copy_(torch.from_numpy(pos_embed_2d).float())
        self.pos_embed_3d.data.copy_(torch.from_numpy(pos_embed_3d).float())
```

### Algorithm Steps
1. **Mixed Batch Sampling**: 50% images, 50% videos per batch
2. **Modality Detection**: Automatic based on input dimensions
3. **Adaptive Masking**: 90% for images, 95% for videos
4. **Unified Encoding**: Same transformer for both modalities
5. **Modality Heads**: Separate reconstruction heads only
6. **Joint Training**: Alternating or mixed batches

### Implementation Details
- Architecture: ViT-Huge with 632M parameters
- Patch Size: 16×16 for images, 2×16×16 for videos
- Hidden Dim: 1280 for ViT-H
- Decoder Depth: 8 blocks (shared)
- Training: IN-1K + SSv2 joint training
- Epochs: 1600 for full convergence
- Batch Size: 4096 total (2048 each modality)
- Hardware: 128 V100 GPUs

### Integration Notes
```python
# Key to unified training:
# In train_omnimae.py:

class MixedModalityDataLoader:
    """Dataloader that mixes images and videos"""
    def __init__(self, image_dataset, video_dataset, mix_ratio=0.5):
        self.image_loader = DataLoader(image_dataset, ...)
        self.video_loader = DataLoader(video_dataset, ...)
        self.mix_ratio = mix_ratio
        
    def __iter__(self):
        image_iter = iter(self.image_loader)
        video_iter = iter(self.video_loader)
        
        while True:
            # Sample modality based on mix ratio
            if random.random() < self.mix_ratio:
                try:
                    batch = next(image_iter)
                    batch['modality'] = 'image'
                except StopIteration:
                    break
            else:
                try:
                    batch = next(video_iter)
                    batch['modality'] = 'video'
                except StopIteration:
                    break
                    
            yield batch

# Efficient mixed training:
def train_omnimae(model, mixed_loader, epochs):
    # Different learning rates for different modalities
    param_groups = [
        {'params': model.patch_embed.parameters(), 'lr': 1e-3},
        {'params': model.blocks.parameters(), 'lr': 1e-3},
        {'params': model.decoder_pred_2d.parameters(), 'lr': 2e-3},
        {'params': model.decoder_pred_3d.parameters(), 'lr': 2e-3},
    ]
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.05)
    
    for epoch in range(epochs):
        for batch in mixed_loader:
            data = batch['data']
            modality = batch['modality']
            
            # Forward with modality info
            loss = model(data, modality=modality)
            
            # Scale loss based on modality complexity
            if modality == 'video':
                loss = loss * 0.8  # Videos are harder
                
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

# Inference on mixed data:
def extract_features(model, data):
    """Extract features from either images or videos"""
    model.eval()
    
    # Auto-detect modality
    if data.dim() == 4:  # Image batch
        modality = 'image'
    elif data.dim() == 5:  # Video batch
        modality = 'video'
    else:
        raise ValueError(f"Unknown input shape: {data.shape}")
    
    # Use only encoder for features
    with torch.no_grad():
        features = model.encode(data, modality=modality)
        
    return features
```

### Speed/Memory Tradeoffs
- Training: 7 days on 128 V100s (896 GPU-days)
- Image Inference: 11ms per image (224×224)
- Video Inference: 89ms per 16-frame clip
- Memory: Unified model uses 40% less than two separate models
- Mixed batching overhead: ~5% slower than single modality

---

## Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data

### Summary
Depth Anything achieves robust monocular depth estimation by scaling data collection to 62M unlabeled images through an automated annotation engine. The method uses a teacher-student framework where a MiDaS-trained teacher generates pseudo-labels for massive unlabeled data, while the student learns with strong augmentations and auxiliary semantic supervision. This achieves state-of-the-art zero-shot performance across diverse benchmarks without pursuing novel architectures.

### Key Improvements
1. **Data Scale**: 1.5M labeled → 62M+ unlabeled images
2. **NYUv2 RMSE**: MiDaS 0.282 → 0.206 (27% reduction)
3. **KITTI δ₁**: 0.955 → 0.982 (relative depth accuracy)
4. **Inference Speed**: 20ms → 12ms (ViT-S on V100)
5. **Robustness**: Handles transparent objects, reflections, low light

### How It Works

```python
def depth_anything_training(labeled_data, unlabeled_data, teacher_model):
    """
    Depth Anything training with pseudo-labeling at scale
    
    Key: Strong augmentations + semantic priors + massive data
    Loss = L_depth + λ_semantic * L_semantic + λ_reg * L_regularization
    """
    # 1. Initialize student from pre-trained DINOv2
    student = DPTModel(
        encoder='dinov2_vitl14',  # Semantic-rich features
        decode_channels=[256, 512, 1024, 1024],
        use_bn=False,
        non_negative=True
    )
    
    # 2. Generate pseudo-labels with teacher
    print(f"Generating labels for {len(unlabeled_data)} images...")
    pseudo_labels = []
    
    with torch.no_grad():
        for batch in tqdm(unlabeled_data):
            # Teacher predictions (no augmentation)
            teacher_depth = teacher_model(batch)
            
            # Filter low-confidence predictions
            confidence = compute_depth_confidence(teacher_depth)
            mask = confidence > 0.9  # High threshold
            
            pseudo_labels.append({
                'depth': teacher_depth,
                'mask': mask,
                'images': batch
            })
    
    # 3. Student training with strong augmentations
    for epoch in range(num_epochs):
        for batch_idx, (labeled_batch, unlabeled_batch) in enumerate(
            zip(labeled_data, cycle(pseudo_labels))
        ):
            # Labeled data branch
            images_l, depth_gt = labeled_batch
            depth_pred_l = student(images_l)
            
            # Standard depth loss on labeled data
            loss_labeled = compute_scale_invariant_loss(
                depth_pred_l, depth_gt
            )
            
            # Unlabeled data branch with augmentations
            images_u = unlabeled_batch['images']
            pseudo_depth = unlabeled_batch['depth']
            confidence_mask = unlabeled_batch['mask']
            
            # Strong augmentations for student
            images_aug = apply_strong_augmentations(images_u)
            depth_pred_u = student(images_aug)
            
            # Robust loss on pseudo-labels
            loss_unlabeled = compute_robust_loss(
                depth_pred_u, 
                pseudo_depth,
                confidence_mask
            )
            
            # Auxiliary semantic consistency loss
            features_original = student.encoder(images_u)
            features_aug = student.encoder(images_aug)
            
            loss_semantic = F.mse_loss(
                features_aug, 
                features_original.detach()
            )
            
            # Total loss with balancing
            loss = loss_labeled + 0.5 * loss_unlabeled + 0.1 * loss_semantic
            
            loss.backward()
            optimizer.step()

def apply_strong_augmentations(images):
    """Strong augmentations that preserve depth relationships"""
    B, C, H, W = images.shape
    
    augmented = images.clone()
    
    for i in range(B):
        # Color jittering (depth-invariant)
        if random.random() < 0.8:
            augmented[i] = ColorJitter(
                brightness=0.4, 
                contrast=0.4,
                saturation=0.4,
                hue=0.1
            )(augmented[i])
        
        # Gaussian blur
        if random.random() < 0.5:
            sigma = random.uniform(0.1, 2.0)
            augmented[i] = gaussian_blur(augmented[i], sigma)
        
        # CutMix-style augmentation (depth-aware)
        if random.random() < 0.5:
            j = random.randint(0, B-1)
            lam = random.uniform(0.3, 0.7)
            
            # Mix images and depths consistently
            bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
            augmented[i, :, bbx1:bbx2, bby1:bby2] = images[j, :, bbx1:bbx2, bby1:bby2]
    
    return augmented

def compute_scale_invariant_loss(pred, target, valid_mask=None):
    """Scale-invariant log depth loss"""
    # Convert to log space
    pred_log = torch.log(pred.clamp(min=1e-8))
    target_log = torch.log(target.clamp(min=1e-8))
    
    # Compute difference
    diff = pred_log - target_log
    
    if valid_mask is not None:
        diff = diff[valid_mask]
    
    # Scale-invariant loss
    loss = torch.sqrt((diff ** 2).mean() - 0.85 * (diff.mean() ** 2))
    
    return loss

class DepthAnythingDataEngine:
    """Automated data collection and annotation engine"""
    def __init__(self, teacher_model, quality_threshold=0.9):
        self.teacher = teacher_model
        self.threshold = quality_threshold
        
    def collect_and_annotate(self, image_urls, batch_size=1000):
        """Download, filter, and annotate images at scale"""
        annotated_data = []
        
        for batch_start in range(0, len(image_urls), batch_size):
            batch_urls = image_urls[batch_start:batch_start + batch_size]
            
            # Parallel download
            images = parallel_download(batch_urls, num_workers=32)
            
            # Quality filtering
            valid_images = []
            for img in images:
                if self.is_valid_image(img):
                    valid_images.append(img)
            
            # Batch inference with teacher
            with torch.no_grad():
                depths = self.teacher(torch.stack(valid_images))
                
            # Confidence-based filtering
            for img, depth in zip(valid_images, depths):
                confidence = self.compute_confidence(depth)
                if confidence > self.threshold:
                    annotated_data.append({
                        'image': img,
                        'depth': depth,
                        'confidence': confidence
                    })
        
        return annotated_data
    
    def compute_confidence(self, depth_map):
        """Estimate pseudo-label quality"""
        # Edge consistency check
        edges = sobel_edges(depth_map)
        edge_sharpness = edges.std()
        
        # Smoothness in homogeneous regions
        smoothness = compute_local_smoothness(depth_map)
        
        # Depth range validity
        depth_range = depth_map.max() - depth_map.min()
        valid_range = 0.1 < depth_range < 100  # meters
        
        confidence = edge_sharpness * smoothness * float(valid_range)
        return confidence.clamp(0, 1)
```

### Algorithm Steps
1. **Teacher Initialization**: Train on 1.5M labeled images (MiDaS)
2. **Data Collection**: Crawl 62M diverse unlabeled images
3. **Pseudo-Labeling**: Generate depth maps with teacher model
4. **Quality Filtering**: Keep high-confidence predictions only
5. **Student Training**: Learn from pseudo-labels with augmentations
6. **Semantic Regularization**: Enforce feature consistency

### Implementation Details
- Encoder: DINOv2 ViT-S/B/L (pretrained features)
- Decoder: DPT heads with 4 scales
- Resolution: 518×518 training, any size inference
- Batch Size: 128 on 8 A100 GPUs
- Learning Rate: 5e-5 with polynomial decay
- Training Time: 35 hours for ViT-L
- Augmentations: Color jitter, blur, CutMix
- Loss Weights: λ_semantic=0.1, λ_robust=0.5

### Integration Notes
```python
# Using Depth Anything:
# In depth_estimation.py:

from depth_anything import DepthAnything
from depth_anything.transform import Resize, Normalize

# Initialize model
model = DepthAnything.from_pretrained('LiheYoung/depth_anything_vitl14')
model.cuda()
model.eval()

# Preprocessing
transform = Compose([
    Resize(
        width=518,
        height=518,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,  # For ViT
        resize_method='minimal'
    ),
    Normalize(mean=[0.485, 0.456, 0.406], 
              std=[0.229, 0.224, 0.225])
])

# Inference on any image
def estimate_depth(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Preprocess
    input_tensor = transform(image).unsqueeze(0).cuda()
    
    # Predict depth
    with torch.no_grad():
        depth = model(input_tensor)
    
    # Resize to original resolution
    depth = F.interpolate(
        depth.unsqueeze(1),
        size=(image.shape[0], image.shape[1]),
        mode='bicubic',
        align_corners=False
    ).squeeze()
    
    return depth.cpu().numpy()

# Fine-tuning for metric depth:
def finetune_metric_depth(model, metric_dataset):
    # Freeze encoder initially
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    # Train decoder only
    optimizer = torch.optim.AdamW(
        model.decoder.parameters(), 
        lr=1e-4
    )
    
    for epoch in range(10):
        for images, depths, intrinsics in metric_dataset:
            pred = model(images)
            
            # Align scale and shift
            scale, shift = compute_scale_shift_least_squares(
                pred, depths
            )
            pred_aligned = pred * scale + shift
            
            loss = F.l1_loss(pred_aligned, depths)
            loss.backward()
            optimizer.step()
    
    # Unfreeze encoder for fine-tuning
    for param in model.encoder.parameters():
        param.requires_grad = True
```

### Speed/Memory Tradeoffs
- Training: 35 hours on 8 A100s (280 GPU-hours) 
- Inference ViT-S: 12ms @ 518×518 (83 FPS)
- Inference ViT-B: 20ms @ 518×518 (50 FPS)
- Inference ViT-L: 37ms @ 518×518 (27 FPS)
- Memory: 2GB (ViT-S), 4GB (ViT-B), 8GB (ViT-L)
- Pseudo-labeling: 500 images/second on 8 GPUs

---

## Video Depth Anything: Consistent Depth Estimation for Super-Long Videos

### Summary
Video Depth Anything extends Depth Anything V2 to handle arbitrarily long videos with temporal consistency through a streaming architecture that caches temporal attention states. The method processes videos frame-by-frame while maintaining hidden states across temporal attention layers, achieving 143 FPS on small models with minimal memory overhead. The key innovation is decoupling spatial and temporal processing to enable streaming inference without quality degradation.

### Key Improvements
1. **Video Length**: Limited sequences → Arbitrary length videos
2. **Temporal Consistency**: Frame flickering → Smooth transitions
3. **Processing Speed**: 14ms/frame (ViT-L) on A100
4. **Memory Efficiency**: Constant memory regardless of video length
5. **Quality Preservation**: No degradation on super-long videos

### How It Works

```python
def video_depth_anything_streaming(video_frames, model, cache_size=512):
    """
    Streaming video depth estimation with temporal consistency
    
    Key: Cache temporal attention states across frames
    Maintains consistency without processing entire video at once
    """
    # 1. Initialize temporal state cache
    temporal_cache = TemporalAttentionCache(
        max_frames=cache_size,
        feature_dim=model.hidden_dim,
        num_layers=model.num_temporal_layers
    )
    
    depth_predictions = []
    
    # 2. Process video in streaming fashion
    for frame_idx, frame in enumerate(video_frames):
        # Spatial encoding (frame-independent)
        spatial_features = model.spatial_encoder(frame)
        
        # 3. Temporal attention with cached states
        if frame_idx > 0:
            # Retrieve relevant temporal context
            temporal_context = temporal_cache.get_context(
                frame_idx, 
                window_size=model.temporal_window
            )
            
            # Apply temporal attention
            temporally_consistent_features = model.temporal_attention(
                query=spatial_features,
                key_value_cache=temporal_context,
                position_offset=frame_idx
            )
        else:
            # First frame - no temporal context
            temporally_consistent_features = spatial_features
        
        # 4. Decode depth with temporal consistency
        depth = model.depth_decoder(temporally_consistent_features)
        
        # 5. Update temporal cache for future frames
        temporal_cache.update(
            frame_idx=frame_idx,
            features=temporally_consistent_features,
            attention_states=model.get_attention_states()
        )
        
        # 6. Optional temporal smoothing
        if frame_idx > 0 and model.use_temporal_smoothing:
            depth = temporal_smooth(
                current_depth=depth,
                previous_depth=depth_predictions[-1],
                motion_weight=compute_motion_weight(frame, video_frames[frame_idx-1])
            )
        
        depth_predictions.append(depth)
    
    return depth_predictions

class TemporalAttentionCache:
    """Efficient cache for temporal attention states"""
    def __init__(self, max_frames, feature_dim, num_layers):
        self.max_frames = max_frames
        self.feature_dim = feature_dim
        self.num_layers = num_layers
        
        # Circular buffer for memory efficiency
        self.cache = {
            'features': torch.zeros(max_frames, feature_dim),
            'attention_states': [
                torch.zeros(max_frames, feature_dim) 
                for _ in range(num_layers)
            ],
            'positions': torch.zeros(max_frames, dtype=torch.long)
        }
        self.write_idx = 0
        self.num_cached = 0
        
    def update(self, frame_idx, features, attention_states):
        """Update cache with new frame information"""
        # Circular buffer write
        cache_idx = self.write_idx % self.max_frames
        
        self.cache['features'][cache_idx] = features
        self.cache['positions'][cache_idx] = frame_idx
        
        for layer_idx, state in enumerate(attention_states):
            self.cache['attention_states'][layer_idx][cache_idx] = state
        
        self.write_idx += 1
        self.num_cached = min(self.num_cached + 1, self.max_frames)
        
    def get_context(self, current_frame_idx, window_size=32):
        """Retrieve relevant temporal context"""
        # Find frames within temporal window
        valid_frames = []
        
        for i in range(self.num_cached):
            cache_idx = i % self.max_frames
            frame_position = self.cache['positions'][cache_idx]
            
            # Check if within window
            if current_frame_idx - window_size <= frame_position < current_frame_idx:
                valid_frames.append({
                    'features': self.cache['features'][cache_idx],
                    'states': [s[cache_idx] for s in self.cache['attention_states']],
                    'position': frame_position
                })
        
        return valid_frames

def temporal_smooth(current_depth, previous_depth, motion_weight):
    """Smooth depth transitions based on motion"""
    # Higher motion = less smoothing
    alpha = 0.9 * (1 - motion_weight)  # 0.1 to 0.9
    
    # Exponential moving average
    smoothed = alpha * previous_depth + (1 - alpha) * current_depth
    
    # Preserve edges
    edge_mask = compute_depth_edges(current_depth) > 0.1
    smoothed[edge_mask] = current_depth[edge_mask]
    
    return smoothed

class VideoDepthAnything(nn.Module):
    """Architecture for consistent video depth estimation"""
    def __init__(self, variant='small'):
        super().__init__()
        
        if variant == 'small':
            self.spatial_encoder = ViTS(embed_dim=384, depth=12)
            self.hidden_dim = 384
            self.num_temporal_layers = 4
        else:  # large
            self.spatial_encoder = ViTL(embed_dim=1024, depth=24)
            self.hidden_dim = 1024
            self.num_temporal_layers = 8
            
        # Temporal consistency modules
        self.temporal_attention = nn.ModuleList([
            TemporalCrossAttention(
                dim=self.hidden_dim,
                num_heads=16 if variant == 'large' else 8,
                window_size=32
            ) for _ in range(self.num_temporal_layers)
        ])
        
        # Depth decoder
        self.depth_decoder = DPTDecoder(
            in_channels=self.hidden_dim,
            features=[96, 192, 384, 768] if variant == 'large' else [48, 96, 192, 384]
        )
        
        self.temporal_window = 32
        self.use_temporal_smoothing = True
```

### Algorithm Steps
1. **Frame Encoding**: Extract spatial features independently
2. **Temporal Context**: Retrieve cached attention states
3. **Cross-Frame Attention**: Apply temporal consistency
4. **Depth Decoding**: Generate temporally consistent depth
5. **State Caching**: Store attention states for future frames
6. **Optional Smoothing**: Motion-adaptive temporal filtering

### Implementation Details
- Models: Small (28.4M) and Large (381.8M params)
- Temporal Window: 32 frames look-back
- Cache Size: 512 frames (configurable)
- Precision: FP16 by default, FP32 optional
- Input Resolution: Any size (adaptive)
- Streaming Mode: Process infinitely long videos
- Batch Processing: Optional for offline videos

### Integration Notes
```python
# Using Video Depth Anything:
# In video_depth_estimation.py:

from video_depth_anything import VideoDepthAnything
import cv2

# Initialize model
model = VideoDepthAnything.from_pretrained(
    'depth-anything/video-depth-anything-v2-large',
    variant='large',
    precision='fp16'
)
model.cuda()
model.eval()

# Streaming inference for long videos
def process_video_stream(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    
    # Initialize streaming mode
    model.init_streaming_mode(cache_size=512)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Streaming depth estimation
        with torch.no_grad():
            depth = model.process_frame_streaming(
                frame_rgb,
                frame_index=frame_count
            )
        
        # Visualize depth
        depth_vis = colorize_depth(depth, cmap='viridis')
        
        # Initialize output writer on first frame
        if out is None:
            h, w = depth_vis.shape[:2]
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        
        out.write(depth_vis)
        frame_count += 1
        
        # Memory stays constant regardless of video length
        if frame_count % 1000 == 0:
            print(f"Processed {frame_count} frames, "
                  f"Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    cap.release()
    out.release()

# Batch processing for known-length videos
def process_video_batch(video_path, batch_size=16):
    """Process entire video with temporal consistency"""
    frames = load_video_frames(video_path)
    
    depths = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        
        with torch.no_grad():
            # Process batch with full temporal context
            batch_depths = model.process_batch(
                batch,
                start_index=i,
                temporal_context=depths[-32:] if depths else None
            )
        
        depths.extend(batch_depths)
    
    return depths

# Fine-tuning for specific video types
def finetune_temporal_consistency(model, video_dataset):
    """Fine-tune temporal modules while keeping spatial frozen"""
    # Freeze spatial encoder
    for param in model.spatial_encoder.parameters():
        param.requires_grad = False
    
    # Only train temporal modules
    temporal_params = []
    for module in model.temporal_attention:
        temporal_params.extend(module.parameters())
    
    optimizer = torch.optim.AdamW(temporal_params, lr=1e-5)
    
    for video in video_dataset:
        frames = video['frames']
        
        # Compute temporal consistency loss
        depths = model(frames)
        
        # Temporal gradient loss
        temporal_loss = compute_temporal_gradient_loss(depths)
        
        # Motion-aware consistency
        motion_loss = compute_motion_consistency_loss(
            depths, frames
        )
        
        loss = temporal_loss + 0.5 * motion_loss
        loss.backward()
        optimizer.step()
```

### Speed/Memory Tradeoffs
- Small Model: 7.5ms/frame (133 FPS) with FP16
- Large Model: 14ms/frame (71 FPS) with FP16
- Memory (Small): 6.8GB constant for any video length
- Memory (Large): 23.6GB constant for any video length
- Cache Overhead: ~500MB for 512-frame cache
- Streaming vs Offline: <5% quality difference

---

## 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering

### Summary
4D Gaussian Splatting (4D-GS) combines 3D Gaussians with 4D neural voxels to achieve real-time rendering of dynamic scenes at 82 FPS (800×800 resolution). The method uses a decomposed neural voxel encoding inspired by HexPlane to efficiently encode spatiotemporal features, which are then used by a lightweight MLP to predict Gaussian deformations at novel timestamps. This enables both high training efficiency (30 minutes) and real-time playback without per-frame optimization.

### Key Improvements
1. **Rendering Speed**: 5 FPS (NeRF) → 82 FPS at 800×800
2. **Training Time**: 48 hours → 30 minutes (HyperNeRF scenes)
3. **Quality**: PSNR 29.5 → 32.8 dB on D-NeRF dataset
4. **Memory**: 10GB → 2.3GB for typical scenes
5. **Temporal Consistency**: Flickering artifacts eliminated

### How It Works

```python
def forward_4d_gaussian_splatting(timestamp, camera_params):
    """
    4D-GS: Deform 3D Gaussians using 4D neural voxels
    
    Key: HexPlane decomposition + lightweight deformation MLP
    Enables real-time rendering by predicting Gaussian changes
    """
    # 1. Initialize canonical 3D Gaussians
    gaussians_canonical = {
        'positions': self.gaussian_positions,      # [N, 3]
        'rotations': self.gaussian_rotations,      # [N, 4] quaternions
        'scales': self.gaussian_scales,            # [N, 3]
        'opacities': self.gaussian_opacities,      # [N, 1]
        'sh_coeffs': self.gaussian_sh_coeffs       # [N, K, 3] spherical harmonics
    }
    
    # 2. Query 4D neural voxels using HexPlane decomposition
    # Decompose 4D space into 6 planes: XY-T, XZ-T, YZ-T, XY, XZ, YZ
    voxel_features = []
    
    for plane_idx, (plane_type, plane_features) in enumerate(self.hexplanes):
        if plane_type == 'XY-T':
            # Spatial-temporal plane
            coords = torch.stack([
                gaussians_canonical['positions'][:, 0],  # X
                gaussians_canonical['positions'][:, 1],  # Y
                timestamp.expand(N)                       # T
            ], dim=-1)
        elif plane_type == 'XZ-T':
            coords = torch.stack([
                gaussians_canonical['positions'][:, 0],  # X
                gaussians_canonical['positions'][:, 2],  # Z
                timestamp.expand(N)                       # T
            ], dim=-1)
        # ... other planes
        
        # Trilinear interpolation on feature planes
        sampled_features = F.grid_sample(
            plane_features,
            coords.unsqueeze(1).unsqueeze(1),
            mode='bilinear',
            align_corners=True
        ).squeeze()
        
        voxel_features.append(sampled_features)
    
    # 3. Aggregate features via learned blending
    aggregated_features = self.feature_aggregator(
        torch.cat(voxel_features, dim=-1)
    )  # [N, feature_dim]
    
    # 4. Predict Gaussian deformations with lightweight MLP
    deformations = self.deformation_mlp(aggregated_features)
    
    # Decompose deformations
    delta_pos = deformations[:, :3]           # Position offset
    delta_rot = deformations[:, 3:7]          # Rotation offset (quaternion)
    delta_scale = deformations[:, 7:10]       # Scale offset
    delta_opacity = deformations[:, 10:11]    # Opacity offset
    delta_sh = deformations[:, 11:].reshape(N, K, 3)  # SH offset
    
    # 5. Apply deformations to canonical Gaussians
    deformed_gaussians = {
        'positions': gaussians_canonical['positions'] + delta_pos,
        'rotations': quaternion_multiply(
            gaussians_canonical['rotations'], 
            quaternion_exp(delta_rot)
        ),
        'scales': gaussians_canonical['scales'] * torch.exp(delta_scale),
        'opacities': torch.sigmoid(
            torch.logit(gaussians_canonical['opacities']) + delta_opacity
        ),
        'sh_coeffs': gaussians_canonical['sh_coeffs'] + delta_sh
    }
    
    # 6. Render using differentiable Gaussian splatting
    rendered_image = gaussian_splatting_render(
        deformed_gaussians,
        camera_params,
        image_size=(800, 800)
    )
    
    return rendered_image, deformed_gaussians

class HexPlaneEncoder(nn.Module):
    """Efficient 4D encoding via plane decomposition"""
    def __init__(self, bounds, resolutions, feature_dim=32):
        super().__init__()
        
        # 6 feature planes for 4D decomposition
        self.planes = nn.ModuleDict({
            'XY_T': self._create_plane(resolutions['XY'], resolutions['T'], feature_dim),
            'XZ_T': self._create_plane(resolutions['XZ'], resolutions['T'], feature_dim),
            'YZ_T': self._create_plane(resolutions['YZ'], resolutions['T'], feature_dim),
            'XY': self._create_plane(resolutions['XY'], 1, feature_dim),
            'XZ': self._create_plane(resolutions['XZ'], 1, feature_dim),
            'YZ': self._create_plane(resolutions['YZ'], 1, feature_dim)
        })
        
        # Feature aggregation network
        self.aggregator = nn.Sequential(
            nn.Linear(6 * feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
    def _create_plane(self, spatial_res, temporal_res, feature_dim):
        """Create a feature plane with given resolution"""
        if temporal_res > 1:
            # Spatial-temporal plane
            return nn.Parameter(
                torch.randn(1, feature_dim, temporal_res, spatial_res, spatial_res) * 0.1
            )
        else:
            # Spatial-only plane
            return nn.Parameter(
                torch.randn(1, feature_dim, spatial_res, spatial_res) * 0.1
            )
    
    def forward(self, positions, timestamp):
        """Extract features for Gaussians at given time"""
        features = []
        
        # Sample from each plane
        for plane_name, plane_features in self.planes.items():
            if '_T' in plane_name:
                # Spatial-temporal sampling
                features.append(self._sample_st_plane(
                    plane_features, positions, timestamp, plane_name
                ))
            else:
                # Spatial-only sampling
                features.append(self._sample_spatial_plane(
                    plane_features, positions, plane_name
                ))
        
        # Aggregate features
        combined = torch.cat(features, dim=-1)
        return self.aggregator(combined)

def train_4d_gaussians(dataset, num_iterations=30000):
    """Efficient training of 4D Gaussian Splatting"""
    # Initialize components
    gaussians = initialize_gaussians_from_pointcloud(
        dataset.get_initial_pointcloud(),
        num_gaussians=100000
    )
    
    hexplane_encoder = HexPlaneEncoder(
        bounds=dataset.get_bounds(),
        resolutions={'XY': 128, 'XZ': 128, 'YZ': 128, 'T': 150},
        feature_dim=32
    )
    
    deformation_mlp = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 11 + K * 3)  # pos(3) + rot(4) + scale(3) + opacity(1) + sh(K*3)
    )
    
    # Optimization
    optimizer = torch.optim.Adam([
        {'params': gaussians.parameters(), 'lr': 1e-3},
        {'params': hexplane_encoder.parameters(), 'lr': 1e-2},
        {'params': deformation_mlp.parameters(), 'lr': 1e-3}
    ])
    
    for iteration in range(num_iterations):
        # Sample random timestamp and camera
        timestamp = torch.rand(1) * dataset.duration
        camera = dataset.sample_camera(timestamp)
        gt_image = dataset.get_image(timestamp, camera)
        
        # Forward pass
        rendered, deformed_gaussians = forward_4d_gaussian_splatting(
            timestamp, camera, gaussians, hexplane_encoder, deformation_mlp
        )
        
        # Losses
        l1_loss = F.l1_loss(rendered, gt_image)
        ssim_loss = 1 - ssim(rendered, gt_image)
        
        # Regularization on deformations
        deform_reg = compute_deformation_regularization(
            gaussians, deformed_gaussians, weight=0.01
        )
        
        total_loss = l1_loss + 0.2 * ssim_loss + deform_reg
        
        # Optimization step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Adaptive Gaussian control
        if iteration % 500 == 0:
            gaussians = adaptive_gaussian_control(
                gaussians, 
                rendered_gradients,
                split_threshold=0.01,
                prune_threshold=0.005
            )
```

### Algorithm Steps
1. **Initialization**: Create canonical 3D Gaussians from point cloud
2. **HexPlane Setup**: Initialize 6 feature planes (XY-T, XZ-T, etc.)
3. **Feature Query**: Sample 4D voxel features at Gaussian positions
4. **Deformation Prediction**: MLP predicts per-Gaussian changes
5. **Gaussian Update**: Apply predicted deformations
6. **Differentiable Rendering**: Splat deformed Gaussians to image

### Implementation Details
- Gaussian Count: 50k-200k depending on scene complexity
- HexPlane Resolution: 128³ spatial, 150 temporal
- Feature Dimensions: 32 per plane, 64 aggregated
- MLP Architecture: 3 layers, 128-256 hidden units
- Training: 30 minutes on single RTX 3090
- Optimization: Adam with learning rate scheduling
- Regularization: L2 on deformations, opacity decay

### Integration Notes
```python
# Using 4D Gaussian Splatting:
# In dynamic_rendering.py:

from gaussian_4d import GaussianSplatting4D

# Initialize model
model = GaussianSplatting4D(
    initial_points=point_cloud,
    num_gaussians=100000,
    sh_degree=3,
    hexplane_resolution=(128, 128, 128, 150),
    feature_dim=32
)

# Training from multi-view video
def train_on_video(video_data, cameras, num_epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(num_epochs):
        for t in range(video_data.num_frames):
            for cam_idx, camera in enumerate(cameras):
                # Get ground truth
                gt_image = video_data.get_frame(t, cam_idx)
                
                # Render prediction
                timestamp = t / video_data.fps
                rendered = model.render(timestamp, camera)
                
                # Loss computation
                loss = compute_rendering_loss(rendered, gt_image)
                loss.backward()
                
                # Gradient-based densification
                if epoch % 10 == 0:
                    model.densify_and_prune(
                        max_grad=0.0002,
                        min_opacity=0.005
                    )
        
        optimizer.step()
        optimizer.zero_grad()

# Real-time playback
def realtime_viewer(model, camera_trajectory, fps=30):
    """Interactive viewer with camera control"""
    clock = pygame.time.Clock()
    
    for t in range(0, model.duration * fps):
        timestamp = t / fps
        
        # Get camera from trajectory or user input
        camera = camera_trajectory.get_camera(timestamp)
        
        # Render at 82 FPS (800x800)
        with torch.no_grad():
            frame = model.render(timestamp, camera)
        
        # Display
        display_frame(frame)
        clock.tick(fps)

# Export to video with custom camera path
def export_novel_view_video(model, camera_path, output_path):
    writer = cv2.VideoWriter(output_path, ...)
    
    for t in np.linspace(0, model.duration, model.duration * 30):
        camera = camera_path.interpolate(t)
        frame = model.render(t, camera).cpu().numpy()
        writer.write(frame)
```

### Speed/Memory Tradeoffs
- Training: 8 min (D-NeRF), 30 min (HyperNeRF) on RTX 3090
- Rendering: 82 FPS @ 800×800, 143 FPS @ 400×400
- Memory: 2.3GB typical scene, up to 5GB for complex
- HexPlane Storage: 200MB for 128³×150 resolution
- Gaussian Storage: 1.5GB for 100k Gaussians
- Quality vs Speed: Can trade Gaussians for FPS (50k → 120 FPS)
