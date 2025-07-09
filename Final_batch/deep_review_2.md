# Deep Technical Reviews - Dynamic Scenes & 4D Reconstruction

## 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering

### Summary
4D-GS extends 3D Gaussian Splatting to dynamic scenes by introducing a Gaussian deformation field network that models both position and shape changes over time. The method maintains a single set of canonical 3D Gaussians and uses a spatial-temporal structure encoder (HexPlane-inspired) combined with lightweight MLPs to predict per-Gaussian deformations at novel timestamps. This achieves 82 FPS at 800×800 resolution on RTX 3090 while maintaining quality comparable to state-of-the-art methods.

### Key Improvements
1. **Rendering Speed**: 1.5 FPS (TiNeuVox) → 82 FPS at 800×800 resolution (54.7x speedup)
2. **Training Time**: 28 minutes (TiNeuVox) → 8 minutes (3.5x faster)
3. **Storage**: 48 MB (TiNeuVox) → 18 MB (2.7x reduction)
4. **Quality (PSNR)**: 32.67 dB (TiNeuVox) → 34.05 dB (+1.38 dB improvement)
5. **Memory Complexity**: O(tN) (Dynamic3DGS) → O(N + F) (linear to constant in time)

### How It Works

**Core Algorithm**
```python
def gaussian_deformation_field(gaussian_positions, timestamp, hexplane_encoder, deformation_mlps):
    """
    Mathematical formulation: G'(t) = G + ΔG(t)
    Key difference from baseline: Single canonical Gaussian set with learned deformations
    """
    # Extract spatial-temporal features using HexPlane decomposition
    features = []
    for plane in [(x,y), (x,z), (y,z), (x,t), (y,t), (z,t)]:
        # Bilinear interpolation on 2D planes at multiple resolutions
        plane_features = hexplane_encoder.query_plane(gaussian_positions, timestamp, plane)
        features.append(plane_features)
    
    # Merge features with tiny MLP
    merged_features = feature_mlp(torch.cat(features, dim=-1))  # [N, hidden_dim]
    
    # Multi-head deformation prediction
    delta_position = position_mlp(merged_features)     # [N, 3]
    delta_rotation = rotation_mlp(merged_features)     # [N, 4] 
    delta_scale = scale_mlp(merged_features)          # [N, 3]
    
    return delta_position, delta_rotation, delta_scale
```

**Mathematical Foundation**
- Deformed Gaussian at time t: `G'(t) = {X + ΔX(t), r + Δr(t), s + Δs(t), σ, C}`
- HexPlane feature encoding: `f = ∏ interp(R_l(i,j))` for all 6 plane pairs
- Computational complexity: O(N) for N Gaussians per frame (constant time)

**Critical Implementation Details**
- 6 multi-resolution planes: 3 spatial (xy, xz, yz) + 3 temporal (xt, yt, zt)
- Base resolution: 64, upsampled by factors of 2 and 4
- Feature dimension: 32 per plane
- Deformation MLPs: 3 separate 2-layer networks (128 hidden units each)

### Algorithm Steps
1. **Initialization**: Optimize static 3D Gaussians for 3000 iterations
2. **Feature Encoding**: Query 6 HexPlane modules at Gaussian centers
3. **Deformation Prediction**: Apply position, rotation, scale MLPs
4. **Gaussian Transform**: Add deformations to canonical Gaussians
5. **Rendering**: Standard 3D-GS differential splatting on deformed Gaussians

### Implementation Details
- Architecture: HexPlane (6×64×64×32) + 3 MLPs (32→128→3/4/3)
- Learning rates: 1.6e-3 (Gaussians), 1.6e-4 (deformation network)
- Warmup: 3000 iterations of static 3D-GS before enabling deformations
- Loss: L1 color + total variation on HexPlane grids
- Dependencies: PyTorch, CUDA 11.0+, diff-gaussian-rasterization
- Hardware: Single RTX 3090 (24GB VRAM)
- Training time: 8 minutes for 200 frames

### Integration Notes
```python
# In original 3D-GS rendering loop:
# - gaussians = optimize_gaussians(views)
# + if iteration < 3000:
# +     gaussians = optimize_gaussians(views)  # Static warmup
# + else:
# +     deformations = deformation_field(gaussians.xyz, timestamp)
# +     gaussians_deformed = apply_deformations(gaussians, deformations)
# +     rendered = rasterize(gaussians_deformed, viewpoint)
```

Key modifications:
- Replace per-timestamp Gaussian storage with deformation field
- Add HexPlane encoder initialization (6 planes, multi-resolution)
- Modify optimizer to include deformation network parameters
- Add temporal sampling strategy for training batches

### Speed/Memory Tradeoffs
- Training: 10 min (3D-GS) → 8 min (20% faster due to shared features)
- Inference: 82 FPS @ 800×800, 30 FPS @ 1352×1014
- Memory: 10MB (canonical Gaussians) + 8MB (deformation network)
- Quality settings: Can reduce HexPlane resolution (64→32) for 1.5x speedup
- Scaling: Linear in number of Gaussians, constant in sequence length

---

## Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis

### Summary
Spacetime Gaussians extends 3D Gaussian Splatting to 4D by adding temporal opacity functions and polynomial motion/rotation trajectories to each Gaussian. Instead of spherical harmonics, it uses splatted neural features (9D vs 48D for SH) that encode base color, view-dependent, and time-dependent information, processed through a lightweight MLP. The method achieves 140 FPS at 1352×1014 resolution while maintaining compact storage through aggressive pruning and feature compression.

### Key Improvements
1. **Rendering Speed**: 0.3 FPS (K-Planes) → 140 FPS at 1352×1014 (466x speedup)
2. **Quality (PSNR)**: 31.63 dB (K-Planes) → 32.05 dB (+0.42 dB)
3. **Storage per Frame**: 1.03 MB (K-Planes) → 0.67 MB (35% reduction)
4. **8K Performance**: 60 FPS at 8K resolution (lite version on RTX 4090)
5. **Feature Size**: 48D (3-degree SH) → 9D features (81% reduction)

### How It Works

**Core Algorithm**
```python
def spacetime_gaussian(position, time, gaussian_params):
    """
    Mathematical formulation: α(x,t) = σ(t) * exp(-0.5 * (x-μ(t))^T Σ(t)^-1 (x-μ(t)))
    Key difference from baseline: Time-varying opacity and polynomial trajectories
    """
    # Temporal opacity using 1D Gaussian
    temporal_opacity = gaussian_params.sigma_s * np.exp(
        -gaussian_params.s_tau * (time - gaussian_params.mu_tau)**2
    )
    
    # Polynomial motion trajectory (3rd degree)
    position_t = sum(
        gaussian_params.b[k] * (time - gaussian_params.mu_tau)**k 
        for k in range(4)  # k=0 to 3
    )
    
    # Polynomial rotation (1st degree for efficiency)
    quaternion_t = sum(
        gaussian_params.c[k] * (time - gaussian_params.mu_tau)**k
        for k in range(2)  # k=0 to 1
    )
    
    # Feature computation instead of SH
    features = torch.cat([
        gaussian_params.f_base,                          # [3] base RGB
        gaussian_params.f_dir,                          # [3] view-dependent
        (time - gaussian_params.mu_tau) * gaussian_params.f_time  # [3] time-dependent
    ])  # Total: [9]
    
    return temporal_opacity, position_t, quaternion_t, features
```

**Mathematical Foundation**
- Temporal opacity: `σ(t) = σ_s * exp(-s_τ |t - μ_τ|²)`
- Motion trajectory: `μ(t) = Σ(k=0 to 3) b_k * (t - μ_τ)^k`
- Rotation: `q(t) = Σ(k=0 to 1) c_k * (t - μ_τ)^k`
- Computational complexity: O(N) for N Gaussians, polynomial evaluation is O(1)

**Critical Implementation Details**
- Features: 9D vectors (3 base + 3 view + 3 time) vs 48D for SH
- MLP: 2-layer network (9 → 32 → 3) for final color
- Polynomial degrees: 3 for position, 1 for rotation (balance between quality/speed)
- Aggressive pruning: More stringent than 3D-GS to maintain compact size

### Algorithm Steps
1. **Initialization**: SfM points from all timestamps (not just first frame)
2. **Temporal Modeling**: Assign temporal center μ_τ and scale s_τ to each Gaussian
3. **Motion Fitting**: Optimize polynomial coefficients for position/rotation
4. **Feature Splatting**: Rasterize 9D features instead of RGB colors
5. **MLP Rendering**: Convert splatted features to final colors
6. **Guided Sampling**: Add Gaussians in high-error regions using coarse depth

### Implementation Details
- Architecture: Per-Gaussian params + shared 2-layer MLP (9→32→3)
- Polynomial coefficients: 4×3 for position, 2×4 for rotation per Gaussian
- Training time: 40-60 minutes for 50 frames on A6000 GPU
- Loss: L1 + D-SSIM on rendered images
- Pruning: Aggressive opacity-based removal to maintain size
- Dependencies: PyTorch, CUDA 11.0+, custom differentiable rasterizer
- Hardware: NVIDIA A6000 (48GB) for training, RTX 4090 for 8K demos

### Integration Notes
```python
# Modified Gaussian structure:
class SpacetimeGaussian:
    # Spatial parameters (unchanged)
    position_base: torch.Tensor  # [3]
    scale: torch.Tensor         # [3]
    opacity_spatial: float
    
    # Temporal parameters (new)
    temporal_center: float      # μ_τ
    temporal_scale: float       # s_τ
    motion_coeffs: torch.Tensor # [4, 3] for cubic position
    rotation_coeffs: torch.Tensor # [2, 4] for linear quaternion
    
    # Features instead of SH (new)
    feature_base: torch.Tensor  # [3]
    feature_dir: torch.Tensor   # [3]
    feature_time: torch.Tensor  # [3]

# In rendering loop:
# - colors = compute_sh(gaussians, viewdir)
# + features = compute_features(gaussians, time)
# + features_splatted = splat_features(features, gaussians)
# + colors = feature_mlp(features_splatted, viewdir)
```

Key modifications:
- Replace SH coefficients with 9D feature vectors
- Add temporal parameters to each Gaussian
- Implement polynomial evaluation for motion
- Add guided sampling based on error maps

### Speed/Memory Tradeoffs
- Training: 40-60 min for 50 frames (vs 8 min for 4D-GS on 200 frames)
- Inference: 140 FPS @ 1352×1014, 60 FPS @ 8K (lite version)
- Memory: 0.67 MB/frame with aggressive pruning
- Quality settings: Lite version drops MLP for 1.5x speedup
- Scaling: Linear in Gaussians, but pruning keeps count low
- Initialization frames: Using every 4th frame reduces size by 58% with minimal quality loss

---

## MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds

### Summary
MoSca introduces a sparse graph-based deformation representation that lifts 2D foundation model predictions (depth, tracking, epipolar error) into a structured 4D Motion Scaffold. The scaffold consists of 6-DoF trajectory nodes connected by curve distance topology, which drives Gaussian deformations via dual quaternion blending. This enables global fusion of all temporal observations while handling unknown camera poses through tracklet-based bundle adjustment, achieving state-of-the-art performance on challenging monocular datasets.

### Key Improvements
1. **Quality (PSNR)**: 17.32 dB (Shape-of-Motion) → 19.32 dB on DyCheck (+2.0 dB)
2. **Camera-free Operation**: No COLMAP needed, solves poses and focal length
3. **Rendering Speed**: 37.8 FPS at 2x resolution (vs <1 FPS for NeRF methods)
4. **Compression Ratio**: 46:1 (Gaussians to motion nodes)
5. **Tracking Accuracy**: 77.1% (BootsTAPIR) → 85.2% after optimization

### How It Works

**Core Algorithm - Motion Scaffold Construction**
```python
def motion_scaffold_deformation(query_pos, query_time, mosca_nodes):
    """
    Mathematical formulation: W(x,w;t_src,t_dst) = DQB({w_i, ΔQ_i})
    Key difference from baseline: Sparse trajectory graph with physics regularization
    """
    # 1. Find nearest node at source time
    nearest_node_idx = argmin([
        np.linalg.norm(node.translation[t_src] - query_pos)
        for node in mosca_nodes
    ])
    
    # 2. Get neighborhood via curve distance topology
    neighbors = mosca_nodes[nearest_node_idx].edges  # K-NN in trajectory space
    
    # 3. Compute RBF skinning weights
    weights = []
    for neighbor_idx in neighbors:
        node = mosca_nodes[neighbor_idx]
        weight = np.exp(-np.linalg.norm(query_pos - node.translation[t_src])**2 
                       / (2 * node.radius**2))
        weights.append(weight)
    
    # 4. Compute SE(3) transformations
    transforms = []
    for neighbor_idx in neighbors:
        node = mosca_nodes[neighbor_idx]
        delta_Q = node.Q[t_dst] @ np.linalg.inv(node.Q[t_src])  # SE(3) difference
        transforms.append(delta_Q)
    
    # 5. Dual Quaternion Blending for manifold interpolation
    blended_dq = sum([
        w * to_dual_quaternion(Q) for w, Q in zip(weights, transforms)
    ])
    blended_dq /= np.linalg.norm(blended_dq)  # Normalize on DQ manifold
    
    return from_dual_quaternion(blended_dq)
```

**Mathematical Foundation**
- Motion nodes: `v^(m) = ([Q_1^(m), Q_2^(m), ..., Q_T^(m)], r^(m))` where Q ∈ SE(3)
- Curve distance: `D_curve(m,n) = max_t ||t_t^(m) - t_t^(n)||`
- Dual quaternion blending: Preserves rigid transformations on SE(3) manifold
- ARAP regularization: Preserves local rigidity during deformation

**Critical Implementation Details**
- Foundation models: BootsTAPIR (tracking), Metric3D-v2/UniDepth (depth)
- Node topology: K=8 nearest neighbors based on curve distance
- Multi-level topology pyramid for ARAP optimization
- Learnable per-Gaussian skinning weight corrections Δw

### Algorithm Steps
1. **Foundation Model Inference**: Extract depth D, tracks T, epipolar errors M
2. **Camera Initialization**: Tracklet-based bundle adjustment on static tracks
3. **MoSca Lifting**: Initialize nodes from 3D-lifted tracks with identity rotations
4. **Geometric Optimization**: ARAP + velocity/acceleration regularization
5. **Gaussian Attachment**: Back-project depth at all frames, attach to MoSca
6. **Photometric Optimization**: Render and optimize with RGB, depth, track losses
7. **Node Control**: Densify high-error regions, prune low-contribution nodes

### Implementation Details
- Architecture: Sparse graph (3K nodes) + dense Gaussians (100K)
- Node parameters: T×SE(3) transforms + radius per node
- Training phases: Geometric optimization → Photometric optimization
- Loss weights: λ_arap=1.0, λ_vel=0.1, λ_acc=0.1, λ_rgb=1.0, λ_track=0.1
- Dependencies: PyTorch, diff-gaussian-rasterization, foundation models
- Hardware: Single GPU, handles iPhone DyCheck videos
- Bundle adjustment: Optimizes both poses and per-frame depth scales

### Integration Notes
```python
# MoSca node structure:
class MotionScaffoldNode:
    def __init__(self, track_3d, node_id):
        self.id = node_id
        self.Q = [np.eye(4) for _ in range(T)]  # SE(3) per timestep
        self.radius = 0.1  # RBF control radius
        self.edges = []  # Topology connections
        
        # Initialize translations from lifted 3D tracks
        for t, pos_3d in enumerate(track_3d):
            if pos_3d is not None:
                self.Q[t][:3, 3] = pos_3d

# Gaussian deformation:
def deform_gaussian_to_time(gaussian, target_time):
    # Get base skinning weights from RBF
    base_weights = compute_rbf_weights(
        gaussian.position, 
        gaussian.ref_time,
        mosca_nodes
    )
    
    # Apply learnable correction
    final_weights = base_weights + gaussian.weight_correction
    
    # Compute deformation via DQB
    transform = motion_scaffold_deformation(
        gaussian.position,
        gaussian.ref_time,
        target_time,
        mosca_nodes
    )
    
    # Transform Gaussian
    gaussian.position = transform @ gaussian.position
    gaussian.rotation = transform[:3, :3] @ gaussian.rotation
```

Key modifications from standard 3DGS:
- Add reference timestamp and skinning weights to each Gaussian
- Replace per-frame reconstruction with global fusion
- Implement tracklet-based camera solver
- Add ARAP regularization for motion coherence

### Speed/Memory Tradeoffs
- Training: Geometric opt (minutes) + Photometric opt (varies by length)
- Inference: 37.8 FPS at 2x resolution on DyCheck
- Memory: ~3K nodes vs ~100K Gaussians (46:1 compression)
- Quality settings: Can reduce nodes for faster but less accurate deformation
- Fusion strategy: Global fusion vs local 4-8 frame windows (2.3 dB gain)
- Foundation models: Better trackers/depth improve quality at inference cost

---

## MoDGS: Dynamic Gaussian Splatting from Casually-Captured Monocular Videos with Depth Priors

### Summary
MoDGS addresses the challenge of dynamic scene reconstruction from casually captured monocular videos where cameras are static or move slowly. It leverages single-view depth estimation (GeoWizard) with a novel ordinal depth loss that maintains depth ordering relationships rather than absolute values. The method introduces 3D-aware initialization by lifting 2D optical flow to 3D using depth maps, training an invertible deformation MLP before Gaussian optimization. This achieves 22.64 PSNR on DyNeRF dataset, significantly outperforming methods that rely on teleporting camera motions.

### Key Improvements
1. **Quality (PSNR)**: 19.55 dB (Deformable-GS) → 22.64 dB on DyNeRF (+3.09 dB)
2. **Robustness**: Works with static/slow cameras vs requiring teleporting motions
3. **Depth Consistency**: Ordinal loss handles scale-inconsistent depth estimates
4. **3D Initialization**: 21.27 → 22.96 PSNR with 3D-aware init (+1.69 dB)
5. **Depth Supervision**: 21.77 → 22.96 PSNR ordinal vs Pearson (+1.19 dB)

### How It Works

**Core Algorithm - Ordinal Depth Loss**
```python
def ordinal_depth_loss(rendered_depth, estimated_depth, num_pairs=100000):
    """
    Mathematical formulation: ℓ = ||tanh(α(D̂(u1) - D̂(u2))) - R(D(u1), D(u2))||
    Key difference from baseline: Preserves depth ordering rather than absolute values
    """
    # Sample random pixel pairs
    h, w = rendered_depth.shape
    u1 = torch.randint(0, h*w, (num_pairs,))
    u2 = torch.randint(0, h*w, (num_pairs,))
    
    # Compute order indicator for estimated depth
    def order_indicator(d1, d2):
        return torch.where(d1 > d2, 1.0, -1.0)
    
    estimated_order = order_indicator(
        estimated_depth.flatten()[u1],
        estimated_depth.flatten()[u2]
    )
    
    # Compute soft ordering for rendered depth using tanh
    alpha = 10.0  # Sharpness parameter
    rendered_diff = rendered_depth.flatten()[u1] - rendered_depth.flatten()[u2]
    rendered_order = torch.tanh(alpha * rendered_diff)
    
    # Loss encourages consistent ordering
    loss = torch.norm(rendered_order - estimated_order)
    
    return loss
```

**3D-Aware Initialization**
```python
def initialize_deformation_field(depth_maps, optical_flows, camera_params):
    """
    Lifts 2D flow to 3D and trains invertible deformation MLP
    """
    # 1. Rectify depth scales across frames
    static_mask = compute_static_regions(optical_flows)
    scales = []
    for t in range(1, T):
        # Project static regions and solve for scale
        scale = least_squares_depth_alignment(
            depth_maps[0], depth_maps[t], 
            static_mask, camera_params
        )
        scales.append(scale)
    
    # 2. Lift 2D flow to 3D flow
    for i, j in frame_pairs:
        # Convert depths to 3D points
        points_i = backproject(depth_maps[i] * scales[i], camera_params[i])
        
        # Use optical flow to find correspondences
        flow_2d = optical_flows[f"{i}->{j}"]
        points_j = backproject_with_flow(
            depth_maps[j] * scales[j], 
            flow_2d, 
            camera_params[j]
        )
        
        # Create 3D flow
        flow_3d[f"{i}->{j}"] = points_j - points_i
    
    # 3. Train invertible deformation MLP
    deform_mlp = InvertibleMLP()  # Based on NICE architecture
    for epoch in range(init_epochs):
        loss = 0
        for i, j in frame_pairs:
            # Forward and backward consistency
            x_i = sample_points(points_i)
            x_j_pred = deform_mlp.forward(x_i, time=j) @ deform_mlp.inverse(x_i, time=i)
            loss += ||x_j_pred - (x_i + flow_3d[f"{i}->{j}"][x_i])||²
```

**Mathematical Foundation**
- Order indicator: `R(D(u1), D(u2)) = +1 if D(u1) > D(u2), else -1`
- Ordinal consistency: Preserves relative depth ordering across frames
- Scale rectification: `D_t = a_t * D_t_raw + b_t` aligned on static regions
- Invertible MLP: Ensures bijective mapping between canonical and deformed space

**Critical Implementation Details**
- Depth estimator: GeoWizard (metric depth with scale ambiguity)
- Flow estimator: RAFT for dense inter-frame correspondences
- Deformation network: Invertible MLP (NICE-based) sharing weights across time
- Static region detection: Flow magnitude thresholding or SAM2 segmentation

### Algorithm Steps
1. **Depth Estimation**: Extract depth maps D_t using GeoWizard
2. **Flow Estimation**: Compute 2D optical flows F_{ti->tj} using RAFT
3. **Scale Rectification**: Align depth scales using static regions
4. **3D Flow Lifting**: Convert 2D flow + depth to 3D correspondences
5. **Deformation Init**: Train invertible MLP on 3D flows
6. **Gaussian Init**: Deform all depth points to canonical space, downsample
7. **Joint Optimization**: Train Gaussians + deformation with ordinal depth loss

### Implementation Details
- Architecture: Canonical Gaussians + Invertible MLP deformation
- Deformation MLP: 8 layers, 256 hidden units, NICE coupling layers
- Ordinal loss: α=10, 100K random pixel pairs per iteration
- Loss weights: λ_render=1.0, λ_ordinal=0.1
- Initialization: 1000 iterations for deformation field pre-training
- Downsampling: Voxel-based to reduce canonical points
- Training time: Several hours per scene (comparable to baselines)

### Integration Notes
```python
# Modified training loop with ordinal depth:
def train_modgs(video_frames, camera_params):
    # Phase 1: Initialize deformation field
    depth_maps = [geowizard(frame) for frame in video_frames]
    flows = compute_raft_flows(video_frames)
    
    deform_field = initialize_deformation_field(
        depth_maps, flows, camera_params
    )
    
    # Phase 2: Initialize Gaussians in canonical space
    canonical_points = []
    for t, depth in enumerate(depth_maps):
        points_t = backproject(depth, camera_params[t])
        # Deform to canonical space using initialized field
        canonical = deform_field.inverse(points_t, time=t)
        canonical_points.extend(canonical)
    
    # Downsample and create Gaussians
    gaussians = initialize_gaussians(
        voxel_downsample(canonical_points, voxel_size=0.01)
    )
    
    # Phase 3: Joint optimization
    for iteration in range(max_iters):
        t = sample_timestamp()
        
        # Deform Gaussians to time t
        deformed_positions = deform_field(
            gaussians.positions, time=t
        )
        
        # Render
        rendered_rgb, rendered_depth = splat_gaussians(
            deformed_positions, camera_params[t]
        )
        
        # Losses
        loss = λ_render * l1_loss(rendered_rgb, video_frames[t])
        loss += λ_ordinal * ordinal_depth_loss(
            rendered_depth, depth_maps[t]
        )
        
        loss.backward()
```

Key modifications from standard dynamic 3DGS:
- Replace Pearson correlation with ordinal depth loss
- Add 3D-aware initialization phase before Gaussian optimization
- Use invertible MLP for bijective deformation
- Handle scale-inconsistent depth via ordering preservation

### Speed/Memory Tradeoffs
- Training: Hours per scene (bottleneck: depth estimation preprocessing)
- Inference: Real-time rendering after training (standard 3DGS speed)
- Memory: Canonical Gaussians + lightweight MLP (< 100MB typical)
- Quality vs Speed: Can reduce ordinal pairs for faster training
- Depth quality: Better depth estimators improve results but increase preprocessing
- Camera motion: Static cameras rely heavily on depth priors vs moving cameras

---

## Deblur4DGS: 4D Gaussian Splatting from Blurry Monocular Video

### Summary
Deblur4DGS tackles 4D reconstruction from motion-blurred monocular videos by transforming continuous dynamic representation estimation into exposure time estimation. It uses learnable exposure time parameters to interpolate between Gaussians at integer timestamps, avoiding explicit motion trajectory modeling. The method introduces blur-aware variable canonical Gaussians that select sharper frames as references and applies exposure, multi-frame, and multi-resolution consistency regularizations to prevent trivial solutions. Achieves 28.88 PSNR on 720x1080 resolution while maintaining real-time rendering.

### Key Improvements
1. **Quality (PSNR)**: 27.49 dB (Shape-of-Motion) → 28.88 dB on 720x1080 (+1.39 dB)
2. **Motion Blur Handling**: First 4D Gaussian Splatting method for blurry videos
3. **Rendering Speed**: 96 FPS at 720x1080 (vs 12 FPS for DybluRF)
4. **Exposure Time Estimation**: Transforms continuous dynamics to simple parameter learning
5. **Multi-task Application**: Deblurring, frame interpolation, video stabilization

### How It Works

**Core Algorithm - Exposure Time Estimation**
```python
def exposure_time_interpolation(canonical_gaussians, timestamp, exposure_params):
    """
    Mathematical formulation: D_t,i = w_t,i ⊙ D_t + (1-w_t,i) ⊙ D_neighbor
    Key difference from baseline: Parameter-based interpolation vs explicit motion modeling
    """
    # Get learnable exposure parameters for this timestamp
    w_start = exposure_params[f"{timestamp}_start"]  # w_t,1
    w_end = exposure_params[f"{timestamp}_end"]      # w_t,N
    
    # Assume uniform motion: |w_start| = |w_end| = w_t/2
    w_t = 2 * max(abs(w_start), abs(w_end))
    
    # Interpolate exposure weights for N sub-timesteps
    continuous_gaussians = []
    for i in range(1, N+1):
        # Linear interpolation of exposure weights
        alpha = (i-1) / (N-1)
        w_i = (1 - alpha) * (-w_t/2) + alpha * (+w_t/2)
        
        # Choose neighbor timestamp based on position
        if i <= N/2:
            neighbor_t = timestamp - 1
            neighbor_weight = w_i
            current_weight = 1 - w_i
        else:
            neighbor_t = timestamp + 1
            neighbor_weight = w_i
            current_weight = 1 - w_i
        
        # Interpolate Gaussians
        current_gaussians = canonical_gaussians[timestamp]
        neighbor_gaussians = canonical_gaussians[neighbor_t]
        
        interpolated = (current_weight * current_gaussians + 
                       neighbor_weight * neighbor_gaussians)
        continuous_gaussians.append(interpolated)
    
    return continuous_gaussians
```

**Blur-Aware Variable Canonical Selection**
```python
def select_variable_canonical_gaussians(video_frames, masks, num_segments=5):
    """
    Selects canonical Gaussians from sharpest frames in each segment
    """
    T = len(video_frames)
    segment_size = T // num_segments
    canonical_indices = []
    
    for seg in range(num_segments):
        start_idx = seg * segment_size
        end_idx = min((seg + 1) * segment_size, T)
        
        # Compute blur levels for dynamic regions
        blur_levels = []
        for t in range(start_idx, end_idx):
            # Laplacian variance as sharpness measure
            dynamic_region = video_frames[t] * masks[t]
            laplacian = cv2.Laplacian(dynamic_region, cv2.CV_64F)
            blur_level = np.var(laplacian)
            blur_levels.append(blur_level)
        
        # Select sharpest frame in this segment
        sharpest_idx = start_idx + np.argmax(blur_levels)
        canonical_indices.append(sharpest_idx)
    
    return canonical_indices
```

**Mathematical Foundation**
- Motion blur formation: `B(u,v) = (1/N) * Σ I_i(u,v)` (averaging N sharp images)
- Exposure interpolation: Linear interpolation in exposure parameter space
- Blur metric: `b_t = Σ (ΔB_t(u,v) - ΔB̄_t)²` where ΔB_t is Laplacian
- Regularization: Exposure L_e, multi-frame L_mfc, multi-resolution L_mrc

**Critical Implementation Details**
- Camera motion predictor: Tiny MLP for stable SE(3) interpolation
- Gaussian deformation: Rigid transformation matrices (Shape-of-Motion style)
- Exposure discretization: N=11 sub-timesteps per frame
- Blur-aware segments: L=5 segments with H=3 frame neighborhood search

### Algorithm Steps
1. **Preprocessing**: Detect dynamic regions, compute blur levels per frame
2. **Variable Canonical Selection**: Choose sharpest frames in each segment
3. **Camera Motion Pre-training**: Train MLP predictor on static regions
4. **Static Gaussian Initialization**: Optimize static scene representation
5. **Joint Optimization**: Learn exposure parameters + canonical Gaussians + deformation
6. **Continuous Rendering**: Interpolate using exposure parameters for any timestamp

### Implementation Details
- Architecture: Variable canonical Gaussians + Camera motion MLP + Exposure parameters
- Camera predictor: 3-layer MLP (64 hidden units) for SE(3) interpolation
- Exposure parameters: 2 per frame (start/end) with uniform motion assumption
- Loss weights: λ_e=0.1, λ_mfc=2.0, λ_mrc=1.0, β=0.2 (SSIM weight)
- Training: 200 epochs pre-training + 200 epochs joint optimization
- Learning rates: 5e-4→1e-5 (camera), 1e-1→1e-5 (exposure), standard for others
- Memory: Comparable to Shape-of-Motion with added exposure parameters

### Integration Notes
```python
# Modified training with motion blur simulation:
def train_deblur4dgs(blurry_video, dynamic_masks):
    # Phase 1: Variable canonical selection
    canonical_indices = select_variable_canonical_gaussians(
        blurry_video, dynamic_masks
    )
    
    # Phase 2: Camera motion pre-training
    camera_predictor = CameraMotionMLP()
    static_gaussians = train_static_scene(
        blurry_video, dynamic_masks, camera_predictor
    )
    
    # Phase 3: Joint optimization
    canonical_gaussians = {idx: initialize_gaussians(blurry_video[idx]) 
                          for idx in canonical_indices}
    exposure_params = {f"{t}_start": 0.0, f"{t}_end": 0.0 
                      for t in range(len(blurry_video))}
    
    for epoch in range(200):
        for t, blurry_frame in enumerate(blurry_video):
            # Generate continuous Gaussians within exposure
            continuous_gaussians = exposure_time_interpolation(
                canonical_gaussians, t, exposure_params
            )
            
            # Render sharp sub-images
            sharp_images = []
            for i, gaussians_i in enumerate(continuous_gaussians):
                camera_pose_i = camera_predictor.interpolate(
                    t, i, N=len(continuous_gaussians)
                )
                sharp_img = render_gaussian_splatting(
                    gaussians_i, static_gaussians, camera_pose_i
                )
                sharp_images.append(sharp_img)
            
            # Average to simulate blur
            synthetic_blur = torch.mean(torch.stack(sharp_images), dim=0)
            
            # Losses
            loss = reconstruction_loss(synthetic_blur, blurry_frame)
            loss += exposure_regularization(exposure_params[f"{t}_start"], 
                                          exposure_params[f"{t}_end"])
            loss += multi_frame_consistency(sharp_images, optical_flow_net)
            loss += multi_resolution_consistency(sharp_images, low_res_model)
            
            loss.backward()
```

Key modifications from standard 4D Gaussian Splatting:
- Replace single canonical with variable segment-based selection
- Add learnable exposure time parameters for interpolation
- Include motion blur simulation in training loop
- Apply multi-scale consistency from low-resolution models

### Speed/Memory Tradeoffs
- Training: 400 epochs total (200 pre-train + 200 joint), comparable to baselines
- Inference: 96 FPS at 720x1080 (real-time), 8x faster than DybluRF
- Memory: Base 4DGS + exposure parameters (2T floats) + camera MLP (~1KB)
- Quality vs Speed: Can reduce N (sub-timesteps) for faster training/rendering
- Multi-resolution: Low-res model helps but requires additional training
- Segment count: More segments improve quality but increase canonical Gaussian storage
