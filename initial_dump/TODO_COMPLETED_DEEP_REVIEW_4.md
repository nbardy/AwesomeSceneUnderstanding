# Technical Extraction Report - Papers 18-22
## Advanced Gaussian Splatting Techniques for Dynamic Scenes and Special Conditions

---

## Paper 18: WildGS-SLAM - Monocular Gaussian Splatting SLAM in Dynamic Environments

**Summary**: WildGS-SLAM introduces uncertainty-aware geometric mapping for robust SLAM in dynamic environments, using DINOv2 features and a shallow MLP to generate per-pixel uncertainty maps that guide both tracking and mapping while filtering out dynamic objects.

**Key Improvements** (numbered list with specific metrics):
1. PSNR improvement: 20.59 vs 17.23 (Splat-SLAM baseline)
2. Training speed: ~1 hour for full sequence
3. Rendering speed: Real-time performance maintained
4. Memory usage: Downsampled to 1/3 resolution for efficiency
5. Quality metrics: SSIM 0.783 vs 0.699, LPIPS 0.209 vs 0.346

**How It Works** (detailed math and algorithms):
- Uncertainty Loss Function:
  ```
  L_uncertainty = λ_ssim * L_SSIM + λ_reg * L_reg + λ_depth * L_depth
  L_depth = ||D_rendered - D_metric||_1
  ```
- Dense Bundle Adjustment with uncertainty weighting:
  ```
  E_tracking = Σ_pixels w_uncertainty * (I_observed - I_rendered)²
  ```
- Algorithm steps:
  1. Extract DINOv2 features (1/14 resolution)
  2. Pass through shallow MLP to generate uncertainty map
  3. Apply uncertainty weights in DBA optimization
  4. Update 3D Gaussian map with weighted rendering loss

**Implementation Details**:
- Key hyperparameters:
  - DINOv2 feature resolution: 1/14 of original
  - Mapping resolution: 1/3 of original
  - Metric depth from Metric3D V2
- Data structures:
  - Uncertainty MLP: 3-layer network
  - Per-pixel uncertainty maps
  - 3D Gaussian primitives with standard attributes
- Computational complexity: O(n) for uncertainty computation per frame

**Integration with Spacetime Gaussians**:
- Modify tracking module to include uncertainty weighting
- Add uncertainty MLP parallel to existing feature extraction
- Update rendering loss computation with uncertainty masks
- Compatible with temporal opacity functions

**Speed/Memory Tradeoffs**:
- Training time impact: +10-15 minutes for uncertainty MLP
- Rendering speed impact: <5% slower due to uncertainty computation
- Memory requirements: +200MB for DINOv2 features
- Quality vs speed: Can disable uncertainty for 20% speedup

---

## Paper 19: FreeTimeGS - Free Gaussian Primitives at Anytime and Anywhere

**Summary**: FreeTimeGS allows Gaussian primitives to appear at arbitrary times and locations with motion functions and temporal opacity, achieving 467 FPS rendering while handling complex dynamic scenes with large motions.

**Key Improvements** (numbered list with specific metrics):
1. PSNR improvement: +2-3 dB on dynamic regions
2. Training speed: 30,000 iterations (~1 hour on RTX 4090)
3. Rendering speed: 467 FPS at 1080p (vs ~100 FPS for 4DGS)
4. Memory usage: Reduced temporal redundancy by 40%
5. Quality metrics: Superior DSSIM and LPIPS in all benchmarks

**How It Works** (detailed math and algorithms):
- Motion Function:
  ```
  μ'_x(t) = μ_x + v * (t - μ_t)
  where v ∈ ℝ³ is velocity, μ_x is initial position, μ_t is initial time
  ```
- Temporal Opacity Function:
  ```
  α(t) = σ * exp(-||t - μ_t||² / (2s²))
  where s is duration, σ is initial opacity
  ```
- 4D Regularization Loss:
  ```
  L_reg = λ * Σ_i max(0, α_i - α_threshold)
  ```
- Algorithm:
  1. Initialize Gaussians with position, time, velocity
  2. Move primitives according to motion function
  3. Apply temporal opacity modulation
  4. Optimize with rendering + regularization loss

**Implementation Details**:
- Key hyperparameters:
  - 4D regularization weight: 1e-2
  - Image/SSIM/perceptual loss weights: 0.8/0.2/0.01
  - Gradient/opacity sampling weights: 0.5/0.5
- Data structures:
  - Extended Gaussian attributes: {μ_x, μ_t, s, v, S, R, σ, SH}
  - Periodic relocation sampling scores
- Adam optimizer with 3DGS settings

**Integration with Spacetime Gaussians**:
- Replace static Gaussians with motion-enabled primitives
- Add velocity and duration parameters to each Gaussian
- Modify rendering to include temporal movement
- Compatible with existing optimization pipeline

**Speed/Memory Tradeoffs**:
- Training time impact: +30 minutes for motion optimization
- Rendering speed impact: 4.5x faster than baseline
- Memory requirements: +8 bytes per Gaussian (velocity)
- Quality vs speed: Adjustable temporal resolution

---

## Paper 20: Deformable Radial Kernel Splatting

**Summary**: Introduces learnable radial bases with adjustable angles and scales to overcome Gaussian symmetry limitations, dramatically reducing primitive count while maintaining quality through accurate ray-primitive intersections and efficient culling.

**Key Improvements** (numbered list with specific metrics):
1. PSNR improvement: Maintains quality with 60-80% fewer primitives
2. Training speed: Similar to standard 3DGS
3. Rendering speed: Slightly slower due to complex intersections
4. Memory usage: 60-80% reduction in primitive count
5. Quality metrics: Better edge sharpness and boundary definition

**How It Works** (detailed math and algorithms):
- Deformable Radial Kernel:
  ```
  K(x) = Σ_i w_i * B_i(||x - c||, θ, s)
  where B_i are learnable radial bases, θ is angle, s is scale
  ```
- Ray-Primitive Intersection:
  ```
  Analytical solution for non-Gaussian kernels
  Adaptive sampling based on kernel complexity
  ```
- Kernel Culling Strategy:
  ```
  Hierarchical spatial culling
  View-dependent primitive selection
  ```

**Implementation Details**:
- Key hyperparameters:
  - Number of radial bases: 8-16
  - Angle range: [-π, π]
  - Scale factors: learnable per primitive
- Data structures:
  - Radial kernel parameters per primitive
  - Spatial acceleration structure
- Computational complexity: O(k*n) where k is bases count

**Integration with Spacetime Gaussians**:
- Replace Gaussian kernels with deformable radial kernels
- Extend temporal functions to work with new primitives
- Modify ray marching for accurate intersections
- May require custom CUDA kernels

**Speed/Memory Tradeoffs**:
- Training time impact: +20% due to complex optimization
- Rendering speed impact: 15-20% slower
- Memory requirements: -60% due to fewer primitives
- Quality vs speed: Tunable kernel complexity

---

## Paper 21: CoCoGaussian - Circle of Confusion for Defocused Images

**Summary**: Models defocus blur using Circle of Confusion as Gaussians, enabling reconstruction from blurry images with learnable aperture and adaptive scaling for reflective/refractive surfaces, achieving best LPIPS scores across benchmarks.

**Key Improvements** (numbered list with specific metrics):
1. PSNR improvement: Dataset-dependent, strong on high-res
2. Training speed: Standard 3DGS training time
3. Rendering speed: Real-time maintained
4. Memory usage: +200k additional points after 2.5k iterations
5. Quality metrics: LPIPS reduction of 0.04 vs BAGS baseline

**How It Works** (detailed math and algorithms):
- CoC Diameter Calculation:
  ```
  d_CoC = |A * f * (S - D) / (D * (S - f))|
  where A is aperture, f is focal length, S is focus distance, D is depth
  ```
- CoC Gaussian Generation:
  ```
  For each base Gaussian, generate M CoC Gaussians:
  G_coc_i = G_base + r * e^(i*2π/M) * n_perp
  where n_perp is perpendicular to viewing direction
  ```
- Adaptive Scaling Factor:
  ```
  d'_CoC = d_CoC * λ_scale (learnable)
  Constrained to [0.5, 2.0] for stability
  ```

**Implementation Details**:
- Key hyperparameters:
  - M (CoC Gaussians per base): 8-16
  - Depth pruning threshold: adaptive
  - Scaling factor limits: [0.5, 2.0]
- Data structures:
  - CoC Gaussian sets
  - Learnable aperture parameters
  - Per-Gaussian scaling factors
- Works with RTX 3090/V100 GPUs

**Integration with Spacetime Gaussians**:
- Add CoC generation after base Gaussian creation
- Extend temporal functions to CoC sets
- Modify rendering to handle defocus
- Compatible with motion blur models

**Speed/Memory Tradeoffs**:
- Training time impact: +15% for CoC generation
- Rendering speed impact: Negligible with optimization
- Memory requirements: +M× Gaussians (8-16×)
- Quality vs speed: Adjustable M parameter

---

## Paper 22: DiET-GS - Event Camera Motion Deblurring

**Summary**: Two-stage framework using event streams and diffusion priors for motion deblurring, introducing Event Double Integral constraints to achieve accurate colors and fine details while maintaining real-time 3DGS rendering capabilities.

**Key Improvements** (numbered list with specific metrics):
1. PSNR improvement: Outperforms all baselines
2. Training speed: Two-stage process, ~2 hours total
3. Rendering speed: Real-time maintained (3DGS base)
4. Memory usage: +Event buffer storage
5. Quality metrics: Superior in MUSIQ, CLIP-IQA scores

**How It Works** (detailed math and algorithms):
- Event Double Integral Model:
  ```
  B = ∫∫ E(t) dt dt + L
  where B is blurry image, E(t) is event stream, L is latent sharp image
  ```
- Stage 1 Loss (DiET-GS):
  ```
  L1 = λ_color*L_C + λ_detail*L_I + λ_edi*L_EDI
  L_EDI = ||B - (∫∫ E dt dt + R(θ))||²
  ```
- Stage 2 Enhancement (DiET-GS++):
  ```
  I_enhanced = R(θ) + f_g(R(θ), D_prior)
  where f_g are learnable parameters, D_prior is diffusion prior
  ```

**Implementation Details**:
- Key hyperparameters:
  - Event accumulation window: 50ms
  - Diffusion prior strength: 0.1-0.3
  - Stage transition: 15k iterations
- Data structures:
  - Event buffer with timestamps
  - Diffusion feature maps
  - Two-stage optimizer states
- Requires event camera data

**Integration with Spacetime Gaussians**:
- Add event accumulation module
- Integrate EDI constraints in loss
- Two-stage training pipeline
- Compatible with temporal representations

**Speed/Memory Tradeoffs**:
- Training time impact: +1 hour for two stages
- Rendering speed impact: None (preprocessing)
- Memory requirements: +2GB for event buffer
- Quality vs speed: Stage 2 optional for speed

---

## Summary of Integration Strategies

### Combined Architecture Recommendations:
1. **Base**: Use FreeTimeGS motion primitives for temporal flexibility
2. **Uncertainty**: Add WildGS-SLAM uncertainty for dynamic filtering
3. **Primitives**: Consider DRK for complex geometries (selective)
4. **Blur Handling**: Integrate CoCoGaussian for defocus, DiET-GS for motion blur
5. **Optimization Order**: Motion → Uncertainty → Blur → Enhancement

### Performance Projections for Combined System:
- Training: 3-4 hours for full pipeline
- Rendering: 200-300 FPS (with all features)
- Memory: +4-5GB over base 3DGS
- Quality: Significant improvements in all challenging scenarios

### Implementation Priority:
1. FreeTimeGS (foundation for dynamics)
2. WildGS-SLAM uncertainty (robustness)
3. CoCoGaussian or DiET-GS (based on blur type)
4. DRK (optional for specific scenes)