# 📚 Gaussian Splatting Papers Index
## TODO
- [ ] (Currently no pending papers to review)
---
## Reviewed Papers
### 1. **Gaussian Opacity Fields (GOF)**
- **arXiv**: https://arxiv.org/abs/2404.10772
- **Project**: https://niujinshuchong.github.io/gaussian-opacity-fields/
- **Summary**: Neural opacity fields for improved geometry extraction from 3D Gaussians
- **Key Innovation**: Direct surface extraction without post-processing
- **Improvements**: +2-3 dB PSNR for geometry tasks
- **Details**: See [TODO_COMPLETED_DEEP_REVIEW_1.md](TODO_COMPLETED_DEEP_REVIEW_1.md#1-gaussian-opacity-fields-gof)
### 2. **Deblurring 3D Gaussian Splatting**
- **arXiv**: https://arxiv.org/abs/2401.00834
- **Project**: https://benhenryl.github.io/Deblurring-3D-Gaussian-Splatting/
- **Summary**: MLP-based blur compensation for sharp reconstruction from blurry images
- **Key Innovation**: Exposure-aware training with covariance adjustment
- **Improvements**: +1.5 dB PSNR on blurry inputs
- **Details**: See [TODO_COMPLETED_DEEP_REVIEW_1.md](TODO_COMPLETED_DEEP_REVIEW_1.md#2-deblurring-3d-gaussian-splatting)
### 3. **Deblur Gaussian Splatting SLAM**
- **arXiv**: https://arxiv.org/pdf/2503.12572
- **Summary**: Real-time SLAM system with motion blur handling
- **Key Innovation**: Sub-frame trajectory modeling for blur compensation
- **Improvements**: 30 FPS tracking with blur handling
- **Details**: See [TODO_COMPLETED_DEEP_REVIEW_1.md](TODO_COMPLETED_DEEP_REVIEW_1.md#3-deblur-gaussian-splatting-slam)
### 4. **Mip-Splatting** ⭐ (CVPR 2024 Best Paper)
- **arXiv**: https://arxiv.org/abs/2405.02468
- **Project**: https://github.com/autonomousvision/mip-splatting
- **Summary**: Anti-aliasing for 3D Gaussian Splatting
- **Key Innovation**: 3D smoothing filter + 2D Mip filter
- **Improvements**: +0.5-1.0 dB PSNR, <5% performance overhead
- **Details**: See [TODO_COMPLETED_DEEP_REVIEW_1.md](TODO_COMPLETED_DEEP_REVIEW_1.md#4-mip-splatting)
### 5. **Wild Gaussians**
- **arXiv**: https://arxiv.org/abs/2407.08447
- **Project**: https://wild-gaussians.github.io/
- **Summary**: Robust 3DGS for uncontrolled capture conditions
- **Key Innovation**: DINO-based appearance modeling
- **Improvements**: Handles 50% lighting variation
- **Details**: See [TODO_COMPLETED_DEEP_REVIEW_1.md](TODO_COMPLETED_DEEP_REVIEW_1.md#5-wild-gaussians)
### 6. **Spacetime Gaussians** 🚀 (Baseline)
- **arXiv**: https://arxiv.org/abs/2312.16812
- **Project**: https://oppo-us-research.github.io/SpacetimeGaussians-website/
- **Summary**: Dynamic 3D Gaussians with temporal modeling
- **Key Innovation**: Polynomial motion trajectories
- **Improvements**: 8K @ 60 FPS claimed (4K @ 30 FPS realistic)
- **Details**: See [TODO_COMPLETED_DEEP_REVIEW_1.md](TODO_COMPLETED_DEEP_REVIEW_1.md#6-spacetime-gaussians-baseline)
### 7. **MoDecGS**
- **arXiv**: https://arxiv.org/abs/2501.03714
- **Project**: https://kaist-viclab.github.io/MoDecGS-site/
- **Summary**: Memory-efficient dynamic Gaussians
- **Key Innovation**: 70% compression via hierarchical decomposition
- **Details**: See [TODO_COMPLETED_DEEP_REVIEW_2.md](TODO_COMPLETED_DEEP_REVIEW_2.md#paper-7-modecgs)
### 8. **3D-4DGS**
- **arXiv**: https://arxiv.org/abs/2411.11324
- **Project**: https://ohsngjun.github.io/3D-4DGS/
- **Summary**: Adaptive static/dynamic separation
- **Key Innovation**: Automatic 4D→3D conversion for efficiency
- **Improvements**: 30-60% memory saving
- **Details**: See [TODO_COMPLETED_DEEP_REVIEW_2.md](TODO_COMPLETED_DEEP_REVIEW_2.md#paper-8-3d-4dgs)
### 9. **ReconFusion** ❌ (NeRF Method)
- **arXiv**: https://arxiv.org/abs/2312.02981
- **Project**: https://reconfusion.github.io/
- **Note**: Not a Gaussian Splatting method
### 10. **LM-Gaussian**
- **arXiv**: https://arxiv.org/abs/2409.03456
- **Project**: https://runningneverstop.github.io/lm-gaussian.github.io/
- **Summary**: Few-shot 3DGS with vision model priors
- **Key Innovation**: 3-view reconstruction capability
- **Details**: See [TODO_COMPLETED_DEEP_REVIEW_2.md](TODO_COMPLETED_DEEP_REVIEW_2.md#paper-10-lm-gaussian)
### 11. **Gaussian Tracer**
- **arXiv**: https://arxiv.org/abs/2407.07090
- **Project**: https://gaussiantracer.github.io/
- **Summary**: Ray tracing for 3D Gaussians
- **Key Innovation**: Enables advanced camera models and effects
- **Improvements**: 55-190 FPS with ray tracing
- **Details**: See [TODO_COMPLETED_DEEP_REVIEW_2.md](TODO_COMPLETED_DEEP_REVIEW_2.md#paper-11-gaussian-tracer)
### 12. **Ref-Gaussian**
- **arXiv**: https://arxiv.org/abs/2412.19282
- **Project**: https://fudan-zvg.github.io/ref-gaussian/
- **Summary**: Inter-reflections for Gaussian Splatting
- **Key Innovation**: Physically-based deferred rendering
- **Improvements**: +2-3 dB on reflective scenes
- **Details**: See [TODO_COMPLETED_DEEP_REVIEW_2.md](TODO_COMPLETED_DEEP_REVIEW_2.md#paper-12-ref-gaussian)
### 13. **GaussianShader**
- **arXiv**: https://arxiv.org/abs/2311.17977
- **Project**: https://asparagus15.github.io/GaussianShader.github.io/
- **Summary**: Lightweight shading for reflective surfaces
- **Key Innovation**: Normal estimation from Gaussian geometry
- **Improvements**: +1.57 dB PSNR, 5-10% overhead
- **Details**: See [TODO_COMPLETED_DEEP_REVIEW_3.md](TODO_COMPLETED_DEEP_REVIEW_3.md#paper-13-gaussianshader)
### 14. **3DGS-DR**
- **arXiv**: https://arxiv.org/abs/2411.02482
- **Project**: https://gapszju.github.io/3DGS-DR/
- **Summary**: Deferred reflection for efficient rendering
- **Key Innovation**: Screen-space reflection computation
- **Improvements**: +2.3 dB PSNR, 30% faster convergence
- **Details**: See [TODO_COMPLETED_DEEP_REVIEW_3.md](TODO_COMPLETED_DEEP_REVIEW_3.md#paper-14-3dgs-dr-deferred-reflection)
### 15. **IRGS**
- **arXiv**: https://arxiv.org/abs/2411.16758
- **Project**: https://fudan-zvg.github.io/IRGS/
- **Summary**: Full rendering equation for Gaussians
- **Key Innovation**: Monte Carlo integration for global illumination
- **Improvements**: +3.1 dB PSNR but only 2-5 FPS
- **Details**: See [TODO_COMPLETED_DEEP_REVIEW_3.md](TODO_COMPLETED_DEEP_REVIEW_3.md#paper-15-irgs-inter-reflective-gaussian-splatting)
### 16. **RaySplatting**
- **Project**: https://github.com/KByrski/RaySplatting
- **Summary**: RTX-accelerated ray tracing viewer
- **Key Innovation**: Direct ray-Gaussian intersection
- **Improvements**: 10-20 FPS with RTX features
- **Details**: See [TODO_COMPLETED_DEEP_REVIEW_3.md](TODO_COMPLETED_DEEP_REVIEW_3.md#paper-16-raysplatting)
### 17. **3DGRUT** ⭐ (NVIDIA)
- **arXiv**: https://arxiv.org/abs/2411.04637
- **Project**: https://github.com/nv-tlabs/3dgrut
- **Summary**: Production framework with advanced camera support
- **Key Innovation**: Unscented Transform for complex projections
- **Improvements**: 347 FPS rasterization performance
- **Details**: See [TODO_COMPLETED_DEEP_REVIEW_3.md](TODO_COMPLETED_DEEP_REVIEW_3.md#paper-17-3dgrut)
### 18. **WildGS-SLAM**
- **Project**: https://wildgs-slam.github.io/
- **Summary**: Uncertainty-aware SLAM for dynamic environments with Gaussian representation
- **Key Innovation**: Visual-inertial odometry with dynamic object handling
- **Improvements**: PSNR 20.59 vs 17.23 baseline, real-time performance
- **Details**: See [TODO_COMPLETED_DEEP_REVIEW_4.md](TODO_COMPLETED_DEEP_REVIEW_4.md#paper-18-wildgs-slam)
### 19. **FreeTimeGS**
- **Project**: http://zju3dv.github.io/freetimegs/
- **Summary**: Free-moving Gaussians in space-time for efficient dynamic modeling
- **Key Innovation**: Gaussian motion functions with temporal opacity
- **Improvements**: 467 FPS at 1080p, 4.5x faster rendering
- **Details**: See [TODO_COMPLETED_DEEP_REVIEW_4.md](TODO_COMPLETED_DEEP_REVIEW_4.md#paper-19-freetimegs)
### 20. **Deformable Radial Kernel Splatting (DRK)**
- **arXiv**: https://arxiv.org/abs/2412.11752
- **Summary**: Radial kernel primitives overcoming Gaussian shape limitations
- **Key Innovation**: Superposition of learnable radial basis functions
- **Improvements**: 60-80% primitive reduction, maintained quality
- **Details**: See [TODO_COMPLETED_DEEP_REVIEW_4.md](TODO_COMPLETED_DEEP_REVIEW_4.md#paper-20-deformable-radial-kernel-splatting-drk)
### 21. **CoCoGaussian**
- **Project**: https://jho-yonsei.github.io/CoCoGaussian/
- **Summary**: Circle of Confusion modeling for defocused image handling
- **Key Innovation**: Depth-adaptive blur kernel with focus distance modeling
- **Improvements**: 0.04 LPIPS improvement on defocused inputs
- **Details**: See [TODO_COMPLETED_DEEP_REVIEW_4.md](TODO_COMPLETED_DEEP_REVIEW_4.md#paper-21-cocogaussian)
### 22. **DiET-GS**
- **Project**: https://diet-gs.github.io/
- **Summary**: Event camera integration for motion deblurring
- **Key Innovation**: Event Double Integral with diffusion priors
- **Improvements**: Superior on high-speed motion scenarios
- **Details**: See [TODO_COMPLETED_DEEP_REVIEW_4.md](TODO_COMPLETED_DEEP_REVIEW_4.md#paper-22-diet-gs)
### 23. **BARD-GS**
- **Project**: https://vulab-ai.github.io/BARD-GS/
- **Summary**: Motion blur-aware dynamic reconstruction
- **Key Innovation**: Two-stage blur decomposition (camera + object motion)
- **Improvements**: Handles complex motion blur in dynamic scenes
- **Details**: See [TODO_COMPLETED_DEEP_REVIEW_5.md](TODO_COMPLETED_DEEP_REVIEW_5.md#paper-23-bard-gs)
### 24. **vkraygs**
- **Project**: https://github.com/facebookresearch/vkraygs
- **Summary**: Vulkan-based hardware-accelerated ray-based renderer
- **Key Innovation**: GPU ray tracing with MIP-aware adaptive rendering
- **Improvements**: Hardware acceleration path for Gaussian rendering
- **Details**: See [TODO_COMPLETED_DEEP_REVIEW_5.md](TODO_COMPLETED_DEEP_REVIEW_5.md#paper-24-vkraygs)
### 25. **4D-Fly** (CVPR 2025)
- **Paper**: CVPR 2025 paper
- **Summary**: Fast 4D reconstruction from single monocular video
- **Key Innovation**: Single camera input with rapid reconstruction
- **Improvements**: Fast reconstruction (specific metrics not available)
- **Details**: See [TODO_COMPLETED_DEEP_REVIEW_5.md](TODO_COMPLETED_DEEP_REVIEW_5.md#paper-25-4d-fly)
### 26. **DASS**
- **Project**: https://www.liuzhening.top/DASS
- **Summary**: Dynamic spatial streaming for efficient training
- **Key Innovation**: Three-stage pipeline with selective inheritance
- **Improvements**: 20% faster training, streaming capability
- **Details**: See [TODO_COMPLETED_DEEP_REVIEW_5.md](TODO_COMPLETED_DEEP_REVIEW_5.md#paper-26-dass)
### 27. **4D-LangSplat**
- **Project**: https://4d-langsplat.github.io/
- **Summary**: Multimodal semantic understanding for dynamic scenes
- **Key Innovation**: MLLM-generated captions for semantic features
- **Improvements**: Open-vocabulary queries on dynamic scenes
- **Details**: See [TODO_COMPLETED_DEEP_REVIEW_5.md](TODO_COMPLETED_DEEP_REVIEW_5.md#paper-27-4d-langsplat)
---
## 📊 Technical Summary
### Quality Improvements Range:
- PSNR gains: +0.5 to +3.1 dB depending on method
- SSIM improvements: +0.02 to +0.05
- LPIPS reductions: 10-30% better perceptual quality
### Performance Metrics Range:
- Training: 30 minutes to 3 hours typical
- Rendering: 2 FPS (IRGS) to 347 FPS (3DGRUT)
- Memory: 70-85% reduction possible with compression methods
### By Category:
- **Quality Enhancement**: Mip-Splatting, GOF, GaussianShader, 3DGS-DR, IRGS, Ref-Gaussian, DRK
- **Dynamic Scenes**: Spacetime Gaussians, MoDecGS, 3D-4DGS, FreeTimeGS, DASS, BARD-GS
- **Robustness**: Wild Gaussians, Deblurring methods, WildGS-SLAM, CoCoGaussian, DiET-GS
- **Ray Tracing**: Gaussian Tracer, RaySplatting, 3DGRUT, vkraygs
- **Sparse Views**: LM-Gaussian, 4D-Fly
- **Semantic/Multimodal**: 4D-LangSplat
### Production Ready:
- ✅ Mip-Splatting
- ✅ 3DGRUT
- ✅ Basic 3DGS + simple extensions
### Research/Experimental:
- ⚠️ Spacetime Gaussians (baseline but memory intensive)
- ⚠️ Most other methods (see deep reviews for tradeoffs)
---
## 💻 Implementation Examples
### Base Setup
```python
# Spacetime Gaussians base
temporal_gaussians = SpacetimeGaussians(poly_degree=3)
# Add Mip-Splatting for quality
temporal_gaussians.add_filter(MipSplatting(
    lambda_3d=0.001,
    sigma_max=10.0  # Critical: missing from paper
))
```
### Rendering Enhancement
```python
# Deferred rendering option
renderer = DeferredRenderer(temporal_gaussians)
renderer.add_pass(NormalPass())
renderer.add_pass(ReflectionPass(strength=0.3))
# OR lightweight shading
shader = GaussianShader(
    normal_residual_weight=0.1,
    reflection_mlp_dims=[32, 64, 64, 1]
)
```
### Memory Optimization
```python
# Compression
compressor = MoDecGS(
    global_deform=HexplaneEncoder(),
    temporal_segments=8
)
# Adaptive dimensions
optimizer = Adaptive4Dto3D(
    temporal_threshold=0.05,
    check_interval=1000
)
```
### Appearance Modeling
```python
# For varying conditions
appearance = WildGaussians(
    gaussian_embed_dim=32,
    image_embed_dim=16,
    dino_threshold=0.7
)
```
---
## 📊 Realistic Performance Expectations
For iPhone multi-camera capture (4 cameras, 1920x1080, 5 minutes):
- **Training**: 2-3 hours on H100
- **Memory**: 16-24 GB with optimizations
- **Rendering**: 35-40 FPS with features
- **Quality**: +5-7 dB over static 3DGS
---
## 🔧 Integration Considerations
- **Memory bandwidth** is often the bottleneck
- **Profile everything** before committing to a method
- **Default hyperparameters** rarely optimal
- Each method has **quality vs speed vs memory tradeoffs**
---
## Downloaded Papers
Available in `research/papers/`:
- spacetime_gaussians.pdf
- mip_splatting.pdf
- wild_gaussians.pdf
- gaussian_opacity_fields.pdf
