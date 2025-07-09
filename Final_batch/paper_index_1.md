# Paper Index 1 - 3D Gaussian Splatting & Core Techniques

## Gaussian Opacity Fields (GOF)

**arXiv:** https://arxiv.org/abs/2404.10772  
**GitHub:** N/A  
**Project Page:** https://niujinshuchong.github.io/gaussian-opacity-fields/

**Abstract:** Neural opacity fields for improved geometry extraction from 3D Gaussians with direct surface extraction without post-processing

**Tags:** 3d gaussian splatting, opacity fields, geometry extraction, neural rendering

---

## Mip-Splatting

**arXiv:** https://arxiv.org/abs/2405.02468  
**GitHub:** https://github.com/autonomousvision/mip-splatting  
**Project Page:** N/A

**Abstract:** Anti-aliasing for 3D Gaussian Splatting with 3D smoothing filter + 2D Mip filter

**Tags:** 3d gaussian splatting, anti-aliasing, cvpr best paper, real-time rendering

---

## Wild Gaussians

**arXiv:** https://arxiv.org/abs/2407.08447  
**GitHub:** N/A  
**Project Page:** https://wild-gaussians.github.io/

**Abstract:** Robust 3DGS for uncontrolled capture conditions with DINO-based appearance modeling

**Tags:** 3d gaussian splatting, robustness, wild conditions, appearance modeling

---

## PolyGS

**arXiv:** https://arxiv.org/abs/2312.09479  
**GitHub:** N/A  
**Project Page:** https://research.nvidia.com/labs/toronto-ai/polygs/

**Abstract:** Mesh extraction from Gaussian splats using polygonal decomposition

**Tags:** 3d gaussian splatting, mesh extraction, polygonal representation

---

## Neural-GS

**arXiv:** https://arxiv.org/abs/2312.05264  
**GitHub:** N/A  
**Project Page:** https://neural-gs.github.io/

**Abstract:** Neural rendering enhancement for Gaussian splatting

**Tags:** 3d gaussian splatting, neural rendering, quality enhancement

---

## Scaffold-GS

**arXiv:** https://arxiv.org/abs/2312.00109  
**GitHub:** N/A  
**Project Page:** https://city-super.github.io/scaffold-gs/

**Abstract:** Structured anchoring for better 3D scene representation

**Tags:** 3d gaussian splatting, structured representation, scene anchoring

---

## GS-IR (Inverse Rendering)

**arXiv:** https://arxiv.org/abs/2311.16473  
**GitHub:** N/A  
**Project Page:** https://gs-ir.github.io/

**Abstract:** Inverse rendering with Gaussian splatting for material and lighting estimation

**Tags:** 3d gaussian splatting, inverse rendering, material estimation

---

## RT-GS

**arXiv:** https://arxiv.org/abs/2312.03307  
**GitHub:** N/A  
**Project Page:** https://zhangganlin.github.io/rt-gs/

**Abstract:** Real-time Gaussian splatting with adaptive LOD

**Tags:** 3d gaussian splatting, real-time rendering, level of detail

---

## Compact3D

**arXiv:** https://arxiv.org/abs/2311.18159  
**GitHub:** N/A  
**Project Page:** https://maincold2.github.io/c3d/

**Abstract:** Compressing radiance fields for deployment

**Tags:** 3d gaussian splatting, compression, model deployment

---

## LightGaussian

**arXiv:** https://arxiv.org/abs/2311.17518  
**GitHub:** N/A  
**Project Page:** https://lightgaussian.github.io/

**Abstract:** Lightweight Gaussian splatting for mobile devices

**Tags:** 3d gaussian splatting, mobile deployment, lightweight models

---

## SuGaR

**arXiv:** https://arxiv.org/abs/2311.12775  
**GitHub:** N/A  
**Project Page:** https://anttwo.github.io/sugar/

**Abstract:** Surface-aligned Gaussian splatting for better geometry

**Tags:** 3d gaussian splatting, surface alignment, geometry reconstruction

---

## LP-3DGS: Learning to Prune 3D Gaussian Splatting

**arXiv:** https://arxiv.org/abs/2405.18784  
**GitHub:** https://github.com/ASU-ESIC-FAN-Lab/LP-3DGS  
**Project Page:** https://paperswithcode.com/paper/lp-3dgs-learning-to-prune-3d-gaussian

**Abstract:** Recently, 3D Gaussian Splatting (3DGS) has become one of the mainstream methodologies for novel view synthesis (NVS) due to its high quality and fast rendering speed. However, as a point-based scene representation, 3DGS potentially generates a large number of Gaussians to fit the scene, leading to high memory usage. Improvements that have been proposed require either an empirical and preset pruning ratio or importance score threshold to prune the point cloud. Such hyperparameter requires multiple rounds of training to optimize and achieve the maximum pruning ratio, while maintaining the rendering quality for each scene. In this work, we propose learning-to-prune 3DGS (LP-3DGS), where a trainable binary mask is applied to the importance score that can find optimal pruning ratio automatically. Instead of using the traditional straight-through estimator (STE) method to approximate the binary mask gradient, we redesign the masking function to leverage the Gumbel-Sigmoid method, making it differentiable and compatible with the existing training process of 3DGS. Extensive experiments have shown that LP-3DGS consistently produces a good balance that is both efficient and high quality.

**Tags:** 3d gaussian splatting, pruning, memory optimization, automatic learning, model compression

---

## PUP 3D-GS: Principled Uncertainty Pruning for 3D Gaussian Splatting

**arXiv:** https://arxiv.org/abs/2406.10219  
**GitHub:** https://github.com/j-alex-hanson/gaussian-splatting-pup  
**Project Page:** https://pup3dgs.github.io/

**Abstract:** The paper addresses compression challenges in 3D Gaussian Splatting (3D-GS) by proposing a principled spatial sensitivity pruning score and introducing a multi-round prune-refine pipeline. The method can prune 90% of Gaussians, increases average rendering speed by 3.56×, retains more foreground details, and achieves higher image quality metrics. It computes Gaussian sensitivity using log determinant of the Hessian to determine which Gaussians to remove from the scene representation.

**Tags:** gaussian splatting, pruning, compression, uncertainty, hessian based

---

## DashGaussian: Optimizing 3D Gaussian Splatting in 200 Seconds

**arXiv:** https://arxiv.org/abs/2503.18402  
**GitHub:** https://github.com/YouyuChen0207/DashGaussian  
**Project Page:** https://dashgaussian.github.io/

**Abstract:** 3D Gaussian Splatting (3DGS) renders pixels by rasterizing Gaussian primitives, where the rendering resolution and the primitive number, concluded as the optimization complexity, dominate the time cost in primitive optimization. In this paper, we propose DashGaussian, a scheduling scheme over the optimization complexity of 3DGS that strips redundant complexity to accelerate 3DGS optimization. Specifically, we formulate 3DGS optimization as progressively fitting 3DGS to higher levels of frequency components in the training views, and propose a dynamic rendering resolution scheme that largely reduces the optimization complexity based on this formulation. Besides, we argue that a specific rendering resolution should cooperate with a proper primitive number for a better balance between computing redundancy and fitting quality, where we schedule the growth of the primitives to synchronize with the rendering resolution. DashGaussian significantly boosts the training speed of various 3DGS backbones by 45.7% on average without trading off rendering quality. Equipping DashGaussian to prior-art 3DGS methods, we reduce the optimization time of a 3DGS model with millions of primitives to 200 seconds on a consumer-grade GPU.

**Tags:** 3D Gaussian Splatting, optimization acceleration, rendering resolution, training efficiency, real-time

---

## Grendel-GS: On Scaling Up 3D Gaussian Splatting Training

**arXiv:** https://arxiv.org/abs/2406.18533  
**GitHub:** https://github.com/nyu-systems/Grendel-GS  
**Project Page:** https://daohanlu.github.io/scaling-up-3dgs/

**Abstract:** We present Grendel-GS, a distributed training system for 3D Gaussian Splatting designed to enable scaling laws through distributed system support. The system partitions 3D Gaussian Splatting parameters and parallelizes computation across multiple GPUs, using sparse all-to-all communication and dynamic load balancing. We demonstrate faster training times, support for more Gaussians in GPU memory, and reconstruction of larger, higher-resolution scenes. Our approach includes novel hyperparameter scaling rules based on an "Independent Gradients Hypothesis" for distributed 3D GS training.

**Tags:** distributed training, gaussian splatting, scalability, system optimization, parallel computing

---

## InstantSplat: Sparse-view Gaussian Splatting in Seconds

**arXiv:** https://arxiv.org/abs/2403.20309  
**GitHub:** https://github.com/NVlabs/InstantSplat  
**Project Page:** https://instantsplat.github.io/

**Abstract:** While neural 3D reconstruction has advanced substantially, its performance significantly degrades with sparse-view data, which limits its broader applicability, since SfM is often unreliable in sparse-view scenarios where feature matches are scarce. In this paper, we introduce InstantSplat, a novel approach for addressing sparse-view 3D scene reconstruction at lightning-fast speed. InstantSplat employs a self-supervised framework that optimizes 3D scene representation and camera poses by unprojecting 2D pixels into 3D space and aligning them using differentiable neural rendering. The optimization process is initialized with a large-scale trained geometric foundation model, which provides dense priors that yield initial points through model inference, after which we further optimize all scene parameters using photometric errors. To mitigate redundancy introduced by the prior model, we propose a co-visibility-based geometry initialization, and a Gaussian-based bundle adjustment is employed to rapidly adapt both the scene representation and camera parameters without relying on a complex adaptive density control process. Overall, InstantSplat is compatible with multiple point-based representations for view synthesis and surface reconstruction. It achieves an acceleration of over 30x in reconstruction and improves visual quality (SSIM) from 0.3755 to 0.7624 compared to traditional SfM with 3D-GS.

**Tags:** 3d reconstruction, gaussian splatting, sparse view, self-supervised, neural rendering

---

## CoCoGaussian: Leveraging Circle of Confusion for Gaussian Splatting from Defocused Images

**arXiv:** https://arxiv.org/abs/2412.16028  
**GitHub:** https://github.com/Jho-Yonsei/CoCoGaussian  
**Project Page:** https://jho-yonsei.github.io/CoCoGaussian/

**Abstract:** 3D Gaussian Splatting (3DGS) has attracted significant attention for its high-quality novel view rendering, inspiring research to address real-world challenges. While conventional methods depend on sharp images for accurate scene reconstruction, real-world scenarios are often affected by defocus blur due to finite depth of field, making it essential to account for realistic 3D scene representation. In this study, we propose CoCoGaussian, a Circle of Confusion-aware Gaussian Splatting that enables precise 3D scene representation using only defocused images. CoCoGaussian addresses the challenge of defocus blur by modeling the Circle of Confusion (CoC) through a physically grounded approach based on the principles of photographic defocus. Exploiting 3D Gaussians, we compute the CoC diameter from depth and learnable aperture information, generating multiple Gaussians to precisely capture the CoC shape. Furthermore, we introduce a learnable scaling factor to enhance robustness and provide more flexibility in handling unreliable depth in scenes with reflective or refractive surfaces. Experiments on both synthetic and real-world datasets demonstrate that CoCoGaussian achieves state-of-the-art performance across multiple benchmarks.

**Tags:** 3D Gaussian Splatting, defocus blur, depth of field, circle of confusion, photography

---

## Summary

This index contains 17 core 3D Gaussian Splatting papers covering:
- **Quality enhancement**: Mip-Splatting, GOF, Wild Gaussians, Neural-GS
- **Optimization and compression**: LP-3DGS, PUP 3D-GS, DashGaussian, LightGaussian, Compact3D
- **Performance improvements**: RT-GS, Grendel-GS, InstantSplat
- **Surface reconstruction**: SuGaR, Scaffold-GS, PolyGS
- **Specialized applications**: GS-IR, CoCoGaussian

These papers represent the foundational and most impactful contributions to 3D Gaussian Splatting technology, focusing on core techniques, quality improvements, optimization strategies, and practical deployment considerations.