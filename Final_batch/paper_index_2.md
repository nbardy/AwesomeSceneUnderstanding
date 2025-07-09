# Paper Index 2 - Dynamic Scenes & 4D Reconstruction

## 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering

**arXiv:** https://arxiv.org/abs/2310.08528  
**GitHub:** https://github.com/hustvl/4DGaussians  
**Project Page:** https://guanjunwu.github.io/4dgs/

**Abstract:** Representing and rendering dynamic scenes has been an important but challenging task. Especially, to accurately model complex motions, high efficiency is usually hard to guarantee. To achieve real-time dynamic scene rendering while also enjoying high training and storage efficiency, we propose 4D Gaussian Splatting (4D-GS) as a holistic representation for dynamic scenes rather than applying 3D-GS for each individual frame. In 4D-GS, a novel explicit representation containing both 3D Gaussians and 4D neural voxels is proposed. A decomposed neural voxel encoding algorithm inspired by HexPlane is proposed to efficiently build Gaussian features from 4D neural voxels and then a lightweight MLP is applied to predict Gaussian deformations at novel timestamps. Our 4D-GS method achieves real-time rendering under high resolutions, 82 FPS at an 800×800 resolution on an RTX 3090 GPU while maintaining comparable or better quality than previous state-of-the-art methods.

**Tags:** dynamic scene rendering, 4d gaussian splatting, real-time rendering, neural radiance fields, temporal modeling

---

## Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis

**arXiv:** https://arxiv.org/abs/2312.16812  
**GitHub:** https://github.com/oppo-us-research/SpacetimeGaussians  
**Project Page:** https://oppo-us-research.github.io/SpacetimeGaussians-website/

**Abstract:** Novel view synthesis of dynamic scenes has been an intriguing yet challenging problem. Despite recent advancements, simultaneously achieving high-resolution photorealistic results, real-time rendering, and compact storage remains a formidable task. To address these challenges, we propose Spacetime Gaussian Feature Splatting as a novel dynamic scene representation, composed of three pivotal components. First, we formulate expressive Spacetime Gaussians by enhancing 3D Gaussians with temporal opacity and parametric motion/rotation. This enables Spacetime Gaussians to capture static, dynamic, as well as transient content within a scene. Second, we introduce splatted feature rendering, which replaces spherical harmonics with neural features. These features facilitate the modeling of view- and time-dependent appearance while maintaining small size. Third, we leverage the guidance of training error and coarse depth to sample new Gaussians in areas that are challenging to converge with existing pipelines. Experiments on several established real-world datasets demonstrate that our method achieves state-of-the-art rendering quality and speed, while retaining compact storage. At 8K resolution, our lite-version model can render at 60 FPS on an Nvidia RTX 4090 GPU.

**Tags:** 3d gaussian splatting, dynamic view synthesis, real-time rendering, neural radiance fields, computer vision

---

## MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds

**arXiv:** https://arxiv.org/abs/2405.17421  
**GitHub:** https://github.com/JiahuiLei/MoSca  
**Project Page:** https://www.cis.upenn.edu/~leijh/projects/mosca/

**Abstract:** We introduce 4D Motion Scaffolds (MoSca), a modern 4D reconstruction system designed to reconstruct and synthesize novel views of dynamic scenes from monocular videos captured casually in the wild. To address such a challenging and ill-posed inverse problem, we leverage prior knowledge from foundational vision models and lift the video data to a novel Motion Scaffold (MoSca) representation, which compactly and smoothly encodes the underlying motions/deformations. The scene geometry and appearance are then disentangled from the deformation field and are encoded by globally fusing the Gaussians anchored onto the MoSca and optimized via Gaussian Splatting. Additionally, camera focal length and poses can be solved using bundle adjustment without the need of any other pose estimation tools. Experiments demonstrate state-of-the-art performance on dynamic rendering benchmarks and its effectiveness on real videos.

**Tags:** 4D reconstruction, gaussian splatting, dynamic scene reconstruction, monocular video analysis, computer vision

---

## MoDGS: Dynamic Gaussian Splatting from Casually-captured Monocular Videos with Depth Priors

**arXiv:** https://arxiv.org/abs/2406.00434  
**GitHub:** https://github.com/MobiusLqm/MoDGS  
**Project Page:** https://modgs.github.io/

**Abstract:** In this paper, we propose MoDGS, a new pipeline to render novel views of dynamic scenes from a casually captured monocular video. Previous monocular dynamic NeRF or Gaussian Splatting methods strongly rely on the rapid movement of input cameras to construct multiview consistency but struggle to reconstruct dynamic scenes on casually captured input videos whose cameras are either static or move slowly. MoDGS adopts recent single-view depth estimation methods to guide the learning of the dynamic scene. Specifically, MoDGS uses a single-view depth estimator to estimate a depth map for every frame and computes both rendering loss and an ordinal depth loss to learn the model. Additionally, a novel 3D-aware initialization method is proposed to learn a reasonable deformation field, which helps learn a robust deformation field from a monocular video. Moreover, a new robust depth loss is proposed to guide the learning of dynamic scene geometry. Comprehensive experiments demonstrate that MoDGS is able to render high-quality novel view images of dynamic scenes from just a casually captured monocular video, which outperforms state-of-the-art methods by a significant margin.

**Tags:** dynamic scene reconstruction, gaussian splatting, monocular video, depth estimation, novel view synthesis

---

## Deblur4DGS: 4D Gaussian Splatting from Blurry Monocular Video

**arXiv:** https://arxiv.org/abs/2412.06424  
**GitHub:** https://github.com/ZcsrenlongZ/Deblur4DGS  
**Project Page:** https://deblur4dgs.github.io/

**Abstract:** Recent 4D reconstruction methods have yielded impressive results but rely on sharp videos as supervision. However, motion blur often occurs in videos due to camera shake and object movement, while existing methods render blurry results when using such videos for reconstructing 4D models. Although a few NeRF-based approaches attempted to address the problem, they struggled to produce high-quality results, due to the inaccuracy in estimating continuous dynamic representations within the exposure time. To this end, we propose Deblur4DGS, the first 4D Gaussian Splatting framework to reconstruct a high-quality 4D model from blurry monocular video. In particular, with the explicit motion trajectory modeling based on 3D Gaussian Splatting, we propose to transform the challenging continuous dynamic representation estimation within an exposure time into the exposure time estimation, where a series of regularizations are suggested to tackle the under-constrained optimization. Moreover, we introduce the blur-aware variable canonical Gaussians to represent objects with large motion better. Beyond novel-view synthesis, Deblur4DGS can be applied to improve blurry video from multiple perspectives, including deblurring, frame interpolation, and video stabilization. Extensive experiments on the above four tasks show that Deblur4DGS significantly outperforms state-of-the-art 4D reconstruction methods.

**Tags:** 4d gaussian splatting, motion blur, video reconstruction, dynamic scene, real-time rendering

---

## VideoLifter: Lifting Videos to 3D with Fast Hierarchical Stereo Alignment

**arXiv:** https://arxiv.org/abs/2501.01949  
**GitHub:** https://github.com/VITA-Group/VideoLifter  
**Project Page:** https://videolifter.github.io/

**Abstract:** The paper focuses on 3D reconstruction from monocular video and introduces a method that leverages geometric priors from a learnable model to incrementally optimize a globally sparse to dense 3D representation. The method reduces training time by over 82% while segmenting video sequences, matching frames, constructing 3D fragments, and hierarchically aligning them into a unified 3D model.

**Tags:** 3d reconstruction, monocular video, hierarchical alignment, stereo matching, fast training

---

## Can Video Diffusion Model Reconstruct 4D Geometry? (Sora3R)

**arXiv:** https://arxiv.org/abs/2503.21082  
**GitHub:** N/A  
**Project Page:** https://wayne-mai.github.io/publication/sora3r_arxiv_2025/

**Abstract:** Reconstructing dynamic 3D scenes (i.e., 4D geometry) from monocular video is an important yet challenging problem. Conventional multiview geometry-based approaches often struggle with dynamic motion, whereas recent learning-based methods either require specialized 4D representation or sophisticated optimization. This paper introduces Sora3R, a novel framework that taps into the rich spatiotemporal priors of large-scale video diffusion models to directly infer 4D pointmaps from casual videos.

**Tags:** 4d reconstruction, video diffusion models, dynamic 3d scenes, monocular video, feedforward inference

---

## D²USt3R: Enhancing 3D Reconstruction with 4D Pointmaps for Dynamic Scenes

**arXiv:** https://arxiv.org/abs/2504.06264  
**GitHub:** https://github.com/cvlab-kaist/DDUSt3R  
**Project Page:** https://cvlab-kaist.github.io/DDUSt3R/

**Abstract:** We address 3D reconstruction in dynamic scenes, where object motions degrade pointmap regression methods originally designed for static scenes. We propose D²USt3R that regresses 4D pointmaps capturing both static and dynamic 3D scene geometry in a feed-forward manner. By explicitly incorporating both spatial and temporal aspects, our approach successfully encapsulates spatio-temporal dense correspondence to the proposed 4D pointmaps, enhancing performance of downstream tasks. Given a pair of input views, D²USt3R accurately establishes dense correspondence not only in static regions but also in dynamic regions, enabling full reconstruction of a dynamic scene.

**Tags:** dynamic 3d reconstruction, 4d pointmaps, spatio-temporal correspondence, scene reconstruction, motion handling

---

## Gaussian Splashing: Unified Particles for Versatile Motion Synthesis and Rendering

**arXiv:** https://arxiv.org/abs/2401.15318  
**GitHub:** http://amysteriouscat.github.io/GaussianSplashing  
**Project Page:** https://gaussiansplashing.github.io/

**Abstract:** We demonstrate the feasibility of integrating physics-based animations of solids and fluids with 3D Gaussian Splatting (3DGS) to create novel effects in virtual scenes. Our method uses a unified particle representation that can handle both solid and fluid dynamics while maintaining the rendering quality of 3DGS. This enables the creation of realistic physics-based animations with high-quality rendering for applications in virtual production and interactive media.

**Tags:** physics simulation, gaussian splatting, fluid dynamics, motion synthesis, unified particles

---

## Gaussian-Flow: 4D Reconstruction from Monocular Video using Gaussian Flow Fields

**arXiv:** https://arxiv.org/abs/2310.12031  
**GitHub:** https://github.com/theEricMa/Gaussian-Flow  
**Project Page:** https://gaussian-flow.github.io/

**Abstract:** 4D reconstruction from monocular video using Gaussian flow fields with optical flow guidance for dynamic scene representation.

**Tags:** 4d reconstruction, gaussian flow, monocular video, optical flow, dynamic scenes

---

## Feed-Forward Bullet-Time Reconstruction of Dynamic Scenes from Monocular Videos

**arXiv:** https://arxiv.org/abs/2412.03526  
**GitHub:** N/A  
**Project Page:** https://research.nvidia.com/labs/toronto-ai/bullet-timer/

**Abstract:** Recent advancements in static feed-forward scene reconstruction have demonstrated significant progress in high-quality novel view synthesis. However, these models often struggle with generalizability across diverse environments and fail to effectively handle dynamic content. We present BTimer (short for BulletTimer), the first motion-aware feed-forward model for real-time reconstruction and novel view synthesis of dynamic scenes. Our approach reconstructs the full scene in a 3D Gaussian Splatting representation at a given target ('bullet') timestamp by aggregating information from all the context frames. Such a formulation allows BTimer to gain scalability and generalization by leveraging both static and dynamic scene datasets. Given a casual monocular dynamic video, BTimer reconstructs a bullet-time scene within 150ms while reaching state-of-the-art performance on both static and dynamic scene datasets, even compared with optimization-based approaches.

**Tags:** 3D Gaussian Splatting, dynamic scenes, novel view synthesis, real-time reconstruction, monocular video

---

## GS-DiT: Advancing Video Generation with Pseudo 4D Gaussian Fields through Efficient Dense 3D Point Tracking

**arXiv:** https://arxiv.org/abs/2501.02690  
**GitHub:** https://github.com/wkbian/GS-DiT  
**Project Page:** https://wkbian.github.io/Projects/GS-DiT/

**Abstract:** GS-DiT addresses 4D video control in video generation, enabling sophisticated lens techniques like multi-camera shooting and dolly zoom that are currently unsupported by existing methods. The method avoids the need for expensive multi-view videos by bringing pseudo 4D Gaussian fields to video generation, inspired by Monocular Dynamic novel View Synthesis (MDVS). The framework constructs a pseudo 4D Gaussian field with dense 3D point tracking and renders the Gaussian field for all video frames. A pretrained DiT (Diffusion Transformer) is then finetuned to generate videos following the guidance of the rendered video. An efficient Dense 3D Point Tracking (D3D-PT) method is introduced that outperforms SpatialTracker in accuracy and accelerates inference speed by two orders of magnitude.

**Tags:** video generation, 4d control, gaussian splatting, diffusion transformers, dense 3d tracking

---

## Diffusion4D: Fast Spatial-temporal Consistent 4D Generation via Video Diffusion Models

**arXiv:** https://arxiv.org/abs/2405.16645  
**GitHub:** https://github.com/VITA-Group/Diffusion4D  
**Project Page:** https://vita-group.github.io/Diffusion4D/

**Abstract:** The paper presents Diffusion4D, a method for fast spatial-temporal consistent 4D generation using video diffusion models. The approach supports image-to-4D, text-to-4D, and 3D-to-4D generation tasks, creating dynamic 3D content with temporal consistency. The method leverages video diffusion models to generate 4D content efficiently while maintaining spatial-temporal coherence.

**Tags:** 4D generation, video diffusion, spatial-temporal consistency, dynamic content, generative models

---

## FreeTimeGS: Free-moving Gaussians in Space-time for Dynamic Scene Reconstruction

**arXiv:** Not available  
**GitHub:** https://github.com/JiahuiLei/FreeTimeGS  
**Project Page:** http://zju3dv.github.io/freetimegs/

**Abstract:** Free-moving Gaussians in space-time for efficient dynamic modeling with Gaussian motion functions and temporal opacity, achieving 467 FPS at 1080p with 4.5x faster rendering than baseline methods.

**Tags:** dynamic scene reconstruction, gaussian splatting, temporal modeling, motion functions, real-time rendering

---

## MoDecGS: Memory-efficient Dynamic Gaussian Splatting with Motion Decomposition

**arXiv:** https://arxiv.org/abs/2501.03714  
**GitHub:** https://github.com/kaist-viclab/MoDecGS  
**Project Page:** https://kaist-viclab.github.io/MoDecGS-site/

**Abstract:** Memory-efficient dynamic Gaussians with motion decomposition achieving 70% compression via hierarchical decomposition while maintaining rendering quality for dynamic scene reconstruction.

**Tags:** dynamic scenes, gaussian splatting, memory efficiency, motion decomposition, compression

---

## BARD-GS: Motion Blur-Aware Dynamic Gaussian Splatting for Real-Time Rendering

**arXiv:** Not available  
**GitHub:** https://github.com/vulab-ai/BARD-GS  
**Project Page:** https://vulab-ai.github.io/BARD-GS/

**Abstract:** Motion blur-aware dynamic reconstruction with two-stage blur decomposition handling both camera and object motion blur for complex dynamic scenes with superior quality compared to existing methods.

**Tags:** motion blur, dynamic scenes, gaussian splatting, real-time rendering, blur decomposition

---

## Summary

This index contains 16 papers focusing on dynamic scenes and 4D reconstruction:

**Core 4D Reconstruction Methods:**
- 4D Gaussian Splatting: Real-time dynamic scene rendering
- Spacetime Gaussians: Feature splatting for dynamic view synthesis
- MoSca: Motion scaffolds for monocular video reconstruction
- MoDGS: Depth-guided dynamic scene reconstruction
- Sora3R: Video diffusion models for 4D geometry
- D²USt3R: 4D pointmaps for dynamic scene reconstruction

**Deblurring & Motion Handling:**
- Deblur4DGS: 4D reconstruction from blurry video
- BARD-GS: Motion blur-aware dynamic reconstruction

**Video & Motion Synthesis:**
- VideoLifter: Hierarchical stereo alignment for video lifting
- Gaussian Splashing: Physics-based motion synthesis
- GS-DiT: 4D video generation with pseudo Gaussian fields
- Diffusion4D: Spatial-temporal consistent 4D generation

**Advanced Dynamic Techniques:**
- Gaussian-Flow: Flow-based 4D reconstruction
- BTimer: Feed-forward bullet-time reconstruction
- FreeTimeGS: Free-moving Gaussians in space-time
- MoDecGS: Memory-efficient dynamic Gaussians

Research emphasizes real-time performance, temporal consistency, and advanced optimization techniques for dynamic 3D scene understanding and reconstruction.