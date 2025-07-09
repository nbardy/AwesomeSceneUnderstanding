# Paper Index 4 - Camera Effects & Novel View Synthesis

## DOF-GS: Adjustable Depth-of-Field 3D Gaussian Splatting for Post-Capture Refocusing, Defocus Rendering and Blur Removal

**arXiv:** https://arxiv.org/abs/2405.17351  
**GitHub:** https://github.com/leoShen917/DoF-Gaussian  
**Project Page:** https://dof-gaussian.github.io/

**Abstract:** 3D Gaussian Splatting (3DGS) techniques have recently enabled high-quality 3D scene reconstruction and real-time novel view synthesis. These approaches, however, are limited by the pinhole camera model and lack effective modeling of defocus effects. Departing from this, we introduce DOF-GS--a new 3DGS-based framework with a finite-aperture camera model and explicit, differentiable defocus rendering, enabling it to function as a post-capture control tool. By training with multi-view images with moderate defocus blur, DOF-GS learns inherent camera characteristics and reconstructs sharp details of the underlying scene, particularly, enabling rendering of varying DOF effects through on-demand aperture and focal distance control, post-capture and optimization. Additionally, our framework extracts circle-of-confusion cues during optimization to identify in-focus regions in input views, enhancing the reconstructed 3D scene details.

**Tags:** 3d gaussian splatting, depth of field, camera modeling, post-capture effects, defocus rendering

---

## CoCoGaussian: Leveraging Circle of Confusion for Gaussian Splatting from Defocused Images

**arXiv:** https://arxiv.org/abs/2412.16028  
**GitHub:** https://github.com/Jho-Yonsei/CoCoGaussian  
**Project Page:** https://jho-yonsei.github.io/CoCoGaussian/

**Abstract:** 3D Gaussian Splatting (3DGS) has attracted significant attention for its high-quality novel view rendering, inspiring research to address real-world challenges. While conventional methods depend on sharp images for accurate scene reconstruction, real-world scenarios are often affected by defocus blur due to finite depth of field, making it essential to account for realistic 3D scene representation. In this study, we propose CoCoGaussian, a Circle of Confusion-aware Gaussian Splatting that enables precise 3D scene representation using only defocused images. CoCoGaussian addresses the challenge of defocus blur by modeling the Circle of Confusion (CoC) through a physically grounded approach based on the principles of photographic defocus. Exploiting 3D Gaussians, we compute the CoC diameter from depth and learnable aperture information, generating multiple Gaussians to precisely capture the CoC shape. Furthermore, we introduce a learnable scaling factor to enhance robustness and provide more flexibility in handling unreliable depth in scenes with reflective or refractive surfaces. Experiments on both synthetic and real-world datasets demonstrate that CoCoGaussian achieves state-of-the-art performance across multiple benchmarks.

**Tags:** 3d gaussian splatting, defocus blur, depth of field, circle of confusion, photography

---

## Deblurring 3D Gaussian Splatting

**arXiv:** https://arxiv.org/abs/2401.00834  
**GitHub:** https://github.com/benhenryL/Deblurring-3D-Gaussian-Splatting  
**Project Page:** https://benhenryl.github.io/Deblurring-3D-Gaussian-Splatting/

**Abstract:** Recent studies in Radiance Fields have paved the robust way for novel view synthesis with their photorealistic rendering quality. Nevertheless, they usually employ neural networks and volumetric rendering, which are costly to train and impede their broad use in various real-time applications due to the lengthy rendering time. Lately 3D Gaussians splatting-based approach has been proposed to model the 3D scene, and it achieves remarkable visual quality while rendering the images in real-time. However, it suffers from severe degradation in the rendering quality if the training images are blurry. Blurriness commonly occurs due to the lens defocusing, object motion, and camera shake, and it inevitably intervenes in clean image acquisition. Several previous studies have attempted to render clean and sharp images from blurry input images using neural fields. The majority of those works, however, are designed only for volumetric rendering-based neural radiance fields and are not straightforwardly applicable to rasterization-based 3D Gaussian splatting methods. Thus, we propose a novel real-time deblurring framework, Deblurring 3D Gaussian Splatting, using a small Multi-Layer Perceptron (MLP) that manipulates the covariance of each 3D Gaussian to model the scene blurriness. While Deblurring 3D Gaussian Splatting can still enjoy real-time rendering, it can reconstruct fine and sharp details from blurry images. A variety of experiments have been conducted on the benchmark, and the results have revealed the effectiveness of our approach for deblurring.

**Tags:** 3d gaussian splatting, deblurring, real-time rendering, image reconstruction, neural rendering

---

## Deblur-GS: 3D Gaussian Splatting from Camera Motion Blurred Images

**arXiv:** N/A  
**GitHub:** https://github.com/Chaphlagical/Deblur-GS  
**Project Page:** https://chaphlagical.icu/Deblur-GS/

**Abstract:** Novel view synthesis has undergone a revolution thanks to the radiance field method. The introduction of 3D Gaussian splatting (3DGS) has successfully addressed the issues of prolonged training times and slow rendering speeds associated with the Neural Radiance Field (NeRF), all while preserving the quality of reconstructions. However, 3DGS remains heavily reliant on the quality of input images and their initial camera pose initialization. In cases where input images are blurred, the reconstruction results suffer from blurriness and artifacts. In this paper, we propose the Deblur-GS method for reconstructing 3D Gaussian points to create a sharp radiance field from a camera motion blurred image set. We model the problem of motion blur as a joint optimization challenge involving camera trajectory estimation and time sampling. We cohesively optimize the parameters of the Gaussian points and the camera trajectory during the shutter time.

**Tags:** 3d gaussian splatting, camera motion blur, novel view synthesis, joint optimization, 3d reconstruction

---

## DeblurGS: Gaussian Splatting for Camera Motion Blur

**arXiv:** https://arxiv.org/abs/2404.11358  
**GitHub:** https://github.com/taekkii/deblurgs  
**Project Page:** N/A

**Abstract:** Novel view synthesis has become a significant area of research in computer vision, with 3D Gaussian Splatting (3DGS) emerging as a particularly promising method. However, the effectiveness of 3DGS is heavily dependent on the quality of the input images, which are often compromised by motion blur, especially in handheld or dynamic camera scenarios. This limitation significantly hinders the synthesis of sharp and detailed novel views. To address this challenge, we introduce DeblurGS, a method specifically designed to reconstruct sharp 3D Gaussian fields from motion-blurred images. DeblurGS operates by optimizing the parameters of 3D Gaussians while simultaneously estimating the camera motion trajectories that induced the blur. By accurately modeling the camera movements, DeblurGS can effectively reverse the blurring process, resulting in significantly improved novel view synthesis quality. We demonstrate that DeblurGS achieves superior performance over existing methods, producing sharp and high-quality novel views even from severely blurred input images, thus expanding the applicability of 3D Gaussian Splatting in real-world scenarios.

**Tags:** 3d gaussian splatting, motion deblurring, camera trajectory, novel view synthesis, computer vision

---

## ViewCrafter: Taming Video Diffusion Models for High-fidelity Novel View Synthesis

**arXiv:** https://arxiv.org/abs/2409.02048  
**GitHub:** https://github.com/Drexubery/ViewCrafter  
**Project Page:** https://drexubery.github.io/ViewCrafter/

**Abstract:** Despite recent advancements in neural 3D reconstruction, the dependence on dense multi-view captures restricts their broader applicability. In this work, we propose ViewCrafter, a novel method for synthesizing high-fidelity novel views of generic scenes from single or sparse images with the prior of video diffusion model. Our method takes advantage of the powerful generation capabilities of video diffusion model and the coarse 3D clues offered by point-based representation to generate high-quality video frames with precise camera pose control. To further enlarge the generation range of novel views, we tailored an iterative view synthesis strategy together with a camera trajectory planning algorithm to progressively extend the 3D clues and the areas covered by the novel views. With ViewCrafter, we can facilitate various applications, such as immersive experiences with real-time rendering by efficiently optimizing a 3D-GS representation using the reconstructed 3D point.

**Tags:** novel view synthesis, video diffusion models, 3d gaussian splatting, sparse view reconstruction, camera control

---

## MultiDiff: Consistent Novel View Synthesis from a Single Image

**arXiv:** https://arxiv.org/abs/2406.18524  
**GitHub:** N/A  
**Project Page:** https://sirwyver.github.io/MultiDiff/

**Abstract:** We introduce MultiDiff, a novel approach for consistent novel view synthesis of scenes from a single RGB image. The task of synthesizing novel views from a single reference image is highly ill-posed by nature, as there exist multiple, plausible explanations for unobserved areas. To address this issue, we incorporate strong priors in form of monocular depth predictors and video-diffusion models. Monocular depth enables us to condition our model on warped reference images for the target views, increasing geometric stability. The video-diffusion prior provides a strong proxy for 3D scenes, allowing the model to learn continuous and pixel-accurate correspondences across generated images. In contrast to approaches relying on autoregressive image generation that are prone to drifts and error accumulation, MultiDiff jointly synthesizes a sequence of frames yielding high-quality and multi-view consistent results -- even for long-term scene generation with large camera movements, while reducing inference time by an order of magnitude. For additional consistency and image quality improvements, we introduce a novel, structured noise distribution. Our experimental results demonstrate that MultiDiff outperforms state-of-the-art methods on the challenging, real-world datasets RealEstate10K and ScanNet. Finally, our model naturally supports multi-view consistent editing without the need for further tuning.

**Tags:** novel view synthesis, diffusion models, single image, multi-view consistency, monocular depth

---

## NVS-Solver: Video Diffusion Model as Zero-Shot Novel View Synthesizer

**arXiv:** https://arxiv.org/abs/2405.15364  
**GitHub:** https://github.com/ZHU-Zhiyu/NVS_Solver  
**Project Page:** https://mengyou2.github.io/NVS_Solver/

**Abstract:** By harnessing the potent generative capabilities of pre-trained large video diffusion models, we propose NVS-Solver, a new novel view synthesis (NVS) paradigm that operates without the need for training. NVS-Solver adaptively modulates the diffusion sampling process with the given views to enable the creation of remarkable visual experiences from single or multiple views of static scenes or monocular videos of dynamic scenes. Specifically, built upon our theoretical modeling, we iteratively modulate the score function with the given scene priors represented with warped input views to control the video diffusion process. Moreover, by theoretically exploring the boundary of the estimation error, we achieve the modulation in an adaptive fashion according to the view pose and the number of diffusion steps. Extensive evaluations on both static and dynamic scenes substantiate the significant superiority of our NVS-Solver over state-of-the-art methods both quantitatively and qualitatively.

**Tags:** novel view synthesis, video diffusion, zero-shot, 3d reconstruction, camera control

---

## InstantSplat: Sparse-view Gaussian Splatting in Seconds

**arXiv:** https://arxiv.org/abs/2403.20309  
**GitHub:** https://github.com/NVlabs/InstantSplat  
**Project Page:** https://instantsplat.github.io/

**Abstract:** While neural 3D reconstruction has advanced substantially, its performance significantly degrades with sparse-view data, which limits its broader applicability, since SfM is often unreliable in sparse-view scenarios where feature matches are scarce. In this paper, we introduce InstantSplat, a novel approach for addressing sparse-view 3D scene reconstruction at lightning-fast speed. InstantSplat employs a self-supervised framework that optimizes 3D scene representation and camera poses by unprojecting 2D pixels into 3D space and aligning them using differentiable neural rendering. The optimization process is initialized with a large-scale trained geometric foundation model, which provides dense priors that yield initial points through model inference, after which we further optimize all scene parameters using photometric errors. To mitigate redundancy introduced by the prior model, we propose a co-visibility-based geometry initialization, and a Gaussian-based bundle adjustment is employed to rapidly adapt both the scene representation and camera parameters without relying on a complex adaptive density control process. Overall, InstantSplat is compatible with multiple point-based representations for view synthesis and surface reconstruction. It achieves an acceleration of over 30x in reconstruction and improves visual quality (SSIM) from 0.3755 to 0.7624 compared to traditional SfM with 3D-GS.

**Tags:** 3d reconstruction, gaussian splatting, sparse view, self-supervised, neural rendering

---

## Generative Camera Dolly: Extreme Monocular Dynamic Novel View Synthesis

**arXiv:** https://arxiv.org/abs/2405.14868  
**GitHub:** https://github.com/basilevh/gcd  
**Project Page:** https://gcd.cs.columbia.edu/

**Abstract:** Accurate reconstruction of complex dynamic scenes from just a single viewpoint continues to be a challenging task in computer vision. Current dynamic novel view synthesis methods typically require videos from many different camera viewpoints, necessitating careful recording setups, and significantly restricting their utility in the wild as well as in terms of embodied AI applications. We propose GCD, a controllable monocular dynamic view synthesis pipeline that leverages large-scale diffusion priors to, given a video of any scene, generate a synchronous video from any other chosen perspective, conditioned on a set of relative camera pose parameters. The model does not require depth as input, and does not explicitly model 3D scene geometry, instead performing end-to-end video-to-video translation. Much like a camera dolly in film-making, the approach essentially conceives a virtual camera that can move around with up to six degrees of freedom, reveal significant portions of the scene that are otherwise unseen, reconstruct hidden objects behind occlusions, all within complex dynamic scenes.

**Tags:** diffusion models, dynamic scenes, novel view synthesis, video generation, camera control

---

## AnyCam: Learning to Recover Camera Poses and Intrinsics from Casual Videos

**arXiv:** https://arxiv.org/abs/2503.23282  
**GitHub:** https://github.com/Brummi/anycam  
**Project Page:** https://fwmb.github.io/anycam/

**Abstract:** Estimating camera motion and intrinsics from casual videos is a core challenge in computer vision. Traditional bundle-adjustment based methods, such as SfM and SLAM, struggle to perform reliably on arbitrary data. Although specialized SfM approaches have been developed for handling dynamic scenes, they either require intrinsics or computationally expensive test-time optimization and often fall short in performance. Recently, methods like Dust3r have reformulated the SfM problem in a more data-driven way. While such techniques show promising results, they are still 1) not robust towards dynamic objects and 2) require labeled data for supervised training. As an alternative, we propose AnyCam, a fast transformer model that directly estimates camera poses and intrinsics from a dynamic video sequence in feed-forward manner. Our intuition is that such a network can learn strong priors over realistic camera motions. To scale up our training, we rely on an uncertainty-based loss formulation and pre-trained depth and flow networks instead of motion or trajectory supervision. This allows us to use diverse, unlabelled video datasets obtained mostly from YouTube. Additionally, we ensure that the predicted trajectory does not accumulate drift over time through a lightweight trajectory refinement step. We test AnyCam on established datasets, where it delivers accurate camera poses and intrinsics both qualitatively and quantitatively. Furthermore, even with trajectory refinement, AnyCam is significantly faster than existing works for SfM in dynamic settings. Finally, by combining camera information, uncertainty, and depth, our model can produce high-quality 4D pointclouds in a feed-forward fashion.

**Tags:** camera pose estimation, structure from motion, transformer, dynamic scenes, video analysis

---

## PF3plat: Pose-Free Feed-Forward 3D Gaussian Splatting

**arXiv:** https://arxiv.org/abs/2410.22128  
**GitHub:** https://github.com/cvlab-kaist/PF3plat  
**Project Page:** https://cvlab-kaist.github.io/PF3plat/

**Abstract:** PF3plat estimates multi-view consistent depth, accurate camera pose, and photorealistic novel views from uncalibrated image collections. The method addresses unstable training process in pixel-aligned 3D Gaussian Splatting and works with wide-baseline images without additional data. It uses coarse alignment of Gaussians using pre-trained depth and correspondence models, followed by fine-alignment through learnable refinement modules.

**Tags:** pose free, feed forward, gaussian splatting, 3d reconstruction, uncalibrated images

---

## Feed-Forward Bullet-Time Reconstruction of Dynamic Scenes from Monocular Videos

**arXiv:** https://arxiv.org/abs/2412.03526  
**GitHub:** N/A  
**Project Page:** https://research.nvidia.com/labs/toronto-ai/bullet-timer/

**Abstract:** Recent advancements in static feed-forward scene reconstruction have demonstrated significant progress in high-quality novel view synthesis. However, these models often struggle with generalizability across diverse environments and fail to effectively handle dynamic content. We present BTimer (short for BulletTimer), the first motion-aware feed-forward model for real-time reconstruction and novel view synthesis of dynamic scenes. Our approach reconstructs the full scene in a 3D Gaussian Splatting representation at a given target ('bullet') timestamp by aggregating information from all the context frames. Such a formulation allows BTimer to gain scalability and generalization by leveraging both static and dynamic scene datasets. Given a casual monocular dynamic video, BTimer reconstructs a bullet-time scene within 150ms while reaching state-of-the-art performance on both static and dynamic scene datasets, even compared with optimization-based approaches.

**Tags:** 3d gaussian splatting, dynamic scenes, novel view synthesis, real-time reconstruction, monocular video

---

## Multi-View Regulated Gaussian Splatting for Novel View Synthesis

**arXiv:** https://arxiv.org/abs/2410.02103  
**GitHub:** https://github.com/xiaobiaodu/MVGS  
**Project Page:** https://xiaobiaodu.github.io/mvgs-project/

**Abstract:** The paper addresses limitations in existing single-view optimization paradigms for novel view synthesis. It introduces multi-view coherent (MVC) constraint to enhance 3D Gaussian multi-view consistency, proposes a cross-intrinsic guidance scheme for optimization in a coarse-to-fine manner, and develops a multi-view cross-ray densification strategy to handle views with minimal overlap. The method provides a plug-and-play optimizer that can improve existing Gaussian-based methods by ~1 dB PSNR.

**Tags:** gaussian splatting, novel view synthesis, multi-view consistency, optimization, plug-and-play

---

## DepthSplat: Connecting Gaussian Splatting and Depth

**arXiv:** https://arxiv.org/abs/2410.13862  
**GitHub:** https://github.com/cvg/depthsplat  
**Project Page:** https://haofeixu.github.io/depthsplat/

**Abstract:** DepthSplat enables cross-task interactions between Gaussian splatting and depth estimation. Better depth leads to improved novel view synthesis, and unsupervised depth pre-training with Gaussian splatting reduces depth prediction error. The method demonstrates how connecting these two tasks can benefit both domains through novel cross-task learning strategies.

**Tags:** gaussian splatting, depth estimation, novel view synthesis, cross-task learning, 3d reconstruction

---

## Deblur4DGS: 4D Gaussian Splatting from Blurry Monocular Video

**arXiv:** https://arxiv.org/abs/2412.06424  
**GitHub:** https://github.com/ZcsrenlongZ/Deblur4DGS  
**Project Page:** https://deblur4dgs.github.io/

**Abstract:** Recent 4D reconstruction methods have yielded impressive results but rely on sharp videos as supervision. However, motion blur often occurs in videos due to camera shake and object movement, while existing methods render blurry results when using such videos for reconstructing 4D models. Although a few NeRF-based approaches attempted to address the problem, they struggled to produce high-quality results, due to the inaccuracy in estimating continuous dynamic representations within the exposure time. To this end, we propose Deblur4DGS, the first 4D Gaussian Splatting framework to reconstruct a high-quality 4D model from blurry monocular video. In particular, with the explicit motion trajectory modeling based on 3D Gaussian Splatting, we propose to transform the challenging continuous dynamic representation estimation within an exposure time into the exposure time estimation, where a series of regularizations are suggested to tackle the under-constrained optimization. Moreover, we introduce the blur-aware variable canonical Gaussians to represent objects with large motion better. Beyond novel-view synthesis, Deblur4DGS can be applied to improve blurry video from multiple perspectives, including deblurring, frame interpolation, and video stabilization. Extensive experiments on the above four tasks show that Deblur4DGS significantly outperforms state-of-the-art 4D reconstruction methods.

**Tags:** 4d gaussian splatting, motion blur, video reconstruction, dynamic scene, real-time rendering

---

## Summary

This index contains 16 papers focusing on Camera Effects & Novel View Synthesis:

- **Camera Effects**: DOF-GS, CoCoGaussian for depth-of-field and defocus modeling
- **Deblurring Techniques**: Three specialized approaches (Deblurring 3D GS, Deblur-GS, DeblurGS) plus 4D extension (Deblur4DGS)
- **Novel View Synthesis**: ViewCrafter, MultiDiff, NVS-Solver with diffusion-based approaches
- **Zero-shot Methods**: NVS-Solver for training-free novel view synthesis
- **Camera Control**: Generative Camera Dolly (GCD), AnyCam for pose estimation
- **Feed-forward Methods**: PF3plat, BTimer for rapid reconstruction
- **Sparse View Reconstruction**: InstantSplat for efficient sparse-view processing
- **Multi-view Consistency**: MVGS for improved view synthesis optimization
- **Cross-task Learning**: DepthSplat connecting depth estimation and 3D reconstruction

Research areas emphasize real-time performance, robust handling of challenging input conditions (blur, sparse views, defocus), and advanced camera modeling for professional-quality novel view synthesis applications.