# Awesome Scene Understanding

> A comprehensive, curated collection of cutting-edge research papers in 3D scene understanding, neural rendering, and computer vision.

## 📋 Description

This repository provides a unified, well-organized resource for researchers, students, and practitioners working in 3D scene understanding. We've consolidated and standardized research from multiple sources into a comprehensive collection covering 65+ papers across 5 key research domains.

**What makes this collection special:**
- **Technical deep reviews** that cut through marketing hype to show actual implementations
- **Standardized format** with abstracts, code repositories, and project pages
- **Cross-referenced navigation** between brief summaries and detailed technical analyses
- **Implementation-focused** approach with working code examples and performance metrics
- **Honest assessment** of trade-offs, limitations, and real-world applicability

The collection focuses on the latest advances in 3D Gaussian Splatting, dynamic scene reconstruction, semantic understanding, novel view synthesis, and self-supervised video learning.

## 📊 Repository Statistics

- **Total Papers**: 65+
- **Research Categories**: 5 major domains
- **Deep Reviews**: 5 comprehensive analyses
- **GitHub Repositories**: 45+ with available code
- **Project Pages**: 55+ with demos and additional resources

## 🗂️ Navigation

- [3D Gaussian Splatting & Core Techniques](#3d-gaussian-splatting--core-techniques) (17 papers)
- [Dynamic Scenes & 4D Reconstruction](#dynamic-scenes--4d-reconstruction) (16 papers)
- [Semantic Understanding & Applications](#semantic-understanding--applications) (15 papers)
- [Camera Effects & Novel View Synthesis](#camera-effects--novel-view-synthesis) (16 papers)
- [Video Understanding & Self-Supervised Learning](#video-understanding--self-supervised-learning) (15 papers)

---

## 3D Gaussian Splatting & Core Techniques

*Core techniques, quality enhancements, optimization strategies, and foundational methods for 3D Gaussian Splatting*

- **[Gaussian Opacity Fields (GOF)]** - https://niujinshuchong.github.io/gaussian-opacity-fields/ | [Deep Review](Final_batch/deep_review_1.md#gaussian-opacity-fields-gof) | [Brief Summary](Final_batch/paper_index_1.md#gaussian-opacity-fields-gof)

- **[Mip-Splatting]** - https://arxiv.org/abs/2405.02468 | [Deep Review](Final_batch/deep_review_1.md#mip-splatting) | [Brief Summary](Final_batch/paper_index_1.md#mip-splatting)

- **[Wild Gaussians]** - https://wild-gaussians.github.io/ | [Deep Review](Final_batch/deep_review_1.md#wild-gaussians) | [Brief Summary](Final_batch/paper_index_1.md#wild-gaussians)

- **[PolyGS]** - https://research.nvidia.com/labs/toronto-ai/polygs/ | [Deep Review](Final_batch/deep_review_1.md#polygs) | [Brief Summary](Final_batch/paper_index_1.md#polygs)

- **[Neural-GS]** - https://neural-gs.github.io/ | [Deep Review](Final_batch/deep_review_1.md#neural-gs) | [Brief Summary](Final_batch/paper_index_1.md#neural-gs)

- **[Scaffold-GS]** - https://city-super.github.io/scaffold-gs/ | [Deep Review](Final_batch/deep_review_1.md#scaffold-gs) | [Brief Summary](Final_batch/paper_index_1.md#scaffold-gs)

- **[GS-IR (Inverse Rendering)]** - https://gs-ir.github.io/ | [Deep Review](Final_batch/deep_review_1.md#gs-ir-inverse-rendering) | [Brief Summary](Final_batch/paper_index_1.md#gs-ir-inverse-rendering)

- **[RT-GS]** - https://zhangganlin.github.io/rt-gs/ | [Deep Review](Final_batch/deep_review_1.md#rt-gs) | [Brief Summary](Final_batch/paper_index_1.md#rt-gs)

- **[Compact3D]** - https://maincold2.github.io/c3d/ | [Deep Review](Final_batch/deep_review_1.md#compact3d) | [Brief Summary](Final_batch/paper_index_1.md#compact3d)

- **[LightGaussian]** - https://lightgaussian.github.io/ | [Deep Review](Final_batch/deep_review_1.md#lightgaussian) | [Brief Summary](Final_batch/paper_index_1.md#lightgaussian)

- **[SuGaR]** - https://anttwo.github.io/sugar/ | [Deep Review](Final_batch/deep_review_1.md#sugar) | [Brief Summary](Final_batch/paper_index_1.md#sugar)

- **[LP-3DGS: Learning to Prune 3D Gaussian Splatting]** - https://paperswithcode.com/paper/lp-3dgs-learning-to-prune-3d-gaussian | [Deep Review](Final_batch/deep_review_1.md#lp-3dgs-learning-to-prune-3d-gaussian-splatting) | [Brief Summary](Final_batch/paper_index_1.md#lp-3dgs-learning-to-prune-3d-gaussian-splatting)

- **[PUP 3D-GS: Principled Uncertainty Pruning for 3D Gaussian Splatting]** - https://pup3dgs.github.io/ | [Deep Review](Final_batch/deep_review_1.md#pup-3d-gs-principled-uncertainty-pruning-for-3d-gaussian-splatting) | [Brief Summary](Final_batch/paper_index_1.md#pup-3d-gs-principled-uncertainty-pruning-for-3d-gaussian-splatting)

- **[DashGaussian: Optimizing 3D Gaussian Splatting in 200 Seconds]** - https://dashgaussian.github.io/ | [Deep Review](Final_batch/deep_review_1.md#dashgaussian-optimizing-3d-gaussian-splatting-in-200-seconds) | [Brief Summary](Final_batch/paper_index_1.md#dashgaussian-optimizing-3d-gaussian-splatting-in-200-seconds)

- **[Grendel-GS: On Scaling Up 3D Gaussian Splatting Training]** - https://daohanlu.github.io/scaling-up-3dgs/ | [Deep Review](Final_batch/deep_review_1.md#grendel-gs-on-scaling-up-3d-gaussian-splatting-training) | [Brief Summary](Final_batch/paper_index_1.md#grendel-gs-on-scaling-up-3d-gaussian-splatting-training)

- **[InstantSplat: Sparse-view Gaussian Splatting in Seconds]** - https://instantsplat.github.io/ | [Deep Review](Final_batch/deep_review_1.md#instantsplat-sparse-view-gaussian-splatting-in-seconds) | [Brief Summary](Final_batch/paper_index_1.md#instantsplat-sparse-view-gaussian-splatting-in-seconds)

- **[CoCoGaussian: Leveraging Circle of Confusion for Gaussian Splatting from Defocused Images]** - https://jho-yonsei.github.io/CoCoGaussian/ | [Deep Review](Final_batch/deep_review_1.md#cocogaussian-leveraging-circle-of-confusion-for-gaussian-splatting-from-defocused-images) | [Brief Summary](Final_batch/paper_index_1.md#cocogaussian-leveraging-circle-of-confusion-for-gaussian-splatting-from-defocused-images)

---

## Dynamic Scenes & 4D Reconstruction

*Real-time dynamic scene rendering, temporal modeling, motion blur handling, and advanced optimization for time-varying 3D content*

- **[4D Gaussian Splatting for Real-Time Dynamic Scene Rendering]** - https://guanjunwu.github.io/4dgs/ | [Deep Review](Final_batch/deep_review_2.md#4d-gaussian-splatting-for-real-time-dynamic-scene-rendering) | [Brief Summary](Final_batch/paper_index_2.md#4d-gaussian-splatting-for-real-time-dynamic-scene-rendering)

- **[Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis]** - https://oppo-us-research.github.io/SpacetimeGaussians-website/ | [Deep Review](Final_batch/deep_review_2.md#spacetime-gaussian-feature-splatting-for-real-time-dynamic-view-synthesis) | [Brief Summary](Final_batch/paper_index_2.md#spacetime-gaussian-feature-splatting-for-real-time-dynamic-view-synthesis)

- **[MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds]** - https://www.cis.upenn.edu/~leijh/projects/mosca/ | [Deep Review](Final_batch/deep_review_2.md#mosca-dynamic-gaussian-fusion-from-casual-videos-via-4d-motion-scaffolds) | [Brief Summary](Final_batch/paper_index_2.md#mosca-dynamic-gaussian-fusion-from-casual-videos-via-4d-motion-scaffolds)

- **[MoDGS: Dynamic Gaussian Splatting from Casually-captured Monocular Videos with Depth Priors]** - https://modgs.github.io/ | [Deep Review](Final_batch/deep_review_2.md#modgs-dynamic-gaussian-splatting-from-casually-captured-monocular-videos-with-depth-priors) | [Brief Summary](Final_batch/paper_index_2.md#modgs-dynamic-gaussian-splatting-from-casually-captured-monocular-videos-with-depth-priors)

- **[Deblur4DGS: 4D Gaussian Splatting from Blurry Monocular Video]** - https://deblur4dgs.github.io/ | [Deep Review](Final_batch/deep_review_2.md#deblur4dgs-4d-gaussian-splatting-from-blurry-monocular-video) | [Brief Summary](Final_batch/paper_index_2.md#deblur4dgs-4d-gaussian-splatting-from-blurry-monocular-video)

- **[VideoLifter: Lifting Videos to 3D with Fast Hierarchical Stereo Alignment]** - https://videolifter.github.io/ | [Deep Review](Final_batch/deep_review_2.md#videolifter-lifting-videos-to-3d-with-fast-hierarchical-stereo-alignment) | [Brief Summary](Final_batch/paper_index_2.md#videolifter-lifting-videos-to-3d-with-fast-hierarchical-stereo-alignment)

- **[Can Video Diffusion Model Reconstruct 4D Geometry? (Sora3R)]** - https://wayne-mai.github.io/publication/sora3r_arxiv_2025/ | [Deep Review](Final_batch/deep_review_2.md#can-video-diffusion-model-reconstruct-4d-geometry-sora3r) | [Brief Summary](Final_batch/paper_index_2.md#can-video-diffusion-model-reconstruct-4d-geometry-sora3r)

- **[D²USt3R: Enhancing 3D Reconstruction with 4D Pointmaps for Dynamic Scenes]** - https://cvlab-kaist.github.io/DDUSt3R/ | [Deep Review](Final_batch/deep_review_2.md#d²ust3r-enhancing-3d-reconstruction-with-4d-pointmaps-for-dynamic-scenes) | [Brief Summary](Final_batch/paper_index_2.md#d²ust3r-enhancing-3d-reconstruction-with-4d-pointmaps-for-dynamic-scenes)

- **[Gaussian Splashing: Unified Particles for Versatile Motion Synthesis and Rendering]** - https://gaussiansplashing.github.io/ | [Deep Review](Final_batch/deep_review_2.md#gaussian-splashing-unified-particles-for-versatile-motion-synthesis-and-rendering) | [Brief Summary](Final_batch/paper_index_2.md#gaussian-splashing-unified-particles-for-versatile-motion-synthesis-and-rendering)

- **[Gaussian-Flow: 4D Reconstruction from Monocular Video using Gaussian Flow Fields]** - https://gaussian-flow.github.io/ | [Deep Review](Final_batch/deep_review_2.md#gaussian-flow-4d-reconstruction-from-monocular-video-using-gaussian-flow-fields) | [Brief Summary](Final_batch/paper_index_2.md#gaussian-flow-4d-reconstruction-from-monocular-video-using-gaussian-flow-fields)

- **[Feed-Forward Bullet-Time Reconstruction of Dynamic Scenes from Monocular Videos]** - https://research.nvidia.com/labs/toronto-ai/bullet-timer/ | [Deep Review](Final_batch/deep_review_2.md#feed-forward-bullet-time-reconstruction-of-dynamic-scenes-from-monocular-videos) | [Brief Summary](Final_batch/paper_index_2.md#feed-forward-bullet-time-reconstruction-of-dynamic-scenes-from-monocular-videos)

- **[GS-DiT: Advancing Video Generation with Pseudo 4D Gaussian Fields through Efficient Dense 3D Point Tracking]** - https://wkbian.github.io/Projects/GS-DiT/ | [Deep Review](Final_batch/deep_review_2.md#gs-dit-advancing-video-generation-with-pseudo-4d-gaussian-fields-through-efficient-dense-3d-point-tracking) | [Brief Summary](Final_batch/paper_index_2.md#gs-dit-advancing-video-generation-with-pseudo-4d-gaussian-fields-through-efficient-dense-3d-point-tracking)

- **[Diffusion4D: Fast Spatial-temporal Consistent 4D Generation via Video Diffusion Models]** - https://vita-group.github.io/Diffusion4D/ | [Deep Review](Final_batch/deep_review_2.md#diffusion4d-fast-spatial-temporal-consistent-4d-generation-via-video-diffusion-models) | [Brief Summary](Final_batch/paper_index_2.md#diffusion4d-fast-spatial-temporal-consistent-4d-generation-via-video-diffusion-models)

- **[FreeTimeGS: Free-moving Gaussians in Space-time for Dynamic Scene Reconstruction]** - http://zju3dv.github.io/freetimegs/ | [Deep Review](Final_batch/deep_review_2.md#freetimegs-free-moving-gaussians-in-space-time-for-dynamic-scene-reconstruction) | [Brief Summary](Final_batch/paper_index_2.md#freetimegs-free-moving-gaussians-in-space-time-for-dynamic-scene-reconstruction)

- **[MoDecGS: Memory-efficient Dynamic Gaussian Splatting with Motion Decomposition]** - https://kaist-viclab.github.io/MoDecGS-site/ | [Deep Review](Final_batch/deep_review_2.md#modecgs-memory-efficient-dynamic-gaussian-splatting-with-motion-decomposition) | [Brief Summary](Final_batch/paper_index_2.md#modecgs-memory-efficient-dynamic-gaussian-splatting-with-motion-decomposition)

- **[BARD-GS: Motion Blur-Aware Dynamic Gaussian Splatting for Real-Time Rendering]** - https://vulab-ai.github.io/BARD-GS/ | [Deep Review](Final_batch/deep_review_2.md#bard-gs-motion-blur-aware-dynamic-gaussian-splatting-for-real-time-rendering) | [Brief Summary](Final_batch/paper_index_2.md#bard-gs-motion-blur-aware-dynamic-gaussian-splatting-for-real-time-rendering)

---

## Semantic Understanding & Applications

*3D semantic segmentation, language-guided 3D understanding, human avatars, SLAM systems, and physics-based simulations*

- **[CLIP-GS: CLIP-Informed Gaussian Splatting for Real-time and View-consistent 3D Semantic Understanding]** - https://gbliao.github.io/CLIP-GS.github.io/ | [Deep Review](Final_batch/deep_review_3.md#clip-gs-clip-informed-gaussian-splatting-for-real-time-and-view-consistent-3d-semantic-understanding) | [Brief Summary](Final_batch/paper_index_3.md#clip-gs-clip-informed-gaussian-splatting-for-real-time-and-view-consistent-3d-semantic-understanding)

- **[LangSplat: 3D Language Gaussian Splatting]** - https://langsplat.github.io/ | [Deep Review](Final_batch/deep_review_3.md#langsplat-3d-language-gaussian-splatting) | [Brief Summary](Final_batch/paper_index_3.md#langsplat-3d-language-gaussian-splatting)

- **[Segment Any 3D Gaussians (SAGA)]** - https://jumpat.github.io/SAGA/ | [Deep Review](Final_batch/deep_review_3.md#segment-any-3d-gaussians-saga) | [Brief Summary](Final_batch/paper_index_3.md#segment-any-3d-gaussians-saga)

- **[SA-GS (Segment Anything in Gaussians)]** - https://jumpat.github.io/SA-GS/ | [Deep Review](Final_batch/deep_review_3.md#sa-gs-segment-anything-in-gaussians) | [Brief Summary](Final_batch/paper_index_3.md#sa-gs-segment-anything-in-gaussians)

- **[HeadGaS]** - https://kennyblh.github.io/HeadGaS/ | [Deep Review](Final_batch/deep_review_3.md#headgas) | [Brief Summary](Final_batch/paper_index_3.md#headgas)

- **[HumanGaussian]** - https://alvinliu0.github.io/projects/HumanGaussian | [Deep Review](Final_batch/deep_review_3.md#humangaussian) | [Brief Summary](Final_batch/paper_index_3.md#humangaussian)

- **[MonoGaussian]** - https://arxiv.org/abs/2312.00435 | [Deep Review](Final_batch/deep_review_3.md#monogaussian) | [Brief Summary](Final_batch/paper_index_3.md#monogaussian)

- **[3DGS-Avatar]** - https://neuralbodies.github.io/3DGS-Avatar/ | [Deep Review](Final_batch/deep_review_3.md#3dgs-avatar) | [Brief Summary](Final_batch/paper_index_3.md#3dgs-avatar)

- **[GaussianAvatars]** - https://shenhanqian.github.io/gaussian-avatars | [Deep Review](Final_batch/deep_review_3.md#gaussianavatars) | [Brief Summary](Final_batch/paper_index_3.md#gaussianavatars)

- **[GS-SLAM]** - https://gs-slam.github.io/ | [Deep Review](Final_batch/deep_review_3.md#gs-slam) | [Brief Summary](Final_batch/paper_index_3.md#gs-slam)

- **[Photo-SLAM]** - https://huajianup.github.io/research/PhotoSLAM/ | [Deep Review](Final_batch/deep_review_3.md#photo-slam) | [Brief Summary](Final_batch/paper_index_3.md#photo-slam)

- **[Deblur Gaussian Splatting SLAM]** - https://arxiv.org/pdf/2503.12572 | [Deep Review](Final_batch/deep_review_3.md#deblur-gaussian-splatting-slam) | [Brief Summary](Final_batch/paper_index_3.md#deblur-gaussian-splatting-slam)

- **[Text-GS]** - https://jwcho5576.github.io/text-gs.github.io/ | [Deep Review](Final_batch/deep_review_3.md#text-gs) | [Brief Summary](Final_batch/paper_index_3.md#text-gs)

- **[PhysGaussian]** - https://xpandora.github.io/PhysGaussian/ | [Deep Review](Final_batch/deep_review_3.md#physgaussian) | [Brief Summary](Final_batch/paper_index_3.md#physgaussian)

- **[Gaussian Splashing: Unified Particles for Versatile Motion Synthesis and Rendering]** - https://gaussiansplashing.github.io/ | [Deep Review](Final_batch/deep_review_3.md#gaussian-splashing-unified-particles-for-versatile-motion-synthesis-and-rendering) | [Brief Summary](Final_batch/paper_index_3.md#gaussian-splashing-unified-particles-for-versatile-motion-synthesis-and-rendering)

---

## Camera Effects & Novel View Synthesis

*Advanced camera modeling, depth-of-field effects, motion deblurring, sparse-view reconstruction, and novel view synthesis techniques*

- **[DOF-GS: Adjustable Depth-of-Field 3D Gaussian Splatting for Post-Capture Refocusing, Defocus Rendering and Blur Removal]** - https://dof-gaussian.github.io/ | [Deep Review](Final_batch/deep_review_4.md#dof-gs-adjustable-depth-of-field-3d-gaussian-splatting-for-post-capture-refocusing-defocus-rendering-and-blur-removal) | [Brief Summary](Final_batch/paper_index_4.md#dof-gs-adjustable-depth-of-field-3d-gaussian-splatting-for-post-capture-refocusing-defocus-rendering-and-blur-removal)

- **[CoCoGaussian: Leveraging Circle of Confusion for Gaussian Splatting from Defocused Images]** - https://jho-yonsei.github.io/CoCoGaussian/ | [Deep Review](Final_batch/deep_review_4.md#cocogaussian-leveraging-circle-of-confusion-for-gaussian-splatting-from-defocused-images) | [Brief Summary](Final_batch/paper_index_4.md#cocogaussian-leveraging-circle-of-confusion-for-gaussian-splatting-from-defocused-images)

- **[Deblurring 3D Gaussian Splatting]** - https://benhenryl.github.io/Deblurring-3D-Gaussian-Splatting/ | [Deep Review](Final_batch/deep_review_4.md#deblurring-3d-gaussian-splatting) | [Brief Summary](Final_batch/paper_index_4.md#deblurring-3d-gaussian-splatting)

- **[Deblur-GS: 3D Gaussian Splatting from Camera Motion Blurred Images]** - https://chaphlagical.icu/Deblur-GS/ | [Deep Review](Final_batch/deep_review_4.md#deblur-gs-3d-gaussian-splatting-from-camera-motion-blurred-images) | [Brief Summary](Final_batch/paper_index_4.md#deblur-gs-3d-gaussian-splatting-from-camera-motion-blurred-images)

- **[DeblurGS: Gaussian Splatting for Camera Motion Blur]** - https://arxiv.org/abs/2404.11358 | [Deep Review](Final_batch/deep_review_4.md#deblurgs-gaussian-splatting-for-camera-motion-blur) | [Brief Summary](Final_batch/paper_index_4.md#deblurgs-gaussian-splatting-for-camera-motion-blur)

- **[ViewCrafter: Taming Video Diffusion Models for High-fidelity Novel View Synthesis]** - https://drexubery.github.io/ViewCrafter/ | [Deep Review](Final_batch/deep_review_4.md#viewcrafter-taming-video-diffusion-models-for-high-fidelity-novel-view-synthesis) | [Brief Summary](Final_batch/paper_index_4.md#viewcrafter-taming-video-diffusion-models-for-high-fidelity-novel-view-synthesis)

- **[MultiDiff: Consistent Novel View Synthesis from a Single Image]** - https://sirwyver.github.io/MultiDiff/ | [Deep Review](Final_batch/deep_review_4.md#multidiff-consistent-novel-view-synthesis-from-a-single-image) | [Brief Summary](Final_batch/paper_index_4.md#multidiff-consistent-novel-view-synthesis-from-a-single-image)

- **[NVS-Solver: Video Diffusion Model as Zero-Shot Novel View Synthesizer]** - https://mengyou2.github.io/NVS_Solver/ | [Deep Review](Final_batch/deep_review_4.md#nvs-solver-video-diffusion-model-as-zero-shot-novel-view-synthesizer) | [Brief Summary](Final_batch/paper_index_4.md#nvs-solver-video-diffusion-model-as-zero-shot-novel-view-synthesizer)

- **[InstantSplat: Sparse-view Gaussian Splatting in Seconds]** - https://instantsplat.github.io/ | [Deep Review](Final_batch/deep_review_4.md#instantsplat-sparse-view-gaussian-splatting-in-seconds) | [Brief Summary](Final_batch/paper_index_4.md#instantsplat-sparse-view-gaussian-splatting-in-seconds)

- **[Generative Camera Dolly: Extreme Monocular Dynamic Novel View Synthesis]** - https://gcd.cs.columbia.edu/ | [Deep Review](Final_batch/deep_review_4.md#generative-camera-dolly-extreme-monocular-dynamic-novel-view-synthesis) | [Brief Summary](Final_batch/paper_index_4.md#generative-camera-dolly-extreme-monocular-dynamic-novel-view-synthesis)

- **[AnyCam: Learning to Recover Camera Poses and Intrinsics from Casual Videos]** - https://fwmb.github.io/anycam/ | [Deep Review](Final_batch/deep_review_4.md#anycam-learning-to-recover-camera-poses-and-intrinsics-from-casual-videos) | [Brief Summary](Final_batch/paper_index_4.md#anycam-learning-to-recover-camera-poses-and-intrinsics-from-casual-videos)

- **[PF3plat: Pose-Free Feed-Forward 3D Gaussian Splatting]** - https://cvlab-kaist.github.io/PF3plat/ | [Deep Review](Final_batch/deep_review_4.md#pf3plat-pose-free-feed-forward-3d-gaussian-splatting) | [Brief Summary](Final_batch/paper_index_4.md#pf3plat-pose-free-feed-forward-3d-gaussian-splatting)

- **[Feed-Forward Bullet-Time Reconstruction of Dynamic Scenes from Monocular Videos]** - https://research.nvidia.com/labs/toronto-ai/bullet-timer/ | [Deep Review](Final_batch/deep_review_4.md#feed-forward-bullet-time-reconstruction-of-dynamic-scenes-from-monocular-videos) | [Brief Summary](Final_batch/paper_index_4.md#feed-forward-bullet-time-reconstruction-of-dynamic-scenes-from-monocular-videos)

- **[Multi-View Regulated Gaussian Splatting for Novel View Synthesis]** - https://xiaobiaodu.github.io/mvgs-project/ | [Deep Review](Final_batch/deep_review_4.md#multi-view-regulated-gaussian-splatting-for-novel-view-synthesis) | [Brief Summary](Final_batch/paper_index_4.md#multi-view-regulated-gaussian-splatting-for-novel-view-synthesis)

- **[DepthSplat: Connecting Gaussian Splatting and Depth]** - https://haofeixu.github.io/depthsplat/ | [Deep Review](Final_batch/deep_review_4.md#depthsplat-connecting-gaussian-splatting-and-depth) | [Brief Summary](Final_batch/paper_index_4.md#depthsplat-connecting-gaussian-splatting-and-depth)

- **[Deblur4DGS: 4D Gaussian Splatting from Blurry Monocular Video]** - https://deblur4dgs.github.io/ | [Deep Review](Final_batch/deep_review_4.md#deblur4dgs-4d-gaussian-splatting-from-blurry-monocular-video) | [Brief Summary](Final_batch/paper_index_4.md#deblur4dgs-4d-gaussian-splatting-from-blurry-monocular-video)

---

## Video Understanding & Self-Supervised Learning

*Self-supervised representation learning, video foundation models, depth estimation, video generation, and temporal consistency methods*

- **[V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video]** - https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/ | [Deep Review](Final_batch/deep_review_5.md#v-jepa-revisiting-feature-prediction-for-learning-visual-representations-from-video) | [Brief Summary](Final_batch/paper_index_5.md#v-jepa-revisiting-feature-prediction-for-learning-visual-representations-from-video)

- **[V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning]** - https://ai.meta.com/vjepa/ | [Deep Review](Final_batch/deep_review_5.md#v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning) | [Brief Summary](Final_batch/paper_index_5.md#v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning)

- **[VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training]** - https://arxiv.org/abs/2203.12602 | [Deep Review](Final_batch/deep_review_5.md#videomae-masked-autoencoders-are-data-efficient-learners-for-self-supervised-video-pre-training) | [Brief Summary](Final_batch/paper_index_5.md#videomae-masked-autoencoders-are-data-efficient-learners-for-self-supervised-video-pre-training)

- **[VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking]** - https://huggingface.co/collections/OpenGVLab/videomae-v2-678631493ab2f0c4642d842d | [Deep Review](Final_batch/deep_review_5.md#videomae-v2-scaling-video-masked-autoencoders-with-dual-masking) | [Brief Summary](Final_batch/paper_index_5.md#videomae-v2-scaling-video-masked-autoencoders-with-dual-masking)

- **[OmniMAE: Single Model Masked Pretraining on Images and Videos]** - https://facebookresearch.github.io/omnivore/ | [Deep Review](Final_batch/deep_review_5.md#omnimae-single-model-masked-pretraining-on-images-and-videos) | [Brief Summary](Final_batch/paper_index_5.md#omnimae-single-model-masked-pretraining-on-images-and-videos)

- **[Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data]** - https://depth-anything.github.io/ | [Deep Review](Final_batch/deep_review_5.md#depth-anything-unleashing-the-power-of-large-scale-unlabeled-data) | [Brief Summary](Final_batch/paper_index_5.md#depth-anything-unleashing-the-power-of-large-scale-unlabeled-data)

- **[Video Depth Anything: Consistent Depth Estimation for Super-Long Videos]** - https://videodepthanything.github.io/ | [Deep Review](Final_batch/deep_review_5.md#video-depth-anything-consistent-depth-estimation-for-super-long-videos) | [Brief Summary](Final_batch/paper_index_5.md#video-depth-anything-consistent-depth-estimation-for-super-long-videos)

- **[4D Gaussian Splatting for Real-Time Dynamic Scene Rendering]** - https://guanjunwu.github.io/4dgs/ | [Deep Review](Final_batch/deep_review_5.md#4d-gaussian-splatting-for-real-time-dynamic-scene-rendering) | [Brief Summary](Final_batch/paper_index_5.md#4d-gaussian-splatting-for-real-time-dynamic-scene-rendering)

- **[Reangle-A-Video: 4D Video Generation as Video-to-Video Translation]** - https://hyeonho99.github.io/reangle-a-video/ | [Deep Review](Final_batch/deep_review_5.md#reangle-a-video-4d-video-generation-as-video-to-video-translation) | [Brief Summary](Final_batch/paper_index_5.md#reangle-a-video-4d-video-generation-as-video-to-video-translation)

- **[TrajectoryCrafter: Redirecting Camera Trajectory for Monocular Videos via Diffusion Models]** - https://trajectorycrafter.github.io/ | [Deep Review](Final_batch/deep_review_5.md#trajectorycrafter-redirecting-camera-trajectory-for-monocular-videos-via-diffusion-models) | [Brief Summary](Final_batch/paper_index_5.md#trajectorycrafter-redirecting-camera-trajectory-for-monocular-videos-via-diffusion-models)

- **[Hierarchical Masked 3D Diffusion Model for Video Outpainting]** - https://fanfanda.github.io/M3DDM/ | [Deep Review](Final_batch/deep_review_5.md#hierarchical-masked-3d-diffusion-model-for-video-outpainting) | [Brief Summary](Final_batch/paper_index_5.md#hierarchical-masked-3d-diffusion-model-for-video-outpainting)

- **[GlobalPaint: Spatiotemporal Coherent Video Outpainting with Global Feature Guidance]** - https://globalpaint.github.io/GlobalPaint/ | [Deep Review](Final_batch/deep_review_5.md#globalpaint-spatiotemporal-coherent-video-outpainting-with-global-feature-guidance) | [Brief Summary](Final_batch/paper_index_5.md#globalpaint-spatiotemporal-coherent-video-outpainting-with-global-feature-guidance)

- **[OutDreamer: Video Outpainting with a Diffusion Transformer]** - https://arxiv.org/abs/2506.22298 | [Deep Review](Final_batch/deep_review_5.md#outdreamer-video-outpainting-with-a-diffusion-transformer) | [Brief Summary](Final_batch/paper_index_5.md#outdreamer-video-outpainting-with-a-diffusion-transformer)

- **[NormalCrafter: Learning Temporally Consistent Normals from Video Diffusion Priors]** - https://normalcrafter.github.io/ | [Deep Review](Final_batch/deep_review_5.md#normalcrafter-learning-temporally-consistent-normals-from-video-diffusion-priors) | [Brief Summary](Final_batch/paper_index_5.md#normalcrafter-learning-temporally-consistent-normals-from-video-diffusion-priors)

- **[Diffusion4D: Fast Spatial-temporal Consistent 4D Generation via Video Diffusion Models]** - https://vita-group.github.io/Diffusion4D/ | [Deep Review](Final_batch/deep_review_5.md#diffusion4d-fast-spatial-temporal-consistent-4d-generation-via-video-diffusion-models) | [Brief Summary](Final_batch/paper_index_5.md#diffusion4d-fast-spatial-temporal-consistent-4d-generation-via-video-diffusion-models)

---

## 📚 Additional Resources

### Deep Reviews
For comprehensive technical analysis and comparisons:
- [Deep Review 1: 3D Gaussian Splatting & Core Techniques](Final_batch/deep_review_1.md)
- [Deep Review 2: Dynamic Scenes & 4D Reconstruction](Final_batch/deep_review_2.md)
- [Deep Review 3: Semantic Understanding & Applications](Final_batch/deep_review_3.md)
- [Deep Review 4: Camera Effects & Novel View Synthesis](Final_batch/deep_review_4.md)
- [Deep Review 5: Video Understanding & Self-Supervised Learning](Final_batch/deep_review_5.md)

### Paper Indices
For quick reference and paper summaries:
- [Paper Index 1: 3D Gaussian Splatting & Core Techniques](Final_batch/paper_index_1.md)
- [Paper Index 2: Dynamic Scenes & 4D Reconstruction](Final_batch/paper_index_2.md)
- [Paper Index 3: Semantic Understanding & Applications](Final_batch/paper_index_3.md)
- [Paper Index 4: Camera Effects & Novel View Synthesis](Final_batch/paper_index_4.md)
- [Paper Index 5: Video Understanding & Self-Supervised Learning](Final_batch/paper_index_5.md)

---

## 🎯 Research Trends & Key Insights

### Emerging Patterns
- **Real-time Performance**: Strong emphasis on real-time rendering (60+ FPS) across all categories
- **Self-Supervised Learning**: Growing adoption of masked autoencoders and feature prediction
- **4D Understanding**: Increasing focus on temporal consistency and dynamic scene modeling
- **Foundation Models**: Scaling up to billion-parameter models with large-scale pretraining
- **Cross-modal Integration**: Combining vision, language, and 3D understanding

### Technical Innovations
- **Gaussian Optimization**: Advanced pruning, compression, and memory efficiency techniques
- **Temporal Modeling**: Novel approaches to motion blur, deformation, and temporal consistency
- **Semantic Integration**: CLIP-based 3D understanding and language-guided 3D generation
- **Camera Modeling**: Sophisticated depth-of-field, blur handling, and pose estimation
- **Video Diffusion**: Leveraging large-scale video priors for 3D and 4D generation

### Applications
- **Entertainment**: Real-time rendering, virtual production, immersive experiences
- **Robotics**: World models, planning, and manipulation
- **Autonomous Systems**: SLAM, localization, and scene understanding
- **Content Creation**: Novel view synthesis, video generation, and editing
- **Research**: Foundation models for 3D understanding and generation

---

*Last updated: 2025-07-09*
*Total papers reviewed: 65+*
*Research domains covered: 5*