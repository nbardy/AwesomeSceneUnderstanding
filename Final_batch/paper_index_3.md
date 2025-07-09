# Paper Index 3 - Semantic Understanding & Applications

## CLIP-GS: CLIP-Informed Gaussian Splatting for Real-time and View-consistent 3D Semantic Understanding

**arXiv:** https://arxiv.org/abs/2404.14249  
**GitHub:** https://github.com/gbliao/CLIP-GS  
**Project Page:** https://gbliao.github.io/CLIP-GS.github.io/

**Abstract:** The recent 3D Gaussian Splatting (GS) exhibits high-quality and real-time synthesis of novel views in 3D scenes. Currently, it primarily focuses on geometry and appearance modeling, while lacking the semantic understanding of scenes. To bridge this gap, we present CLIP-GS, which integrates semantics from Contrastive Language-Image Pre-Training (CLIP) into Gaussian Splatting to efficiently comprehend 3D environments without annotated semantic data. In specific, rather than straightforwardly learning and rendering high-dimensional semantic features of 3D Gaussians, which significantly diminishes the efficiency, we propose a Semantic Attribute Compactness (SAC) approach. SAC exploits the inherent unified semantics within objects to learn compact yet effective semantic representations of 3D Gaussians, enabling highly efficient rendering (>100 FPS). Additionally, to address the semantic ambiguity, we introduce 3DCS which imposes cross-view semantic consistency constraints by leveraging refined, self-predicted pseudo-labels derived from the trained 3D Gaussian model, thereby enhancing precise and view-consistent segmentation results. Extensive experiments demonstrate that our method remarkably outperforms existing state-of-the-art approaches, achieving improvements of 17.29% and 20.81% in mIoU metric on Replica and ScanNet datasets, respectively, while maintaining real-time rendering speed. Furthermore, our approach exhibits superior performance even with sparse input data, verifying the robustness of our method.

**Tags:** 3d gaussian splatting, semantic understanding, clip, real-time rendering, scene understanding

---

## LangSplat: 3D Language Gaussian Splatting

**arXiv:** https://arxiv.org/abs/2312.16084  
**GitHub:** https://github.com/minghanqin/LangSplat  
**Project Page:** https://langsplat.github.io/

**Abstract:** Human lives in a 3D world and commonly uses natural language to interact with a 3D scene. Modeling a 3D language field to support open-ended language queries in 3D has gained increasing attention recently. This paper introduces LangSplat, which constructs a 3D language field that enables precise and efficient open-vocabulary querying within 3D spaces. Unlike existing methods that ground CLIP language embeddings in a NeRF model, LangSplat advances the field by utilizing a collection of 3D Gaussians, each encoding language features distilled from CLIP, to represent the language field. Existing methods struggle with imprecise and vague 3D language fields, which fail to discern clear boundaries between objects. We delve into this issue and propose to learn hierarchical semantics using SAM, thereby eliminating the need for extensively querying the language field across various scales and the regularization of DINO features. Extensive experiments on open-vocabulary 3D object localization and semantic segmentation demonstrate that LangSplat significantly outperforms the previous state-of-the-art method LERF. Notably, LangSplat is extremely efficient, achieving a 119 × speedup compared to LERF.

**Tags:** 3d language field, gaussian splatting, clip, open vocabulary, semantic segmentation

---

## Segment Any 3D Gaussians (SAGA)

**arXiv:** https://arxiv.org/abs/2312.00860  
**GitHub:** https://github.com/Jumpat/SegAnyGAussians  
**Project Page:** https://jumpat.github.io/SAGA/

**Abstract:** SAGA presents a highly efficient 3D promptable segmentation method based on 3D Gaussian Splatting that can segment 3D targets from 2D visual prompts within 4 ms. It uses scale-gated affinity feature for multi-granularity segmentation and employs a scale-aware contrastive training strategy. The method distills 2D segmentation capabilities into 3D affinity features and introduces soft scale gate mechanism for handling segmentation ambiguity.

**Tags:** 3d segmentation, gaussian splatting, promptable segmentation, real-time, multi-granularity

---

## SA-GS (Segment Anything in Gaussians)

**arXiv:** https://arxiv.org/abs/2312.11473  
**GitHub:** N/A  
**Project Page:** https://jumpat.github.io/SA-GS/

**Abstract:** Interactive 3D scene segmentation using Gaussian splatting

**Tags:** 3d gaussian splatting, segmentation, interactive editing

---

## HeadGaS

**arXiv:** https://arxiv.org/abs/2312.02902  
**GitHub:** N/A  
**Project Page:** https://kennyblh.github.io/HeadGaS/

**Abstract:** Real-time head avatar generation with Gaussian splatting

**Tags:** 3d gaussian splatting, head avatars, real-time rendering

---

## HumanGaussian

**arXiv:** https://arxiv.org/abs/2311.17061  
**GitHub:** N/A  
**Project Page:** https://alvinliu0.github.io/projects/HumanGaussian

**Abstract:** Real-time human modeling with Gaussian splatting

**Tags:** 3d gaussian splatting, human modeling, real-time rendering

---

## MonoGaussian

**arXiv:** https://arxiv.org/abs/2312.00435  
**GitHub:** https://github.com/mulinmeng/MonoGaussian  
**Project Page:** N/A

**Abstract:** Head avatar reconstruction from monocular video

**Tags:** 3d gaussian splatting, head avatars, monocular reconstruction

---

## 3DGS-Avatar

**arXiv:** https://arxiv.org/abs/2312.09228  
**GitHub:** N/A  
**Project Page:** https://neuralbodies.github.io/3DGS-Avatar/

**Abstract:** Animatable avatar creation with Gaussian splatting

**Tags:** 3d gaussian splatting, avatars, animation

---

## GaussianAvatars

**arXiv:** https://arxiv.org/abs/2312.02069  
**GitHub:** N/A  
**Project Page:** https://shenhanqian.github.io/gaussian-avatars

**Abstract:** Photorealistic head avatars with controllable expressions

**Tags:** 3d gaussian splatting, head avatars, expression control

---

## GS-SLAM

**arXiv:** https://arxiv.org/abs/2311.11700  
**GitHub:** N/A  
**Project Page:** https://gs-slam.github.io/

**Abstract:** Real-time dense SLAM with Gaussian splatting

**Tags:** 3d gaussian splatting, slam, real-time tracking

---

## Photo-SLAM

**arXiv:** https://arxiv.org/abs/2311.16728  
**GitHub:** N/A  
**Project Page:** https://huajianup.github.io/research/PhotoSLAM/

**Abstract:** Real-time photorealistic SLAM with Gaussian splatting

**Tags:** 3d gaussian splatting, slam, photorealistic rendering

---

## Deblur Gaussian Splatting SLAM

**arXiv:** https://arxiv.org/pdf/2503.12572  
**GitHub:** N/A  
**Project Page:** N/A

**Abstract:** Real-time SLAM system with motion blur handling using sub-frame trajectory modeling

**Tags:** gaussian splatting, slam, motion blur, real-time tracking

---

## Text-GS

**arXiv:** https://arxiv.org/abs/2311.17701  
**GitHub:** N/A  
**Project Page:** https://jwcho5576.github.io/text-gs.github.io/

**Abstract:** Text-driven 3D Gaussian generation from language descriptions

**Tags:** 3d gaussian splatting, text-to-3d, language-driven generation

---

## PhysGaussian

**arXiv:** https://arxiv.org/abs/2311.12198  
**GitHub:** N/A  
**Project Page:** https://xpandora.github.io/PhysGaussian/

**Abstract:** Physics-based deformable Gaussian objects

**Tags:** 3d gaussian splatting, physics simulation, deformable objects

---

## Gaussian Splashing: Unified Particles for Versatile Motion Synthesis and Rendering

**arXiv:** https://arxiv.org/abs/2401.15318  
**GitHub:** http://amysteriouscat.github.io/GaussianSplashing  
**Project Page:** https://gaussiansplashing.github.io/

**Abstract:** We demonstrate the feasibility of integrating physics-based animations of solids and fluids with 3D Gaussian Splatting (3DGS) to create novel effects in virtual scenes. Our method uses a unified particle representation that can handle both solid and fluid dynamics while maintaining the rendering quality of 3DGS. This enables the creation of realistic physics-based animations with high-quality rendering for applications in virtual production and interactive media.

**Tags:** physics simulation, gaussian splatting, fluid dynamics, motion synthesis, unified particles

---

## GS-IR (Inverse Rendering)

**arXiv:** https://arxiv.org/abs/2311.16473  
**GitHub:** N/A  
**Project Page:** https://gs-ir.github.io/

**Abstract:** Inverse rendering with Gaussian splatting for material and lighting estimation

**Tags:** 3d gaussian splatting, inverse rendering, material estimation

---