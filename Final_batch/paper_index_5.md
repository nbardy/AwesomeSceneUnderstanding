# Paper Index 5 - Video Understanding & Self-Supervised Learning

## V-JEPA: Revisiting Feature Prediction for Learning Visual Representations from Video

**arXiv:** https://arxiv.org/abs/2404.08471  
**GitHub:** https://github.com/facebookresearch/jepa  
**Project Page:** https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/

**Abstract:** This paper explores feature prediction as a stand-alone objective for unsupervised learning from video and introduces V-JEPA, a collection of vision models trained solely using a feature prediction objective, without the use of pretrained image encoders, text, negative examples, reconstruction, or other sources of supervision. We train on 2 million videos and study how the size of the model and the amount of data affect performance on downstream tasks. We find that feature prediction can lead to versatile visual representations that perform well across downstream image and video tasks without adaption of the model's weights; i.e., using a frozen backbone. The largest model (ViT-H/16) achieves 81.9% on Kinetics-400, 72.2% on Something-Something-v2, and 77.9% on ImageNet1K.

**Tags:** self-supervised learning, video understanding, representation learning, feature prediction, joint-embedding predictive architecture

---

## V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning

**arXiv:** https://arxiv.org/abs/2506.09985  
**GitHub:** https://github.com/facebookresearch/vjepa2  
**Project Page:** https://ai.meta.com/vjepa/

**Abstract:** We present V-JEPA 2, an action-free joint-embedding-predictive architecture pre-trained on a video and image dataset comprising over 1 million hours of internet video. V-JEPA 2 achieves strong performance on motion understanding (77.3 top-1 accuracy on Something-Something v2) and state-of-the-art performance on human action anticipation (39.7 recall-at-5 on Epic-Kitchens-100) surpassing previous task-specific models. On video question-answering at 8B parameter scale, V-JEPA 2 achieves 84.0 on PerceptionTest and 76.9 on TempCompass. We also present V-JEPA 2-AC, a latent action-conditioned world model post-trained using less than 62 hours of unlabeled robot videos from the Droid dataset. V-JEPA 2-AC was deployed zero-shot on Franka arms in two different labs and enabled picking and placing of objects using planning with image goals. This was achieved without collecting any data from the robots in these environments, and without any task-specific training or reward.

**Tags:** self-supervised learning, video understanding, world models, robotic planning, action-conditioned prediction

---

## VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training

**arXiv:** https://arxiv.org/abs/2203.12602  
**GitHub:** https://github.com/MCG-NJU/VideoMAE  
**Project Page:** N/A

**Abstract:** Pre-training video transformers on extra large-scale datasets is generally required to achieve premier performance on relatively small datasets. In this paper, we show that video masked autoencoders (VideoMAE) are data-efficient learners for self-supervised video pre-training (SSVP). We are inspired by the recent ImageMAE and propose customized video tube masking and reconstruction. Specifically, we present the following findings: (1) An extremely high proportion of masking ratio (i.e., 90% to 95%) still yields favorable performance of VideoMAE. The temporally redundant video content enables higher masking ratio than in images. (2) VideoMAE achieves impressive results on very small datasets (i.e., around 3k-4k videos) without using any extra data. This suggests that VideoMAE could be a new paradigm for data-efficient video pre-training. (3) VideoMAE shows that data quality is more important than data quantity for SSVP. Domain-specific video data yields better performance than general-purpose video data. VideoMAE can achieve 87.4% on Kinetics-400, 75.4% on Something-Something V2, 91.3% on UCF101, and 62.6% on HMDB51 without using any extra data.

**Tags:** masked autoencoders, self-supervised learning, video understanding, data-efficient learning, video pre-training

---

## VideoMAE V2: Scaling Video Masked Autoencoders with Dual Masking

**arXiv:** https://arxiv.org/abs/2303.16727  
**GitHub:** https://github.com/OpenGVLab/VideoMAEv2  
**Project Page:** https://huggingface.co/collections/OpenGVLab/videomae-v2-678631493ab2f0c4642d842d

**Abstract:** This paper shows that video masked autoencoder (VideoMAE) is a scalable and general self-supervised pre-trainer for building video foundation models. We present a dual masking strategy for efficient pre-training, with an encoder operating on a subset of video tokens and a decoder processing another subset of video tokens. We scaled the pre-training of VideoMAE V2 to billion-level ViT-g model with million-level data size, achieving state-of-the-art performance on Kinetics (90.0% on K400) and Something-Something (68.7% on V1). The pre-trained models are publicly available.

**Tags:** masked autoencoders, video foundation models, dual masking, scalable pre-training, billion-parameter models

---

## OmniMAE: Single Model Masked Pretraining on Images and Videos

**arXiv:** https://arxiv.org/abs/2206.08356  
**GitHub:** https://github.com/facebookresearch/omnivore  
**Project Page:** https://facebookresearch.github.io/omnivore/

**Abstract:** Transformer-based architectures have become competitive across a variety of visual domains, most notably images and videos. While prior work studies these modalities in isolation, having a common architecture suggests that one can train a single unified model for multiple visual modalities. Prior attempts at unified modeling typically use architectures tailored for vision tasks, or obtain worse performance compared to single modality models. In this work, we show that masked autoencoding can be used to train a simple Vision Transformer on images and videos, without requiring any labeled data. This single model learns visual representations that are comparable to or better than single-modality representations on both image and video benchmarks, while using a much simpler architecture. Furthermore, this model can be learned by dropping 90% of the image and 95% of the video patches, enabling extremely fast training of huge model architectures. In particular, we show that our single ViT-Huge model can be finetuned to achieve 86.6% on ImageNet and 75.5% on the challenging Something Something-v2 video benchmark, setting a new state-of-the-art.

**Tags:** masked autoencoders, multi-modal learning, unified vision models, self-supervised learning, cross-modal pre-training

---

## Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data

**arXiv:** https://arxiv.org/abs/2401.10891  
**GitHub:** https://github.com/LiheYoung/Depth-Anything  
**Project Page:** https://depth-anything.github.io/

**Abstract:** This work presents Depth Anything, a highly practical solution for robust monocular depth estimation. Without pursuing novel technical modules, we aim to build a simple yet powerful foundation model dealing with any images under any circumstances. To this end, we scale up the dataset by designing a data engine to collect and automatically annotate large-scale unlabeled data (~62M), which significantly enlarges the data coverage and thus is able to reduce the generalization error. We investigate two simple yet effective strategies that make data scaling-up promising: creating a more challenging optimization target through data augmentation tools and developing auxiliary supervision to enforce the model to inherit rich semantic priors from pre-trained encoders.

**Tags:** monocular depth estimation, computer vision, foundation model, data scaling, zero-shot generalization

---

## Video Depth Anything: Consistent Depth Estimation for Super-Long Videos

**arXiv:** https://arxiv.org/abs/2501.12375  
**GitHub:** https://github.com/DepthAnything/Video-Depth-Anything  
**Project Page:** https://videodepthanything.github.io/

**Abstract:** The project presents a method for depth estimation that can be applied to arbitrarily long videos and exhibits strong quality, temporal consistency, and generalization ability.

**Tags:** video depth estimation, temporal consistency, long video processing, depth anything v2, computer vision

---

## 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering

**arXiv:** https://arxiv.org/abs/2310.08528  
**GitHub:** https://github.com/hustvl/4DGaussians  
**Project Page:** https://guanjunwu.github.io/4dgs/

**Abstract:** Representing and rendering dynamic scenes has been an important but challenging task. Especially, to accurately model complex motions, high efficiency is usually hard to guarantee. To achieve real-time dynamic scene rendering while also enjoying high training and storage efficiency, we propose 4D Gaussian Splatting (4D-GS) as a holistic representation for dynamic scenes rather than applying 3D-GS for each individual frame. In 4D-GS, a novel explicit representation containing both 3D Gaussians and 4D neural voxels is proposed. A decomposed neural voxel encoding algorithm inspired by HexPlane is proposed to efficiently build Gaussian features from 4D neural voxels and then a lightweight MLP is applied to predict Gaussian deformations at novel timestamps. Our 4D-GS method achieves real-time rendering under high resolutions, 82 FPS at an 800×800 resolution on an RTX 3090 GPU while maintaining comparable or better quality than previous state-of-the-art methods.

**Tags:** dynamic scene rendering, 4d gaussian splatting, real-time rendering, neural radiance fields, temporal modeling

---

## Reangle-A-Video: 4D Video Generation as Video-to-Video Translation

**arXiv:** https://arxiv.org/abs/2503.09151  
**GitHub:** https://github.com/HyeonHo99/Reangle-Video  
**Project Page:** https://hyeonho99.github.io/reangle-a-video/

**Abstract:** We introduce Reangle-A-Video, a unified framework for generating synchronized multi-view videos from a single input video. Unlike mainstream approaches that train multi-view video diffusion models on large-scale 4D datasets, our method reframes the multi-view video generation task as video-to-videos translation, leveraging publicly available image and video diffusion priors. Our system operates in two stages: (1) Multi-View Motion Learning: An image-to-video diffusion transformer is synchronously fine-tuned in a self-supervised manner to distill view-invariant motion from a set of warped videos. (2) Multi-View Consistent Image-to-Images Translation: The first frame of the input video is warped and inpainted into various camera perspectives under an inference-time cross-view consistency guidance using DUSt3R, generating multi-view consistent starting images.

**Tags:** 4d video generation, multi-view synthesis, video diffusion, camera control, video-to-video translation

---

## TrajectoryCrafter: Redirecting Camera Trajectory for Monocular Videos via Diffusion Models

**arXiv:** https://arxiv.org/abs/2503.05638  
**GitHub:** https://github.com/TrajectoryCrafter/TrajectoryCrafter  
**Project Page:** https://trajectorycrafter.github.io/

**Abstract:** We present TrajectoryCrafter, a novel approach to redirect camera trajectories for monocular videos. By disentangling deterministic view transformations from stochastic content generation, our method achieves precise control over user-specified camera trajectories. We propose a novel dual-stream conditional video diffusion model that concurrently integrates point cloud renders and source videos as conditions, ensuring accurate view transformations and coherent 4D content generation. Instead of leveraging scarce multi-view videos, we curate a hybrid training dataset combining web-scale monocular videos with static multi-view datasets, by an innovative double-reprojection strategy, significantly fostering robust generalization across diverse scenes.

**Tags:** camera control, video diffusion, trajectory redirection, 4d consistency, view synthesis

---

## Hierarchical Masked 3D Diffusion Model for Video Outpainting

**arXiv:** https://arxiv.org/abs/2309.02119  
**GitHub:** https://github.com/alimama-creative/M3DDM-Video-Outpainting  
**Project Page:** https://fanfanda.github.io/M3DDM/

**Abstract:** Video outpainting aims to adequately complete missing areas at the edges of video frames. Compared to image outpainting, it presents an additional challenge as the model should maintain the temporal consistency of the filled area. In this paper, we introduce a masked 3D diffusion model for video outpainting. We use the technique of mask modeling to train the 3D diffusion model. This allows us to use multiple guide frames to connect the results of multiple video clip inferences, thus ensuring temporal consistency and reducing jitter between adjacent frames. Meanwhile, we extract the global frames of the video as prompts and guide the model to obtain information other than the current video clip using cross-attention. We also introduce a hybrid coarse-to-fine inference pipeline to alleviate the artifact accumulation problem. The existing coarse-to-fine pipeline only uses the infilling strategy, which brings degradation because the time interval of the sparse frames is too large. Our pipeline benefits from bidirectional learning of the mask modeling and thus can employ a hybrid strategy of infilling and interpolation when generating sparse frames. Experiments show that our method achieves state-of-the-art results in video outpainting tasks.

**Tags:** video outpainting, 3d diffusion models, temporal consistency, cross-attention, coarse-to-fine pipeline

---

## GlobalPaint: Spatiotemporal Coherent Video Outpainting with Global Feature Guidance

**arXiv:** N/A  
**GitHub:** N/A  
**Project Page:** https://globalpaint.github.io/GlobalPaint/

**Abstract:** Video outpainting, the process of filling in missing regions at the edges of videos based on existing context, poses substantial challenges in maintaining both local and global coherence. In this paper, we introduce GlobalPaint, a novel approach for video outpainting. GlobalPaint adopts a hierarchical processing framework and employs a diffusion-based model enriched with Enhanced Spatiotemporal (EST) modules and guided by global features. Our EST modules extend pretrained spatial layers by incorporating 3D windowed attention layers alongside conventional 1D temporal layers, ensuring seamless integration. To enhance global coherence, GlobalPaint efficiently distills OpenCLIP features into manageable global features, integrating them into the outpainting process through cross-attention operations. Comprehensive evaluations on benchmark datasets demonstrate that GlobalPaint surpasses state-of-the-art models in terms of both image quality and motion naturalness.

**Tags:** video outpainting, enhanced spatiotemporal modules, global feature guidance, openclip integration, diffusion models

---

## OutDreamer: Video Outpainting with a Diffusion Transformer

**arXiv:** https://arxiv.org/abs/2506.22298  
**GitHub:** N/A  
**Project Page:** N/A

**Abstract:** Video outpainting is a challenging task of generating new video content by extending beyond the boundaries of an original input video, requiring both temporal and spatial consistency. The paper introduces a novel diffusion transformer (DiT) framework with two main components: an efficient video control branch and a conditional outpainting branch. The method aims to improve video content generation by addressing limitations in existing latent diffusion models.

**Tags:** video outpainting, diffusion transformer, computer vision, video generation, temporal consistency

---

## NormalCrafter: Learning Temporally Consistent Normals from Video Diffusion Priors

**arXiv:** https://arxiv.org/abs/2504.11427  
**GitHub:** https://github.com/Binyr/NormalCrafter  
**Project Page:** https://normalcrafter.github.io/

**Abstract:** Surface normal estimation serves as a cornerstone for computer vision applications. While numerous efforts have been devoted to static image scenarios, ensuring temporal coherence in video-based normal estimation remains a formidable challenge. The paper proposes Semantic Feature Regularization (SFR) to align diffusion features with semantic cues, introduces a two-stage training protocol leveraging latent and pixel space learning, and aims to generate temporally consistent normal sequences with intricate details.

**Tags:** surface normal estimation, temporal consistency, video diffusion, semantic features, normal sequences

---

## Diffusion4D: Fast Spatial-temporal Consistent 4D Generation via Video Diffusion Models

**arXiv:** https://arxiv.org/abs/2405.16645  
**GitHub:** https://github.com/VITA-Group/Diffusion4D  
**Project Page:** https://vita-group.github.io/Diffusion4D/

**Abstract:** The paper presents Diffusion4D, a method for fast spatial-temporal consistent 4D generation using video diffusion models. The approach supports image-to-4D, text-to-4D, and 3D-to-4D generation tasks, creating dynamic 3D content with temporal consistency. The method leverages video diffusion models to generate 4D content efficiently while maintaining spatial-temporal coherence.

**Tags:** 4D generation, video diffusion, spatial-temporal consistency, dynamic content, generative models

---

## Can Video Diffusion Model Reconstruct 4D Geometry? (Sora3R)

**arXiv:** https://arxiv.org/abs/2503.21082  
**GitHub:** N/A  
**Project Page:** https://wayne-mai.github.io/publication/sora3r_arxiv_2025/

**Abstract:** Reconstructing dynamic 3D scenes (i.e., 4D geometry) from monocular video is an important yet challenging problem. Conventional multiview geometry-based approaches often struggle with dynamic motion, whereas recent learning-based methods either require specialized 4D representation or sophisticated optimization. This paper introduces Sora3R, a novel framework that taps into the rich spatiotemporal priors of large-scale video diffusion models to directly infer 4D pointmaps from casual videos.

**Tags:** 4d reconstruction, video diffusion models, dynamic 3d scenes, monocular video, feedforward inference

---