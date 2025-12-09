# EMOD

**EMOD: A Unified EEG Emotion Representation Framework Leveraging V-A Guided Contrastive Learning**

Paper is available at https://arxiv.org/abs/2511.05863 and accepted by AAAI 2026 (oral).

---

## Abstract
Emotion recognition from EEG signals is essential for affective computing and has been widely explored using deep learning. While recent deep learning approaches have achieved strong performance on single EEG emotion datasets, their generalization across datasets remains limited due to the heterogeneity in annotation schemes and data formats. Existing models typically require dataset-specific architectures tailored to input structure and lack semantic alignment across diverse emotion labels. To address these challenges, we propose EMOD: A Unified EEG Emotion Representation Framework Leveraging Valence-Arousal (V-A) Guided Contrastive Learning. EMOD learns transferable and emotion-aware representations from heterogeneous datasets by bridging both semantic and structural gaps. Specifically, we project discrete and continuous emotion labels into a unified V-A space and formulate a soft-weighted supervised contrastive loss that encourages emotionally similar samples to cluster in the latent space. To accommodate variable EEG formats, EMOD employs a flexible backbone comprising a Triple-Domain Encoder followed by a Spatial-Temporal Transformer, enabling robust extraction and integration of temporal, spectral, and spatial features. We pretrain EMOD on 8 public EEG datasets and evaluate its performance on three benchmark datasets. Experimental results show that EMOD achieves the state-of-the-art performance, demonstrating strong adaptability and generalization across diverse EEG-based emotion recognition scenarios.

---

## Framework
![EMOD Framework](./overview.pdf)

---

## Citation

If you find EMOD useful in your research, please cite:

```bibtex
@misc{chen2025emodunifiedeegemotion,
      title={EMOD: A Unified EEG Emotion Representation Framework Leveraging V-A Guided Contrastive Learning}, 
      author={Yuning Chen and Sha Zhao and Shijian Li and Gang Pan},
      year={2025},
      eprint={2511.05863},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.05863}, 
}
