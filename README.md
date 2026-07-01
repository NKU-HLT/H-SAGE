([简体中文](./README_zh.md)|English)

# H-SAGE: Holistic Speaker-Aware Guided Experts for MoE-based Multi-Talker ASR



H-SAGE (Holistic Speaker-Aware Guided Experts) is an architecture designed for Multi-Talker Automatic Speech Recognition (MTASR), aiming to improve transcription performance under overlapped speech scenarios. Built upon GLAD, H-SAGE further enhances both global acoustic modeling and expert routing mechanisms.

Our design is motivated by two key observations:

- **Stronger global information extraction benefits speaker discrimination**. Rich global acoustic representations can provide the encoder with clearer speaker activity cues, enabling better separation of multiple speakers and improving recognition performance.
- **Expert routing should jointly consider global and local information**. Rather than relying solely on local acoustic features, expert allocation should leverage both global contextual information and local fine-grained cues for more effective routing decisions.

## Overall Framework

The overall architecture of H-SAGE is illustrated in Figure 1.

<p align="center"> <img src="assets/H-SAGE.png" width="600"/> </p>

<p align="center"><em>Figure 1: Overview of the proposed H-SAGE framework. (a) SA-Encoder extracts global representations and is explicitly supervised by OA Loss. (b) Detailed architecture of SA-Encoder based on a self-attention mechanism. (c) Holistic Gating assigns expert weights by jointly considering global and local information.</em></p>

### Enhancing Global Representation Learning

Existing MoE-based MTASR approaches, such as GLAD, obtain global information through simple linear projections without explicit supervision, which limits their ability to model long-range dependencies and speaker dynamics.

To address this limitation, we replace the original global feature extractor with a Speaker-Aware Encoder (SA-Encoder). Built upon self-attention, SA-Encoder captures long-range acoustic dependencies and generates more expressive global representations.

Moreover, we introduce an Overlap-Aware Loss (OA Loss) to explicitly supervise the extracted global features. By predicting frame-level acoustic states—including silence, single-speaker speech, and overlapped speech—the model is encouraged to learn speaker-aware global representations.

<p align="center"> <img src="assets/OALoss.png" width="600"/> </p>

<p align="center"><em>Figure 2: Overlap-Aware Loss</em></p>

This design demonstrates that stronger global representation learning can effectively guide the encoder to distinguish different speakers, leading to improved MTASR performance. It also suggests that more powerful feature extractors and supervision objectives remain promising directions for further advancing multi-talker speech recognition.

### Jointly Leveraging Global and Local Information

Beyond improving global modeling, we further revisit the expert routing strategy.

As shown in Figure 1(c), conventional MoE architectures mainly rely on local features for expert assignment, which may overlook useful global contextual information.

To address this issue, we propose Holistic Gating, where expert probabilities are computed by jointly integrating local acoustic cues and global contextual representations.

Experimental results demonstrate that incorporating both global and local information into expert routing leads to more effective expert collaboration and further improves recognition performance in MoE-based MTASR.

Additional analyses are provided in the ablation study section.


## Contact
If you have any questions or are interested in collaboration, feel free to contact us via email:

guoyujie02@mail.nankai.edu.cn

## Citation
If you find our work or code helpful, please consider citing our paper and giving this repository a ⭐.

```
TODO
```

## Acknowledgements

This project is built upon the [ESPnet](https://github.com/espnet/espnet) framework.

We would like to thank the following open-source projects, which inspired and supported parts of our implementation:

- [LibrispeechMix](https://github.com/NaoyukiKanda/LibriSpeechMix)
- [Speaker-Aware-CTC](https://github.com/kjw11/Speaker-Aware-CTC)
- [GLAD](https://github.com/NKU-HLT/H-SAGE)