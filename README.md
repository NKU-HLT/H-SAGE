([简体中文](./README_zh.md)|English)

# H-SAGE: Holistic Speaker-Aware Guided Experts for MoE-based Multi-Talker ASR



H-SAGE (Holistic Speaker-Aware Guided Experts) is an architecture designed for Multi-Talker Automatic Speech Recognition (MTASR), aiming to improve transcription performance under overlapped speech scenarios. Built upon [GLAD](https://arxiv.org/abs/2509.13093), H-SAGE further enhances both global acoustic modeling and expert routing mechanisms.

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

Moreover, we introduce an Overlap-Aware Loss (OA Loss) to explicitly supervise the extracted global features. By predicting frame-level acoustic states, the model is encouraged to learn speaker-aware global representations.

<p align="center"> <img src="assets/OALoss.png" width="600"/> </p>

<p align="center"><em>Figure 2: Overlap-Aware Loss</em></p>

This design demonstrates that stronger global representation learning can effectively guide the encoder to distinguish different speakers, leading to improved MTASR performance. It also suggests that more powerful feature extractors and supervision objectives remain promising directions for further advancing multi-talker speech recognition.

### Jointly Leveraging Global and Local Information

Beyond improving global modeling, we further revisit the expert routing strategy.

As shown in Figure 1(c), conventional MoE architectures mainly rely on local features for expert assignment, which may overlook useful global contextual information.

To address this issue, we propose Holistic Gating, where expert probabilities are computed by jointly integrating local acoustic cues and global contextual representations.

Experimental results demonstrate that incorporating both global and local information into expert routing leads to more effective expert collaboration and further improves recognition performance in MoE-based MTASR.

Additional analyses are provided in the ablation study section.

## Q&A

**Q1: Since H-SAGE was finalized in December 2025 and GLAD has been updated afterward, how does H-SAGE perform under the new GLAD architecture? Are there corresponding experimental results?**

The recent update of GLAD introduces a key modification: each MoLE now adopts an independent Global Gating Router, whereas in the original version, all MoLEs shared a single global router. This change primarily affects the granularity of routing decisions and is orthogonal to the core design of H-SAGE.

To ensure a fair comparison, we re-train H-SAGE on top of the updated GLAD framework and evaluate both methods under identical experimental settings. We use the same datasets as in the original paper. Due to GPU memory constraints, we slightly reduce the batch size while keeping all other training hyperparameters and optimization settings unchanged. The implementation details can be found in the H-SAGE-NEW directory.

The experimental results are reported as follows. As shown in the results, H-SAGE consistently achieves comparable or better performance under the updated GLAD architecture. In particular, it demonstrates more stable improvements in low- and medium-overlap conditions as well as in the 3-speaker generalization setting, indicating strong robustness and transferability.

<div style="overflow-x: auto;">
  <table class="custom-table">
    <thead>
      <tr>
        <th rowspan="2">Method</th>
        <th rowspan="2">Parm.(M)</th>
        <th colspan="2">Librispeech</th>
        <th colspan="6">LibrispeechMix-2mix</th>
        <th colspan="6">LibrispeechMix-3mix (Generalization)</th>
      </tr>
      <tr>
        <th>Dev</th><th>Test</th>
        <th>Dev</th><th>Test</th><th>low</th><th>mid</th><th>high</th><th>OA-WER</th>
        <th>Dev</th><th>Test</th><th>low</th><th>mid</th><th>high</th><th>OA-WER</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>GLAD</td><td>35.31</td><td>3.4</td><td>3.8</td><td>6.2</td><td>6.2</td><td>5.0</td><td>6.3</td><td>9.2</td><td>6.8</td><td>20.9</td><td>21.1</td><td>14.7</td><td>21.1</td><td><b>27.3</b></td><td>21.0</td>
      </tr>
      <tr>
        <td>H-SAGE</td><td>35.83</td><td><b>3.3</b></td><td><b>3.5</b></td><td><b>6.0</b></td><td><b>5.8</b></td><td><b>4.4</b></td><td><b>5.9</b></td><td><b>9.1</b></td><td><b>6.5</b></td><td><b>19.9</b></td><td><b>19.9</b></td><td><b>13.3</b></td><td><b>19.6</b></td><td><b>27.3</b></td><td><b>20.3</b></td>
      </tr>
    </tbody>
  </table>
</div>

**Q2: Why do you adopt OA Loss instead of designing a dedicated speaker-aware loss?**
First, the purpose of introducing explicit supervision (i.e., OA Loss) is to verify that enhancing global representation learning can improve MTASR performance. From this perspective, the choice of supervision is not unique, and other loss designs may also potentially lead to performance gains.

Second, in terms of implementation, we adopt OA Loss mainly due to its good generalization capability and relatively low design complexity. OA Loss models acoustic states, which leads to a simpler model structure and better adaptability across different scenarios and speaker configurations. Therefore, we select OA Loss as our form of explicit supervision.

It is worth noting that the acoustic state labels are constructed based on the duration and temporal offsets of mixed speech segments. As a result, the supervision signals are coarse-grained and approximate, and single-speaker segments may still contain silence or other ambiguities. Nevertheless, even under such noisy and weak supervision conditions, the model consistently achieves performance improvements, further validating the effectiveness and soundness of the proposed design.

## Using H-SAGE

This project is developed based on the [ESPnet](https://github.com/espnet/espnet) framework.

**Step 1**:
Replace the `egs2`, `espnet`, and `espnet2` directories in this repository under H-SAGE-NEW or H-SAGE with the corresponding directories in the official [ESPnet](https://github.com/espnet/espnet) repository. Then modify the configuration files according to your environment (e.g., dataset paths).

**Step 2**:

Prepare the training dataset. In this work, we construct a mixed-speech dataset based on LibriSpeech with a total duration of approximately 1,770 hours, containing both single-speaker and two-speaker utterances. After preparing the data, run the corresponding data preprocessing stages in `run.sh` under egs2 to generate features.

In addition, an auxiliary target file for OA Loss must be generated in advance and placed under the directory specified by `${data_feats}/${train_set}` in run.sh. The file name must be `id2audiomask`. The file is organized in a line-by-line format, where each line corresponds to one utterance:

```
utterance_id audio_len sample_rate start_1,duration_1 ... start_n,duration_n
```

where:
- utterance_id: unique utterance identifier 
- audio_len: audio length in number of samples 
- sample_rate: sampling rate 
- start_i,duration_i: start time (in seconds) and duration (in seconds) of the i-th speaker in the mixed utterance

Examples:

Single-speaker: train-960-1mix-043830 243040 16000

Two-speaker: train-960-2mix-020034 257785 16000 0.0,5.12 3.2815494853243217,12.8300625


**Step 3**:
After data preparation, run `run.sh` under egs2. Specifically, execute Stage 11 to Stage 13. Before running Stage 12–13, use average_ckpt.py to average the last five checkpoints.

**Step 4**:
Use `run_pi_scoring.sh` for model evaluation. The evaluation script is adapted from Speaker-Aware-CTC, and we thank the authors for their open-source contribution.


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