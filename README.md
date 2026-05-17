<div align="center">

<img src="banner.png" alt="TalkDrive Banner" width="100%">

# TalkDrive — AI Talking Head Runner

<a href="https://colab.research.google.com/github/Bilal140202/TalkDrive_by_BilalAnsari/blob/main/notebooks/TalkDrive_by_BilalAnsari.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

TalkDrive — Colab runner for SoulX FlashHead 1.3B. Turn any portrait photo into a talking-head video from audio, with face-crop fix and 16:9 compositing. By Bilal Ansari.

</div>

---

### Architecture Overview

```text
Portrait photo  ──►┐
                   │  SoulX FlashHead 1.3B (DiT + Wav2Vec2)  ──► 512×512 clip
Audio / speech  ──►┘
                                           │
                                  16:9 Compositor
                                           │
                              Final video (original resolution)
```
