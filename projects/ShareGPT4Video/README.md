# <img src="https://raw.githubusercontent.com/ShareGPT4V/ShareGPT4V-Resources/master/images/share4video_tight.png" style="vertical-align: -10px;" :height="50px" width="50px"> ShareGPT4Video: Improving Video Understanding and Generation with Better Captions

⭐️ [**Star to follow our team's projects !**](https://github.com/InternLM/InternLM-XComposer)

---

🚀🚀🚀 Official implementation of **ShareGPT4Video: Improving Video Understanding and Generation with Better Captions**.

Here is a video for introducing ShareGPT4Video clearly:

[![video]](https://user-images.githubusercontent.com/56393454/334307144-1b106612-447b-4e35-bb9f-9b904fa3a464.mp4)

- **Authors**: [Lin Chen*](https://lin-chen.site), [Xilin Wei*]() [Jinsong Li*](https://li-jinsong.github.io/), [Xiaoyi Dong](https://scholar.google.com/citations?user=FscToE0AAAAJ&hl=en), [Pan Zhang](https://panzhang0212.github.io/), [Yuhang Zang](https://yuhangzang.github.io/), [Zehui Chen](https://lovesnowbest.site/), [Haodong Duan](https://kennymckormick.github.io/), [Bin Lin](https://scholar.google.com.hk/citations?user=GCOVDKoAAAAJ&hl=en), [Zhenyu Tang](), [Li Yuan](https://yuanli2333.github.io/), [Yu Qiao](https://scholar.google.co.uk/citations?user=gFtI-8QAAAAJ&hl=en), [Dahua Lin](http://dahua.site/), [Feng Zhao📧](https://scholar.google.com/citations?hl=en&user=r6CvuOUAAAAJ), [Jiaqi Wang 📧](https://myownskyw7.github.io/)
- **Institutes**: University of Science and Technology of China; Shanghai AI Laboratory; Peking University;
- **Resources**: [[Paper]()] [[Project Page](https://sharegpt4video.github.io/)] [[ShareGPT4Video Dataset]()]
- **Models**: [[🤗ShareGPT4Video-8B](https://huggingface.co/Lin-Chen/sharegpt4video-8b)] [[ShareCaptioner-Video]()]
- **Demo**: [[🤗ShareGPT4Video-8B]()] [[🤗ShareCaptioner-Video]()]

## 💡 Highlights

- 🔥 A **large-scale** **highly descriptive** video-text dataset, **40K** GPT4-Vision-generated video captions, around **400K** implicit video split captions
- 🔥 A **general video captioner for various video durations, resolutions, aspect ratios**, approaching GPT4-Vision's caption capability, featuring two inference mode targeted for quality and efficiency, separately.
- 🔥 A superior large video-language model **ShareGPT4Video-8B**, lasting **4 hours** on 8xA100 GPUs of training respectively.
- 🔥 **Improving Text-to-Video performance** with high-quality video captions generate by our ShareCaptioner-Video

## 📜 News

**[2024/5/27]** The [ShareGPT4Video-8B](https://huggingface.co/Lin-Chen/sharegpt4video-8b) model is released!

**[2024/5/26]** The [ShareGPT4Video dataset](https://huggingface.co/datasets/ShareGPT4Video/ShareGPT4Video) and [project page](https://sharegpt4video.github.io/) are released!

## 👨‍💻 Todo

- [ ] Training and evaluation code for ShareGPT4Video-8B
- [ ] Local ShareCaptioner-Video
- [ ] Web demo and local demo of ShareGPT4V-8B
- [x] Checkpoints of ShareGPT4Video-8B

## ❤️ Acknowledgments
- [LLaVA](https://github.com/haotian-liu/LLaVA): the codebase we built upon. Thanks for their wonderful work.
- [Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan): an excellent open-source codebase for Sora-like text-to-video implementation. Thanks for their wonderful work.
- [Open-LLaVA-NeXT](https://github.com/xiaoachen98/Open-LLaVA-NeXT): an open-source codebase for re-producing the training procedure of LLaVA-NeXT series.
