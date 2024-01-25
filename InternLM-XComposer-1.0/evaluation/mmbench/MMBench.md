# MMBench & CCBench

[MMBench](https://opencompass.org.cn/leaderboard-multimodal) is a comprehensive evaluation pipeline comprised of meticulously curated multimodal dataset and a novel circulareval strategy using ChatGPT. It is comprised of 20 ability dimensions defined by MMBench. It also contains chinese version with translated question.

CCBench is an extension of MMBench with newly design questions about Chinese traditional culture, including Calligraphy Painting, Cultural Relic, Food & Clothes, Historical Figures, Scenery & Building, Sketch Reasoning and Traditional Show.


InternLM-XComposer-VL achieves SOTAs on MMbench, MMBench-CN and CCBench.

MMBench-test

| Rank |      Model      |          Version         |  Score  |
|:----:|:---------------:|:------------------------:|:-------:|
| ️  1  | InternLM-XComposer-VL | InternLM-7B | 74.4 |
|   2  |    Pink  |        Vicuna-7B            | 74.1 |
|   3  |      JiuTian      |        FLANT5-XXL        | 71.8 |
|   4  |  WeMM   |      InternLM-7B      | 69.0 |
|   5  |     mPLUG-Owl     |    LLaMA2 7B            |  68.5 |

MMBench-CN-test

| Rank |      Model      |          Version         |  Score  |
|:----:|:---------------:|:------------------------:|:-------:|
| ️  1  | InternLM-XComposer-VL | InternLM-7B | 72.4 |
|   2  |    QWen-VL-Chat | Qwen-7B | 56.3 |
|   3  |    LLaVA       | LLaMA 7B  |36.6 |
|   4  |    VosualGLM   | ChatGLM 6B | 25.6 |
|   5  |    mPLUG-Owl | LLaMA2 7B  | 24.9 |

CCBench

| Rank |      Model      |          Version         |  Score  |
|:----:|:---------------:|:------------------------:|:-------:|
| ️  1  | InternLM-XComposer-VL | InternLM-7B | 47.6 |
|   2  |    QWen-VL-Chat | Qwen-7B | 39.3 |
|   3  |    mPLUG-Owl | LLaMA2 7B  | 12.9 |
|   3  |    InstructBLIP       |        Vicuna 7B  | 12.1 |
|   4  |    VosualGLM   | ChatGLM 6B | 9.2  |



## How To Reproduce Results 

1. Download dev or test set from the [MMBench repo](https://github.com/open-compass/opencompass/blob/mm/docs/en/MMBench.md)
2. Evaluate InternLM-XComposer-VL results by executing `python eval.py`
3. Submit it to the [MMBench Submission Website](https://opencompass.org.cn/mmbench-submission)

