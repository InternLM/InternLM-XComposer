# MME Benchmark

[MME](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) is a comprehensive and influential evaluation benchmark for multimodal large language models. It measures both perception and cognition abilities on a total of 14 subtasks, including existence, count, position, color, poster, celebrity, scene, landmark, artwork, OCR, commonsense reasoning, numerical calculation, text translation, and code reasoning.

InternLM-XComposer-VL achieves SOTAs on the overall perforamnce of perception and cognition evaluation.

Overall Evaluation

| Rank |      Model      |          Version         |  Score  |
|:----:|:---------------:|:------------------------:|:-------:|
| ️  1  | [InternLM-XComposer-VL](https://github.com/InternLM/InternLM-XComposer) | [InternLM-7B](https://github.com/InternLM/InternLM-XComposer) | 1919.5 |
|   2  | Qwen-VL-Chat    |        Qwen-7B            | 1848.3 |
|   3  |      MMICL      |         FlanT5xxl        | 1810.7 |
|   4  |    Skywork-MM   |      Skywork-MM-13B      | 1775.5 |
|   5  |       BLIVA     |    FlanT5xxl             | 1669.2 |


Full Metrics

```
=========== Perception ===========
total score: 1528.4488795518207
         existence  score: 190.0
         count  score: 158.33333333333331
         position  score: 126.66666666666666
         color  score: 165.0
         posters  score: 161.9047619047619
         celebrity  score: 150.2941176470588
         scene  score: 159.75
         landmark  score: 165.25
         artwork  score: 126.25
         OCR  score: 125.0
=========== Cognition ===========
total score: 391.07142857142856
         commonsense_reasoning  score: 138.57142857142858
         numerical_calculation  score: 55.0
         text_translation  score: 112.5
         code_reasoning  score: 85.0
```

## How To Reproduce Results of MME Benchmark

1. Download MME images and eval_tool from the [MME repo](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/blob/Evaluation/README.md)
2. Rearrange images by executing `python get_images.py`
3. Evaluate InternLM-XComposer-VL results by executing `python eval.py`
4. Calculate MME results by executing `python calculation.py --results_dir InternLM-XComposer-VL`, which the calculation script comes from the MME eval_tool.

