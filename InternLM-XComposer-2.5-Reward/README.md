<p align="center">
    <img src="assets/logo_en.png" width="650"/>
</p>
<p align="center">
    <b><font size="6">InternLM-XComposer2.5-Reward (IXC-2.5-Reward) </font></b>
</p>


<div align="center">
        InternLM-XComposer2.5-Reward <a href="https://huggingface.co/internlm/internlm-xcomposer2d5-7b-reward">🤗</a> <a href="https://modelscope.cn/models/Shanghai_AI_Laboratory/internlm-xcomposer2d5-7b-reward"><img src="../assets/modelscope_logo.png" width="20px"></a> &nbsp｜ IXC-2.5-Reward Technical Report <a href="https://arxiv.org/pdf/2501.xxxxx">  📄 </a>  
 

[English](./README.md) | [简体中文](./README_CN.md)

</div>

<p align="center">
    👋 join us on <a href="https://discord.gg/xa29JuW87d" target="_blank">Discord</a> and <a href="https://r.vansin.top/?r=internwx" target="_blank">WeChat</a>
</p>

<p align="center">
<a href="https://trendshift.io/repositories/5245" target="_blank"><img src="https://trendshift.io/api/badge/repositories/5245" alt="InternLM%2FInternLM-XComposer | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<br>

## Requirements

- python 3.8 and above
- pytorch 1.12 and above, 2.0 and above are recommended
- CUDA 11.4 and above are recommended (this is for GPU users)
- [flash-attention2](https://github.com/Dao-AILab/flash-attention) is required for high-resolution usage of InternLM-XComposer2.5.
  <br>

## Installation

Before running the code, make sure you have set up the environment and installed the required packages. Make sure you meet the above requirements, and then install the dependent libraries.
Please refer to the [installation instructions](../docs/install.md)

## Quickstart

Here is an example of how to use the model to get the reward score of a chat, compare two chats, or rank multiple chats.

```python
import torch
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained(
    "internlm/internlm-xcomposer2d5-7b-reward", 
    device_map="cuda", 
    torch_dtype=torch.float16, 
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("internlm/internlm-xcomposer2d5-7b-reward", trust_remote_code=True)
model.tokenizer = tokenizer

chat_1 = [
    {"role": "user", "content": 'I want to buy a car from the input image, analyze the advantages and weaknesses.'},
    {"role": "assistant", "content": "The car in the image is a Mercedes-Benz G-Class. This is a luxury SUV known for its ruggedness, off-road capability, and iconic boxy design. Here's an analysis of its advantages and weaknesses: Advantages: Off-road capability: The G-Class is built on a robust ladder frame chassis and has a powerful engine, making it highly capable in off-road conditions. It features a low-range transfer case, locking differentials, and a high ground clearance. Luxury and comfort: Despite its rugged exterior, the G-Class offers a luxurious interior with high-quality materials, comfortable seats, and advanced technology features. Iconic design: The distinctive boxy shape of the G-Class has made it a cult classic. It's instantly recognizable and has a timeless appeal. Strong resale value: Due to its exclusivity and desirability, the G-Class tends to hold its value well over time. Weaknesses: Fuel efficiency: The G-Class is not known for its fuel efficiency, especially with the larger engine options. Ride quality: The firm suspension is designed for off-road performance, which can make the ride a bit bumpy on paved roads. Price: The G-Class is a very expensive vehicle, even for a luxury SUV. Size and maneuverability: The large size of the G-Class can make it challenging to park and maneuver in tight spaces."}
]
chat_2 = [
    {"role": "user", "content": 'I want to buy a car from the input image, analyze the advantages and weaknesses.'},
    {"role": "assistant", "content": "Based on the image, it appears to be a Ferrari F8 Tributo. Advantages: Performance: Ferraris are renowned for their exceptional performance. The F8 Tributo is no exception, boasting a powerful twin-turbocharged V8 engine that delivers exhilarating acceleration and top speed. Handling: Ferraris are known for their precise handling and agility. The F8 Tributo continues this tradition, with advanced aerodynamics and sophisticated suspension systems ensuring a thrilling driving experience. Design: Ferrari is synonymous with iconic and head-turning designs. The F8 Tributo is a testament to this, featuring sleek lines, aggressive styling, and a captivating presence. Exclusivity: Owning a Ferrari is a status symbol. It represents luxury, exclusivity, and a passion for high-performance automobiles. Weaknesses: Price: Ferraris come with a hefty price tag. The F8 Tributo is no exception, making it an investment for those with significant financial resources. Fuel Efficiency: High-performance sports cars like the F8 Tributo are not known for their fuel efficiency. You can expect lower miles per gallon compared to everyday vehicles. Maintenance Costs: Owning a Ferrari comes with associated maintenance costs, which can be higher than those for regular cars. Practicality: The F8 Tributo is primarily a two-seater sports car, making it less practical for everyday use or carrying passengers. Ride Comfort: While the F8 Tributo offers a thrilling driving experience, its stiff suspension might not be ideal for long-distance comfort."}
]
image = ['./examples/cars1.jpg']
hd_num = 9


# get reward score for a single chat
with torch.autocast(device_type='cuda', dtype=torch.float16):
    score1 = model.get_score(chat_1, image, hd_num=hd_num)
    score2 = model.get_score(chat_2, image, hd_num=hd_num)
print("score1: ", score1)
print("score2: ", score2)
# >>> score1:  5.76
# >>> score2:  -2.84375


# batch inference, get multiple scores at once
with torch.autocast(device_type='cuda', dtype=torch.float16):
    scores = model.get_scores([chat_1, chat_2], [image, image], hd_num=hd_num)
print("scores: ", scores)
# >>> scores:  [5.76171875, -2.845703125]


# compare whether chat_1 is better than chat_2
with torch.autocast(device_type='cuda', dtype=torch.float16):
    compare_res = model.compare(chat_1, image, chat_2, image, hd_num=hd_num)
print("compare_res: ", compare_res)
# >>> compare_res:  True


# rank multiple chats, it will return the ranking index of each chat
# the chat with the highest score will have ranking index as 0
with torch.autocast(device_type='cuda', dtype=torch.float16):
    rank_res = model.rank([chat_1, chat_2], [image, image], hd_num=hd_num)
print("rank_res: ", rank_res)  # lower index means higher score
# >>> rank_res:  [0, 1]  
```

## Citation

If you find our models / code / papers useful in your research, please consider giving ⭐ and citations 📝, thx :)

```BibTeX
@article{internlmxcomposer2_5_Reward,
      title={InternLM-XComposer2.5-Reward: A Simple Yet Effective Multi-Modal Reward Model}, 
      author={Yuhang Zang and Xiaoyi Dong and Pan Zhang and Yuhang Cao and Ziyu Liu and Shengyuan Ding and Shenxi Wu and Yubo Ma and Haodong Duan and Wenwei Zhang and Kai Chen and Dahua Lin and Jiaqi Wang},
      journal={arXiv preprint arXiv:2501.xxxxx},
      year={2024}
}
```

```BibTeX
@article{internlmxcomposer2_5_OL,
      title={InternLM-XComposer2.5-OmniLive: A Comprehensive Multimodal System for Long-term Streaming Video and Audio Interactions}, 
      author={Pan Zhang and Xiaoyi Dong and Yuhang Cao and Yuhang Zang and Rui Qian and Xilin Wei and Lin Chen and Yifei Li and Junbo Niu and Shuangrui Ding and Qipeng Guo and Haodong Duan and Xin Chen and Han Lv and Zheng Nie and Min Zhang and Bin Wang and Wenwei Zhang and Xinyue Zhang and Jiaye Ge and Wei Li and Jingwen Li and Zhongying Tu and Conghui He and Xingcheng Zhang and Kai Chen and Yu Qiao and Dahua Lin and Jiaqi Wang},
      journal={arXiv preprint arXiv:2412.09596},
      year={2024}
}
```

```BibTeX
@article{internlmxcomposer2_5,
      title={InternLM-XComposer-2.5: A Versatile Large Vision Language Model Supporting Long-Contextual Input and Output}, 
      author={Pan Zhang and Xiaoyi Dong and Yuhang Zang and Yuhang Cao and Rui Qian and Lin Chen and Qipeng Guo and Haodong Duan and Bin Wang and Linke Ouyang and Songyang Zhang and Wenwei Zhang and Yining Li and Yang Gao and Peng Sun and Xinyue Zhang and Wei Li and Jingwen Li and Wenhai Wang and Hang Yan and Conghui He and Xingcheng Zhang and Kai Chen and Jifeng Dai and Yu Qiao and Dahua Lin and Jiaqi Wang},
      journal={arXiv preprint arXiv:2407.03320},
      year={2024}
}
```

```BibTeX
@article{internlmxcomposer2_4khd,
      title={InternLM-XComposer2-4KHD: A Pioneering Large Vision-Language Model Handling Resolutions from 336 Pixels to 4K HD},
      author={Xiaoyi Dong and Pan Zhang and Yuhang Zang and Yuhang Cao and Bin Wang and Linke Ouyang and Songyang Zhang and Haodong Duan and Wenwei Zhang and Yining Li and Hang Yan and Yang Gao and Zhe Chen and Xinyue Zhang and Wei Li and Jingwen Li and Wenhai Wang and Kai Chen and Conghui He and Xingcheng Zhang and Jifeng Dai and Yu Qiao and Dahua Lin and Jiaqi Wang},
      journal={arXiv preprint arXiv:2404.06512},
      year={2024}
}
```

```BibTeX
@article{internlmxcomposer2,
      title={InternLM-XComposer2: Mastering Free-form Text-Image Composition and Comprehension in Vision-Language Large Model},
      author={Xiaoyi Dong and Pan Zhang and Yuhang Zang and Yuhang Cao and Bin Wang and Linke Ouyang and Xilin Wei and Songyang Zhang and Haodong Duan and Maosong Cao and Wenwei Zhang and Yining Li and Hang Yan and Yang Gao and Xinyue Zhang and Wei Li and Jingwen Li and Kai Chen and Conghui He and Xingcheng Zhang and Yu Qiao and Dahua Lin and Jiaqi Wang},
      journal={arXiv preprint arXiv:2401.16420},
      year={2024}
}
```

```BibTeX
@article{internlmxcomposer,
      title={InternLM-XComposer: A Vision-Language Large Model for Advanced Text-image Comprehension and Composition},
      author={Pan Zhang and Xiaoyi Dong and Bin Wang and Yuhang Cao and Chao Xu and Linke Ouyang and Zhiyuan Zhao and Shuangrui Ding and Songyang Zhang and Haodong Duan and Wenwei Zhang and Hang Yan and Xinyue Zhang and Wei Li and Jingwen Li and Kai Chen and Conghui He and Xingcheng Zhang and Yu Qiao and Dahua Lin and Jiaqi Wang},
      journal={arXiv preprint arXiv:2309.15112},
      year={2023}
}
```

<br>

## License & Contact Us

The code is licensed under Apache-2.0, while model weights are fully open for academic research and also allow **free** commercial usage. To apply for a commercial license, please fill in the [application form (English)](https://wj.qq.com/s2/12727483/5dba/)/[申请表（中文）](https://wj.qq.com/s2/12725412/f7c1/). For other questions or collaborations, please contact <internlm@pjlab.org.cn>.
