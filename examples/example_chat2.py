import torch
from transformers import AutoModel, AutoTokenizer

torch.set_grad_enabled(False)


# init model and tokenizer
model = AutoModel.from_pretrained('internlm/internlm-xcomposer-7b', trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer-7b', trust_remote_code=True)
model.tokenizer = tokenizer

from accelerate import dispatch_model
device_map = {}
device_map['visual_encoder'] = 0
device_map['ln_vision'] = 0
device_map['Qformer'] = 0
device_map['internlm_model.model.embed_tokens'] = 0
device_map['internlm_model.model.norm'] = 0
device_map['internlm_model.lm_head'] = 0
device_map['query_tokens'] = 0
device_map['flag_image_start'] = 0
device_map['flag_image_end'] = 0
device_map['internlm_proj.weight'] = 0
device_map['internlm_proj.bias'] = 0
for i in range(14):
    device_map[f'internlm_model.model.layers.{i}'] = 0
for i in range(14, 32):
    device_map[f'internlm_model.model.layers.{i}'] = 1
model = dispatch_model(model, device_map=device_map)

# example image
image = 'examples/images/aiyinsitan.jpg'

# Single-Turn Pure-Text Dialogue
text = '请介绍下爱因斯坦的生平'
response = model.generate(text)
print(f'User: {text}')
print(f'Bot: {response}')
# 阿尔伯特·爱因斯坦（Albert Einstein，1879年3月14日-1955年4月18日）是德国出生的理论物理学家。他提出了狭义相对论和广义相对论，
# 这两个理论对现代物理学产生了深远的影响。爱因斯坦还发现了光电效应定律，并因此获得了1921年的诺贝尔物理学奖。
# 爱因斯坦于1879年3月14日出生于德国巴登-符腾堡州乌尔姆市的一个犹太人家庭。他在瑞士苏黎世联邦理工学院学习物理学和数学， # 并于1905年发表了一系列重要论文，其中包括狭义相对论和光电效应定律。
# 1915年，爱因斯坦发表了广义相对论，该理论解释了引力是如何通过时空弯曲来影响物体的运动。这一理论改变了人们对宇宙的认识，并为现代宇宙学奠定了基础。
# 1933年，爱因斯坦因为他的犹太血统而受到纳粹党的迫害，被迫离开德国。他最终定居在美国，并在那里度过了他的余生。1955年4月18日，爱因斯坦在普林斯顿去世，享年76岁。
# 爱因斯坦的贡献对现代物理学产生了深远的影响，他被认为是20世纪最伟大的科学家之一。

# # Single-Turn Text-Image Dialogue
# text = '请问这张图片里面的人是谁？并介绍下他。'
# image = 'examples/images/aiyinsitan.jpg'
# response = model.generate(text, image)
# print(f'User: {text}')
# print(f'Bot: {response}')
# # 图片里的人是阿尔伯特·爱因斯坦（Albert Einstein），一位著名的物理学家和理论物理学家。他于1879年3月14日出生于德国巴登-符腾堡州的乌尔姆市，
# # 并在那里度过了他的 童年和少年时代。爱因斯坦在瑞士苏黎世联邦理工学院学习物理学，并于1905年发表了一系列重要论文，
# # 其中包括狭义相对论和质能方程E=mc^2。1921年，爱因斯坦获得了诺贝尔物理学奖，以表彰他对光电效应的发现和对狭义相对论的贡献。
#
# # Multi-Turn Text-Image Dialogue
# # 1st turn
# text = '图片里面的是谁？'
# response, history = model.chat(text=text, image=image, history=None)
# print(f'User: {text}')
# print(f'Bot: {response}')
# # 图片里面的人物是阿尔伯特·爱因斯坦（Albert Einstein），一位著名的物理学家和理论物理学家。
#
# # 2nd turn
# text = '他有哪些成就?'
# response, history = model.chat(text=text, image=None, history=history)
# print(f'User: {text}')
# print(f'Bot: {response}')
# # 阿尔伯特·爱因斯坦是20世纪最伟大的物理学家之一，他提出了狭义相对论和广义相对论，为现代物理学的发展做出了巨大的贡献。
# # 此外，他还提出了光量子理论、质能关系等重要理论，对现代物理学的发展产生了深远的影响。
#
# # 3rd turn
# text = '他是最伟大的物理学家吗?'
# response, history = model.chat(text=text, image=None, history=history)
# print(f'User: {text}')
# print(f'Bot: {response}')
# # 是的，阿尔伯特·爱因斯坦是20世纪最伟大的物理学家之一。他提出了狭义相对论和广义相对论，为现代物理学的发展做出了巨大的贡献。
