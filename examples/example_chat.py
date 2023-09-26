import torch
from transformers import AutoModel, AutoTokenizer

torch.set_grad_enabled(False)

# init model and tokenizer
model = AutoModel.from_pretrained('chat', trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained('chat', trust_remote_code=True)
model.tokenizer = tokenizer

# example image
image = 'examples/images/aiyinsitan.jpg'

# Single-Turn Pure-Text Dialogue
text = '请介绍下爱因斯坦的生平'
response = model.generate(text)
print(f'User: {text}')
print(f'Bot: {response}')
# '阿尔伯特·爱因斯坦（Albert Einstein，1879年3月14日－1955年4月18日），德国裔瑞士籍物理学家。他创立了现代物理学的两大支柱理论：
# 相对论和量子力学， 而质能等价公式E=mc2便是他的相对论思想的明证，因而被公认为是继伽利略、牛顿之后最伟大的物理学家。
# 1999年，爱因斯坦被美国《时代周刊》评选为20世纪的“世纪人物”，他在物理学上的贡献，使他在世界各地受到人们的尊敬。'

# Single-Turn Text-Image Dialogue
text = '请问这张图片里面的人是谁？并介绍下他。'
image = 'examples/images/aiyinsitan.jpg'
response = model.generate(text, image)
print(f'User: {text}')
print(f'Bot: {response}')
# 图片中的男子是阿尔伯特·爱因斯坦（Albert Einstein），一位著名的物理学家和理论物理学家。他于1879年3月14日出生于德国巴登-符腾堡州的乌尔姆市，
# 1955 年4月18日逝世于美国新泽西州普林斯顿市。爱因斯坦在20世纪初提出了狭义相对论和广义相对论，对现代物理学的发展产生了深远影响。

# Multi-Turn Text-Image Dialogue
# 1st turn
text = '图片里面的是谁？'
response, history = model.chat(text=text, image=image, history=None)
print(f'User: {text}')
print(f'Bot: {response}')
# 阿尔伯特·爱因斯坦。

# 2nd turn
text = '他有哪些成就?'
response, history = model.chat(text=text, image=None, history=history)
print(f'User: {text}')
print(f'Bot: {response}')
# 阿尔伯特·爱因斯坦是20世纪最伟大的物理学家之一，他提出了狭义相对论和广义相对论，对现代物理学的发展产生了深远影响。
# 此外，他还提出了著名的质能方程E=mc²，为核能的开发提供了理论基础。

# 3rd turn
text = '他是最伟大的物理学家吗?'
response, history = model.chat(text=text, image=None, history=history)
print(f'User: {text}')
print(f'Bot: {response}')
# 是的，阿尔伯特·爱因斯坦是20世纪最伟大的物理学家之一。
