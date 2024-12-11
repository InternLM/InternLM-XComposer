---
license: other
pipeline_tag: visual-question-answering
---


<p align="center">
    <img src="logo_en.png" width="600"/>
<p>

<p align="center">
    <b><font size="6">InternLM-XComposer-2.5</font></b> 
<p>

<div align="center">

[💻Github Repo](https://github.com/InternLM/InternLM-XComposer)

[Online Demo](https://huggingface.co/spaces/Willow123/InternLM-XComposer)

[Paper](https://huggingface.co/papers/2407.03320)

</div>

**InternLM-XComposer2.5** excels in various text-image comprehension and composition applications, achieving GPT-4V level capabilities with merely 7B LLM backend. IXC2.5 is trained with 24K interleaved image-text contexts, it can seamlessly extend to 96K long contexts via RoPE extrapolation. This long-context capability allows IXC-2.5 to excel in tasks requiring extensive input and output contexts. 


### Import from Transformers
To load the InternLM-XComposer2-4KHD model using Transformers, use the following code:
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
ckpt_path = "internlm/internlm-xcomposer2d5-7b"
tokenizer = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True).cuda()
# Set `torch_dtype=torch.floatb16` to load model in bfloat16, otherwise it will be loaded as float32 and might cause OOM Error.
model = AutoModelForCausalLM.from_pretrained(ckpt_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
model = model.eval()
```

## Quickstart

We provide a simple example to show how to use InternLM-XComposer2.5 with 🤗 Transformers. 

<details>
  <summary>
    <b>Video Understanding</b>
  </summary>

```python
import torch
from transformers import AutoModel, AutoTokenizer

torch.set_grad_enabled(False)

# init model and tokenizer
model = AutoModel.from_pretrained('internlm/internlm-xcomposer2d5-7b', torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2d5-7b', trust_remote_code=True)
model.tokenizer = tokenizer

query = 'Here are some frames of a video. Describe this video in detail'
image = ['./examples/liuxiang.mp4',]
with torch.autocast(device_type='cuda', dtype=torch.float16):
    response, his = model.chat(tokenizer, query, image, do_sample=False, num_beams=3, use_meta=True)
print(response)
#The video opens with a shot of an athlete, dressed in a red and yellow uniform with the word "CHINA" emblazoned across the front, preparing for a race. 
#The athlete, Liu Xiang, is seen in a crouched position, focused and ready, with the Olympic rings visible in the background, indicating the prestigious setting of the Olympic Games. As the race commences, the athletes are seen sprinting towards the hurdles, their determination evident in their powerful strides. 
#The camera captures the intensity of the competition, with the athletes' numbers and times displayed on the screen, providing a real-time update on their performance. The race reaches a climax as Liu Xiang, still in his red and yellow uniform, triumphantly crosses the finish line, his arms raised in victory. 
#The crowd in the stands erupts into cheers, their excitement palpable as they witness the athlete's success. The video concludes with a close-up shot of Liu Xiang, still basking in the glory of his victory, as the Olympic rings continue to symbolize the significance of the event.

query = 'tell me the athlete code of Liu Xiang'
image = ['./examples/liuxiang.mp4',]
with torch.autocast(device_type='cuda', dtype=torch.float16):
    response, _ = model.chat(tokenizer, query, image, history=his, do_sample=False, num_beams=3, use_meta=True)
print(response)
#The athlete code of Liu Xiang, as displayed on his uniform in the video, is "1363".
```

</details>

<details>
  <summary>
    <b>Multi-Image Mutli-Tune Dialog</b>
  </summary>

```python
import torch
from transformers import AutoModel, AutoTokenizer

torch.set_grad_enabled(False)

# init model and tokenizer
model = AutoModel.from_pretrained('internlm/internlm-xcomposer2d5-7b', torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2d5-7b', trust_remote_code=True)
model.tokenizer = tokenizer

query = 'Image1 <ImageHere>; Image2 <ImageHere>; Image3 <ImageHere>; I want to buy a car from the three given cars, analyze their advantages and weaknesses one by one'
image = ['./examples/cars1.jpg',
        './examples/cars2.jpg',
        './examples/cars3.jpg',]
with torch.autocast(device_type='cuda', dtype=torch.float16):
    response, his = model.chat(tokenizer, query, image, do_sample=False, num_beams=3, use_meta=True)
print(response)
#To analyze the advantages and disadvantages of each car, we need to consider factors such as brand reputation, performance, design, cost, and maintenance. 
#1. Mercedes-Benz: - Advantages: Known for its luxury and reliability, the Mercedes-Benz brand offers a high level of comfort, advanced technology, and superior craftsmanship. The vehicle in the image appears to be an SUV, which is versatile for both city driving and off-road conditions. - Disadvantages: Typically, Mercedes-Benz vehicles are more expensive compared to other brands, and they may require more frequent maintenance due to their luxury status. 
#2. Bugatti: - Advantages: Bugatti is renowned for producing some of the fastest and most powerful cars in the world. The vehicle in the image is a sports car, likely offering an exhilarating driving experience with its high-performance engine and advanced aerodynamics. - Disadvantages: Bugatti cars are extremely expensive, making them less accessible to the average consumer. They also require specialized knowledge for maintenance and may not be suitable for everyday driving due to their high-performance nature. 
#3. BMW: - Advantages: BMW is a well-respected manufacturer known for its innovative engineering and design. The car in the image appears to be an electric vehicle (EV), which offers advantages such as lower emissions, reduced noise pollution, and potentially lower operating costs. - Disadvantages: EVs like the one in the image may have a shorter range compared to traditional gasoline-powered vehicles, limiting their suitability for long-distance travel. 
#Additionally, the initial cost of purchasing an EV can be higher than that of a conventional car. In conclusion, each car has its own set of advantages and disadvantages. The Mercedes-Benz offers luxury and versatility, the Bugatti provides high performance and exclusivity, and the BMW represents innovation in electric vehicle technology. The choice ultimately depends on the buyer's priorities, such as comfort, performance, environmental impact, and budget.

query = 'Image4 <ImageHere>; How about the car in Image4'
image.append('./examples/cars4.jpg')        
with torch.autocast(device_type='cuda', dtype=torch.float16):
    response, _ = model.chat(tokenizer, query, image, do_sample=False, num_beams=3, history= his, use_meta=True)
print(response)
#The car in Image4 is a red sports car, which appears to be a Ferrari. Ferrari is a renowned Italian manufacturer known for producing some of the most iconic and high-performance sports cars in the world. - Advantages: Ferrari vehicles are synonymous with speed, luxury, and engineering excellence. 
#The car in the image likely offers an exhilarating driving experience with its powerful engine, advanced aerodynamics, and high-quality craftsmanship. The red color adds to the car's aesthetic appeal, making it stand out on the road. - Disadvantages: Ferrari cars are extremely expensive, making them less accessible to the average consumer. 
#They also require specialized knowledge for maintenance and may not be suitable for everyday driving due to their high-performance nature. In conclusion, the Ferrari in Image4 represents a pinnacle of automotive engineering and design, offering unmatched performance and luxury. 
#However, its high cost and specialized maintenance requirements make it less practical for everyday use compared to the other vehicles in the images.
```


</details>

<details>
  <summary>
    <b>High Resolution Image Understanding</b>
  </summary>

```python
import torch
from transformers import AutoModel, AutoTokenizer

torch.set_grad_enabled(False)

# init model and tokenizer
model = AutoModel.from_pretrained('internlm/internlm-xcomposer2d5-7b', torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2d5-7b', trust_remote_code=True)
model.tokenizer = tokenizer

query = 'Analyze the given image in a detail manner'
image = ['./examples/dubai.png']
with torch.autocast(device_type='cuda', dtype=torch.float16):
    response, _ = model.chat(tokenizer, query, image, do_sample=False, num_beams=3, use_meta=True)
print(response)
#The infographic is a visual representation of various facts about Dubai. It begins with a statement about Palm Jumeirah, highlighting it as the largest artificial island visible from space. It then provides a historical context, noting that in 1968, there were only a few cars in Dubai, contrasting this with the current figure of more than 1.5 million vehicles. 
#The infographic also points out that Dubai has the world's largest Gold Chain, with 7 of the top 10 tallest hotels located there. Additionally, it mentions that the crime rate is near 0%, and the income tax rate is also 0%, with 20% of the world's total cranes operating in Dubai. Furthermore, it states that 17% of the population is Emirati, and 83% are immigrants.
#The Dubai Mall is highlighted as the largest shopping mall in the world, with 1200 stores. The infographic also notes that Dubai has no standard address system, with no zip codes, area codes, or postal services. It mentions that the Burj Khalifa is so tall that its residents on top floors need to wait longer to break fast during Ramadan. 
#The infographic also includes information about Dubai's climate-controlled City, with the Royal Suite at Burj Al Arab costing $24,000 per night. Lastly, it notes that the net worth of the four listed billionaires is roughly equal to the GDP of Honduras.

```

</details>


<details>
  <summary>
    <b>Instruction to Webpage</b>
  </summary>

```python
import torch
from transformers import AutoModel, AutoTokenizer

torch.set_grad_enabled(False)

# init model and tokenizer
model = AutoModel.from_pretrained('internlm/internlm-xcomposer2d5-7b', torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2d5-7b', trust_remote_code=True)
model.tokenizer = tokenizer

query = 'A website for Research institutions. The name is Shanghai AI lab. Top Navigation Bar is blue.Below left, an image shows the logo of the lab. In the right, there is a passage of text below that describes the mission of the laboratory.There are several images to show the research projects of Shanghai AI lab.'
with torch.autocast(device_type='cuda', dtype=torch.float16):
    response = model.write_webpage(query, seed=202, task='Instruction-aware Webpage Generation', repetition_penalty=3.0)
print(response)
# see the Instruction-aware Webpage Generation.html 
```
 
See the [Instruction to Webpage](https://github.com/InternLM/InternLM-XComposer/blob/main/examples/Instruction-aware_Webpage_Generation.html) results here.
</details>

<details>
  <summary>
    <b>Resume to Webpage</b>
  </summary>

```python
import torch
from transformers import AutoModel, AutoTokenizer

torch.set_grad_enabled(False)

# init model and tokenizer
model = AutoModel.from_pretrained('internlm/internlm-xcomposer2d5-7b', torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2d5-7b', trust_remote_code=True)
model.tokenizer = tokenizer

## the input should be a resume in markdown format
query = './examples/resume.md'
with torch.autocast(device_type='cuda', dtype=torch.float16):
    response = model.resume_2_webpage(query, seed=202, repetition_penalty=3.0)
print(response)
```
See the [Resume to Webpage](https://github.com/InternLM/InternLM-XComposer/blob/main/examples/Resume-to-Personal_Page.html) results here.


</details>


<details>
  <summary>
    <b>Screenshot to Webpage</b>
  </summary>

```python
import torch
from transformers import AutoModel, AutoTokenizer

torch.set_grad_enabled(False)

# init model and tokenizer
model = AutoModel.from_pretrained('internlm/internlm-xcomposer2d5-7b', torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2d5-7b', trust_remote_code=True)
model.tokenizer = tokenizer

query = 'Generate the HTML code of this web image with Tailwind CSS.'
image = ['./examples/screenshot.jpg']
with torch.autocast(device_type='cuda', dtype=torch.float16):
    response = model.screen_2_webpage(query, image, seed=202, repetition_penalty=3.0)
print(response)
```
See the [Screenshot to Webpage](https://github.com/InternLM/InternLM-XComposer/blob/main/examples/Screenshot-to-Webpage.html) results here.

</details>



<details>
  <summary>
    <b>Write Article</b>
  </summary>

```python
import torch
from transformers import AutoModel, AutoTokenizer

torch.set_grad_enabled(False)

# init model and tokenizer
model = AutoModel.from_pretrained('internlm/internlm-xcomposer2d5-7b', torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2d5-7b', trust_remote_code=True)
model.tokenizer = tokenizer

query = '阅读下面的材料，根据要求写作。 电影《长安三万里》的出现让人感慨，影片并未将重点全落在大唐风华上，也展现了恢弘气象的阴暗面，即旧门阀的资源垄断、朝政的日益衰败与青年才俊的壮志难酬。高适仕进无门，只能回乡>沉潜修行。李白虽得玉真公主举荐，擢入翰林，但他只是成为唐玄宗的御用文人，不能真正实现有益于朝政的志意。然而，片中高潮部分《将进酒》一节，人至中年、挂着肚腩的李白引众人乘仙鹤上天，一路从水面、瀑布飞升至银河进入仙>宫，李白狂奔着与仙人们碰杯，最后大家纵身飞向漩涡般的九重天。肉身的微贱、世路的“天生我材必有用，坎坷，拘不住精神的高蹈。“天生我材必有用，千金散尽还复来。” 古往今来，身处闲顿、遭受挫折、被病痛折磨，很多人都曾经历>了人生的“失意”，却反而成就了他们“诗意”的人生。对正在追求人生价值的当代青年来说，如何对待人生中的缺憾和困顿?诗意人生中又有怎样的自我坚守和自我认同?请结合“失意”与“诗意”这两个关键词写一篇文章。 要求:选准角度，确定>立意，明确文体，自拟标题;不要套作，不得抄袭;不得泄露个人信息;不少于 800 字。'
with torch.autocast(device_type='cuda', dtype=torch.float16):
    response = model.write_artical(query, seed=8192)
print(response)
#诗意人生，贵在坚守
#《菜根谭》有云:“闲时要有吃紧的心思,忙里要留吃闲工夫。”人生在世,总有失意之时,当面对缺憾和困顿,诗意地生活着才能为人生增添一抹亮色。何谓诗意地生活? 所谓诗意地生活，便是在于坚守本心、直面遗憾、超越自我,在失意中寻找人生价值。
#诗意地生活,需坚守本心,淡然处之。
#陶渊明曾执意辞去彭泽县令,归隐田园,“采菊东篱下,悠然见南山”,在山水间寄情自娱；王维面对仕途失意,终日沉醉于诗酒之中,“兴来每独往,胜事空自知”,在诗酒中闲逸自如;李白仕途不顺,被赐金放还,但他依旧豪气干云,“天生我才必有用,千金散尽还复来”,在失意中坦然豁达。坚守本心，便能在遭遇失意之时守住自己的精神家园,让生活充满诗意。反之,若不能坚守本心,而只是一味迎合世俗以求得升迁,那纵使身居高位,亦会丧失生活的乐趣。
#诗意地生活,需直面遗憾,超越自我。
#“西塞山前白鹭飞,桃花流水鳜鱼肥。青箬笠,绿柳枝,半斤酒,一纶丝。五湖四海皆如此,何妨到此处归。”白居易的《渔歌子》写出了多少人的愿望:没有权势纷扰,没有贫困凄凉,只有青山绿水、白鹭鸥鸟作伴,如此自由自在的生活令人神往。然而,白居易却并没有因此真的归隐山林,而是直面人生,超越自我,写下了一首首诗意而富有现实关怀的作品。如果白居易只顾逃避人生,那又怎会拥有“大弦嘈嘈如急雨,小弦切切如私语”的绝美比喻呢?如果白居易只顾归隐山林,那又怎会写出“此曲只应天上有,人间哪得配白居易”这样的诗句呢?
#诗意地生活,需直面遗憾,坚守本心。
#李文波患有渐冻症,医生说他活不过五年,但他没有因此放弃对音乐的热爱,而是与病魔作斗争,演奏出美妙的乐曲;孙家林自幼患有脑瘫,但他不甘于命运的捉弄,终成全国最美教师;史铁生饱受疾病折磨,但他仍能发出“我常常在我的心头清点,我有什么?”的叩问,并由此走上文学道路,为后世留下丰厚的文化遗产。这些人没有逃避,而是选择直面人生的缺憾,在坚守本心的同时超越自我,最终实现了自己的价值。
#诗意地生活,是于失意中坚守本心,于缺憾中超越自我。当面对人生的缺憾与挫折,坚守本心、超越自我的同时,也必将书写属于自己的辉煌篇章。
#愿你我都能诗意地生活着!

query = 'Please write a blog based on the title: French Pastries: A Sweet Indulgence'
with torch.autocast(device_type='cuda', dtype=torch.float16):
    response = model.write_artical(query, seed=8192)
print(response)
#French Pastries: A Sweet Indulgence
#The French are well known for their love of pastries, and it’s a love that is passed down through generations. When one visits France, they are treated to an assortment of baked goods that can range from the delicate macaron to the rich and decadent chocolate mousse. While there are many delicious types of pastries found in France, five stand out as being the most iconic. Each of these pastries has its own unique qualities that make it special.
#1. Croissant
#One of the most famous pastries from France is the croissant. It is a buttery, flaky pastry that is best enjoyed fresh from the bakery. The dough is laminated with butter, giving it its signature layers. Croissants are typically eaten for breakfast or brunch, often accompanied by coffee or hot chocolate.
#2. Macaron
#The macaron is a small, delicate French confection made from almond flour, powdered sugar, and egg whites. The macaron itself is sandwiched with a ganache or jam filling. They come in a variety of colors and flavors, making them a popular choice for both casual snacking and upscale desserts.
#3. Madeleine
#The madeleine is a small shell-shaped cake that is light and sponge-like. It is often flavored with lemon or orange zest and sometimes dipped in chocolate. Madeleines are perfect for an afternoon snack with tea or coffee.
#4. Éclair
#The éclair is a long, thin pastry filled with cream and topped with chocolate glaze. It is a classic French treat that is both sweet and satisfying. Éclairs can be found in bakeries all over France and are often enjoyed with a cup of hot chocolate.
#5. Tarte Tatin
#The tarte Tatin is an apple tart that is known for its caramelized apples and puff pastry crust. It is named after the Tatin sisters who created the recipe in the late 19th century. Tarte Tatin is best served warm with a scoop of vanilla ice cream.
#These pastries are just a few of the many delicious treats that France has to offer. Whether you are a seasoned traveler or a first-time visitor, indulging in French pastries is a must-do activity. So go ahead, treat yourself—you deserve it!
```

</details>


### Open Source License
The code is licensed under Apache-2.0, while model weights are fully open for academic research and also allow free commercial usage. To apply for a commercial license, please fill in the application form (English)/申请表（中文）. For other questions or collaborations, please contact internlm@pjlab.org.cn.