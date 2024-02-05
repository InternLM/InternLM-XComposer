from PIL import Image

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

import dataclasses
from enum import auto, Enum
from typing import List, Any


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    # system_img: List[Image.Image] = []
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    #ret += role + ": " + message + self.sep
                    ret += role + ":" + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message[0] + seps[i % 2] if isinstance(message, list) else role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == "7132":
            seps = [self.sep, self.sep2]
            ret = self.system 
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message[0] + seps[i % 2] if isinstance(message, list) else role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == "raw":
            seps = [self.sep, self.sep2]
            ret = self.system 
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + message + seps[i % 2]
                else:
                    ret += role 
            return ret
        elif self.sep_style == "intern2":
            ret = self.system
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if isinstance(message, list):
                        ret += '<Img><ImageHere></Img>' + role + message[0].replace('<Img><ImageHere></Img>', '') + self.sep
                    else:
                        ret += role + message + self.sep
                else:
                    ret += role
            return ret

        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                if type(msg) is tuple or type(msg) is list:
                    import base64
                    from io import BytesIO
                    msg, images = msg

                    # type check for images, if not list(e.g. PIL), just put it in a list
                    if type(images) is not list:
                        images = [images]
                    
                    img_str = '''<style>.centerimg{float:left;}.flex_img{align-items: left;display: flex;justify-content: left;}</style><div class='flex_img'>'''
                    for j, image in enumerate(images):
                        max_hw, min_hw = max(image.size), min(image.size)
                        aspect_ratio = max_hw / min_hw
                        max_len, min_len = 800, 400
                        shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
                        longest_edge = int(shortest_edge * aspect_ratio)
                        W, H = image.size
                        if H > W:
                            H, W = longest_edge, shortest_edge
                        else:
                            H, W = shortest_edge, longest_edge
                        image = image.resize((W, H))
                        # image = image.resize((224, 224))
                        buffered = BytesIO()
                        image.save(buffered, format="JPEG")
                        img_b64_str = base64.b64encode(buffered.getvalue()).decode()
                        img_str += f' <div class="centerimg"><img src="data:image/png;base64,{img_b64_str}" alt="user upload image{j}" /></div>'
                    img_str += "</div>"
                    msg = msg.replace('<Img><ImageHere></Img>', img_str)
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            # system_img=self.system_img,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            # "system_img": self.system_img,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[:, -len(stop):])).item():
                return True

        return False


meta = """meta instruction
You are an AI assistant whose name is 浦语.
- 浦语 is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- 浦语 can understand and communicate fluently in the language chosen by the user such as English and 中文.
conversation
"""
CONV_VISION_7132_v2 = Conversation(
    system=meta,
    roles=(" <|User|>", " <|Bot|>"),
    messages=(),
    offset=0,
    sep_style="7132",
    sep="<TOKENS_UNUSED_0>",
    sep2="<TOKENS_UNUSED_1>",
)


meta_instruction = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""
CONV_VISION_INTERN2 = Conversation(
    system="",
    roles=("[UNUSED_TOKEN_146]user\n", "[UNUSED_TOKEN_146]assistant\n"),
    messages=(),
    offset=0,
    sep_style="intern2",
    sep="[UNUSED_TOKEN_145]\n",
)

