import re
import os
import glob
import numpy as np
import soundfile as sf
import torchaudio
from txtsplit import txtsplit

def split_sentence(text, min_len=10, language_str='EN'):
    if language_str in ['EN', 'FR', 'ES', 'SP']:
        sentences = split_sentences_latin(text, min_len=min_len)
    else:
        sentences = split_sentences_zh(text, min_len=min_len)
    return sentences

def split_sentences_latin(text, min_len=10):
    text = re.sub('[。！？；]', '.', text)
    text = re.sub('[，]', ',', text)
    text = re.sub('[“”]', '"', text)
    text = re.sub('[‘’]', "'", text)
    text = re.sub(r"[\<\>\(\)\[\]\"\«\»]+", "", text)
    return [item.strip() for item in txtsplit(text, 512, 512) if item.strip()]

def split_sentences_zh(text, min_len=10):
    text = re.sub('[。！？；]', '.', text)
    text = re.sub('[，]', ',', text)
    # 将文本中的换行符、空格和制表符替换为空格
    text = re.sub('[\n\t ]+', ' ', text)
    # 在标点符号后添加一个空格
    text = re.sub('([,.!?;])', r'\1 $#!', text)
    # 分隔句子并去除前后空格
    # sentences = [s.strip() for s in re.split('(。|！|？|；)', text)]
    sentences = [s.strip() for s in text.split('$#!')]
    if len(sentences[-1]) == 0: del sentences[-1]

    new_sentences = []
    new_sent = []
    count_len = 0
    for ind, sent in enumerate(sentences):
        new_sent.append(sent)
        count_len += len(sent)
        if count_len > min_len or ind == len(sentences) - 1:
            count_len = 0
            new_sentences.append(' '.join(new_sent))
            new_sent = []
    return merge_short_sentences_zh(new_sentences)

def merge_short_sentences_en(sens):
    """Avoid short sentences by merging them with the following sentence.

    Args:
        List[str]: list of input sentences.

    Returns:
        List[str]: list of output sentences.
    """
    sens_out = []
    for s in sens:
        # If the previous sentense is too short, merge them with
        # the current sentence.
        if len(sens_out) > 0 and len(sens_out[-1].split(" ")) <= 2:
            sens_out[-1] = sens_out[-1] + " " + s
        else:
            sens_out.append(s)
    try:
        if len(sens_out[-1].split(" ")) <= 2:
            sens_out[-2] = sens_out[-2] + " " + sens_out[-1]
            sens_out.pop(-1)
    except:
        pass
    return sens_out

def merge_short_sentences_zh(sens):
    # return sens
    """Avoid short sentences by merging them with the following sentence.

    Args:
        List[str]: list of input sentences.

    Returns:
        List[str]: list of output sentences.
    """
    sens_out = []
    for s in sens:
        # If the previous sentense is too short, merge them with
        # the current sentence.
        if len(sens_out) > 0 and len(sens_out[-1]) <= 2:
            sens_out[-1] = sens_out[-1] + " " + s
        else:
            sens_out.append(s)
    try:
        if len(sens_out[-1]) <= 2:
            sens_out[-2] = sens_out[-2] + " " + sens_out[-1]
            sens_out.pop(-1)
    except:
        pass
    return sens_out


if __name__ == '__main__':
    zh_text = "好的，我来给你讲一个故事吧。从前有一个小姑娘，她叫做小红。小红非常喜欢在森林里玩耍，她经常会和她的小伙伴们一起去探险。有一天，小红和她的小伙伴们走到了森林深处，突然遇到了一只凶猛的野兽。小红的小伙伴们都吓得不敢动弹，但是小红并没有被吓倒，她勇敢地走向野兽，用她的智慧和勇气成功地制服了野兽，保护了她的小伙伴们。从那以后，小红变得更加勇敢和自信，成为了她小伙伴们心中的英雄。"
    en_text = "I didn’t know what to do. I said please kill her because it would be better than being kidnapped,” Ben, whose surname CNN is not using for security concerns, said on Wednesday. “It’s a nightmare. I said ‘please kill her, don’t take her there.’"
    sp_text = "¡Claro! ¿En qué tema te gustaría que te hable en español? Puedo proporcionarte información o conversar contigo sobre una amplia variedad de temas, desde cultura y comida hasta viajes y tecnología. ¿Tienes alguna preferencia en particular?"
    fr_text = "Bien sûr ! En quelle matière voudriez-vous que je vous parle en français ? Je peux vous fournir des informations ou discuter avec vous sur une grande variété de sujets, que ce soit la culture, la nourriture, les voyages ou la technologie. Avez-vous une préférence particulière ?"

    print(split_sentence(zh_text, language_str='ZH'))
    print(split_sentence(en_text, language_str='EN'))
    print(split_sentence(sp_text, language_str='SP'))
    print(split_sentence(fr_text, language_str='FR'))
