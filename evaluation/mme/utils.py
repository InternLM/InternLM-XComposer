import torch
from transformers import StoppingCriteria, StoppingCriteriaList
class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False

stop_words_ids = [
                  torch.tensor([103027]).cuda(), ### end of human
                  torch.tensor([103028]).cuda(), ### end of bot
                 ]
stopping_criteria = StoppingCriteriaList(
        [StoppingCriteriaSub(stops=stop_words_ids)])

def generate_answer(model, text, image_path):
    img_embeds = model.encode_img(image_path)
    prompt_segs = text.split('<ImageHere>')
    prompt_seg_tokens = [
        model.tokenizer(seg,
                             return_tensors='pt',
                             add_special_tokens=i == 0).
        to(model.internlm_model.model.embed_tokens.weight.device).input_ids
        for i, seg in enumerate(prompt_segs)
    ]
    prompt_seg_embs = [
        model.internlm_model.model.embed_tokens(seg)
        for seg in prompt_seg_tokens
    ]
    prompt_seg_embs = [prompt_seg_embs[0], img_embeds, prompt_seg_embs[1]]
    
    prompt_embs = torch.cat(prompt_seg_embs, dim=1)
    
    outputs = model.internlm_model.generate(
        inputs_embeds=prompt_embs,
        max_new_tokens=5,
        num_beams=5,
        do_sample=False,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.5,
        length_penalty=1.0,
        temperature=1.0,
        stopping_criteria=stopping_criteria,
    )
    #print (outputs)
    output_token = outputs[0]
    if output_token[0] == 0:
        output_token = output_token[1:]
    if output_token[0] == 1:
        output_token = output_token[1:]
    output_text = model.tokenizer.decode(output_token,
                                              add_special_tokens=False)

    output_text = output_text.split(model.eoa)[0]
    output_text = output_text.split('<|Bot|>')[-1].strip()
    return output_text
