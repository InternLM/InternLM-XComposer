import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from calflops.utils import (
    generate_transformer_input,
    flops_to_string,
    macs_to_string,
    params_to_string
)
from calflops.calculate_pipline import CalFlopsPipline

torch.set_grad_enabled(False)

def calculate_flops(model,
                    input_shape=None,
                    transformer_tokenizer=None,
                    args=[],
                    kwargs={},
                    forward_mode="forward",
                    include_backPropagation=False,
                    compute_bp_factor=2.0,
                    print_results=True,
                    print_detailed=True,
                    output_as_string=True,
                    output_precision=2,
                    output_unit=None,
                    ignore_modules=None):

    assert isinstance(model, nn.Module), "model must be a PyTorch module"

    model.eval()

    is_Transformer = True if "transformers" in str(type(model)) else False
    
    calculate_flops_pipline = CalFlopsPipline(model=model, 
                                                  include_backPropagation=include_backPropagation,
                                                  compute_bp_factor=compute_bp_factor)
    calculate_flops_pipline.start_flops_calculate(ignore_list=ignore_modules)
    
    device = next(model.parameters()).device
    model = model.to(device)
    
    if input_shape is not None:
        assert len(args) == 0 and len(kwargs) == 0, "args and kwargs must be empty value if input_shape is not None, then will be generate random input by inpust_shape"
        assert type(input_shape) is tuple, "input_shape must be a tuple"
        assert len(input_shape) >= 1, "input_shape must have at least one element"
        
        if transformer_tokenizer is None:  # model is not transformers model
            assert is_Transformer is False, "the model is must not transformer model if input_shape is not None and transformer_tokenizer is None"
            try:
                input = torch.ones(()).new_empty(
                    (*input_shape, ),
                    dtype=next(model.parameters()).dtype,
                    device=device,
                )
            except StopIteration:
                input = torch.ones(()).new_empty((*input_shape, ))
            args = [input]
        else:
            assert len(input_shape) == 2, "the format of input_shape must be (batch_size, seq_len) if model is transformers model and auto_generate_transformers_input if True"
            kwargs = generate_transformer_input(input_shape=input_shape,
                                                model_tokenizer=transformer_tokenizer,
                                                device=device)
    else:
        assert transformer_tokenizer or (len(args) > 0 or len(kwargs) > 0),  "input_shape or args or kwargs one of there parameters must specified if auto_generate_input is False"
        if transformer_tokenizer:
            kwargs = generate_transformer_input(input_shape=None,
                                                model_tokenizer=transformer_tokenizer,
                                                device=device)
    
    if kwargs:
        if forward_mode == 'forward':
            _ = model(*args, **kwargs)
        if forward_mode == 'generate':
            _ = model.generate(*args, **kwargs)
    else:
        if forward_mode == 'forward':
            _ = model(*args)
        if forward_mode == 'generate':
            _ = model.generate(*args)

    flops = calculate_flops_pipline.get_total_flops()
    macs = calculate_flops_pipline.get_total_macs()
    params = calculate_flops_pipline.get_total_params()
    
    if print_results:
        return_print = calculate_flops_pipline.print_model_pipline(units=output_unit,
                                                    precision=output_precision,
                                                    print_detailed=print_detailed)

    calculate_flops_pipline.end_flops_calculate()
   
    if include_backPropagation:
        flops = flops * (1 + compute_bp_factor) 
        macs = macs * (1 + compute_bp_factor)

    if output_as_string:
        return flops_to_string(flops, units=output_unit, precision=output_precision),\
               macs_to_string(macs, units=output_unit, precision=output_precision),  \
               params_to_string(params, units=output_unit, precision=output_precision)
    
    return flops, macs, params



# init model and tokenizer
model = AutoModel.from_pretrained('internlm/internlm-xcomposer-7b', trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer-7b', trust_remote_code=True)
model.tokenizer = tokenizer


# inputs
text = 'Please introduce the person in this picture in detail.'
image = 'examples/images/aiyinsitan.jpg'


# test params and flops
flops, _, params = calculate_flops(model=model, kwargs={'text': text, 'image': image}, forward_mode="generate", print_results=False)
print("FLOPs: %s, Params: %s. \n" %(flops, params))