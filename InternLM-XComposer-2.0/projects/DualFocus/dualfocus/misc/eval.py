def process_outputs(outputs, options):
    ori_outputs = outputs
    if len(outputs) > 1:
        if outputs[0].lower() in ('a', 'b', 'c', 'd'):
            outputs = outputs[0]
            # print(f'error outputs: {ori_outputs}, fixed outputs: {outputs}')
        else:
            fixed = False
            for i, option in enumerate(options):
                if outputs.lower() in option.lower() or (option.lower()
                                                         in outputs.lower()):
                    out_option = chr(ord('A') + i)
                    fixed = True
                    break
            if fixed:
                outputs = out_option
                # print(f'error outputs: {ori_outputs}, fixed outputs: {outputs}')
    return outputs
