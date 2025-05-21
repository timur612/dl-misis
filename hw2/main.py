import torch.cuda
from transformers import AutoModelForCausalLM, AutoTokenizer

# from greedy_decoding import greedy_decode
# from sampling import sampling_decode
from sampling_temperature import sampling_decode


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct').eval()
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
    model.to(device)

    input_text_hedgehog = ('<|im_start|>system\nYou are a storyteller. Generate a '
                           'story based on user '
                           'message.<|im_end|>\n<|im_start|>user\nGenerate me a short '
                           'story about a tiny hedgehog named '
                           'Sonic.<|im_end|>\n<|im_start|>assistant\n')
    input_text_json = ('<|im_start|>system\nYou are a JSON machine. Generate a JSON '
                       'with format {"contractor": string with normalized contractor '
                       'name, "sum": decimal, "currency": string with uppercased '
                       '3-letter currency code} based on user '
                       'message.<|im_end|>\n<|im_start|>user\nTransfer 100 rubles and '
                       '50 kopeck to Mike<|im_end|>\n<|im_start|>assistant\n')

    # output_text = greedy_decode(model, tokenizer, input_text_hedgehog, device)
    # output_json = greedy_decode(model, tokenizer, input_text_json, device)
    # print(output_text)
    # print(output_json)
    

    # output_text = sampling_decode(model, tokenizer, input_text_hedgehog, device)
    # print(output_text)
    # output_json = sampling_decode(model, tokenizer, input_text_json, device)
    # print(output_json)


    # output_text = sampling_decode(model, tokenizer, input_text_hedgehog, device, temperature=10.0)
    # print(output_text)
    # output_json = sampling_decode(model, tokenizer, input_text_json, device, temperature=10.0)
    # print(output_json)

    
