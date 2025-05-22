import torch.cuda
from transformers import AutoModelForCausalLM, AutoTokenizer

from beam_search import beam_search_decode


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

    # beam search experiments
    settings = [
        (1, 1.0),
        (4, 1.0),
        (4, 0.5),
        (4, 2.0),
        (8, 1.0),
    ]
    for num_beams, lp in settings:
        out_text = beam_search_decode(model, tokenizer,
                                      input_text_hedgehog, device,
                                      num_beams=num_beams,
                                      length_penalty=lp)
        print(f"--- hedgehog (beams={num_beams}, lp={lp}) ---\n{out_text}\n")

        out_json = beam_search_decode(model, tokenizer,
                                      input_text_json, device,
                                      num_beams=num_beams,
                                      length_penalty=lp)
        print(f"--- json (beams={num_beams}, lp={lp}) ---\n{out_json}\n")


