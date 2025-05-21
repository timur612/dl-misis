import torch

EOS_TOKEN = 151645


def greedy_decode(model, tokenizer, input_text, device, max_token_count=1000):
    encoding = tokenizer(input_text, return_tensors='pt').to(device)
    generated = encoding.copy()

    for _ in range(max_token_count):
        logits = model(input_ids=generated.input_ids,
                       attention_mask=generated.attention_mask).logits
        next_token = torch.argmax(logits[0, -1], dim=-1, keepdim=True)
        if next_token.item() == EOS_TOKEN: break
        generated.input_ids = torch.cat([generated.input_ids, next_token.unsqueeze(0)],
                                        dim=1)
    output_text = tokenizer.decode(generated.input_ids[0], skip_special_tokens=False)
    return output_text

"""
    Once upon a time, in a small, cozy village nestled in the heart of the forest, there lived a tiny hedgehog named Sonic. Sonic was a curious and adventurous creature, always eager to explore the world around him. One day, while wandering through the forest, Sonic stumbled upon a hidden cave.

    Inside the cave, Sonic discovered a treasure chest filled with magical items. As he opened the chest, he was amazed to see that the items were not just ordinary, but enchanted. Sonic was thrilled to find that he could use the items to help others in need.

    From that day on, Sonic became a hero in the village. He used his magical powers to help people in need, and soon, the village was filled with people who were grateful for the help they received from Sonic.

    Sonic's story became a legend, and people from all over the village would tell stories about him. Sonic's adventures and his magic helped to bring joy and hope to the people of the village, and he was loved and respected by all who knew him.

    And so, Sonic continued to be a tiny hedgehog, always on the lookout for new adventures and helping others in need.
"""

"""
    {"contractor": "Mike", "sum": 105, "currency": "rubles"}
"""