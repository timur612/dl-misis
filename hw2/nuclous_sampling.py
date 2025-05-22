import torch

EOS_TOKEN = 151645


def nucleus_decode(model, tokenizer, input_text, device,
                    temperature=1.0, max_token_count=1000, top_p=1.0):
    encoding = tokenizer(input_text, return_tensors='pt').to(device)
    generated = encoding.copy()

    for _ in range(max_token_count):
        logits = model(input_ids=generated.input_ids,
                       attention_mask=generated.attention_mask).logits
        scaled = logits[0, -1] / temperature
        probs = torch.softmax(scaled, dim=-1)

        # --- nucleus (top-p) filtering ---
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs <= top_p
        if not mask[0]:  # always keep top token
            mask[0] = True
        filtered_indices = sorted_indices[mask]
        filtered_probs   = sorted_probs[mask]
        filtered_probs  = filtered_probs / filtered_probs.sum()
        new_probs = torch.zeros_like(probs)
        new_probs[filtered_indices] = filtered_probs

        next_token = torch.multinomial(new_probs, 1)
        if next_token.item() == EOS_TOKEN:
            break
        generated.input_ids = torch.cat([generated.input_ids,
                                         next_token.unsqueeze(0)], dim=1)
    output_text = tokenizer.decode(generated.input_ids[0],
                                   skip_special_tokens=False)
    return output_text