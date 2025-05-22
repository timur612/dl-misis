import torch

EOS_TOKEN = 151645


def beam_search_decode(model, tokenizer, input_text, device,
                       num_beams=4, length_penalty=1.0, max_length=1000):
    # encode
    encoding = tokenizer(input_text, return_tensors='pt').to(device)
    init_ids = encoding.input_ids
    init_mask = encoding.attention_mask

    # each candidate = (seq_ids, score, finished)
    sequences = [(init_ids, 0.0, False)]
    finished = []

    for _ in range(max_length):
        all_candidates = []
        for seq_ids, score, is_done in sequences:
            if is_done:
                all_candidates.append((seq_ids, score, True))
                continue
            # model forward
            mask = torch.ones_like(seq_ids)
            logits = model(input_ids=seq_ids, attention_mask=mask).logits
            log_probs = torch.log_softmax(logits[0, -1], dim=-1)
            # top-k expansions
            topk_logp, topk_idx = torch.topk(log_probs, num_beams)
            for lp, idx in zip(topk_logp.tolist(), topk_idx.tolist()):
                new_seq = torch.cat([seq_ids, torch.tensor([[idx]], device=device)], dim=1)
                new_score = score + lp
                done = (idx == EOS_TOKEN)
                all_candidates.append((new_seq, new_score, done))

        # prune to beam width
        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        sequences = []
        for seq_ids, score, done in ordered:
            if len(sequences) >= num_beams:
                break
            if done:
                finished.append((seq_ids, score))
            else:
                sequences.append((seq_ids, score, False))
        if len(finished) >= num_beams:
            break

    # if nothing finished, treat current beams as finished
    if not finished:
        finished = [(seq_ids, score) for seq_ids, score, _ in sequences]

    # apply length penalty and pick best
    def apply_lp(item):
        seq_ids, sc = item
        length = seq_ids.shape[1]
        return sc / (length ** length_penalty)

    best_seq, _ = max(finished, key=apply_lp)
    return tokenizer.decode(best_seq[0], skip_special_tokens=False)