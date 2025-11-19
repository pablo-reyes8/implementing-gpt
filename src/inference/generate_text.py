import torch
import math

def generate(model, tokenizer, prompt, max_new_tokens=30, temperature=1.0, top_k=50, device="cuda"):
    model.eval()

    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)

    block_size = model.module.block_size if hasattr(model, "module") else model.block_size

    for _ in range(max_new_tokens):
        x_cond = x[:, -block_size:]

        # forward
        logits, _ = model(x_cond, None)
        logits = logits[:, -1, :] / temperature

        # top-k truncation
        if top_k is not None:
            values, _ = torch.topk(logits, top_k)
            logits[logits < values[:, [-1]]] = -float("inf")

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        x = torch.cat([x, next_id], dim=1)

    return tokenizer.decode(x[0].tolist())


def generate_greedy(model, tokenizer, prompt, max_new_tokens=50, device="cuda"):
    model.eval()
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)

    block_size = model.module.block_size if hasattr(model, "module") else model.block_size

    for _ in range(max_new_tokens):
        x_cond = x[:, -block_size:]
        logits, _ = model(x_cond, None)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        x = torch.cat([x, next_id], dim=1)

    return tokenizer.decode(x[0].tolist())


def top_p_sample(logits, p=0.9):
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    cumulative = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

    mask = cumulative > p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False

    sorted_logits[mask] = -float("inf")
    probs = torch.softmax(sorted_logits, dim=-1)
    idx = torch.multinomial(probs, num_samples=1)

    return sorted_idx[torch.arange(logits.size(0)), idx.squeeze()].unsqueeze(1)


def next_token_probs(model, tokenizer, prompt, device="cuda"):
    model.eval()
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)

    logits, _ = model(x, None)
    probs = torch.softmax(logits[:, -1, :], dim=-1)

    return probs[0]


@torch.no_grad()
def compute_perplexity(model, tokenizer, text, device="cuda"):
    model.eval()
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    idx = torch.tensor([ids[:-1]], device=device)
    tgt = torch.tensor([ids[1:]],  device=device)

    logits, loss = model(idx, tgt)
    return math.exp(loss.item())


def generate_beam(model, tokenizer, prompt, max_new_tokens=50, beam=3, device="cuda"):
    model.eval()
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    beams = [(torch.tensor([ids], device=device), 0.0)]  # (sequence, score)

    block_size = model.module.block_size if hasattr(model, "module") else model.block_size

    for _ in range(max_new_tokens):
        candidates = []
        for seq, score in beams:
            x_cond = seq[:, -block_size:]
            logits, _ = model(x_cond, None)
            logp = torch.log_softmax(logits[:, -1, :], dim=-1)[0]

            topk = torch.topk(logp, beam)
            for new_logp, idx in zip(topk.values, topk.indices):
                new_seq = torch.cat([seq, idx.view(1,1)], dim=1)
                candidates.append((new_seq, score + new_logp.item()))

        # keep best beams
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam]

    best = beams[0][0][0].tolist()
    return tokenizer.decode(best)


@torch.no_grad()
def embed_text(model, tokenizer, text, device="cuda"):
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)

    h, _ = model(x, None)      
    emb = h.mean(dim=1)       
    return emb.cpu()


@torch.no_grad()
def explain_generation(model, tokenizer, prompt, device="cuda"):
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    x = torch.tensor([ids], dtype=torch.long, device=device)

    for i in range(1, len(ids)+1):
        logits, _ = model(x[:, :i], None)
        probs = torch.softmax(logits[:, -1, :], dim=-1)[0]
        top5 = torch.topk(probs, 5)

        print(f"TOKEN: {tokenizer.decode([ids[i-1]])}")
        for p, tok in zip(top5.values, top5.indices):
            print(f"  {p.item():.4f} → {tokenizer.decode([tok.item()])}")
        print()