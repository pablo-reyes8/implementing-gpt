import torch
from termcolor import colored
from src.inference.generate_text import *

def interactive_generate_debug(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 30,
    mode: str = "all",      # "all" | "greedy" | "topk" | "topp"
    top_k: int = 50,
    top_p: float = 0.9,
    temperature: float = 1.0,
    device: str = "cuda",
):
    """
    Modo de depuración paso a paso.

    Si mode == "all":
        - Ejecuta la generación 3 veces (greedy, topk, topp) desde el MISMO prompt.
        - Imprime todo separado por bloques.
        - Devuelve un dict con las 3 salidas:
            {
                "greedy": texto_greedy,
                "topk":   texto_topk,
                "topp":   texto_topp,
            }

    Si mode es uno de {"greedy", "topk", "topp"}:
        - Ejecuta solo ese modo.
        - Devuelve el texto generado (str).
    """

    assert mode in {"all", "greedy", "topk", "topp"}, \
        "mode debe ser 'all', 'greedy', 'topk' o 'topp'"

    model.eval()

    # Codificamos una sola vez
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    base_x = torch.tensor([ids], dtype=torch.long, device=device)

    block_size = model.module.block_size if hasattr(model, "module") else model.block_size

    print(colored("══════════════════════════════════════════", "cyan"))
    print(colored(" INTERACTIVE GENERATION DEBUG MODE ", "yellow"))
    print(colored("══════════════════════════════════════════\n", "cyan"))

    print(colored("PROMPT INICIAL:", "green"))
    print(colored(prompt, "white"))
    print()

    def run_single_mode(run_mode: str):
        """
        Ejecuta la generación completa para un modo específico
        (greedy / topk / topp), empezando siempre desde base_x.
        """
        x = base_x.clone()
        full_text = tokenizer.decode(x[0].tolist())

        print(colored(f"\n===== MODO: {run_mode.upper()} =====", "blue"))

        for step in range(1, max_new_tokens + 1):
            x_cond = x[:, -block_size:]

            # forward
            logits, _ = model(x_cond, None)
            logits = logits[:, -1, :] / temperature  

            probs = torch.softmax(logits, dim=-1)[0]  

            # Top-10 para debug (siempre sobre probs completas)
            topk_vals, topk_idx = torch.topk(probs, k=min(10, probs.size(0)))

            if run_mode == "greedy":
                next_id = torch.argmax(probs).unsqueeze(0).unsqueeze(0)

            elif run_mode == "topk":
                # Limitar logits al top_k más alto
                k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, k)
                logits_masked = logits.clone()
                logits_masked[logits_masked < values[:, [-1]]] = -float("inf")
                probs_k = torch.softmax(logits_masked, dim=-1)
                next_id = torch.multinomial(probs_k, num_samples=1)

            elif run_mode == "topp":
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumulative = sorted_probs.cumsum(dim=0)

                mask = cumulative < top_p
                # Siempre incluir al menos el token más probable
                if mask.sum() == 0:
                    mask[0] = True

                filtered_idx = sorted_idx[mask]
                filtered_probs = sorted_probs[mask]
                filtered_probs = filtered_probs / filtered_probs.sum()

                choice = torch.multinomial(filtered_probs, 1)
                chosen_token_id = filtered_idx[choice].item()
                next_id = torch.tensor([[chosen_token_id]], device=device)

            else:
                raise ValueError("run_mode debe ser 'greedy', 'topk' o 'topp'")

            # Actualizar secuencia
            x = torch.cat([x, next_id], dim=1)

            new_token_str = tokenizer.decode([next_id.item()])
            full_text = tokenizer.decode(x[0].tolist())

            print(colored(f"\n──────────────── STEP {step} ────────────────", "magenta"))
            print(colored("Contexto actual:", "cyan"))
            print(colored(full_text, "white"))
            print()

            print(colored("Top-10 probabilidades:", "yellow"))
            for prob, idx in zip(topk_vals.tolist(), topk_idx.tolist()):
                tok_str = tokenizer.decode([idx])
                print(f"  {prob:0.4f}  →  {repr(tok_str)}")
            print()

            print(colored("Token elegido:", "green"))
            print(colored(f"→ '{new_token_str}'", "green"))
            print()

        print(colored(f"══════ FIN MODO {run_mode.upper()} ══════", "cyan"))
        return full_text

    if mode == "all":
        out_greedy = run_single_mode("greedy")
        out_topk   = run_single_mode("topk")
        out_topp   = run_single_mode("topp")

        print(colored("\n════════ FIN GENERACIÓN (ALL) ════════", "cyan"))
        return {
            "greedy": out_greedy,
            "topk":   out_topk,
            "topp":   out_topp,}
    
    else:
        out_text = run_single_mode(mode)
        print(colored("\n════════ FIN GENERACIÓN ════════", "cyan"))
        return out_text