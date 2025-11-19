from collections import Counter
import torch

def inspect_autoregressive_loader(train_loader, tokenizer, 
                                  num_batches=2,  # cuántos batches inspeccionar
                                  max_examples=3, # cuántas secuencias imprimir por batch
                                  max_tokens_print=40):  # cuántos tokens decodificar por ejemplo
    """
    Inspecciona un DataLoader de lenguaje autoregresivo (GPT-style).

    Supone que cada batch es:
        x: [B, T]  (inputs)
        y: [B, T]  (targets desplazados 1 a la derecha en el stream original)

    Hace:
    - Verificar qué tan cierto es que y[:, :-1] == x[:, 1:].
    - Analizar la distribución de y[:, 0] (primer token que el modelo debe predecir).
    - Imprimir ejemplos decodificados (input vs target) para entender el shift.
    """
    total_shift_positions = 0
    total_shift_matches   = 0

    first_input_ids  = []
    first_target_ids = []

    for b_idx, (x, y) in enumerate(train_loader):
        B, T = x.shape

        shift_equal = (y[:, :-1] == x[:, 1:]) 
        total_shift_matches   += shift_equal.sum().item()
        total_shift_positions += shift_equal.numel()

        first_input_ids.extend(x[:, 0].tolist())
        first_target_ids.extend(y[:, 0].tolist())

        print(f"\n=== Batch {b_idx} ===")
        for i in range(min(max_examples, B)):
            inp_ids = x[i].tolist()
            tgt_ids = y[i].tolist()

            inp_ids_short = inp_ids[:max_tokens_print]
            tgt_ids_short = tgt_ids[:max_tokens_print]

            inp_text = tokenizer.decode(inp_ids_short)
            tgt_text = tokenizer.decode(tgt_ids_short)

            print(f"\n--- Ejemplo {i} ---")
            print(f"Input IDs   (primeros {len(inp_ids_short)}): {inp_ids_short}")
            print(f"Target IDs  (primeros {len(tgt_ids_short)}): {tgt_ids_short}")
            print("Input texto (modelo VE):")
            print(repr(inp_text))
            print("Target texto (modelo DEBE predecir):")
            print(repr(tgt_text))

        if b_idx + 1 >= num_batches:
            break

    shift_ratio = total_shift_matches / total_shift_positions
    print("\n================== RESUMEN AUTORREGRESIVO ==================")
    print(f"Total posiciones comparadas (y[:, :-1] vs x[:, 1:]): {total_shift_positions}")
    print(f"Coincidencias: {total_shift_matches}")
    print(f"Proporción de coincidencia (ideal ~1.0): {shift_ratio:.6f}")

    print("\nDistribución de primeros tokens (input vs target):")

    first_input_counts  = Counter(first_input_ids)
    first_target_counts = Counter(first_target_ids)

    def top_tokens(counter, name, k=10):
        print(f"\nTop {k} tokens más frecuentes en {name}:")
        for token_id, cnt in counter.most_common(k):
            text = tokenizer.decode([token_id])
            print(f"  id={token_id:5d} | freq={cnt:6d} | texto={repr(text)}")

    top_tokens(first_input_counts,  "x[:, 0]  (primer token que VE el modelo)")
    top_tokens(first_target_counts, "y[:, 0]  (primer token que DEBE predecir)")

    print("\n============================================================")