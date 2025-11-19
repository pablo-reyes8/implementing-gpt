from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tokenizers import decoders
import os

DATASET_NAME   = "wikitext"
DATASET_CONFIG = "wikitext-103-raw-v1"   

VOCAB_SIZE = 32000         
MIN_FREQ  = 2
BLOCK_SIZE  = 256  # Ventana de contexto         
VAL_FRACTION  = 0.1
TOKENIZER_PATH = Path("wikitext103_tokenizer.json")  

CPU_COUNT   = os.cpu_count() or 2
BATCH_SIZE  = 64
NUM_WORKERS  = 2 if CPU_COUNT <= 2 else min(4, CPU_COUNT - 1)             
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_openwebtext10k():
    """
    AHORA: carga wikitext-103-raw-v1 con splits train / validation.
    """

    ds = load_dataset(DATASET_NAME, DATASET_CONFIG)
    train_ds = ds["train"]
    val_ds   = ds["validation"]
    print(train_ds)
    print(val_ds)
    return train_ds, val_ds


def train_tokenizer(train_ds,
                    vocab_size=VOCAB_SIZE,
                    min_freq=MIN_FREQ,
                    save_path=TOKENIZER_PATH):
    """
    Entrena un tokenizer BPE estilo GPT (byte-level) sobre el split de entrenamiento.
    """
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFKC(),
        normalizers.Lowercase(),])

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    tokenizer.decoder = decoders.ByteLevel() 

    special_tokens = ["<unk>", "<pad>", "<bos>", "<eos>"]

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_freq,
        special_tokens=special_tokens,)

    def batch_iterator():
        for ex in train_ds:
            txt = ex["text"]
            if txt is not None and len(txt.strip()) > 0:
                yield txt

    print("Entrenando tokenizer BPE...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)
    print("Tamaño vocabulario:", tokenizer.get_vocab_size())

    save_path = Path(save_path)
    tokenizer.save(str(save_path))
    print(f"Tokenizer guardado en {save_path.resolve()}")

    return tokenizer


def load_or_train_tokenizer(train_ds):
    """
    Carga el tokenizer si ya existe en disco, si no lo entrena.
    """
    if TOKENIZER_PATH.exists():
        print(f"Cargando tokenizer desde {TOKENIZER_PATH}...")
        tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    else:
        tokenizer = train_tokenizer(train_ds)
    return tokenizer


class GPTTextDataset(Dataset):
    """
    Construye un dataset para LM autoregresivo:
      - Concatena todos los documentos en una secuencia larga de IDs.
      - Añade <eos> al final de cada documento.
      - Parte en chunks de longitud (BLOCK_SIZE + 1).
      - Input = ids[:-1], Target = ids[1:].
    """
    def __init__(self, hf_split, tokenizer, block_size=BLOCK_SIZE):
        super().__init__()
        self.block_size = block_size

        eos_id = tokenizer.token_to_id("<eos>")
        if eos_id is None:
            raise ValueError("El tokenizer no tiene token <eos>.")

        all_ids = []

        print("Tokenizando y concatenando textos...")
        for ex in hf_split:
            txt = ex["text"]
            if txt is None or len(txt.strip()) == 0:
                continue
            enc = tokenizer.encode(txt)
            # enc.ids es una lista de ints
            all_ids.extend(enc.ids + [eos_id])

        self.data = torch.tensor(all_ids, dtype=torch.long)
        print(f"Total de tokens en este split: {len(self.data):,}")

        n_tokens = len(self.data)
        chunk_len = block_size + 1
        n_chunks = n_tokens // chunk_len

        if n_chunks == 0:
            raise ValueError("Muy pocos tokens para formar un solo chunk. "
                             "Baja BLOCK_SIZE o usa más datos.")

        # Cortar a múltiplo exacto de chunk_len y reshape
        self.data = self.data[: n_chunks * chunk_len]
        self.data = self.data.view(n_chunks, chunk_len)

        # inputs y targets precomputados
        self.inputs  = self.data[:, :-1]  # [N, block_size]
        self.targets = self.data[:, 1:]   # [N, block_size]

        print(f"Número de secuencias: {len(self.inputs):,}")
        print(f"Forma inputs:  {self.inputs.shape}")
        print(f"Forma targets: {self.targets.shape}")

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def create_dataloaders(block_size=BLOCK_SIZE,
                       batch_size=BATCH_SIZE,
                       num_workers=NUM_WORKERS):
    train_hf, val_hf = load_openwebtext10k()
    tokenizer = load_or_train_tokenizer(train_hf)

    train_ds = GPTTextDataset(train_hf, tokenizer, block_size=block_size)
    val_ds   = GPTTextDataset(val_hf,   tokenizer, block_size=block_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)

    return train_loader, val_loader, tokenizer