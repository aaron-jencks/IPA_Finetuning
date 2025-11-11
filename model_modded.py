import torch
from torch import Tensor, nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tokenizers import Tokenizer
from sklearn.metrics import fbeta_score


# -----------------------------------------------------------------------------
# Core Model Components

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))


def init_linear(w: Tensor):
    std = 0.5 * (w.size(-1) ** -0.5)
    bound = (3 ** 0.5) * std
    return w.uniform_(-bound, bound)


class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        # half-truncate RoPE by @YouJiacheng (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.register_buffer("cos", theta.cos(), persistent=False)
        self.register_buffer("sin", theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)


# -----------------------------------------------------------------------------
# Batched Attention (replaces FlexAttention)

class CausalSelfAttentionBatched(nn.Module):
    """Standard batched causal attention - no FlexAttention constraints"""

    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim

        # Merged QKV weights - matches the small model structure
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std
        self.qkv_w = nn.Parameter(torch.empty(3, hdim, dim).uniform_(-bound, bound))

        self.rotary = Rotary(head_dim, max_seq_len)
        self.c_proj_weight = nn.Parameter(torch.zeros(dim, hdim))
        self.attn_scale = 0.12
        self.max_seq_len = max_seq_len

    def forward(self, x: Tensor, ve: Tensor | None, sa_lambdas: Tensor):
        """Forward pass with value embedding support

        Args:
            x: Input tensor [B, T, D]
            ve: Value embeddings [B, T, D] or None
            sa_lambdas: Self-attention lambdas [2] for mixing v with ve
        """
        B, T, D = x.shape

        # QKV projection using merged weights
        qkv = F.linear(x, self.qkv_w.flatten(end_dim=1)).view(B, T, 3 * self.num_heads, self.head_dim)
        q, k, v = qkv.chunk(3, dim=-2)

        # Apply normalization and rotary embeddings
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)

        # Value embedding mixing
        if ve is not None:
            v = sa_lambdas[0] * v + sa_lambdas[1] * ve.view_as(v)
        else:
            v = sa_lambdas[0] * v

        # Reshape for attention: [B, num_heads, T, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        attn = (q @ k.transpose(-2, -1)) * self.attn_scale

        # Apply causal mask
        if not hasattr(self, '_causal_mask') or self._causal_mask.size(0) < T:
            causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            self.register_buffer('_causal_mask', causal_mask, persistent=False)

        attn = attn.masked_fill(self._causal_mask[:T, :T], float('-inf'))
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        y = attn @ v  # [B, num_heads, T, head_dim]

        # Reshape back and apply output projection
        y = y.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        y = F.linear(y, self.c_proj_weight)
        return y


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc_weight = nn.Parameter(init_linear(torch.empty(hdim, dim)))
        self.c_proj_weight = nn.Parameter(torch.zeros(dim, hdim))

    def forward(self, x: Tensor):
        x = F.linear(x, self.c_fc_weight)
        x = F.relu(x).square()  # ReSquared activation
        x = F.linear(x, self.c_proj_weight)
        return x


class BlockBatched(nn.Module):
    """Block using standard batched attention - matches small model structure"""

    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        # Skip attention of layer 7 (the 8th layer) - same as small model
        self.attn = CausalSelfAttentionBatched(dim, num_heads, max_seq_len) if layer_idx != 7 else None
        self.mlp = MLP(dim)

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, lambdas: Tensor, sa_lambdas: Tensor):
        """Forward pass matching small model Block signature

        Args:
            x: Input tensor [B, T, D]
            ve: Value embeddings [B, T, D] or None
            x0: Initial embeddings [B, T, D]
            lambdas: Block lambdas [2] for residual scaling
            sa_lambdas: Self-attention lambdas [2] for value mixing
        """
        # Apply residual scaling
        x = lambdas[0] * x + lambdas[1] * x0

        # Self-attention (if not skipped)
        if self.attn is not None:
            x = x + self.attn(norm(x), ve, sa_lambdas)

        # MLP
        x = x + self.mlp(norm(x))
        return x


# -----------------------------------------------------------------------------
# Batched GPT Model (Small Version - 12 layers)

class GPTBatchedSmall(nn.Module):
    """Batched version of small GPT (12 layers) for classification fine-tuning

    This model removes FlexAttention constraints and supports arbitrary batch sizes.
    It uses U-net style skip connections in the second half of the network.
    """

    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int):
        super().__init__()
        assert num_layers == 12, f"Small model expects 12 layers, got {num_layers}"
        assert num_heads == 6, f"Small model expects 6 heads, got {num_heads}"
        assert model_dim == 768, f"Small model expects 768 dimensions, got {model_dim}"

        self.model_dim = model_dim
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers

        # Initialize embeddings
        self.embed = nn.Embedding(vocab_size, model_dim)

        self.value_embeds = nn.ModuleList([
            nn.Embedding(vocab_size, model_dim) for _ in range(3)
        ])

        # Create blocks (12 layers for small model)
        self.blocks = nn.ModuleList([
            BlockBatched(model_dim, num_heads, max_seq_len, i)
            for i in range(num_layers)
        ])

        # Learnable scalars - matching small model structure
        # Note: Small model may have padding for distributed training
        self.scalars = nn.Parameter(torch.cat([
            torch.ones(num_layers),  # skip_weights
            *[torch.tensor([1.0, 0.0]) for _ in range(num_layers)],  # block lambdas
            *[torch.tensor([0.5, 0.5]) for _ in range(num_layers)],  # SA lambdas
        ]))

    @staticmethod
    def from_pretrained_gpt(original_gpt):
        """Create a batched GPT from a FlexAttention GPT model

        Args:
            original_gpt: A small GPT model (12 layers) trained with FlexAttention

        Returns:
            GPTBatchedSmall model with copied weights
        """
        # Get architecture parameters
        model_dim = original_gpt.embed.embedding_dim
        vocab_size = original_gpt.embed.num_embeddings
        num_layers = len(original_gpt.blocks)

        # Verify this is a small model
        assert num_layers == 12, f"Expected 12 layers for small model, got {num_layers}"

        # Infer num_heads from attention layer
        first_attn = next(block.attn for block in original_gpt.blocks if block.attn is not None)
        num_heads = first_attn.num_heads
        assert num_heads == 6, f"Expected 6 heads for small model, got {num_heads}"

        # Get max_seq_len from rotary embeddings
        max_seq_len = first_attn.rotary.cos.size(0)

        # Create new batched model
        batched_model = GPTBatchedSmall(vocab_size, num_layers, num_heads, model_dim, max_seq_len)

        # Copy embedding weights
        batched_model.embed.weight.data.copy_(original_gpt.embed.weight.data)

        for i, (old_ve, new_ve) in enumerate(zip(original_gpt.value_embeds, batched_model.value_embeds)):
            new_ve.weight.data.copy_(old_ve.weight.data)

        # Copy block weights
        for i, (old_block, new_block) in enumerate(zip(original_gpt.blocks, batched_model.blocks)):
            # Copy attention weights if exists
            if old_block.attn is not None and new_block.attn is not None:
                # Small model uses qkv_w, not qkvo_w
                new_block.attn.qkv_w.data.copy_(old_block.attn.qkv_w.data)
                new_block.attn.c_proj_weight.data.copy_(old_block.attn.c_proj.weight.data)

            # Copy MLP weights
            new_block.mlp.c_fc_weight.data.copy_(old_block.mlp.c_fc.weight.data)
            new_block.mlp.c_proj_weight.data.copy_(old_block.mlp.c_proj.weight.data)

        # Copy scalars (handle potential padding)
        # Original scalars might have padding for distributed training
        expected_scalar_size = 5 * num_layers
        if original_gpt.scalars.size(0) > expected_scalar_size:
            # Has padding, copy only the non-padding part
            batched_model.scalars.data.copy_(original_gpt.scalars.data[:expected_scalar_size])
        else:
            batched_model.scalars.data.copy_(original_gpt.scalars.data)

        return batched_model

    def forward_features(self, input_seq: Tensor):
        """Extract features from the model

        Args:
            input_seq: Input token IDs [B, T]

        Returns:
            Features tensor [B, T, model_dim]
        """
        B, T = input_seq.shape

        # Embed tokens
        x = x0 = norm(self.embed(input_seq))

        # Value embedding pattern: 012 ... 012 for 12-layer model
        # Layers 0,1,2 get ve[0,1,2], middle 6 layers get None, layers 9,10,11 get ve[0,1,2]
        ve_raw = [value_embed(input_seq) for value_embed in self.value_embeds]
        ve = [ve_raw[0], ve_raw[1], ve_raw[2]] + [None] * 6 + [ve_raw[0], ve_raw[1], ve_raw[2]]
        assert len(ve) == self.num_layers

        # U-net design: first half stores skip connections, second half adds them back
        skip_connections = []
        n = self.num_layers // 2  # n = 6

        skip_weights = self.scalars[:self.num_layers]
        lambdas = self.scalars[self.num_layers:3 * self.num_layers].view(-1, 2)
        sa_lambdas = self.scalars[3 * self.num_layers:5 * self.num_layers].view(-1, 2)

        for i, block in enumerate(self.blocks):
            # Second half: add skip connections from first half (in reverse order)
            if i >= n:
                x = x + skip_weights[i - n] * skip_connections.pop()

            # Apply block with value embeddings
            x = block(x, ve[i], x0, lambdas[i], sa_lambdas[i])

            # First half: store skip connections
            if i < n:
                skip_connections.append(x)

        # Final normalization
        x = norm(x)
        return x


# -----------------------------------------------------------------------------
# Classification Model Wrapper

class GPTClassificationSmall(nn.Module):
    """Wrapper around pretrained small GPT (12 layers) for classification tasks

    This version uses batched attention for much faster training.

    Args:
        device: Device to run model on
        vocab_file: Path to tokenizer vocab file
        merges_file: Path to tokenizer merges file (can be None for GPT-2)
        data_dir: Directory to save/load processed data
        num_classes: Number of output classes (default: 2 for binary classification)
        num_embed: Model embedding dimension (must be 768 for small model)
        dropout: Dropout rate for classification head
        context_size: Maximum sequence length
        batch_size: Batch size for training
        ipa: Whether to use IPA (unused, for compatibility)
        text_column: Name of text column(s) in dataset - can be single string or list of strings
        label_column: Name of label column in dataset (default: 'label')
    """

    def __init__(self, device, vocab_file, merges_file, data_dir, num_classes=2,
                 num_embed=768, dropout=0.1, context_size=1024, batch_size=16, ipa=False,
                 text_column='sentence', label_column='label'):
        super().__init__()

        assert num_embed == 768, f"Small model requires num_embed=768, got {num_embed}"

        # Validate context size is multiple of 128 (for compatibility)
        if context_size % 128 != 0:
            print(f"Warning: context_size {context_size} is not a multiple of 128. Rounding up.")
            context_size = ((context_size // 128) + 1) * 128

        self.device = device
        self.num_classes = num_classes
        self.context_window = context_size
        self.batch_size = batch_size
        self.dropout_rate = dropout

        # Column names for dataset
        self.text_column = text_column if isinstance(text_column, list) else [text_column]
        self.label_column = label_column

        # Training hyperparameters
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.grad_clip = 1.0
        self.warmup_iter_ratio = 0.1
        self.lr_decay_iter_ratio = 0.9
        self.min_lr = 1e-5

        # Tokenizer
        self.tokenizer = Tokenizer.from_file(str(vocab_file))

        # Data paths
        self.data_dir = Path(data_dir)
        self.train_data_path = self.data_dir / "train.bin"
        self.val_data_path = self.data_dir / "val.bin"

        # Data will be loaded after prepare_if_needed() is called
        self.train_data = None
        self.train_labels = None
        self.val_data = None
        self.val_labels = None

        # Pad token ID (used for masking padding tokens)
        self.pad_token_id = 0  # Tokenizer pads with 0

        # Pretrained model (to be loaded)
        self.pretrained_model = None

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_embed, num_embed // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_embed // 2, num_classes)
        )

        # Class weights for handling imbalanced datasets
        self.class_weights = None

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass for classification

        Args:
            input_ids: Input token IDs [B, T]
            attention_mask: Attention mask [B, T] (optional)
            labels: Labels [B] (optional)

        Returns:
            If labels provided: (logits, loss)
            Otherwise: logits
        """
        # Get features from pretrained model
        features = self.pretrained_model.forward_features(input_ids)  # [B, T, D]

        # Compute attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long()

        # Find the last non-padding token for each sequence
        batch_size = input_ids.shape[0]
        token_indices = torch.arange(input_ids.shape[1], device=input_ids.device)
        last_non_pad_indices = (token_indices * attention_mask).argmax(dim=1)

        # Pool using the last non-padding token
        pooled = features[torch.arange(batch_size, device=features.device), last_non_pad_indices]

        # Classification head
        logits = self.classifier(pooled)  # [B, num_classes]

        if labels is not None:
            loss = F.cross_entropy(logits, labels, weight=self.class_weights)
            return logits, loss
        return logits

    def prepare_if_needed(self, train_dataset, val_dataset, force_tokenization=False):
        """Prepare binary data files for training"""
        if force_tokenization or not self.train_data_path.exists():
            print(f"Tokenizing training data...")
            self._tokenize_and_save(train_dataset, self.train_data_path)

        if force_tokenization or not self.val_data_path.exists():
            print(f"Tokenizing validation data...")
            self._tokenize_and_save(val_dataset, self.val_data_path)

        # Load data once into memory
        print(f"Loading training data into memory...")
        self._load_data_into_memory()

        # Compute class weights
        self._compute_class_weights()

    def _load_data_into_memory(self):
        """Load all data into memory once"""
        print(f"Loading train data from {self.train_data_path}...")
        with open(self.train_data_path, 'rb') as f:
            num_examples = np.fromfile(f, dtype=np.int32, count=1)[0]
            self.train_data = np.fromfile(f, dtype=np.int32,
                                          count=num_examples * self.context_window).reshape(num_examples,
                                                                                            self.context_window)
            self.train_labels = np.fromfile(f, dtype=np.int64, count=num_examples)
        print(f"Loaded {len(self.train_data)} training examples")

        print(f"Loading val data from {self.val_data_path}...")
        with open(self.val_data_path, 'rb') as f:
            num_examples = np.fromfile(f, dtype=np.int32, count=1)[0]
            self.val_data = np.fromfile(f, dtype=np.int32,
                                        count=num_examples * self.context_window).reshape(num_examples,
                                                                                          self.context_window)
            self.val_labels = np.fromfile(f, dtype=np.int64, count=num_examples)
        print(f"Loaded {len(self.val_data)} validation examples")

    def _compute_class_weights(self):
        """Compute class weights from training data"""
        if self.train_labels is None:
            print("Warning: Cannot compute class weights - training labels not loaded")
            return

        unique_labels, counts = np.unique(self.train_labels, return_counts=True)
        print(f"\nClass distribution in training data:")
        for label, count in zip(unique_labels, counts):
            percentage = 100.0 * count / len(self.train_labels)
            print(f"  Class {label}: {count:,} samples ({percentage:.2f}%)")

        total_samples = len(self.train_labels)
        weights = np.zeros(self.num_classes, dtype=np.float32)

        for label, count in zip(unique_labels, counts):
            weights[label] = total_samples / (self.num_classes * count)

        self.class_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        print(f"\nComputed class weights (for balanced loss):")
        for i, weight in enumerate(weights):
            print(f"  Class {i}: {weight:.4f}")
        print()

    def _tokenize_and_save(self, dataset, save_path):
        """Tokenize dataset and save to binary file"""
        save_path.parent.mkdir(parents=True, exist_ok=True)

        all_input_ids = []
        all_labels = []

        for example in dataset:
            # Handle single or multiple text columns
            if len(self.text_column) == 1:
                text = str(example[self.text_column[0]])
            else:
                text = ' <ENDOFTEXT> '.join([str(example[col]) for col in self.text_column])

            # Tokenize text
            encoded = self.tokenizer.encode(text)
            input_ids = encoded.ids[:self.context_window]

            # Pad if necessary
            if len(input_ids) < self.context_window:
                input_ids = input_ids + [0] * (self.context_window - len(input_ids))

            all_input_ids.append(input_ids)
            all_labels.append(example[self.label_column])

        # Convert to numpy arrays
        input_ids_array = np.array(all_input_ids, dtype=np.int32)
        labels_array = np.array(all_labels, dtype=np.int64)

        # Save as binary file
        with open(save_path, 'wb') as f:
            num_examples = np.array([len(all_input_ids)], dtype=np.int32)
            num_examples.tofile(f)
            input_ids_array.tofile(f)
            labels_array.tofile(f)

        print(f"Saved {len(all_input_ids)} examples to {save_path}")
        print(f"Text columns used: {self.text_column}")

    def get_batch(self, split):
        """Get a batch of data

        Args:
            split: 'train' or 'val'

        Returns:
            input_ids: [batch_size, context_window]
            labels: [batch_size]
        """
        if split == 'train':
            if self.train_data is None:
                raise RuntimeError("Training data not loaded! Call prepare_if_needed() first.")
            input_ids = self.train_data
            labels = self.train_labels
        else:
            if self.val_data is None:
                raise RuntimeError("Validation data not loaded! Call prepare_if_needed() first.")
            input_ids = self.val_data
            labels = self.val_labels

        # Sample batch_size examples
        indices = np.random.randint(0, len(input_ids), size=self.batch_size)
        input_ids_batch = torch.from_numpy(input_ids[indices]).to(self.device, dtype=torch.int32)
        labels_batch = torch.from_numpy(labels[indices]).to(self.device, dtype=torch.int64)

        return input_ids_batch, labels_batch

    def estimate_loss(self, ctx, eval_iters):
        """Estimate loss on train and val sets"""
        out = {}
        self.eval()

        for split in ['train', 'val']:
            losses = []
            accuracies = []
            f2_scores = []

            for _ in range(eval_iters):
                with torch.no_grad():
                    with ctx:
                        X, Y = self.get_batch(split)
                        attention_mask = (X != self.pad_token_id).long()
                        logits, loss = self(X, attention_mask=attention_mask, labels=Y)

                        preds = logits.argmax(dim=-1)
                        acc = (preds == Y).float().mean().item()
                        f2 = self._calculate_f2(preds, Y)

                        losses.append(loss.item())
                        accuracies.append(acc)
                        f2_scores.append(f2)

            out[split] = np.mean(losses)
            out[f'{split}_accuracy'] = np.mean(accuracies)
            out[f'{split}_f2'] = np.mean(f2_scores)

        self.train()
        return out

    def _calculate_f2(self, preds, labels, beta=2):
        """Calculate F-beta score"""
        preds_np = preds.cpu().numpy().flatten()
        labels_np = labels.cpu().numpy().flatten()

        try:
            f_beta = fbeta_score(labels_np, preds_np, beta=beta, average='macro', zero_division=0)
            return float(f_beta)
        except Exception:
            return 0.0

    def get_token_count(self):
        """Get total number of tokens in training set"""
        if self.train_data is None:
            with open(self.train_data_path, 'rb') as f:
                num_examples = np.fromfile(f, dtype=np.int32, count=1)[0]
            return num_examples * self.context_window
        return len(self.train_data) * self.context_window

    def get_metadata(self):
        """Get metadata about the tokenizer"""
        return {
            'vocab_size': self.tokenizer.get_vocab_size(),
            'max_length': self.context_window
        }


# -----------------------------------------------------------------------------
# Helper function to load original FlexAttention GPT (for weight conversion)

class GPTFlexAttentionSmall(nn.Module):
    """Minimal implementation to load small FlexAttention checkpoints

    This is just for loading weights, not for actual use.
    """

    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int):
        super().__init__()
        assert num_layers == 12
        assert num_heads == 6
        assert model_dim == 768

        self.embed = nn.Embedding(vocab_size, model_dim)
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])

        # Use a simple placeholder for blocks
        self.blocks = nn.ModuleList(
            [self._create_flex_block(model_dim, num_heads, max_seq_len, i) for i in range(num_layers)])

        # LM head - use exact vocab_size without additional padding
        # The checkpoint already has the properly padded size
        self.lm_head = nn.Module()
        self.lm_head.weight = nn.Parameter(torch.zeros(vocab_size, model_dim))

        # Scalars with potential padding
        self.scalars = nn.Parameter(torch.zeros(5 * num_layers + 8))  # May have padding for world_size=8

    def _create_flex_block(self, dim, num_heads, max_seq_len, layer_idx):
        """Create a minimal block structure for loading weights"""
        block = nn.Module()
        if layer_idx != 7:
            # Create attention module
            attn = nn.Module()
            hdim = num_heads * 128
            attn.qkv_w = nn.Parameter(torch.empty(3, hdim, dim))
            attn.rotary = Rotary(128, max_seq_len)
            attn.num_heads = num_heads
            attn.head_dim = 128
            # c_proj for small model
            attn.c_proj = nn.Module()
            attn.c_proj.weight = nn.Parameter(torch.zeros(dim, hdim))
            block.attn = attn
        else:
            block.attn = None

        # Create MLP
        mlp = nn.Module()
        mlp.c_fc = nn.Module()
        mlp.c_fc.weight = nn.Parameter(torch.empty(4 * dim, dim))
        mlp.c_proj = nn.Module()
        mlp.c_proj.weight = nn.Parameter(torch.zeros(dim, 4 * dim))
        block.mlp = mlp

        return block
