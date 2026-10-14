import mlx.core as mx
import pytest
from mlx_lm.models import llama


def make_tiny_llama(layers=2, hidden=128, inter=256, vocab=256, seed=0, dtype=mx.float32):
    args = llama.ModelArgs(
        model_type="llama",
        hidden_size=hidden,
        num_hidden_layers=layers,
        intermediate_size=inter,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=vocab,
        tie_word_embeddings=True,
    )
    mx.random.seed(seed)
    model = llama.Model(args)
    if dtype != mx.float32:
        model.set_dtype(dtype)
    mx.eval(model.parameters())
    return model


@pytest.fixture
def tiny_llama():
    return make_tiny_llama()


@pytest.fixture
def tokens():
    mx.random.seed(1)
    return mx.random.randint(0, 256, (8, 64))
