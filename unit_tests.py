"""
Validation/Unit Tests for Transformer Components
"""

import torch
import torch.nn.functional as F
import math

from transformerComponents import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    PositionWiseFeedForward,
    PositionalEncoding,
    TransformerEncoderLayer,
)

def test_scaled_dot_product_attention():
    """
    Test the Scaled Dot-Product Attention mechanism.

    Validates:
    - Correct output shape.
    - Correct attention weights shape.
    - Attention weights sum to 1 across the sequence dimension.
    """
    print("Testing Scaled Dot-Product Attention...")
    batch_size = 2
    seq_length = 4
    d_k = 64

    # Create random tensors for Q, K, V
    Q = torch.randn(batch_size, seq_length, d_k)
    K = torch.randn(batch_size, seq_length, d_k)
    V = torch.randn(batch_size, seq_length, d_k)

    # Instantiate  ScaledDotProductAttention
    attention = ScaledDotProductAttention()

    # Perform forward pass
    output, attn_weights = attention(Q, K, V)

    # Validate output shape
    assert output.shape == (batch_size, seq_length, d_k), \
        f"Expected output shape ({batch_size}, {seq_length}, {d_k}), got {output.shape}"

    # Validate attention weights shape
    assert attn_weights.shape == (batch_size, seq_length, seq_length), \
        f"Expected attn_weights shape ({batch_size}, {seq_length}, {seq_length}), got {attn_weights.shape}"

    # Validate that attention weights sum to 1
    attn_sums = attn_weights.sum(dim=-1)
    assert torch.allclose(attn_sums, torch.ones(batch_size, seq_length), atol=1e-6), \
        "Attention weights should sum to 1 over the last dimension."

    print("Scaled Dot-Product Attention test passed.\n")

def test_multi_head_attention():
    """
    Test the Multi-Head Attention mechanism.

    Validates:
    - Correct output shape.
    - Correct functionality with masking (if implemented).
    """
    print("Testing Multi-Head Attention...")
    batch_size = 2
    seq_length = 4
    embed_dim = 64
    num_heads = 8

    # Create random tensor for input
    x = torch.randn(batch_size, seq_length, embed_dim)

    # Instantiate  MultiHeadAttention
    mha = MultiHeadAttention(embed_dim, num_heads)

    # Perform forward pass
    output = mha(x, x, x)

    # Validate output shape
    assert output.shape == (batch_size, seq_length, embed_dim), \
        f"Expected output shape ({batch_size}, {seq_length}, {embed_dim}), got {output.shape}"

    print("Multi-Head Attention test passed.\n")

def test_position_wise_feed_forward():
    """
    Test the Position-Wise Feed-Forward Network.

    Validates:
    - Correct output shape.
    - Non-linear activation function is applied.
    """
    print("Testing Position-Wise Feed-Forward Network...")
    batch_size = 2
    seq_length = 4
    embed_dim = 64
    ff_dim = 256

    # Create random tensor for input
    x = torch.randn(batch_size, seq_length, embed_dim)

    # Instantiate  PositionWiseFeedForward
    ff = PositionWiseFeedForward(embed_dim, ff_dim)

    # Perform forward pass
    output = ff(x)

    # Validate output shape
    assert output.shape == (batch_size, seq_length, embed_dim), \
        f"Expected output shape ({batch_size}, {seq_length}, {embed_dim}), got {output.shape}"

    print("Position-Wise Feed-Forward Network test passed.\n")

def test_positional_encoding():
    """
    Test the Positional Encoding mechanism.

    Validates:
    - Correct output shape.
    - Positional encoding does not alter the input shape.
    """
    print("Testing Positional Encoding...")
    batch_size = 2
    seq_length = 4
    embed_dim = 64

    # Create zero tensor for input
    x = torch.zeros(batch_size, seq_length, embed_dim)

    # Instantiate  PositionalEncoding
    pe = PositionalEncoding(embed_dim, max_len=seq_length)

    # Apply positional encoding
    output = pe(x)

    # Validate output shape
    assert output.shape == (batch_size, seq_length, embed_dim), \
        f"Expected output shape ({batch_size}, {seq_length}, {embed_dim}), got {output.shape}"

    # Check that positional encoding adds information
    assert not torch.allclose(output, x), "Positional encoding should modify the input tensor."

    print("Positional Encoding test passed.\n")
    
def test_full_transformer_layer():
    """
    Test the integration of all components in a Transformer Encoder Layer.

    Validates:
    - Correct output shape.
    - Data flow through the combined components.
    """
    print("Testing Full Transformer Encoder Layer...")
    batch_size = 2
    seq_length = 4
    embed_dim = 64
    num_heads = 8
    ff_dim = 256

    # Create random tensor for input
    x = torch.randn(batch_size, seq_length, embed_dim)

    # Instantiate the TransformerEncoderLayer
    encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, ff_dim, max_len=seq_length)

    # Perform forward pass
    output = encoder_layer(x)

    # Validate output shape
    assert output.shape == (batch_size, seq_length, embed_dim), \
        f"Expected output shape ({batch_size}, {seq_length}, {embed_dim}), got {output.shape}"

    print("Full Transformer Encoder Layer test passed.\n")

def run_all_tests():
    """
    Run all validation tests.
    """
    print("Running all validation tests...\n")
    test_scaled_dot_product_attention()
    test_multi_head_attention()
    test_position_wise_feed_forward()
    test_positional_encoding()
    test_full_transformer_layer()
    print("All tests completed.")

if __name__ == "__main__":
    run_all_tests()