import math
import torch
from torch import nn
import torch.nn.functional as F

'''
Assumptions
Throughout this process, I made a few key assumptions:
1. Attention here is self-attention, and not cross-attention.
2. The embedding dimension passed in multi-head-attention is always divisble by num_heads.
3. Defaulted to self-attention for the encoder (type argument in the Scaled Dot Product Attention class), with masking options available for the decoder.
4. Haven't added dropout, layer normalization, and residual connections - can add later.
'''

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention.
    
    The attention mechanism is defined as:
        attention(Q, K, V) = softmax(Q * K.T / sqrt(d_k)) * V
    
    where Q, K, V are the query, key, and value tensors, respectively.
    
    This class is designed to based on the assumption that it do "self-attention" and
    can also be used for "encoder" (without masking) and "decoder" (with masking) attention.
    """
    def __init__(self, type_='encoder'):
        super(ScaledDotProductAttention, self).__init__()
        self.type_ = type_
        self.isInitialized_tril = False

    def forward(self, Q, K, V):
        """
        Computes the scaled dot-product attention.
        B - batch_size, T - seq_length, d_k - embedding_dimension
        Args:
            Q (torch.Tensor): The query tensor of shape (B, T, d_k).
            K (torch.Tensor): The key tensor of shape (B, T, d_k).
            V (torch.Tensor): The value tensor of shape (B, T, d_k).
        
        Returns:
            output (torch.Tensor): The output tensor of shape (B, T, d_k).
            attention_weights (torch.Tensor): The attention weights of shape (B, T, T).
        """
        assert Q.shape==K.shape==V.shape

        ## COMPUTE SELF-ATTENTION SCOREs AND WEIGHTS
        # Q shape : (B, T, d_k), K shape : (B, T, d_k), K.transpose shape : (B, d_k, T), attention_weights shape : (B, T, T) 
        # Dividing by sqrt(d_k) as proposed in the paper to scale the dot-product attention 
        # @ is matrix multiplication - same as torch.matmul
        attention_scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1])
        
        if self.type_ == "decoder":
            # Apply mask to prevent attending to future tokens for decoder
            if not self.isInitialized_tril or self.tril.size(0) != Q.shape[-2]:
                self.register_buffer('tril', torch.tril(torch.ones(Q.shape[-2], Q.shape[-2], device=Q.device)))
                self.isInitialized_tril = True
            # Required for decoder : Apply the mask such that the future tokens cannot be attended to affect the past ones
            # For encoder - no masking
            attention_scores = attention_scores.masked_fill(self.tril[:Q.shape[-2], :Q.shape[-2]] == 0, float('-inf'))
        # Apply softmax to get the normalized attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Compute the output
        output = attention_weights @ V
        
        return output, attention_weights
    
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Mechanism.
    
    It uses Scaled Dot-Product Attention Mechanism.
    """
    def __init__(self, embed_dim, num_heads):
        """
        Initialize the Multi-Head Attention Mechanism.
        
        Args:
            embed_dim (int): The embedding dimension of the input and output.
            num_heads (int): The number of heads in the multi-head attention.
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        ## head_dim is the embed_dim of each head
        ## Assumption - embed_dim = num_heads * head_dim
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.out = nn.Linear(embed_dim, embed_dim)
        self.scaled_dot_product_attention = ScaledDotProductAttention()

    def forward(self, query, key, value):
        """
        Compute the multi-head attention.
        
        Args:
            query (torch.Tensor): The query tensor of shape (B, T, embed_dim).
            key (torch.Tensor): The key tensor of shape (B, T, embed_dim).
            value (torch.Tensor): The value tensor of shape (B, T, embed_dim).
        
        Returns:
            output (torch.Tensor): The output tensor of shape (B, T, embed_dim).
        """
        # Need of linear layer to query, key, and value tensors : paper - https://arxiv.org/pdf/1706.03762.pdf
        # Reshape the query, key, and value tensors to (B, num_heads, T, head_dim) after applying linear layers
        # to apply the attention mechanism in parallel across the heads.

        query_linear = self.query(query).view(-1, query.shape[1], self.num_heads, self.head_dim) # (B, T, num_heads, head_dim)
        query_linear = query_linear.transpose(1, 2) # (B, num_heads, T, head_dim) 

        key_linear = self.key(key).view(-1, key.shape[1], self.num_heads, self.head_dim) # (B, T, num_heads, head_dim)
        key_linear = key_linear.transpose(1, 2) # (B, num_heads, T, head_dim)

        value_linear = self.value(value).view(-1, value.shape[1], self.num_heads, self.head_dim) # (B, T, num_heads, head_dim)
        value_linear = value_linear.transpose(1, 2) # (B, num_heads, T, head_dim)

        output, attention_weights = self.scaled_dot_product_attention(query_linear, key_linear, value_linear) # output shape : (B, num_heads, T, head_dim)
        output = output.transpose(1, 2) # (B, T, num_heads, head_dim)
        output = output.contiguous() # contiguous copy of the original tensor, ensuring that the memory layout is correct for further operations.  
        output = output.view(-1, query.shape[1], self.embed_dim) # (B, T, embed_dim)
        output = self.out(output) # (B, T, embed_dim)
        return output


class PositionWiseFeedForward(nn.Module):
    """
    Position-Wise Feed-Forward Network.

    The Position-Wise Feed-Forward Network is a fully connected feed-forward network
    applied to each position separately and identically. This network applies two
    linear transformations with a non-linear ReLU activation in between. The output linear
    transformation is applied to the output of the ReLU activation function.

    Args:
        embed_dim (int): The embedding dimension of the model.
        ff_dim (int): The hidden dimension of the feed-forward network.

    Shape:
        - Input: `(B, T, embed_dim)`
        - Output: `(B, T, embed_dim)`
    """
    def __init__(self, embed_dim, ff_dim):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.relu  = nn.ReLU()

    def forward(self, x):
        """
        Compute the position-wise feed-forward network.

        Args:
            x (torch.Tensor): The input tensor of shape (B, T, embed_dim).

        Returns:
            output (torch.Tensor): The output tensor of shape (B, T, embed_dim).
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
        

class PositionalEncoding(nn.Module):
    """
    Positional Encoding.

    The Positional Encoding layer encodes the position of each element in the input sequence
    into a fixed-length vector. This is used to preserve the order of the sequence.

    Args:
        embed_dim (int): The embedding dimension of the model.
        max_len (int): The maximum length of the sequence.

    Shape:
        - Input: `(B, T, embed_dim)`
        - Output: `(B, T, embed_dim)`
    """

    def __init__(self, embed_dim, max_len):
        super(PositionalEncoding, self).__init__()
        # Create the positional encoding tensor
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / embed_dim)) # as in paper

        # Compute the positional encoding
        pe = torch.zeros(max_len, embed_dim)  # (max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices - sinusoidal
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices - cosine

        # Add batch dimension
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)

        # Register as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Add the positional encoding to the input
        return x + self.pe[:, :x.size(1)]

class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer.

    The Transformer Encoder Layer applies self-attention (after using positional encoding) and a feed-forward network to the input sequence.

    Args:
        embed_dim (int): The embedding dimension of the model.
        num_heads (int): The number of heads in the multi-head attention.
        ff_dim (int): The hidden dimension of the feed-forward network.
        max_len (int): The maximum length of the sequence.

    Shape:
        - Input: `(B, T, embed_dim)`
        - Output: `(B, T, embed_dim)`
    """

    def __init__(self, embed_dim, num_heads, ff_dim, max_len):
        super(TransformerEncoderLayer, self).__init__()
        self.positional_encoding = PositionalEncoding(embed_dim, max_len)  # Add Positional Encoding
        self.self_attention = MultiHeadAttention(embed_dim, num_heads)
        self.feed_forward = PositionWiseFeedForward(embed_dim, ff_dim)

    def forward(self, x):
        """
        Compute the forward pass of the Transformer Encoder Layer.

        Args:
            x (torch.Tensor): The input tensor of shape (B, T, embed_dim).

        Returns:
            output (torch.Tensor): The output tensor of shape (B, T, embed_dim).
        """
        # Add positional encoding before attention
        x = self.positional_encoding(x)  # (B, T, embed_dim)
        # Compute self-attention
        x = self.self_attention(x, x, x)  # (B, T, embed_dim)
        # Apply feed-forward network
        x = self.feed_forward(x)  # (B, T, embed_dim)
        return x
