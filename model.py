import math
import torch
import torch.nn as nn

# Define the model
# The model is composed of three main parts:
# 1. The input embedding layer
# 2. The positional encoding layer
# 3. The transformer encoder layer

#the input embedding layer
class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model # dimension of the model
        self.embedding = nn.Embedding(vocab_size, d_model) # embedding layer
        self.vocab_size = vocab_size # size of the vocabulary

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x) * math.sqrt(self.d_model) # formula from the paper

#the positional encoding layer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # dimension of the model
        self.seq_len = seq_len # length of the sequence
        self.dropout = nn.Dropout(dropout) # dropout layer for regularization

        # create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        # calculate the divisors
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # calculate the positional encoding

        # apply the sin to even indices in the array
        pe[:, 0::2] = torch.sin(position * div_term)

        # apply the cos to odd indices in the array
        pe[:, 1::2] = torch.cos(position * div_term)

        # add a batch dimension
        pe = pe.unsqueeze(0) # shape (1, seq_len, d_model)
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + (self.pe[:, :x.shape(1), :]).requires_grad_(False) # add the positional encoding to the input
        return self.dropout(x)


#the transformer encoder layer

#LayerNornalization is used to normalize the input before applying the attention and feedforward layers    
class LayerNorm(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # multiplicative factor
        self.beta = nn.Parameter(torch.zeros(1)) # additive factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x= x.type(dtype=torch.float32)
        mean = x.type(dtype=torch.float32).mean(dim=-1, keepdim=True)
        std = x.type(dtype=torch.float32).std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta
    
#FeedForwardLayer is used to apply a feedforward layer to the input
class FeedForwardLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # dimension of the model
        self.d_ff = d_ff # dimension of the feedforward layer
        self.dropout = nn.Dropout(dropout) # dropout layer for regularization
        self.linear1 = nn.Linear(d_model, d_ff) # linear transformation
        self.linear2 = nn.Linear(d_ff, d_model) # linear transformation
        self.relu = nn.ReLU() # activation function

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # apply the first linear transformation and the activation function
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) --> (batch_size, seq_len, d_model)
        return self.linear2(self.dropout(self.relu(self.linear1(x)))) # formula from the paper

#MultiHeadAttentionLayer is used to apply the multi-head attention mechanism to the input
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # dimension of the model
        self.num_heads = num_heads # number of heads
        self.d_k = d_model // num_heads # dimension of the key and the value
        self.dropout = nn.Dropout(dropout) # dropout layer for regularization

        # Make sure d_model is divisible by h
        assert d_model % num_heads == 0, "d_model is not divisible by num_heads"

        # linear transformations for the queries, the keys, and the values
        self.linear_q = nn.Linear(d_model, d_model, bias=False) #wq
        self.linear_k = nn.Linear(d_model, d_model, bias=False) #wk
        self.linear_v = nn.Linear(d_model, d_model, bias=False) #wv

        # linear transformation for the output
        self.linear_o = nn.Linear(d_model, d_model, bias=False) #wo

    @staticmethod
    def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask, dropout) -> torch.Tensor:
        
        d_k = query.shape[-1]

        # calculate the attention scores
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9) # replace the masked values with a large negative value
        
        # apply the softmax function
        attention_scores = attention_scores.softmax(dim=-1) # (batch_size, num_heads, seq_len, seq_len)

        # apply the dropout
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return torch.matmul(attention_scores, value), attention_scores #attention scores are returned to visualize the attention weights


    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        # apply the linear transformations
        query = self.linear_q(q) # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        key= self.linear_k(k) # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        value = self.linear_v(v) # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)

        # split the queries, the keys, and the values into multiple heads
        # (batch_size, seq_len, d_model) -> (batch_size,  seq_len, num_heads, d_k) -> (batch_size, num_heads, seq_len, d_k)
        q = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1,2) 
        k = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k) 
        v = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k) 

        # calculate the scaled dot-product attention
        # apply the scaled dot-product attention
        x, self.attention_scores = MultiHeadAttentionLayer.scaled_dot_product_attention(q, k, v, mask, self.dropout) # (batch_size * num_heads, seq_len, d_k)

    
        # (batch_size, num_heads, seq_len, d_k) -> (batch_size, seq_len, num_heads, d_k) -> (batch_size, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.d_model) # (batch_size, seq_len, d_model)

        return self.linear_o(x) # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
    
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.norm = LayerNorm() # layer normalization
        self.dropout = nn.Dropout(dropout) # dropout layer for regularization

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        print(x.shape)
        return x + self.dropout(sublayer(self.norm(x))) # formula from the paper
    


#EncoderBlock 
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_layer: MultiHeadAttentionLayer, feed_forward_layer: FeedForwardLayer, dropout: float) -> None:
        super().__init__()
        self.self_attention_layer = self_attention_layer
        self.feed_forward_layer = feed_forward_layer
        self.dropout = nn.Dropout(dropout) # dropout layer for regularization
        self.residual_connection1 = ResidualConnection(dropout) # residual connection for the self-attention layer
        self.residual_connection2 = ResidualConnection(dropout) # residual connection for the feedforward layer

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # apply the first residual connection
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)

        print(f"encoder block x shape {x.shape}")
        x = self.residual_connection1(x, lambda x: self.self_attention_layer(x, x, x, mask)) # formula from the paper

        # apply the second residual connection
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        x = self.residual_connection2(x, self.feed_forward_layer) # formula from the paper

        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask) # apply the encoder block
        return self.norm(x) # apply the layer normalization


class DecoderBlock(nn.Module):
    def __init__(self, self_attention_layer: MultiHeadAttentionLayer, cross_attention_layer: MultiHeadAttentionLayer, feed_forward_layer: FeedForwardLayer, dropout: float) -> None:
        super().__init__()
        self.self_attention_layer = self_attention_layer
        self.cross_attention_layer = cross_attention_layer
        self.feed_forward_layer = feed_forward_layer
        self.dropout = nn.Dropout(dropout)
        self.residual_connection1 = ResidualConnection(dropout)
        self.residual_connection2 = ResidualConnection(dropout)
        self.residual_connection3 = ResidualConnection(dropout)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, source_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        x = self.residual_connection1(x, lambda x: self.self_attention_layer(x, x, x, target_mask)) # self-attention
        x = self.residual_connection2(x, lambda x: self.cross_attention_layer(x, encoder_output, encoder_output, source_mask)) #cross-attention
        x = self.residual_connection3(x, self.feed_forward_layer)
        return x
  
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, source_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output, source_mask, target_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size) # linear transformation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim=-1) # apply the softmax function and take the logarithm
    

class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, source_embedding: InputEmbedding, target_embedding: InputEmbedding, source_positional_encoding, target_positional_encoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.source_embedding = source_embedding
        self.target_embedding = target_embedding
        self.encoder = encoder
        self.decoder = decoder
        self.projection_layer = projection_layer
        self.source_positional_encoding = source_positional_encoding
        self.target_positional_encoding = target_positional_encoding


    def encode(self, source: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        source = self.source_embedding(source) # apply the source embedding
        source = self.source_positional_encoding(source) # apply the positional encoding

        return self.encoder(source, source_mask) # apply the encoder
    
    def decode(self, target: torch.Tensor, encoder_output: torch.Tensor, source_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        target = self.target_embedding(target) # apply the target embedding
        target = self.target_positional_encoding(target) # apply the positional encoding
        return self.decoder(target, encoder_output, source_mask, target_mask) # apply the decoder
    
    def projection(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection_layer(x) # apply the projection layer
    
def build_model(source_vocab_size: int, target_vocab_size: int, source_sequence_len: int, target_sequence_len: int, d_model: int = 512, num_heads: int = 8, d_ff: int = 2048, num_layers: int = 6, dropout: float = 0.1) -> Transformer:
    # create embedding layers
    source_embedding = InputEmbedding(d_model, source_vocab_size) # source embedding
    target_embedding = InputEmbedding(d_model, target_vocab_size) # target embedding

    # create the positional encoding layer
    source_positional_encoding = PositionalEncoding(d_model, source_sequence_len, dropout) # source positional encoding
    target_positional_encoding = PositionalEncoding(d_model, target_sequence_len, dropout) # target positional encoding

    # create the encoder layers
    encoder_blocks = []
    for _ in range(num_layers):
        encoder_self_attention_layer = MultiHeadAttentionLayer(d_model, num_heads, dropout) # self-attention
        encoder_feed_forward_layer = FeedForwardLayer(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_layer, encoder_feed_forward_layer, dropout)
        encoder_blocks.append(encoder_block)
    
    # create the decoder layers
    decoder_blocks = []
    for _ in range(num_layers):
        decoder_self_attention_layer = MultiHeadAttentionLayer(d_model, num_heads, dropout) # self-attention
        decoder_cross_attention_layer = MultiHeadAttentionLayer(d_model, num_heads, dropout) # cross-attention
        decoder_feed_forward_layer = FeedForwardLayer(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_layer, decoder_cross_attention_layer, decoder_feed_forward_layer, dropout)
        decoder_blocks.append(decoder_block)

    # create encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks)) # encoder
    decoder = Decoder(nn.ModuleList(decoder_blocks)) # decoder

    # create the projection layer
    projection_layer = ProjectionLayer(d_model, target_vocab_size) # projection layer

    # create the transformer model
    transformer = Transformer(encoder, decoder, source_embedding, target_embedding, source_positional_encoding, target_positional_encoding,  projection_layer) # transformer model

    #initialize the parameters using the Xavier uniform initializer
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer