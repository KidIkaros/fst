
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from datasets import load_dataset
try:
    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
except ImportError:
    print("SONAR not found. Install it with: pip install sonar-space")

class SinusoidalPositionEncoder(nn.Module):
    """Sinusoidal positional encoding"""
    def __init__(self, encoding_dim, max_seq_len=514):
        super().__init__()
        self.encoding_dim = encoding_dim
        
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, encoding_dim, 2) * (-math.log(10000.0) / encoding_dim))
        
        pe = torch.zeros(max_seq_len, encoding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class FullTokenSonar(nn.Module):
    """
    Full token-level student model with transformer architecture.
    Refactored for Production Grade stability.
    """
    def __init__(self, vocab_size=256206, embed_dim=512, layers=6, num_heads=8, output_dim=1024, dropout=0.1):
        super().__init__()
        
        # A. SMALLER EMBEDDINGS (512 vs 1024)
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        
        # B. POSITIONAL ENCODING
        # Defined *before* usage now to avoid definition order bugs
        self.pos_encoder = SinusoidalPositionEncoder(embed_dim, max_seq_len=514)
        
        # C. COMPACT TRANSFORMER (6 layers vs 24, 8 heads vs 16)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            activation="gelu",
            norm_first=True,
            dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        
        # D. ATTENTION POOLING (Learned importance)
        self.attention_pool = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # E. PROJECTION TO MATCH TEACHER DIMENSION
        self.projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # F. DROPOUT
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass for batched token sequences.
        Replaces the monkey-patched forward_token_batch.
        """
        # 1. Embeddings
        x = self.embed(input_ids)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        # 2. Transformer
        x = self.encoder(x, src_key_padding_mask=attention_mask)
        
        # 3. Attention Pooling
        # Calculate attention scores
        attn_scores = self.attention_pool(x).squeeze(-1)  # [B, L]
        
        # Apply mask (convert bool to float for masking)
        # True (pad) -> -inf, False (real) -> 0
        mask_float = attention_mask.float() * -1e9  
        attn_scores = attn_scores + mask_float
        
        # Softmax weights
        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)  # [B, L, 1]
        
        # Weighted sum
        x_pooled = torch.sum(x * attn_weights, dim=1)  # [B, D]
        
        # 4. Projection
        x = self.projection(x_pooled)
        
        return F.normalize(x, p=2, dim=-1)

def copy_embeddings_from_teacher(teacher, student):
    """Copy embedding weights from teacher to student (The 'Cheat Code')"""
    print("Copying embeddings from teacher...")
    
    teacher_embed = teacher.model.encoder_frontend.embed.weight.data
    # Handle DataParallel if present
    if isinstance(student, nn.DataParallel):
        student_embed = student.module.embed.weight.data
    else:
        student_embed = student.embed.weight.data
        
    # Copy first 512 dimensions
    student_embed[:, :] = teacher_embed[:, :512]  
    
    print(f"Copied embeddings: {teacher_embed.shape} -> {student_embed.shape}")

def evaluate_full_token_sonar(student, teacher, device):
    """Evaluate the trained model"""
    print("\n=== Full TokenSonar Evaluation ===")
    
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a fascinating field of study.",
        "Python is a versatile programming language.",
        "Neural networks learn complex patterns from data.",
        "Artificial intelligence transforms industries."
    ]
    
    student.eval()
    with torch.no_grad():
        similarities = []
        for text in test_texts:
            try:
                # Teacher
                teacher_vec = teacher.predict([text], source_lang="eng_Latn")
                teacher_vec = F.normalize(teacher_vec, p=2, dim=-1)
                
                # Student
                tokenized = teacher.tokenizer.create_encoder()(text)
                input_ids = tokenized.unsqueeze(0).to(device)
                attention_mask = torch.zeros(1, len(tokenized), dtype=torch.bool).to(device)
                
                # Using standard forward call now
                student_vec = student(input_ids, attention_mask)
                
                similarity = F.cosine_similarity(student_vec, teacher_vec).item()
                similarities.append(similarity)
                
                print(f"'{text[:45]}...' -> {similarity:.4f}")
            except Exception as e:
                print(f"Failed to evaluate '{text[:40]}...': {e}")
                
    if similarities:
        print(f"\nAverage similarity: {sum(similarities) / len(similarities):.4f}")

if __name__ == "__main__":
    print("Model definition valid. Import this file to use FullTokenSonar.")
