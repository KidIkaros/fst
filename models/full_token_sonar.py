import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from datasets import load_dataset
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

class FullTokenSonar(nn.Module):
    """
    Full token-level student model with transformer architecture
    """
    def __init__(self, vocab_size=256206, embed_dim=512, layers=6, num_heads=8, output_dim=1024):
        super().__init__()
        
        # A. SMALLER EMBEDDINGS (512 vs 1024)
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        
        # B. POSITIONAL ENCODING
        self.pos_encoder = SinusoidalPositionEncoder(embed_dim, max_seq_len=514)
        
        # C. COMPACT TRANSFORMER (6 layers vs 24, 8 heads vs 16)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            activation="gelu",
            norm_first=True,
            dropout=0.1
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
        self.dropout = nn.Dropout(0.1)

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

def forward_token_batch(self, input_ids, attention_mask):
    """Forward pass for batched token sequences"""
    # Embeddings
    x = self.embed(input_ids)
    x = self.pos_encoder(x)
    x = self.dropout(x)
    
    # Transformer
    x = self.encoder(x, src_key_padding_mask=attention_mask)
    
    # Attention Pooling
    # Calculate attention scores
    attn_scores = self.attention_pool(x).squeeze(-1)  # [B, L]
    
    # Apply mask (convert bool to float for masking)
    mask_float = attention_mask.float() * -1e9  # True (pad) -> -inf, False (real) -> 0
    attn_scores = attn_scores + mask_float
    
    # Softmax weights
    attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)  # [B, L, 1]
    
    # Weighted sum
    x_pooled = torch.sum(x * attn_weights, dim=1)  # [B, D]
    
    # Projection
    x = self.projection(x_pooled)
    return F.normalize(x, p=2, dim=-1)

# Attach method to class
FullTokenSonar.forward_token_batch = forward_token_batch

def copy_embeddings_from_teacher(teacher, student):
    """Copy embedding weights from teacher to student"""
    print("Copying embeddings from teacher...")
    
    teacher_embed = teacher.model.encoder_frontend.embed.weight.data
    student_embed = student.embed.weight.data
    student_embed[:, :] = teacher_embed[:, :512]  # Take first 512 dims
    
    print(f"Copied embeddings: {teacher_embed.shape} -> {student_embed.shape}")

def train_full_token_sonar():
    """Train full token-level student model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Full TokenSonar Training on {device} ---")
    
    # Memory settings
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    
    print("Loading SONAR teacher...")
    teacher = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder", 
        tokenizer="text_sonar_basic_encoder", 
        device=device,
        dtype=torch.float32
    )
    
    print("Creating Full TokenSonar student...")
    student = FullTokenSonar().to(device)
    
    # Copy embeddings
    copy_embeddings_from_teacher(teacher, student)
    
    # Optimizer settings
    optimizer = torch.optim.AdamW(
        student.parameters(), 
        lr=2e-4,  # Moderate learning rate
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler with warmup
    def lr_lambda(step):
        warmup_steps = 1000
        total_steps = 15000
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.1, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print("Loading training data...")
    # Load diverse training data
    dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
    
    texts = []
    for x in dataset:
        t = x['text']
        if len(t) > 30:
            parts = t.split('.')
            for p in parts:
                if len(p.strip()) > 30:
                    texts.append(p.strip())
    
    # Add some code samples for diversity
    code_samples = [
        "def function_name(): return True",
        "class MyClass: def __init__(self): pass",
        "import numpy as np; x = np.array([1,2,3])",
        "if condition: do_something()",
        "for item in items: process(item)",
        "try: risky_operation() except: handle_error()",
        "with open('file.txt') as f: data = f.read()",
        "result = [x*2 for x in items if x > 0]"
    ]
    texts.extend(code_samples * 100)  # Repeat for balance
    
    import random
    random.shuffle(texts)
    texts = texts[:20000]  # Good amount of data
    print(f"Training on {len(texts)} diverse samples")
    
    batch_size = 8  # Conservative for memory
    steps = len(texts) // batch_size
    
    print(f"Starting training with {steps} steps per epoch")
    
    best_similarity = 0.0
    
    for epoch in range(12):  # Reasonable number of epochs
        student.train()
        total_loss = 0
        similarities = []
        
        random.shuffle(texts)
        
        for i in range(steps):
            batch_texts = texts[i*batch_size : (i+1)*batch_size]
            
            # Tokenize batch
            batch_tokens = []
            batch_masks = []
            max_len = 0
            
            for text in batch_texts:
                try:
                    tokenized = teacher.tokenizer.create_encoder()(text)
                    if len(tokenized) < 512:  # Filter very long sequences
                        batch_tokens.append(tokenized)
                        max_len = max(max_len, len(tokenized))
                except:
                    continue
            
            if len(batch_tokens) < batch_size // 2:  # Skip if too many failed
                continue
            
            # Pad sequences
            padded_tokens = []
            attention_masks = []
            
            for tokens in batch_tokens:
                padded = torch.zeros(max_len, dtype=torch.long)
                mask = torch.ones(max_len, dtype=torch.bool)  # True = pad
                
                length = len(tokens)
                padded[:length] = tokens
                mask[:length] = False  # False = real token
                
                padded_tokens.append(padded)
                attention_masks.append(mask)
            
            input_ids = torch.stack(padded_tokens).to(device)
            attention_mask = torch.stack(attention_masks).to(device)
            
            try:
                # Teacher forward
                with torch.no_grad():
                    teacher_vecs = teacher.predict(batch_texts, source_lang="eng_Latn")
                    teacher_vecs = F.normalize(teacher_vecs, p=2, dim=-1)
                
                # Student forward
                student_vecs = student.forward_token_batch(input_ids, attention_mask)
                student_vecs = F.normalize(student_vecs, p=2, dim=-1)
                
                # Loss: cosine similarity
                loss = 1.0 - F.cosine_similarity(student_vecs, teacher_vecs).mean()
                
                # Track similarity
                batch_similarity = F.cosine_similarity(student_vecs, teacher_vecs).mean().item()
                similarities.append(batch_similarity)
                
                # Update
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM at step {i}, skipping")
                    torch.cuda.empty_cache()
                    continue
                else:
                    print(f"Error at step {i}: {e}")
                    continue
            
            if i % 100 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Epoch {epoch} | Step {i} | Loss: {loss.item():.4f} | Sim: {batch_similarity:.4f} | LR: {current_lr:.6f}")
        
        if len(similarities) > 0:
            avg_loss = total_loss / len(similarities)
            avg_similarity = sum(similarities) / len(similarities)
            
            print(f">>> Epoch {epoch} Complete")
            print(f"    Avg Loss: {avg_loss:.4f}")
            print(f"    Avg Similarity: {avg_similarity:.4f}")
            
            # Save best model
            if avg_similarity > best_similarity:
                best_similarity = avg_similarity
                torch.save(student.state_dict(), "full_token_sonar_best.pt")
                print(f"    New best model! Similarity: {best_similarity:.4f}")
            
            # Check convergence
            if avg_similarity > 0.80:
                print(">>> EXCELLENT CONVERGENCE!")
                break
            elif avg_similarity > 0.70:
                print(">>> GOOD CONVERGENCE!")
                break
            
            # Save epoch checkpoint
            torch.save(student.state_dict(), f"full_token_sonar_epoch_{epoch}.pt")
        else:
            print(f"Epoch {epoch} failed - no successful batches")
    
    print(f"Training complete! Best similarity: {best_similarity:.4f}")
    
    # Final evaluation
    evaluate_full_token_sonar(student, teacher, device)

def evaluate_full_token_sonar(student, teacher, device):
    """Evaluate the trained model"""
    print("\n=== Full TokenSonar Evaluation ===")
    
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a fascinating field of study.",
        "In a hole in the ground there lived a hobbit.",
        "Python is a versatile programming language.",
        "The weather today is quite pleasant.",
        "Neural networks learn complex patterns from data.",
        "Software engineering requires careful design.",
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
                
                student_vec = student.forward_token_batch(input_ids, attention_mask)
                student_vec = F.normalize(student_vec, p=2, dim=-1)
                
                similarity = F.cosine_similarity(student_vec, teacher_vec).item()
                similarities.append(similarity)
                
                print(f"'{text[:45]}...' -> {similarity:.4f}")
            except Exception as e:
                print(f"Failed to evaluate '{text[:40]}...': {e}")
                similarities.append(0.0)
    
    if similarities:
        avg_sim = sum(similarities) / len(similarities)
        print(f"\nAverage similarity: {avg_sim:.4f}")
        
        # Model size comparison
        sonar_params = 766_091_264
        student_params = sum(p.numel() for p in student.parameters())
        compression = sonar_params / student_params
        
        print(f"\nModel sizes:")
        print(f"SONAR: {sonar_params:,} parameters")
        print(f"Full TokenSonar: {student_params:,} parameters")
        print(f"Compression: {compression:.1f}x")
        
        if avg_sim > 0.85:
            print("🎉 EXCELLENT: Full TokenSonar successfully compressed SONAR!")
        elif avg_sim > 0.80:
            print("✅ EXCELLENT: Full TokenSonar achieved useful compression!")
        elif avg_sim > 0.70:
            print("✅ GOOD: Full TokenSonar achieved solid compression!")
        elif avg_sim > 0.60:
            print("⚠️ MODERATE: Full TokenSonar works but could be better")
        else:
            print("❌ POOR: Full TokenSonar needs more work")
    else:
        print("No successful evaluations")

if __name__ == "__main__":
    train_full_token_sonar()
