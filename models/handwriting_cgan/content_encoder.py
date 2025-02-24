import torch.nn as nn

class ContentEncoder(nn.Module):
    def __init__(self, vocab_size, char_embedding_dim=128, seq_len=10):
        super(ContentEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, char_embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(char_embedding_dim * seq_len, 256),
            nn.ReLU(),
            nn.Linear(256, 256)
        )
    
    def forward(self, text):
        embeddings = self.embedding(text)  # shape: [batch_size, seq_len, char_embedding_dim]
        embeddings = embeddings.view(embeddings.size(0), -1)  # flatten
        content_features = self.fc(embeddings)
        return content_features
