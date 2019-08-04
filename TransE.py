from torch import nn
from torch.nn import functional as F

class TransE(nn.Module):
    def __init__(self, num_of_entities, num_of_relations, num_of_dimensions):
        super().__init__()
        self.entity_embeddings = nn.Embedding(num_of_entities, num_of_dimensions)
        self.entity_embeddings.weight.data.uniform_(-6 / num_of_dimensions ** 0.5, 6 / num_of_dimensions ** 0.5)

        self.relation_embeddings = nn.Embedding(num_of_relations, num_of_dimensions)
        self.relation_embeddings.weight.data.uniform_(-6 / num_of_dimensions**0.5, 6 / num_of_dimensions ** 0.5)
        self.relation_embeddings.weight.data = F.normalize(self.relation_embeddings.weight.data, p=2, dim=1)

    def forward(self, batch, corrupted_batch):
        pass


