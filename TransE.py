import torch
from torch import nn
from torch.nn import functional as F
from triples import Fb23715k


class TransE(nn.Module):
    def __init__(self, num_of_entities: int, num_of_relations: int, num_of_dimensions: int, norm: int = 2):
        super().__init__()
        self.norm = norm
        with torch.no_grad():
            self.entity_embeddings = nn.Embedding(num_of_entities, num_of_dimensions)
            self.entity_embeddings.weight.data.uniform_(-6 / num_of_dimensions ** 0.5, 6 / num_of_dimensions ** 0.5)

            self.relation_embeddings = nn.Embedding(num_of_relations, num_of_dimensions)
            self.relation_embeddings.weight.data.uniform_(-6 / num_of_dimensions ** 0.5, 6 / num_of_dimensions ** 0.5)
            self.relation_embeddings.weight.data = F.normalize(self.relation_embeddings.weight.data, p=2, dim=1)

    def forward(self, batch: torch.tensor, corrupted_batch: torch.tensor):
        # normalize entity embeddings
        with torch.no_grad():
            self.entity_embeddings.weight.data = F.normalize(self.entity_embeddings.weight.data, p=2, dim=1)

        # destructure batch into head_ids, relation_ids, tail_ids
        batch_head_ids = batch[:, 0]
        batch_relation_ids = batch[:, 1]
        batch_tail_ids = batch[:, 2]

        corr_batch_head_ids = corrupted_batch[:, 0]
        corr_batch_relation_ids = corrupted_batch[:, 1]
        corr_batch_tail_ids = corrupted_batch[:, 2]

        # get corresponding embeddings
        batch_head_embeddings = self.entity_embeddings(batch_head_ids)
        batch_relation_embeddings = self.relation_embeddings(batch_relation_ids)
        batch_tail_embeddings = self.entity_embeddings(batch_tail_ids)

        corr_batch_head_embeddings = self.entity_embeddings(corr_batch_head_ids)
        corr_batch_relation_embeddings = self.relation_embeddings(corr_batch_relation_ids)
        corr_batch_tail_embeddings = self.entity_embeddings(corr_batch_tail_ids)

        batch_energies = batch_head_embeddings + batch_relation_embeddings - batch_tail_embeddings
        corr_batch_energies = corr_batch_head_embeddings + corr_batch_relation_embeddings - corr_batch_tail_embeddings

        return batch_energies, corr_batch_energies
