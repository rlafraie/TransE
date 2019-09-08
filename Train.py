import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
from torch import optim
from TransE import TransE
from triples import Dataset, Datasubset
from typing import List, Dict
from tqdm import tqdm

class Experiment():
    def __init__(self, knowledge_graph: Dataset, num_of_epochs: int = 50, batch_size: int = 100, margin: int = 1,
                 norm: int = 1, learning_rate: float = 0.01, num_of_dimensions: int = 50):
        self.knowledge_graph = knowledge_graph

        self.num_of_dimensions = num_of_dimensions
        self.num_of_epochs = num_of_epochs
        self.batch_size = batch_size
        self.margin = margin
        self.norm = norm
        self.learning_rate = learning_rate

        self.num_of_dimensions = num_of_dimensions
        self.num_of_entities = knowledge_graph.num_of_entities
        self.num_of_relations = knowledge_graph.num_of_relations

        self.transe: TransE = TransE(knowledge_graph.num_of_entities, knowledge_graph.num_of_relations,
                                     num_of_dimensions, norm)
        self.loss: float = 0.0

        self.validation_mean_rank: float = 0.0
        self.best_validation_mean_rank: float = 0.0
        self.validation_mean_rank_filtered: float = 0.0
        self.best_validation_mean_rank_filtered: float = 0.0

        self.validation_hits10: int = 0
        self.validation_hits10_filtered: int = 0

        self.test_mean_rank: int = 0
        self.best_test_mean_rank: int = 0
        self.test_mean_rank_filtered: int = 0
        self.best_test_mean_rank_filtered: int = 0

        self.test_hits10: int = 0
        self.test_hits10_filtered: int = 0

    def train(self, filtered_corrupted_batch=True):
        dataset: TensorDataset = TensorDataset(torch.tensor(self.knowledge_graph.training_triples))
        train_dl: DataLoader = DataLoader(dataset, batch_size=self.batch_size)
        optimizer: optim = optim.SGD(self.transe.parameters(), lr=self.learning_rate)

        for i in range(self.num_of_epochs):
            epoch_loss = 0
            for batch in tqdm(train_dl):
                mini_batch = batch[0]
                corr_mini_batch = \
                    self.knowledge_graph.get_corrupted_training_triples(mini_batch) if filtered_corrupted_batch \
                        else self.knowledge_graph.get_corrupted_batch_unfiltered(mini_batch)

                batch_loss, corr_batch_loss = self.transe(mini_batch, corr_mini_batch)

                loss = F.relu(self.margin + batch_loss.norm(p=self.norm, dim=1)
                              - corr_batch_loss.norm(p=self.norm, dim=1)).sum()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss

            with torch.no_grad():
                self.loss = epoch_loss / train_dl.__len__()

            validation_datasets = [self.knowledge_graph.valid_dataset, self.knowledge_graph.train_dataset,
                        self.knowledge_graph.test_dataset]
            self.validation_mean_rank, hits10 = self.get_mean_rank(validation_datasets)
            # self.validation_mean_rank_filtered = self.get_validation_mean_rank(filtered=True, fast_validation=True)
            print("loss for epoch %s:" % str(i + 1), self.loss)

            if self.best_validation_mean_rank > self.validation_mean_rank:
                self.best_validation_mean_rank = self.validation_mean_rank

            if self.best_validation_mean_rank_filtered > self.validation_mean_rank_filtered:
                self.best_validation_mean_rank_filtered = self.validation_mean_rank_filtered

    @torch.no_grad()
    def get_evluation_scores(self, link_prediction_dataset: Datasubset, filter_datasets: List[Datasubset], filtered=False, fast_testing=True) -> float:
        mean_rank = 0
        hits10_list = []
        link_prediction_triples = link_prediction_dataset.triples
        datasets = filter_datasets + [link_prediction_dataset]

        threshold = 3999 if fast_testing else len(link_prediction_triples) - 1

        for triple in tqdm(link_prediction_triples[:threshold]):
            head_id, relation_id, tail_id = triple[0], triple[1], triple[2]
            mean_rank += self.get_filtered_triple_mean_rank(head_id, relation_id, tail_id, datasets, hits10_list) \
                if filtered else self.get_raw_triple_mean_rank(head_id, relation_id, tail_id, hits10_list)

        return (mean_rank / (threshold + 1), len(hits10_list))

    def get_raw_triple_mean_rank(self, head_id: int, relation_id: int, tail_id: int, hits10_list: List) -> float:
        rank_head, rank_tail = self.get_raw_triple_ranks(head_id, relation_id, tail_id, hits10_list)

        return (rank_head + rank_tail) / 2

    def get_raw_triple_ranks(self, head_id: int, relation_id: int, tail_id: int, hits10_list: List) -> (int, int):
        head_embeddings = self.transe.entity_embeddings(torch.tensor(head_id)).repeat(
            self.knowledge_graph.num_of_entities, 1)
        relation_embeddings = self.transe.relation_embeddings(torch.tensor(relation_id)).repeat(
            self.knowledge_graph.num_of_entities, 1)
        tail_embeddings = self.transe.entity_embeddings(torch.tensor(tail_id)).repeat(
            self.knowledge_graph.num_of_entities, 1)

        head_loss = (self.transe.entity_embeddings.weight.data + relation_embeddings - tail_embeddings).norm(
            p=self.norm, dim=1)
        tail_loss = (head_embeddings + relation_embeddings - self.transe.entity_embeddings.weight.data).norm(
            p=self.norm, dim=1)

        rank_head = (head_loss.sort()[1] == head_id).nonzero().item() + 1
        rank_tail = (tail_loss.sort()[1] == tail_id).nonzero().item() + 1

        if rank_head < 11:
            hits10_list.append(rank_head)
        if rank_tail < 11:
            hits10_list.append(str(rank_tail))

        return rank_head, rank_tail

    def get_filtered_triple_mean_rank(self, head_id: int, relation_id: int, tail_id: int,
                                      datasets: List[Datasubset], hits10_list: List) -> float:
        rank_head, rank_tail = self.get_filtered_ranks(head_id, relation_id, tail_id, datasets)
        if rank_head < 11:
            hits10_list.append(rank_head)

        if rank_tail < 11:
            hits10_list.append(str(rank_tail))

        return (rank_head + rank_tail) / 2

    def get_filtered_ranks(self, head_id: int, relation_id: int, tail_id: int, datasets: List[Datasubset]) -> (int, int):
        head_list = [entity for entity in range(self.knowledge_graph.num_of_entities)]
        tail2head_lookups = [dataset.tail2head_lookup for dataset in datasets]
        head_filter = self.get_filter(tail_id, relation_id, tail2head_lookups)

        head_list = list(set(head_list) - set(head_filter))
        head_list = list(set(head_list) - {head_id}) + [head_id]
        head_list_embeddings = self.transe.entity_embeddings(torch.tensor(head_list))

        head_loss = (head_list_embeddings
                     + self.transe.relation_embeddings(torch.tensor(relation_id)).repeat(len(head_list), 1)
                     - self.transe.entity_embeddings(torch.tensor(tail_id)).repeat(len(head_list), 1)).norm(p=self.norm,
                                                                                                            dim=1)
        tail_list = [entity for entity in range(self.knowledge_graph.num_of_entities)]
        head2tail_lookups = [dataset.head2tail_lookup for dataset in datasets]
        tail_filter = self.get_filter(head_id, relation_id, head2tail_lookups)

        tail_list = list(set(tail_list) - set(tail_filter))
        tail_list = list(set(tail_list) - {tail_id}) + [tail_id]
        tail_list_embeddings = self.transe.entity_embeddings(torch.tensor(tail_list))

        tail_loss = (self.transe.entity_embeddings(torch.tensor(head_id)).repeat(len(tail_list), 1)
                     + self.transe.relation_embeddings(torch.tensor(relation_id)).repeat(len(tail_list), 1)
                     - tail_list_embeddings).norm(p=self.norm, dim=1)

        rank_head = (head_loss.sort()[1] == len(head_list) - 1).nonzero().item() + 1
        rank_tail = (tail_loss.sort()[1] == len(tail_list) - 1).nonzero().item() + 1

        return rank_head, rank_tail

    def get_filter(self, entity_id: int, relation_id: int, lookup_dicts: List[Dict]) -> List:
        entities_filter = []

        for lookup_dict in lookup_dicts:
            if entity_id in lookup_dict:
                if relation_id in lookup_dict[entity_id]:
                    entities_filter.extend(lookup_dict[entity_id][relation_id])

        return entities_filter

    def save_model_params(self, filename):
        file_path = self.knowledge_graph.data_dir / str(filename + '.pickle')
        torch.save(self.transe.state_dict(), file_path)

    def load_model_params(self, filename):
        file_path = self.knowledge_graph.data_dir / str(filename + '.pickle')
        self.transe.load_state_dict(torch.load(file_path))
