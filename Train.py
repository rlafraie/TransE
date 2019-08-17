import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
from torch import optim
from TransE import TransE


class Experiment:
    def __init__(self, knowledge_graph, num_of_epochs=10, batch_size=100, margin=1, norm=1, learning_rate=0.01,
                 num_of_dimensions=50):
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

        self.transe = TransE(knowledge_graph.num_of_entities, knowledge_graph.num_of_relations, num_of_dimensions, norm)
        self.dataset = TensorDataset(torch.tensor(knowledge_graph.training_triples))
        self.train_dl = DataLoader(self.dataset, batch_size=batch_size)

        self.optimizer = optim.SGD(self.transe.parameters(), lr=learning_rate)

        self.loss = 0

        self.validation_mean_rank = 0
        self.best_validation_mean_rank = 0
        self.validation_mean_rank_filtered = 0
        self.best_validation_mean_rank_filtered = 0

        self.validation_hits10 = 0
        self.validation_hits10_filtered = 0

        self.test_mean_rank = 0
        self.best_test_mean_rank = 0
        self.test_mean_rank_filtered = 0
        self.best_test_mean_rank_filtered = 0

        self.test_hits10 = 0
        self.test_hits10_filtered = 0

    def train(self):
        ##Training Iterations##
        for i in range(self.num_of_epochs):
            for batch in self.train_dl:
                mini_batch = batch[0]
                corr_mini_batch = self.knowledge_graph.get_corrupted_training_triples(mini_batch)
                batch_loss, corr_batch_loss = self.transe(mini_batch, corr_mini_batch)

                self.loss = F.relu(
                    self.margin + batch_loss.norm(p=self.norm, dim=1) - corr_batch_loss.norm(p=self.norm, dim=1)).sum()

                self.loss.backward()
                print("------------------------------------------------------------")
                print("loss: ", self.loss)
                print("------------------------------------------------------------")
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.validation_mean_rank = self.get_validation_mean_rank(fast_validation=True)
            # self.validation_mean_rank_filtered = self.get_validation_mean_rank(filtered=True, fast_validation=True)

        if self.best_validation_mean_rank > self.validation_mean_rank:
            self.best_validation_mean_rank = self.validation_mean_rank

        if self.best_validation_mean_rank_filtered > self.validation_mean_rank_filtered:
            self.best_validation_mean_rank_filtered = self.validation_mean_rank_filtered

    @torch.no_grad()
    def get_validation_mean_rank(self, filtered=False, fast_validation=True):
        mean_rank = 0
        processed = 0
        threshold = 3999 if fast_validation else len(self.knowledge_graph.validation_triples) - 1

        if filtered:
            self.validation_hits10_filtered = 0
        else:
            self.validation_hits10 = 0

        for triple in self.knowledge_graph.validation_triples[:threshold]:
            head_id, relation_id, tail_id = triple[0], triple[1], triple[2]
            mean_rank += self.get_filtered_validation_triple_mean_rank(head_id, relation_id, tail_id) \
                if filtered else self.get_raw_validation_triple_mean_rank(head_id, relation_id, tail_id)
            processed += 1
            print("processed " + str(processed) + " of " + str(threshold))

        return mean_rank / threshold + 1

    @torch.no_grad()
    def get_test_mean_rank(self, filtered=False, fast_testing=True):
        mean_rank = 0
        processed = 0
        threshold = 3999 if fast_testing else len(self.knowledge_graph.test_triples) - 1

        if filtered:
            self.test_hits10_filtered = 0
        else:
            self.test_hits10 = 0

        for triple in self.knowledge_graph.test_triples[threshold]:
            head_id, relation_id, tail_id = triple[0], triple[1], triple[2]
            mean_rank += self.get_filtered_test_triple_mean_rank(head_id, relation_id, tail_id) \
                if filtered else self.get_raw_test_triple_mean_rank(head_id, relation_id, tail_id)
            processed += 1
            print("processed " + str(processed) + " of " + str(threshold))

        return mean_rank / threshold

    def get_raw_triple_ranks(self, head_id, relation_id, tail_id):
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

        return rank_head, rank_tail

    def get_raw_validation_triple_mean_rank(self, head_id, relation_id, tail_id):
        rank_head, rank_tail = self.get_raw_triple_ranks(head_id, relation_id, tail_id)

        if rank_head < 11:
            self.validation_hits10 += 1
        if rank_tail < 11:
            self.validation_hits10 += 1

        return (rank_head + rank_tail) / 2

    def get_raw_test_triple_mean_rank(self, head_id, relation_id, tail_id):
        rank_head, rank_tail = self.get_raw_triple_ranks(head_id, relation_id, tail_id)

        if rank_head < 11:
            self.test_hits10 += 1
        if rank_tail < 11:
            self.test_hits10 += 1

        return (rank_head + rank_tail) / 2

    def get_head_validation_filter(self, relation_id, tail_id):
        head_validation_filter = []

        if tail_id in self.knowledge_graph.train_tail2head_lookup:
            if relation_id in self.knowledge_graph.train_tail2head_lookup[tail_id]:
                head_validation_filter.extend(self.knowledge_graph.train_tail2head_lookup[tail_id][relation_id])
        if tail_id in self.knowledge_graph.valid_tail2head_lookup:
            if relation_id in self.knowledge_graph.valid_tail2head_lookup[tail_id]:
                head_validation_filter.extend(self.knowledge_graph.valid_tail2head_lookup[tail_id][relation_id])

        return head_validation_filter

    def get_head_test_filter(self, relation_id, tail_id):
        head_test_filter = self.get_head_validation_filter(relation_id, tail_id)

        if tail_id in self.knowledge_graph.test_tail2head_lookup:
            if relation_id in self.knowledge_graph.test_tail2head_lookup[tail_id]:
                head_test_filter.extend(self.knowledge_graph.test_tail2head_lookup[tail_id][relation_id])

        return head_test_filter

    def get_tail_validation_filter(self, head_id, relation_id):
        tail_validation_filter = []

        if head_id in self.knowledge_graph.train_head2tail_lookup:
            if relation_id in self.knowledge_graph.train_head2tail_lookup[head_id]:
                tail_validation_filter.extend(self.knowledge_graph.train_head2tail_lookup[head_id][relation_id])
        if head_id in self.knowledge_graph.valid_head2tail_lookup:
            if relation_id in self.knowledge_graph.valid_head2tail_lookup[head_id]:
                tail_validation_filter.extend(self.knowledge_graph.valid_head2tail_lookup[head_id][relation_id])

        return tail_validation_filter

    def get_tail_test_filter(self, head_id, relation_id):
        tail_test_filter = self.get_tail_validation_filter(head_id, relation_id)

        if head_id in self.knowledge_graph.test_head2tail_lookup:
            if relation_id in self.knowledge_graph.test_head2tail_lookup[head_id]:
                tail_test_filter.extend(self.knowledge_graph.test_head2tail_lookup[head_id][relation_id])

        return tail_test_filter

    def get_filtered_validation_triple_mean_rank(self, head_id, relation_id, tail_id):
        head_list = [entity for entity in range(self.knowledge_graph.num_of_entities)]
        head_filter = self.get_head_validation_filter(relation_id, tail_id)

        head_list = list(set(head_list) - set(head_filter))
        head_list = list(set(head_list) - {head_id}) + [head_id]
        head_list_embeddings = self.transe.entity_embeddings(torch.tensor(head_list))

        head_loss = (head_list_embeddings
                     + self.transe.relation_embeddings(torch.tensor(relation_id)).repeat(len(head_list), 1)
                     - self.transe.entity_embeddings(torch.tensor(tail_id)).repeat(len(head_list), 1)).norm(p=self.norm,
                                                                                                            dim=1)
        tail_list = [entity for entity in range(self.knowledge_graph.num_of_entities)]
        tail_filter = self.get_tail_validation_filter(head_id, relation_id)

        tail_list = list(set(tail_list) - set(tail_filter))
        tail_list = list(set(tail_list) - {tail_id}) + [tail_id]
        tail_list_embeddings = self.transe.entity_embeddings(torch.tensor(tail_list))

        tail_loss = (self.transe.entity_embeddings(torch.tensor(head_id)).repeat(len(tail_list), 1)
                     + self.transe.relation_embeddings(torch.tensor(relation_id)).repeat(len(tail_list), 1)
                     - tail_list_embeddings).norm(p=self.norm, dim=1)

        rank_head = (head_loss.sort()[1] == len(head_list) - 1).nonzero().item() + 1
        if rank_head < 11:
            self.validation_hits10_filtered += 1

        rank_tail = (tail_loss.sort()[1] == len(tail_list) - 1).nonzero().item() + 1
        if rank_tail < 11:
            self.validation_hits10_filtered += 1

        return (rank_head + rank_tail) / 2

    def get_filtered_test_triple_mean_rank(self, head_id, relation_id, tail_id):
        head_list = [entity for entity in range(self.knowledge_graph.num_of_entities)]
        head_filter = self.get_head_test_filter(relation_id, tail_id)

        head_list = list(set(head_list) - set(head_filter))
        head_list = list(set(head_list) - {head_id}) + [head_id]
        head_list_embeddings = self.transe.entity_embeddings(torch.tensor(head_list))

        head_loss = (head_list_embeddings
                     + self.transe.relation_embeddings(torch.tensor(relation_id)).repeat(len(head_list), 1)
                     - self.transe.entity_embeddings(torch.tensor(tail_id)).repeat(len(head_list), 1)).norm(p=self.norm,
                                                                                                            dim=1)
        tail_list = [entity for entity in range(self.knowledge_graph.num_of_entities)]
        tail_filter = self.get_tail_test_filter(head_id, relation_id)

        tail_list = list(set(tail_list) - set(tail_filter))
        tail_list = list(set(tail_list) - {tail_id}) + [tail_id]
        tail_list_embeddings = self.transe.entity_embeddings(torch.tensor(tail_list))

        tail_loss = (self.transe.entity_embeddings(torch.tensor(head_id)).repeat(len(tail_list), 1)
                     + self.transe.relation_embeddings(torch.tensor(relation_id)).repeat(len(tail_list), 1)
                     - tail_list_embeddings).norm(p=self.norm, dim=1)

        rank_head = (head_loss.sort()[1] == len(head_list) - 1).nonzero().item() + 1
        if rank_head < 11:
            self.test_hits10_filtered += 1

        rank_tail = (tail_loss.sort()[1] == len(tail_list) - 1).nonzero().item() + 1
        if rank_tail < 11:
            self.test_hits10_filtered += 1

        return (rank_head + rank_tail) / 2
