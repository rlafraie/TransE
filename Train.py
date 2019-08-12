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

        self.transe = TransE(knowledge_graph.num_of_entities, knowledge_graph.num_of_relations, num_of_dimensions)
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
                with torch.no_grad():
                    print("------------------------------------------------------------")
                    # print("entity sum: ", self.transe.entity_embeddings.weight.sum())
                    # print("relation sum: ", self.transe.relation_embeddings.weight.sum())
                    # print("entity norm: ", self.transe.entity_embeddings.weight.norm())
                    # print("relation norm: ", self.transe.relation_embeddings.weight.norm())
                    # print("entity_grad sum: ", self.transe.entity_embeddings.weight.grad.max())
                    # print("relation_grad sum: ", self.transe.relation_embeddings.weight.grad.max())
                    # print("entity_grad norm: ", self.transe.entity_embeddings.weight.grad.norm())
                    # print("relation_grad norm: ", self.transe.relation_embeddings.weight.grad.norm())
                    print("loss: ", self.loss)
                    print("------------------------------------------------------------")
                self.optimizer.step()
                self.optimizer.zero_grad()

        # self.validation_mean_rank = self.get_validation_mean_rank()
        # self.validation_mean_rank_filtered = self.get_validation_mean_rank(True)

        if self.best_validation_mean_rank > self.validation_mean_rank:
            self.best_validation_mean_rank = self.validation_mean_rank

        if self.best_validation_mean_rank_filtered > self.validation_mean_rank_filtered:
            self.best_validation_mean_rank_filtered = self.validation_mean_rank_filtered

    def get_validation_mean_rank(self, filtered=False):
        mean_rank = 0
        process = 0

        if filtered:
            self.validation_hits10_filtered = 0
        else:
            self.validation_hits10 = 0

        for triple in self.knowledge_graph.validation_triples:
            mean_rank += self.get_validation_triple_mean_rank(triple[0], triple[1], triple[2], filtered)
            process += 1
            print("processed " + str(process) + " of " + str(len(self.knowledge_graph.validation_triples)))

        return mean_rank / len(self.knowledge_graph.validation_triples)

    def get_validation_triple_mean_rank(self, head_id, relation_id, tail_id, filtered):
        head_list = list(self.knowledge_graph.id2entity_dict.keys())
        tail_list = list(self.knowledge_graph.id2entity_dict.keys())
        if filtered:
            filtered_head_entities = self.knowledge_graph.train_tail2head_lookup[tail_id][relation_id] + \
                                     self.knowledge_graph.valid_tail2head_lookup[tail_id][relation_id]
            head_list = list(set(head_list) - set(filtered_head_entities))

            filtered_tail_entities = self.knowledge_graph.train_head2tail_lookup[head_id][relation_id] + \
                                     self.knowledge_graph.valid_head2tail_lookup[head_id][relation_id]
            tail_list = list(set(tail_list) - set(filtered_tail_entities))

        head_list = list(set(head_list) - set([head_id])) + [head_id]
        tail_list = list(set(tail_list) - set([tail_id])) + [tail_id]
        rank_head_triples = torch.tensor(
            [list(i) for i in zip(head_list, [relation_id] * len(head_list), [tail_id] * len(head_list))])

        rank_tail_triples = torch.tensor(
            [list(i) for i in zip([head_id] * len(tail_list), [relation_id] * len(tail_list), tail_list)])

        rank_head = self.rank(len(head_list) - 1, rank_head_triples)
        if rank_head < 11 and filtered:
            self.validation_hits10_filtered += 1
        elif rank_head < 11 and not filtered:
            self.validation_hits10 += 1

        rank_tail = self.rank(len(tail_list) - 1, rank_tail_triples)
        if rank_tail < 11 and filtered:
            self.validation_hits10_filtered += 1
        elif rank_tail < 11 and not filtered:
            self.validation_hits10 += 1

        return (rank_head + rank_tail) / 2

    def get_test_mean_rank(self, filtered=False):
        mean_rank = 0
        process = 0

        if filtered:
            self.test_hits10_filtered = 0
        else:
            self.test_hits10 = 0

        for triple in self.knowledge_graph.test_triples:
            mean_rank += self.get_test_triple_mean_rank(triple[0], triple[1], triple[2], filtered)
            process += 1
            print("processed " + str(process) + " of " + str(len(self.knowledge_graph.test_triples)))

        return mean_rank / self.knowledge_graph.test_triples

    def get_test_triple_mean_rank(self, head_id, relation_id, tail_id, filtered):
        head_list = list(self.knowledge_graph.id2entity_dict.keys())
        tail_list = list(self.knowledge_graph.id2entity_dict.keys())
        if filtered:
            filtered_head_entities = self.knowledge_graph.train_tail2head_lookup[tail_id][relation_id] + \
                                     self.knowledge_graph.valid_tail2head_lookup[tail_id][relation_id] + \
                                     self.knowledge_graph.test_tail2head_lookup[tail_id][relation_id]
            head_list = list(set(head_list) - set(filtered_head_entities))

            filtered_tail_entities = self.knowledge_graph.train_head2tail_lookup[head_id][relation_id] + \
                                     self.knowledge_graph.valid_head2tail_lookup[head_id][relation_id] + \
                                     self.knowledge_graph.test_head2tail_lookup[head_id][relation_id]
            tail_list = list(set(tail_list) - set(filtered_tail_entities))

        head_list = list(set(head_list) - set([head_id])) + [head_id]
        tail_list = list(set(tail_list) - set([tail_id])) + [tail_id]
        rank_head_triples = torch.tensor(
            [list(i) for i in zip(head_list, [relation_id] * len(head_list), [tail_id] * len(head_list))])

        rank_tail_triples = torch.tensor(
            [list(i) for i in zip([head_id] * len(tail_list), [relation_id] * len(tail_list), tail_list)])

        rank_head = self.rank(len(head_list) - 1, rank_head_triples)
        if rank_head < 11 and filtered:
            self.test_hits10_filtered += 1
        elif rank_head < 11 and not filtered:
            self.test_hits10 += 1

        rank_tail = self.rank(len(tail_list) - 1, rank_tail_triples)
        if rank_tail < 11 and filtered:
            self.test_hits10_filtered += 1
        elif rank_tail < 11 and not filtered:
            self.test_hits10 += 1

        return (rank_head + rank_tail) / 2

    def rank(self, index, rank_triples):
        head_ids = rank_triples[:, 0]
        relation_ids = rank_triples[:, 1]
        tail_ids = rank_triples[:, 2]

        with torch.no_grad():
            head_embeddings = self.transe.entity_embeddings(head_ids)
            relation_embeddings = self.transe.relation_embeddings(relation_ids)
            tail_embeddings = self.transe.entity_embeddings(tail_ids)

            triples_loss = (head_embeddings + relation_embeddings - tail_embeddings).norm(p=self.norm, dim=1)
            rank = (triples_loss.sort()[1] == index).nonzero().item()

        return rank
