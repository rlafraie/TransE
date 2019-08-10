from pathlib import Path
from typing import Dict, List
import torch


class Dataset:
    entity2id_dict: Dict[str, int] = {}
    id2entity_dict: Dict[int, str] = {}
    next_entity_id = 0

    relation2id_dict: Dict[str, int] = {}
    id2relation_dict: Dict[int, str] = {}
    next_relation_id = 0



    training_triples: List[List]
    test_triples: List[List]
    validation_triples: List[List]

    def get_entity_id(self, entity):
        if entity not in self.entity2id_dict:
            self.entity2id_dict[entity] = self.next_entity_id
            self.id2entity_dict[self.next_entity_id] = entity
            entity_id = self.next_entity_id
            self.next_entity_id += 1
        else:
            entity_id = self.entity2id_dict[entity]
        return entity_id

    def get_relation_id(self, relation):
        if relation not in self.relation2id_dict:
            self.relation2id_dict[relation] = self.next_relation_id
            self.id2relation_dict[self.next_relation_id] = relation
            relation_id = self.next_relation_id
            self.next_relation_id += 1
        else:
            relation_id = self.relation2id_dict[relation]
        return relation_id



class Fb23715k(Dataset):
    def __init__(self, data_dir):
        self.training_triples = self.load_triples(Path(data_dir) / 'train.txt')
        self.test_triples = self.load_triples(Path(data_dir) / 'test.txt')
        self.validation_triples = self.load_triples(Path(data_dir) / 'valid.txt')

        self.num_of_entities = len(self.entity2id_dict)
        self.num_of_relations = len(self.relation2id_dict)

        self.train_head2tail_lookup, self.train_tail2head_lookup = self.load_lookup_dictionaries(self.training_triples)
        self.valid_head2tail_lookup, self.valid_tail2head_lookup = self.load_lookup_dictionaries(self.validation_triples)

    def load_triples(self, file):
        triple_list = []

        with file.open() as data:
            for fact in data:
                head_entity, relation, tail_entity = fact.split()

                head_entity_id = self.get_entity_id(head_entity)
                tail_entity_id = self.get_entity_id(tail_entity)
                relation_id = self.get_relation_id(relation)

                triple_list.append([head_entity_id, relation_id, tail_entity_id])

            return triple_list

    def load_lookup_dictionaries(self, triples: List[List]):
        head2tail_lookup = {}
        tail2head_lookup = {}

        for head_id, relation_id, tail_id in triples:

            if head_id not in head2tail_lookup:
                head2tail_lookup[head_id] = {relation_id: [tail_id]}
            else:
                if relation_id not in head2tail_lookup[head_id]:
                    head2tail_lookup[head_id][relation_id] = [tail_id]
                else:
                    if tail_id not in head2tail_lookup[head_id][relation_id]:
                        head2tail_lookup[head_id][relation_id].append(tail_id)

            if tail_id not in tail2head_lookup:
                tail2head_lookup[tail_id] = {relation_id: [head_id]}
            else:
                if relation_id not in tail2head_lookup[tail_id]:
                    tail2head_lookup[tail_id][relation_id] = [head_id]
                else:
                    if head_id not in tail2head_lookup[tail_id][relation_id]:
                        tail2head_lookup[tail_id][relation_id].append(head_id)

        return head2tail_lookup, tail2head_lookup

    def gather_ranking_tail_triples(self, head_id: int, relation_id: int, tail_id: int):
        entity_list = list(self.id2entity_dict.keys())
        filtered_entities = self.train_head2tail_lookup[head_id][relation_id] + self.valid_head2tail_lookup[head_id][relation_id]
        tail_list = list(set(entity_list) - set(filtered_entities)) + [tail_id]

        head_list = [head_id] * len(tail_list)
        relation_list = [relation_id] * len(tail_list)

        return torch.tensor([list(i) for i in zip(head_list, relation_list, tail_list)])

    def gather_ranking_head_triples(self, head_id: int, relation_id: int, tail_id: int):
        entity_list = list(self.id2entity_dict.keys())
        filtered_entities = self.train_tail2head_lookup[tail_id][relation_id] + self.valid_tail2head_lookup[tail_id][relation_id]
        head_list = list(set(entity_list) - set(filtered_entities)) + [head_id]

        tail_list = [tail_id] * len(head_list)
        relation_list = [relation_id] * len(head_list)

        return torch.tensor([list(i) for i in zip(head_list, relation_list, tail_list)])

    def get_corrupted_training_triples(self, triples: torch.tensor):
        return torch.tensor(list(map(lambda x: self.corrupt_training_triple(x[0].item(), x[1].item(), x[2].item()), triples)))

    def corrupt_training_triple(self, head_id: int, relation_id: int, tail_id: int):
        if torch.rand(1).uniform_(0, 1).item() >= 0.5:
            initial_head_id = head_id
            while head_id in self.train_tail2head_lookup[tail_id][relation_id] or head_id == initial_head_id:
                head_id = torch.randint(1, self.num_of_entities + 1, (1,)).item()

        else:
            initial_tail_id = tail_id
            while tail_id in self.train_head2tail_lookup[head_id][relation_id] or tail_id == initial_tail_id:
                tail_id = torch.randint(1, self.num_of_entities + 1, (1,)).item()

        return [head_id, relation_id, tail_id]


class Fb15k(Dataset):
    def __init__(self, data_dir):
        self.training_triples = self.load_triples(Path(data_dir) / 'train.txt')
        self.test_triples = self.load_triples(Path(data_dir) / 'test.txt')
        self.validation_triples = self.load_triples(Path(data_dir) / 'valid.txt')

    def load_triples(self, file):
        triple_list = []

        with file.open() as data:
            for fact in data:
                head_entity, relation, tail_entity = fact.split()

                head_entity_id = self.get_entity_id(head_entity)
                tail_entity_id = self.get_entity_id(tail_entity)
                relation_id = self.get_relation_id(relation)

                triple_list.append([head_entity_id, relation_id, tail_entity_id])

            return triple_list


class Wn18(Dataset):
    def __init__(self, data_dir):
        self.training_triples = self.load_triples(Path(data_dir) / 'train.txt')
        self.test_triples = self.load_triples(Path(data_dir) / 'test.txt')
        self.validation_triples = self.load_triples(Path(data_dir) / 'valid.txt')

    def load_triples(self, file):
        triple_list = []

        with file.open() as f:
            for fact in f:
                head_entity, relation, tail_entity = fact.split()

                head_entity_id = self.get_entity_id(head_entity)
                tail_entity_id = self.get_entity_id(tail_entity)
                relation_id = self.get_relation_id(relation)

                triple_list.append([head_entity_id, relation_id, tail_entity_id])

        return triple_list


