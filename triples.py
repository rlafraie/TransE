from pathlib import Path
from typing import Dict, List
import torch


class Dataset:
    entity2id_dict: Dict[str, int] = {}
    id2entity_dict: Dict[int, str] = {}
    next_entity_id = 1
    
    relation2id_dict: Dict[str, int] = {}
    id2relation_dict: Dict[int, str] = {}
    next_relation_id = 1

    head2tail_lookup = {}
    tail2head_lookup = {}

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

    def set_head2tail_entry(self, head_id, relation_id, tail_id):
        if head_id not in self.head2tail_lookup:
            self.head2tail_lookup[head_id] = {relation_id: [tail_id]}
        else:
            if relation_id not in self.head2tail_lookup[head_id]:
                self.head2tail_lookup[head_id][relation_id] = [tail_id]
            else:
                if tail_id not in self.head2tail_lookup[head_id][relation_id]:
                    self.head2tail_lookup[head_id][relation_id].append(tail_id)
                else:
                    pass

    def set_tail2head_entry(self, head_id, relation_id, tail_id):
        if tail_id not in self.tail2head_lookup:
            self.tail2head_lookup[tail_id] = {relation_id: [head_id]}
        else:
            if relation_id not in self.tail2head_lookup[tail_id]:
                self.tail2head_lookup[tail_id][relation_id] = [head_id]
            else:
                if head_id not in self.tail2head_lookup[tail_id][relation_id]:
                    self.tail2head_lookup[tail_id][relation_id].append(head_id)
                else:
                    pass


class Fb23715k(Dataset):
    def __init__(self, data_dir):
        self.training_triples = self.load_triples(Path(data_dir) / 'train.txt')
        self.test_triples = self.load_triples(Path(data_dir) / 'test.txt')
        self.validation_triples = self.load_triples(Path(data_dir) / 'valid.txt')

        self.num_of_entities = len(self.entity2id_dict)
        self.num_of_relations = len(self.relation2id_dict)

    def load_triples(self, file):
        triple_list = []

        with file.open() as data:
            for fact in data:
                head_entity, relation, tail_entity = fact.split()

                head_entity_id = self.get_entity_id(head_entity)
                tail_entity_id = self.get_entity_id(tail_entity)
                relation_id = self.get_relation_id(relation)

                self.set_head2tail_entry(head_entity_id, relation_id, tail_entity_id)
                self.set_tail2head_entry(head_entity_id, relation_id, tail_entity_id)

                triple_list.append([head_entity_id, relation_id, tail_entity_id])

            return triple_list


    def get_corrupted_triples(self, triples: torch.tensor):
        return torch.tensor(list(map(lambda x: self.corrupt_triple(x[0].item(), x[1].item(), x[2].item()), triples)))

    def corrupt_triple(self, head_id: int, relation_id: int, tail_id: int):
        if torch.rand(1).uniform_(0, 1).item() >= 0.5:
            initial_head_id = head_id
            while head_id in self.tail2head_lookup[tail_id][relation_id] or head_id == initial_head_id:
                head_id = torch.randint(1, self.num_of_entities + 1, (1,)).item()

        else:
            initial_tail_id = tail_id
            while tail_id in self.head2tail_lookup[head_id][relation_id] or tail_id == initial_tail_id:
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

                self.set_head2tail_entry(head_entity_id, relation_id, tail_entity_id)
                self.set_tail2head_entry(head_entity_id, relation_id, tail_entity_id)

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

                self.set_head2tail_entry(head_entity_id, relation_id, tail_entity_id)
                self.set_tail2head_entry(head_entity_id, relation_id, tail_entity_id)

                triple_list.append([head_entity_id, relation_id, tail_entity_id])

        return triple_list


