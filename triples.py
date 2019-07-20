import csv
from pathlib import Path
from typing import Dict, List
import pandas as pd


class Data:

    def __init__(self, data_dir):
        self.data_dir_root = Path.cwd() / 'data'
        self.data_dir = self.data_dir_root / data_dir
        self.entity_dict = dict()
        self.relation_dict = dict()
        self.training_triples = []
        self.validation_triples = []
        self.load_knowledge_graph()

    def load_knowledge_graph(self):
        training_file_name = self.data_dir / "train.txt"
        train_df = pd.read_csv(training_file_name, delimiter='\t',
                               names=['head_entity', 'relation_type', 'tail_entity'])

        # Initialize entity mapping
        entity_list = list(train_df.head_entity) + list(train_df.tail_entity)
        entity_set = set(entity_list)
        self.entity_dict = dict(zip(entity_set, range(1, len(entity_set) + 1)))

        # Initialize relation mapping
        relation_list = list(train_df.relation_type)
        relation_set = set(relation_list)
        self.relation_dict = dict(zip(relation_set, range(1, len(relation_set) + 1)))

        # Map ids to triples in train_df and Initialize training triples
        train_df.head_entity.map(self.entity_dict)
        train_df.tail_entity.map(self.entity_dict)
        train_df.relation_type.map(self.relation_dict)
        self.training_triples = [tuple(fact) for fact in train_df.values]

        # Initialize validation triples
        validation_file_name = self.data_dir / "valid.txt"
        validation_df = pd.read_csv(validation_file_name, delimiter='\t',
                                    names=['head_entity', 'relation_type', 'tail_entity'])
        validation_df.head_entity.map(self.entity_dict)
        validation_df.tail_entity.map(self.entity_dict)
        validation_df.relation_type.map(self.relation_dict)
        self.validation_triples = [tuple(fact) for fact in validation_df.values]


class Fact:
    head_entity_id: int
    relation_id: int
    tail_entity_id: int

    def __init__(self, head_entity_id, relation_id, tail_entity_id):
        self.head_entity_id = head_entity_id
        self.relation_id = relation_id
        self.tail_entity_id = tail_entity_id


class Dataset:
    entity_id_dict: Dict[str, int] = {}
    next_entity_id = 1
    relation_id_dict: Dict[str, int] = {}
    next_relation_id = 1

    training_triples: List[Fact]
    test_triples: List[Fact]
    validation_triples: List[Fact]

    def get_entity_id(self, entity):
        if entity not in self.entity_id_dict:
            self.entity_id_dict[entity] = self.next_entity_id
            entity_id = self.next_entity_id
            self.next_entity_id = + 1
        else:
            entity_id = self.entity_id_dict[entity]
        return entity_id

    def get_relation_id(self, relation):
        if relation not in self.relation_id_dict:
            self.relation_id_dict[relation] = self.next_relation_id
            relation_id = self.next_relation_id
            self.next_relation_id = + 1
        else:
            relation_id = self.relation_id_dict[relation]
        return relation_id


class Fb23715k(Dataset):
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

                triple_list.append(Fact(head_entity_id, relation_id, tail_entity_id))

            return triple_list
