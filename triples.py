
import os
import csv
from pathlib import Path

class Data:

    def __init__(self, data_dir):
        self.data_dir_root = Path.cwd() / 'data'
        self.data_dir = self.data_dir_root / data_dir
        self.entity_dict = dict()
        self.relation_dict = dict()
        self.training_triples = []
        self.validation_triples = []

        self.load_entity_dict()
        self.load_relation_dict()

    def load_entity_dict(self):
        entity_file_name = "entity2id.txt"
        # adjust accordingly: "/Users/rlafraie/PycharmProjects/TransE/data/entity2id.txt"
        entity_file = open(self.data_dir / entity_file_name)
        entity_reader = csv.reader(entity_file, delimiter="\t")
        self.entity_dict = {entity[0]: entity[1] for entity in entity_reader}

    def load_relation_dict(self):
        relation_file_name = "relation2id.txt"
        # adjust accordingly:"/Users/rlafraie/PycharmProjects/TransE/data/relation2id.txt"
        relation_file = open(self.data_dir / relation_file_name)
        relation_reader = csv.reader(relation_file, delimiter="\t")
        self.relation_dict = {relation[0]: relation[1] for relation in relation_reader}

    def load_triples(self):
        training_file_name = "train.txt"
        validation_file_name = "validation.txt"

        with open(self.data_dir / training_file_name) as train_f:
            for r in csv.reader(train_f, delimiter='\t'):
                self.training_triples.append((self.entity_dict[r[0]], self.entity_dict[r[1]],self.relation_dict[r[2]]))

        with open(self.data_dir / validation_file_name) as validation_f:
            for r in csv.reader(validation_f, delimiter='\t'):
                self.validation_triples.append((self.entity_dict[r[0]], self.entity_dict[r[1]],self.relation_dict[r[2]]))


