
import csv
from pathlib import Path
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
        train_df = pd.read_csv(training_file_name, delimiter='\t', names=['head_entity', 'relation_type', 'tail_entity'])

        # Initialize entity mapping
        entity_list = list(train_df.head_entity) + list(train_df.tail_entity)
        entity_set = set(entity_list)
        self.entity_dict = dict(zip(entity_set, range(1, len(entity_set) + 1)))

        # Initialize relation mapping
        relation_list = list(train_df.relation_type)
        relation_set = set(relation_list)
        self.relation_dict = dict(zip(relation_set, range(1, len(relation_set)+1)))

        # Map ids to triples in train_df and Initialize training triples
        train_df.head_entity.map(self.entity_dict)
        train_df.tail_entity.map(self.entity_dict)
        train_df.relation_type.map(self.relation_dict)
        self.training_triples = [tuple(fact) for fact in train_df.values]

        # Initialize validation triples
        validation_file_name = self.data_dir / "validation.txt"
        validation_df = pd.read_csv(validation_file_name, delimiter='\t', names=['head_entity', 'relation_type', 'tail_entity'])
        validation_df.head_entity.map(self.entity_dict)
        validation_df.tail_entity.map(self.entity_dict)
        validation_df.relation_type.map(self.relation_dict)
        self.validation_triples = [tuple(fact) for fact in validation_df.values]
