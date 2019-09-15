from pathlib import Path
from typing import Dict, List
import torch
from collections import defaultdict


class Datasubset:
    triples: List[List] = []
    head2tail_lookup: Dict[int, Dict[int, set]] = {}
    tail2head_lookup: Dict[int, Dict[int, set]] = {}


class KnowledgeGraph:
    data_dir = ''

    entity2id_dict: Dict[str, int] = {}
    id2entity_dict: Dict[int, str] = {}
    next_entity_id = 0

    relation2id_dict: Dict[str, int] = {}
    id2relation_dict: Dict[int, str] = {}
    next_relation_id = 0

    train_dataset: Datasubset = Datasubset()
    valid_dataset: Datasubset = Datasubset()
    test_dataset: Datasubset = Datasubset()

    num_of_entities: int = 0
    num_of_relations: int = 0

    def factory(knowledge_graph_name):
        if knowledge_graph_name == 'Fb15k-237':
            return Fb15k237(Path.cwd() / 'data' / 'fb15k-237')

        if knowledge_graph_name == 'Fb15k':
            return Fb15k(Path.cwd() / 'data' / 'fb15k')

        if knowledge_graph_name == 'Wn18':
            return Wn18(Path.cwd() / 'data' / 'wn18')

    def get_entity_id(self, entity: str) -> int:
        if entity not in self.entity2id_dict:
            self.entity2id_dict[entity] = self.next_entity_id
            self.id2entity_dict[self.next_entity_id] = entity
            entity_id = self.next_entity_id
            self.next_entity_id += 1
        else:
            entity_id = self.entity2id_dict[entity]
        return entity_id

    def get_relation_id(self, relation: str) -> int:
        if relation not in self.relation2id_dict:
            self.relation2id_dict[relation] = self.next_relation_id
            self.id2relation_dict[self.next_relation_id] = relation
            relation_id = self.next_relation_id
            self.next_relation_id += 1
        else:
            relation_id = self.relation2id_dict[relation]
        return relation_id

    def load_lookup_dictionaries(self, triples: List[List]) -> (Dict[int, Dict[int, set]], Dict[int, Dict[int, set]]):
        head2tail_lookup = defaultdict(lambda: defaultdict(set))
        tail2head_lookup = defaultdict(lambda: defaultdict(set))

        for head_id, relation_id, tail_id in triples:
            head2tail_lookup[head_id][relation_id].add(tail_id)
            tail2head_lookup[tail_id][relation_id].add(head_id)

        return head2tail_lookup, tail2head_lookup

    def get_corrupted_training_triples(self, triples: torch.tensor) -> torch.tensor:
        return torch.tensor(
            list(map(lambda x: self.corrupt_training_triple(x[0].item(), x[1].item(), x[2].item()), triples)))

    def corrupt_training_triple(self, head_id: int, relation_id: int, tail_id: int) -> List[int]:
        if torch.rand(1).uniform_(0, 1).item() >= 0.5:
            initial_head_id = head_id
            while head_id in self.train_dataset.tail2head_lookup[tail_id][relation_id] or head_id == initial_head_id:
                head_id = torch.randint(self.num_of_entities, (1,)).item()

        else:
            initial_tail_id = tail_id
            while tail_id in self.train_dataset.head2tail_lookup[head_id][relation_id] or tail_id == initial_tail_id:
                tail_id = torch.randint(self.num_of_entities, (1,)).item()

        return [head_id, relation_id, tail_id]

    def get_corrupted_batch_unfiltered(self, batch: torch.tensor):
        corrupted_batch = batch.clone()
        head_tail_indexes = torch.randint(2, (batch.shape[0],)) * 2
        corrupted_batch[torch.arange(corrupted_batch.shape[0]), head_tail_indexes] = torch.randint(self.num_of_entities,
                                                                                                   (
                                                                                                   corrupted_batch.shape[
                                                                                                       0],))

        # torch.randint statement randomly generates 0 or 1. The outcome is multiplied with 2 to get 0 or 2
        # which is the column index for either the head_id or tail_id. Hence, we create a tensor defining
        # whether we corrupt the head or tail for the corrupted batch and transmit it as an input

        ## print batch and corr batch in forward() to dismiss the possibility that they are refering to the same tensor instance
        return corrupted_batch


class Fb15k237(KnowledgeGraph):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.train_dataset.triples = self.load_triples(Path(data_dir) / 'train.txt')
        self.valid_dataset.triples = self.load_triples(Path(data_dir) / 'valid.txt')
        self.test_dataset.triples = self.load_triples(Path(data_dir) / 'test.txt')

        self.train_dataset.head2tail_lookup, self.train_dataset.tail2head_lookup = self.load_lookup_dictionaries(
            self.train_dataset.triples)
        self.valid_dataset.head2tail_lookup, self.valid_dataset.tail2head_lookup = self.load_lookup_dictionaries(
            self.valid_dataset.triples)
        self.test_dataset.head2tail_lookup, self.test_dataset.tail2head_lookup = self.load_lookup_dictionaries(
            self.test_dataset.triples)

        self.num_of_entities = len(self.entity2id_dict)
        self.num_of_relations = len(self.relation2id_dict)

    def load_triples(self, file: Path) -> List[List[int]]:
        triple_list = []

        with file.open() as data:
            for fact in data:
                head_entity, relation, tail_entity = fact.split()

                head_entity_id = self.get_entity_id(head_entity)
                tail_entity_id = self.get_entity_id(tail_entity)
                relation_id = self.get_relation_id(relation)

                triple_list.append([head_entity_id, relation_id, tail_entity_id])

        return triple_list


class Fb15k(KnowledgeGraph):
    def __init__(self, data_dir):
        self.training_triples = self.load_triples(Path(data_dir) / 'train.txt')
        self.test_triples = self.load_triples(Path(data_dir) / 'test.txt')
        self.validation_triples = self.load_triples(Path(data_dir) / 'valid.txt')

        self.num_of_entities = len(self.entity2id_dict)
        self.num_of_relations = len(self.relation2id_dict)

        self.train_dataset.head2tail_lookup, self.train_dataset.tail2head_lookup = self.load_lookup_dictionaries(
            self.training_triples)
        self.valid_dataset.head2tail_lookup, self.valid_dataset.tail2head_lookup = self.load_lookup_dictionaries(
            self.validation_triples)
        self.test_dataset.head2tail_lookup, self.test_dataset.tail2head_lookup = self.load_lookup_dictionaries(
            self.test_triples)

    def load_triples(self, file: Path) -> List[List[int]]:
        triple_list = []

        with file.open() as data:
            for fact in data:
                head_entity, relation, tail_entity = fact.split()

                head_entity_id = self.get_entity_id(head_entity)
                tail_entity_id = self.get_entity_id(tail_entity)
                relation_id = self.get_relation_id(relation)

                triple_list.append([head_entity_id, relation_id, tail_entity_id])

            return triple_list


class Wn18(KnowledgeGraph):
    def __init__(self, data_dir):
        self.training_triples = self.load_triples(Path(data_dir) / 'train.txt')
        self.test_triples = self.load_triples(Path(data_dir) / 'test.txt')
        self.validation_triples = self.load_triples(Path(data_dir) / 'valid.txt')

        self.num_of_entities = len(self.entity2id_dict)
        self.num_of_relations = len(self.relation2id_dict)

        self.train_dataset.head2tail_lookup, self.train_dataset.tail2head_lookup = self.load_lookup_dictionaries(
            self.training_triples)
        self.valid_dataset.head2tail_lookup, self.valid_dataset.tail2head_lookup = self.load_lookup_dictionaries(
            self.validation_triples)
        self.test_dataset.head2tail_lookup, self.test_dataset.tail2head_lookup = self.load_lookup_dictionaries(
            self.test_triples)

    def load_triples(self, file: Path) -> List[List[int]]:
        triple_list = []

        with file.open() as f:
            for fact in f:
                head_entity, relation, tail_entity = fact.split()

                head_entity_id = self.get_entity_id(head_entity)
                tail_entity_id = self.get_entity_id(tail_entity)
                relation_id = self.get_relation_id(relation)

                triple_list.append([head_entity_id, relation_id, tail_entity_id])

        return triple_list
