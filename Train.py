import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import functional as F
from torch import optim
from TransE import TransE
from triples import KnowledgeGraph, Datasubset
from typing import List, Dict
from random import shuffle
from tqdm import tqdm
from OutputMethods import initialize_log_folder, update_hyper_param_sheet, save_figure


class Experiment:
    def __init__(self, knowledge_graph: KnowledgeGraph, num_of_epochs: int = 50, batch_size: int = 100, margin: int = 1,
                 norm: int = 1, learning_rate: float = 0.01, num_of_dimensions: int = 50, validation_freq: int = 10):
        self.knowledge_graph = knowledge_graph

        self.num_of_dimensions = num_of_dimensions
        self.num_of_epochs = num_of_epochs
        self.validation_freq = validation_freq
        self.batch_size = batch_size
        self.margin = margin
        self.norm = norm
        self.learning_rate = learning_rate
        self.early_stop_threshold = 10

        self.num_of_dimensions = num_of_dimensions
        self.num_of_entities = knowledge_graph.num_of_entities
        self.num_of_relations = knowledge_graph.num_of_relations

        self.transe: TransE = TransE(knowledge_graph.num_of_entities, knowledge_graph.num_of_relations,
                                     num_of_dimensions, norm)

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.transe.to(self.device)

        self.best_mean_rank_score = None
        self.best_mean_rank_epoch = None
        self.best_mean_rank_entity_embeddings = None
        self.best_mean_rank_relation_embeddings = None

    def train(self, filtered_corrupted_batch=False):
        hyper_param_path = initialize_log_folder(self.knowledge_graph.data_dir)
        output_log_file = open(hyper_param_path / 'log.txt', 'w')

        print('Running expirement with hyperparameter id [{}]:'.format(hyper_param_path.name), file=output_log_file)
        print('  Batch size:', self.batch_size, file=output_log_file)
        print('  Number of Epochs:', self.num_of_epochs, file=output_log_file)
        print('  Margin:', self.margin, file=output_log_file)
        print('  Norm: L{}'.format(self.norm), file=output_log_file)
        print('  Learning Rate:', self.learning_rate, file=output_log_file)
        print('  Number of Dimensions:', self.num_of_dimensions, file=output_log_file)
        print(' ', file=output_log_file)
        print(' ', file=output_log_file)

        training_losses = []

        output_losses = []
        training_mean_ranks = []
        training_hits = []
        validation_mean_ranks = []
        validation_hits = []

        dataset = TensorDataset(torch.tensor(self.knowledge_graph.train_dataset.triples))
        train_dl = DataLoader(dataset, batch_size=self.batch_size)
        optimizer = optim.SGD(self.transe.parameters(), lr=self.learning_rate)

        self.best_mean_rank_epoch = 1
        self.best_mean_rank_score = self.get_evaluation_scores(self.knowledge_graph.valid_dataset)[0]
        self.best_mean_rank_entity_embeddings = self.transe.entity_embeddings.weight.data.clone()
        self.best_mean_rank_relation_embeddings = self.transe.relation_embeddings.weight.data.clone()

        for epoch in range(self.num_of_epochs):
            epoch_loss = 0
            for batch in tqdm(train_dl):
                mini_batch = batch[0].to(self.device)
                corr_mini_batch = \
                    self.knowledge_graph.get_corrupted_training_triples(mini_batch).to(self.device) \
                        if filtered_corrupted_batch \
                        else self.knowledge_graph.get_corrupted_batch_unfiltered(mini_batch, self.device)

                batch_loss, corr_batch_loss = self.transe(mini_batch, corr_mini_batch)
                batch_loss.to(self.device)
                corr_batch_loss.to(self.device)

                loss = F.relu(self.margin + batch_loss.norm(p=self.norm, dim=1)
                              - corr_batch_loss.norm(p=self.norm, dim=1)).sum()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                epoch_loss += loss

            with torch.no_grad():
                epoch_loss = (epoch_loss / train_dl.__len__()).item()
                training_losses.append(round(epoch_loss, 4))

            if ((epoch + 1) % self.validation_freq == 0) or ((epoch + 1) == self.num_of_epochs):
                output_losses.append(round(epoch_loss, 4))

                training_mean_rank, training_hits10 = self.get_evaluation_scores(self.knowledge_graph.train_dataset)
                training_mean_ranks.append(training_mean_rank)
                training_hits.append(training_hits10)

                print('Validation for epoch', epoch + 1, file=output_log_file)
                print('     Epoch loss: :      ', round(epoch_loss, 4), file=output_log_file)
                print('     Train Dataset:      hits@10(raw)= {} mean_rank(raw)= {}'
                      .format(training_hits10, training_mean_rank), file=output_log_file)

                validation_mean_rank, validation_hits10 = self.get_evaluation_scores(self.knowledge_graph.valid_dataset)
                validation_mean_ranks.append(validation_mean_rank)
                validation_hits.append(validation_hits10)

                print('     Validation Dataset: hits@10= {} mean_rank= {}'
                      .format(validation_hits10, validation_mean_rank), file=output_log_file)
                print('--------------------------------', file=output_log_file)

                if validation_mean_rank < self.best_mean_rank_score:
                    self.best_mean_rank_epoch = epoch + 1
                    self.best_mean_rank_score = validation_mean_rank
                    self.best_mean_rank_entity_embeddings = self.transe.entity_embeddings.weight.data.clone()
                    self.best_mean_rank_relation_embeddings = self.transe.relation_embeddings.weight.data.clone()
                    self.early_stop_threshold = 10
                    
                else:
                    self.early_stop_threshold -= 1

            if self.early_stop_threshold == 0:
                self.num_of_epochs = self.best_mean_rank_epoch
                self.transe.entity_embeddings.weight.data = self.best_mean_rank_entity_embeddings
                self.transe.relation_embeddings.weight.data = self.best_mean_rank_relation_embeddings

                print('________________________', file=output_log_file)
                print('EARLY STOP at epoch: {}'.format(epoch+1), file=output_log_file)
                print('Best Mean Rank Score: {}'.format(self.best_mean_rank_score), file=output_log_file)
                print('---- @ epoch: {}'.format(self.best_mean_rank_epoch), file=output_log_file)
                print('________________________', file=output_log_file)

                break

        save_figure(hyper_param_path, 'meanRank_raw.png', 'MeanRank (raw) on Training vs. Validation Dataset',
                    'Training Epochs', 'Mean Rank', training_mean_ranks, validation_mean_ranks, output_losses,
                    self.num_of_epochs, self.validation_freq)

        save_figure(hyper_param_path, 'hits10_raw.png', 'Hits@10 (raw) on Training vs. Validation Dataset',
                    'Training Epochs', 'Hits@10', training_hits, validation_hits, output_losses, self.num_of_epochs,
                    self.validation_freq)

        save_figure(hyper_param_path, 'training_loss.png', 'Loss curve during Training',
                    'Training Epochs', 'Training Loss', training_losses, [], [], self.num_of_epochs,
                    self.validation_freq)

        raw_validation_mean_rank, raw_validation_hits = self.get_evaluation_scores(self.knowledge_graph.valid_dataset,
                                                                                   filtered=False, fast_testing=False)
        filtered_validation_mean_rank, filtered_validation_hits = self.get_evaluation_scores(
            self.knowledge_graph.valid_dataset, [self.knowledge_graph.train_dataset],
            filtered=True, fast_testing=False)

        raw_test_mean_rank, raw_test_hits = self.get_evaluation_scores(self.knowledge_graph.test_dataset,
                                                                       filtered=False, fast_testing=False)
        filtered_test_mean_rank, filtered_test_hits = self.get_evaluation_scores(
            self.knowledge_graph.test_dataset, [self.knowledge_graph.train_dataset, self.knowledge_graph.valid_dataset],
            filtered=True, fast_testing=False)

        print('-----------', file=output_log_file)
        print('Test Scores', file=output_log_file)
        print('-----------', file=output_log_file)
        print(' Validation Dataset:', file=output_log_file)
        print('     hits@10(raw)={} mean_rank(raw)={}'.format(raw_validation_hits, raw_validation_mean_rank),
              file=output_log_file)
        print('     hits@10(filtered)={} mean_rank(filtered)={}'.format(filtered_validation_hits,
                                                                        filtered_validation_mean_rank),
              file=output_log_file)
        print('-------------------------------------------------------', file=output_log_file)
        print(' Test Dataset:', file=output_log_file)
        print('     hits@10(raw)={} mean_rank(raw)={}'.format(raw_test_hits, raw_test_mean_rank), file=output_log_file)
        print('     hits@10(filtered)={} mean_rank(filtered)={}'.format(filtered_test_hits,
                                                                        filtered_test_mean_rank), file=output_log_file)
        output_log_file.close()
        hyper_param_config = [hyper_param_path.name, self.num_of_epochs, self.batch_size, self.margin, self.norm,
                              self.learning_rate, self.num_of_dimensions, self.num_of_epochs]
        update_hyper_param_sheet(hyper_param_path.parent, 'hyper_param_mapping.xlsx', hyper_param_config)

        evaluation_scores = [hyper_param_path.name, raw_validation_mean_rank, filtered_validation_mean_rank,
                             raw_validation_hits, filtered_validation_hits, raw_test_mean_rank,
                             filtered_test_mean_rank, raw_test_hits, filtered_test_hits]
        update_hyper_param_sheet(hyper_param_path.parent, 'hyper_param_scores.xlsx', evaluation_scores)

        self.save_model_params(hyper_param_path)

    @torch.no_grad()
    def get_evaluation_scores(self, link_prediction_dataset: Datasubset, filter_datasets: List[Datasubset] = [],
                              filtered=False, fast_testing=True) -> (float, int):
        mean_rank = 0
        hits10 = 0
        link_prediction_triples = link_prediction_dataset.triples
        datasets = filter_datasets + [link_prediction_dataset]

        if fast_testing:
            threshold = 1000
            shuffle(link_prediction_triples)
        else:
            threshold = len(link_prediction_triples) - 1

        for triple in tqdm(link_prediction_triples[:threshold]):
            head_id, relation_id, tail_id = triple[0], triple[1], triple[2]
            triple_mean_rank = self.get_filtered_triple_mean_rank(head_id, relation_id, tail_id, datasets) \
                if filtered else self.get_raw_triple_mean_rank(head_id, relation_id, tail_id)

            if triple_mean_rank <= 10:
                hits10 += 1

            mean_rank += triple_mean_rank

        return round(mean_rank / (threshold + 1), 4), hits10

    def get_raw_triple_mean_rank(self, head_id: int, relation_id: int, tail_id: int) -> float:
        rank_head, rank_tail = self.get_raw_triple_ranks(head_id, relation_id, tail_id)

        return (rank_head + rank_tail) / 2

    def get_raw_triple_ranks(self, head_id: int, relation_id: int, tail_id: int) -> (int, int):
        head_embeddings = self.transe.entity_embeddings(torch.tensor(head_id).to(self.device)).repeat(
            self.knowledge_graph.num_of_entities, 1)
        relation_embeddings = self.transe.relation_embeddings(torch.tensor(relation_id).to(self.device)).repeat(
            self.knowledge_graph.num_of_entities, 1)
        tail_embeddings = self.transe.entity_embeddings(torch.tensor(tail_id).to(self.device)).repeat(
            self.knowledge_graph.num_of_entities, 1)

        head_loss = (self.transe.entity_embeddings.weight.data + relation_embeddings - tail_embeddings).norm(
            p=self.norm, dim=1)
        tail_loss = (head_embeddings + relation_embeddings - self.transe.entity_embeddings.weight.data).norm(
            p=self.norm, dim=1)

        rank_head = (head_loss.sort()[1] == head_id).nonzero().item() + 1
        rank_tail = (tail_loss.sort()[1] == tail_id).nonzero().item() + 1

        return rank_head, rank_tail

    def get_filtered_triple_mean_rank(self, head_id: int, relation_id: int, tail_id: int,
                                      datasets: List[Datasubset]) -> float:
        rank_head, rank_tail = self.get_filtered_ranks(head_id, relation_id, tail_id, datasets)

        return (rank_head + rank_tail) / 2

    def get_filtered_ranks(self, head_id: int, relation_id: int, tail_id: int, datasets: List[Datasubset]) -> (int, int):
        head_list = [entity for entity in range(self.knowledge_graph.num_of_entities)]
        tail2head_lookups = [dataset.tail2head_lookup for dataset in datasets]
        head_filter = self.get_filter(tail_id, relation_id, tail2head_lookups)

        head_list = list(set(head_list) - set(head_filter))
        head_list = list(set(head_list) - {head_id}) + [head_id]
        head_list_embeddings = self.transe.entity_embeddings(torch.tensor(head_list).to(self.device))

        head_loss = (head_list_embeddings
                     + self.transe.relation_embeddings(torch.tensor(relation_id).to(self.device)).repeat(len(head_list),
                                                                                                         1)
                     - self.transe.entity_embeddings(torch.tensor(tail_id).to(self.device)).repeat(len(head_list),
                                                                                                   1)).norm(p=self.norm,
                                                                                                            dim=1)
        tail_list = [entity for entity in range(self.knowledge_graph.num_of_entities)]
        head2tail_lookups = [dataset.head2tail_lookup for dataset in datasets]
        tail_filter = self.get_filter(head_id, relation_id, head2tail_lookups)

        tail_list = list(set(tail_list) - set(tail_filter))
        tail_list = list(set(tail_list) - {tail_id}) + [tail_id]
        tail_list_embeddings = self.transe.entity_embeddings(torch.tensor(tail_list).to(self.device))

        tail_loss = (self.transe.entity_embeddings(torch.tensor(head_id).to(self.device)).repeat(len(tail_list), 1)
                     + self.transe.relation_embeddings(torch.tensor(relation_id)
                                                       .to(self.device)).repeat(len(tail_list), 1)
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

    def save_model_params(self, hyper_param_path):
        file_path = hyper_param_path / 'trained_parameters.pickle'
        torch.save(self.transe.state_dict(), file_path)

    def load_model_params(self, hyper_param_id):
        file_path = self.knowledge_graph.data_dir / 'evaluation_earlyStop' / hyper_param_id / str(
            'trained_parameters.pickle')
        self.transe.load_state_dict(torch.load(file_path))
