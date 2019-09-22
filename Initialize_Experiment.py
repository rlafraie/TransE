from triples import KnowledgeGraph
from Train import Experiment
import copy

# initialize Freebase Dataset
#fb = KnowledgeGraph.factory('Fb15k')
#fb = KnowledgeGraph.factory('Fb15k-237')
#fb = KnowledgeGraph.factory('Wn18')
#fb = KnowledgeGraph.factory('Wn18RR')
#fb = KnowledgeGraph.factory('YAGO3-10')


num_of_epochs = 100
validation_freq = 10
batch_size = 100
margin = 2 #
norm = 1
learning_rate = 0.01
num_of_dimensions = 25

transE_experiment = Experiment(knowledge_graph=fb, num_of_epochs=num_of_epochs, batch_size=batch_size, margin=margin,
                               norm=norm, learning_rate=learning_rate, num_of_dimensions=num_of_dimensions, validation_freq=validation_freq)
transE_experiment.train()



# transE_experiment.num_of_epochs = 1
# transE_experiment.validation_freq = 1
# unfiltered_exp = copy.deepcopy(filtered_exp)

# Either train or load pretrained experiment objects
## TRAIN
# filtered_exp.train(filtered_corrupted_batch=True)
# unfiltered_exp.train(filtered_corrupted_batch=False)


## LOAD PRETRAINED
# filtered_exp.load_model_params('filtered_learned_params')
# unfiltered_exp.load_model_params('unfiltered_learned_params')

# Evaluation
# validation_link_prediction_dataset = transE_experiment.knowledge_graph.valid_dataset
# validation_filter_datasets = [transE_experiment.knowledge_graph.train_dataset]
# filtered_validation_mean_rank, filtered_validation_hits10 = transE_experiment.get_mean_rank(
#     validation_link_prediction_dataset,
#     [validation_filter_datasets], filtered=True, fast_testing=False)
#
# test_link_prediction_dataset = transE_experiment.knowledge_graph.test_dataset
# test_filter_datasets = [transE_experiment.knowledge_graph.train_dataset,
#                         transE_experiment.knowledge_graph.valid_dataset]
# filtered_test_mean_rank, filtered_test_hits10 = transE_experiment.get_mean_rank(test_link_prediction_dataset,
#                                                                                 test_filter_datasets, filtered=True,
#                                                                                 fast_testing=False)

# unfiltered_exp.get_test_mean_rank(filtered=False, fast_testing=False)
