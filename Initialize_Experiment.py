from pathlib import Path
from triples import Fb23715k
from Train import Experiment
import copy

# initialize Freebase Dataset
fb_path = Path.cwd() / 'data' / 'fb15k-237'
fb = Fb23715k(fb_path)

filtered_exp = Experiment(fb)
# unfiltered_exp = copy.deepcopy(filtered_exp)

# Either train or load pretrained experiment objects
## TRAIN
# filtered_exp.train(filtered_corrupted_batch=True)
# unfiltered_exp.train(filtered_corrupted_batch=False)


## LOAD PRETRAINED
# filtered_exp.load_model_params('filtered_learned_params')
# unfiltered_exp.load_model_params('unfiltered_learned_params')

# Evaluation
validation_link_prediction_dataset = filtered_exp.knowledge_graph.valid_dataset
validation_filter_datasets = [filtered_exp.knowledge_graph.train_dataset]
filtered_validation_mean_rank, filtered_validation_hits10 = filtered_exp.get_mean_rank(validation_link_prediction_dataset,
    validation_filter_datasets, filtered=True,fast_testing=False)

test_link_prediction_dataset = filtered_exp.knowledge_graph.test_dataset
test_filter_datasets = [filtered_exp.knowledge_graph.train_dataset, filtered_exp.knowledge_graph.valid_dataset]
filtered_test_mean_rank, filtered_test_hits10 = filtered_exp.get_mean_rank(test_link_prediction_dataset,
                                                                           test_filter_datasets, filtered=True,
                                                                           fast_testing=False)

# unfiltered_exp.get_test_mean_rank(filtered=False, fast_testing=False)
