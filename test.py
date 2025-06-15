from test_model import *
from test_model_util import *

if __name__ == "__main__":

    # test_model
    test_forward_batching()
    test_double_forward_batching()
    test_stepwise_forward_batching()
    
    # test_model_util
    test_find_closest_match()
    test_batch_find_closest_match()    
    test_batch_apply_att_scaling()