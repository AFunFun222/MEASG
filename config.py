import torch
import os
import time
import logging
# a = torch.cuda.is_available()
debug = True
selectData = "Funcom"

data_root_path = "../Data/%s" % selectData
correct_fid = os.path.join(data_root_path, "correct_fids")
dots_files_dir = "../../SSMIF_Data/%s/java_dots" % selectData
code_summary_dir = os.path.join(data_root_path, "source_code2summary")
token_dir = os.path.join(data_root_path, "token_source_code")
ASTs_dir = os.path.join(data_root_path, "ASTs")
GNNData_dir = os.path.join(data_root_path, "GNNData")

plt_photo_dir = "../plt_photo/"


output_root = os.path.join('../output/%s' % selectData, time.strftime('%Y%m%d_%H%M%S', time.localtime()))

model_root = os.path.join(output_root, 'models')
if not os.path.exists(model_root):
    os.makedirs(model_root)


logger = logging.getLogger()
logger.setLevel(level=logging.DEBUG)

console = logging.StreamHandler()
console.setLevel(level=logging.INFO)
logger.addHandler(console)

file = logging.FileHandler(os.path.join(output_root, 'run.log'))
file.setLevel(level=logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s | %(filename)s | line %(lineno)d] - %(levelname)s: %(message)s')
file.setFormatter(formatter)
logger.addHandler(file)

vocab_root = os.path.join(output_root, 'vocab')
if not os.path.exists(vocab_root):
    os.makedirs(vocab_root)

DDP = False

batch_size = 64
valid_batch_size = 128
test_batch_size = 128

learning_rate = 0.000055

lr_decay_rate = 0.9

embedding_dim = 256
hidden_size = 256
decoder_dropout_rate = 0.5
n_epochs = 50

teacher_forcing_ratio = 0.5


use_cuda = torch.cuda.is_available()

device = torch.device('cuda:0')
init_uniform_mag = 0.02
init_normal_std = 1e-4
eps = 1e-12

continue_train = True  


max_ast_length = 2000
max_subast_len = 100
max_code_length = 200  
max_nl_length = 30
min_nl_length = 4

max_decode_steps = 30
early_stopping_patience = 20  


source_vocab_size = 50000  
code_vocab_size = 50000  
ast_vocab_size = 50000  
nl_vocab_size = 50000  

# features
use_pointer_gen = True  
use_teacher_forcing = True 
use_lr_decay = True  
use_early_stopping = True  
a


save_valid_model = True
save_best_model = True
save_test_outputs = True

beam_width = 5  
beam_top_sentences = 1     


log_state_every = 1000
