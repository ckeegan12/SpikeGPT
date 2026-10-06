########################################################################################################
# The RWKV v2-RNN Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import logging
import datetime
import json
from src.model import GPT, GPTConfig
from src.trainer import Trainer, TrainerConfig
from src.utils import Dataset
import torch
import numpy as np
from src.spikingjelly.clock_driven import functional
from src.binidx import MMapIndexedDataset
from accelerate import accelerator
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


# SpikeGPT-1B: the same architecture as the repo's 216M model, scaled to the
# parameter count of Llama 3.2 1B (1,235,814,400). See docs/SpikeGPT_explained.md.
#
#   params = n_layer * (13*n_embd^2 + 11*n_embd) + 4*n_embd + 2*vocab_size*n_embd
#          = 19 * (13*2048^2 + 11*2048) + 4*2048 + 2*50277*2048
#          = 1,242,363,904   (+0.53% vs Llama 3.2 1B)
#
# Launch on multiple GPUs with:  accelerate launch train.py

### Step 1: set training data ##########################################################################

# binidx corpus tokenized with 20B_tokenizer.json (e.g. the pre-tokenized Pile, see readme).
# Give the path WITHOUT the .bin / .idx extension.
datafile_train = "pile_binidx/pile_text_document"

### Step 2: set model size #############################################################################

ctx_len = 1024        # ===> increase T_MAX in model.py if your ctx_len > 1024
n_layer = 19
n_embd = 2048

# 'RWKV' (better for char-level English) or 'RWKV-ffnPre' (better in some cases)
model_type = 'RWKV'

### Step 3: set batch size #############################################################################

# Per-GPU micro batch. Each 1024-token sequence needs an estimated ~5 GB of fp32 activations
# at this size, on top of ~20 GB for weights + gradients + Adam state.
# If you see "CUDA out of memory", reduce it.
batch_size = 8

### Step 4: set learning rate, training mini-epochs #######################################################

lr_init = 3e-4
lr_final = 1e-5
# the mini-epoch is very short and of fixed length (ctx_len * epoch_length_fixed tokens)
n_epoch = 1000
# 0 = never, 1 = every mini-epoch, 2 = every two mini-epochs, etc.
epoch_save_frequency = 10
epoch_save_path = 'SpikeGPT-1B-'

epoch_length_fixed = 10000

########################################################################################################

import src.utils
src.utils.set_seed(42) # remember to change seed if you load a model

np.set_printoptions(precision=4, suppress=True, linewidth=200)
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO,)

grad_norm_clip = 1.0
warmup_tokens = 20 * epoch_length_fixed * ctx_len  # first 2% of n_epoch

betas = (0.9, 0.99)
eps = 4e-9

num_workers = 0

########################################################################################################
# Load data
########################################################################################################

print('loading data... ' + datafile_train)
train_dataset = Dataset(MMapIndexedDataset(datafile_train), ctx_len, epoch_length_fixed)
########################################################################################################
# Train model
########################################################################################################
if __name__ == '__main__':

    model = GPT(GPTConfig(train_dataset.vocab_size, train_dataset.ctx_len, model_type=model_type,
                          n_layer=n_layer, n_embd=n_embd)).cuda()

    # # load a trained model. remember to change random seed
#     m2 = torch.load('medium/trained-30L-768E-936.pth',map_location=torch.device('cpu'))
#     model.load_state_dict(m2)
    valid_dataset = None
    test_dataset = None
    print('model', model_type, 'epoch', n_epoch, 'batchsz', batch_size, 'betas',
          betas, 'eps', eps, 'ctx', ctx_len, 'layer', n_layer, 'embd', n_embd, )
    tconf = TrainerConfig(model_type=model_type, max_epochs=n_epoch, batch_size=batch_size,
                          learning_rate=lr_init, lr_decay=True, lr_final=lr_final, betas=betas, eps=eps, grad_norm_clip=grad_norm_clip,
                          warmup_tokens=warmup_tokens, final_tokens=n_epoch*len(train_dataset)*ctx_len, num_workers=num_workers, epoch_save_frequency=epoch_save_frequency, epoch_save_path=epoch_save_path)
    trainer = Trainer(model, train_dataset, valid_dataset, test_dataset, tconf)

    trainer.train()

    torch.save(model.state_dict(), 'trained-' + str(n_epoch) + '-' + trainer.get_run_name() +
               '-' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S') + '.pth')
