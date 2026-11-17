# Running SpikeGPT-1B on an NVIDIA DGX Spark

Step-by-step instructions for training SpikeGPT-1B (`train.py`) and generating text (`run.py`) from a
DGX Spark terminal.

## What to expect from a DGX Spark

| | DGX Spark |
|---|---|
| GPU | 1× GB10 Grace Blackwell (compute capability **12.1**, `sm_121`) |
| Memory | 128 GB **unified** memory, shared by the CPU and the GPU |
| CPU architecture | ARM64 (`aarch64`) |
| CUDA | 13.0 |

This has three consequences for this repo:

1. Every Python package with GPU code must be a **CUDA 13, ARM64** build. CUDA 12 builds (for
   example `cupy-cuda12x`) will not load.
2. 128 GB is enough to train the full 1B model on one machine: about 20 GB for weights, gradients
   and Adam state, plus an estimated ~5 GB of activations per 1024-token sequence (~40 GB at the
   default `batch_size = 8`). Because the memory is shared with the operating system, running out
   of it can freeze the whole machine instead of just stopping the job (step 2 covers this).
3. **It is slow for a full run.** With one GPU and fp32 training, the full 10.24B-token run in
   `train.py` is roughly **1–2 months** by my estimate. That figure is not measured. Use the Spark
   to check that everything works and to run short experiments, and do the full run on Empire AI.
   Step 8 shows how to measure the real speed.

---

## Step 1: Check the machine

```bash
uname -m                 # expect: aarch64
nvidia-smi               # expect: NVIDIA GB10, CUDA Version: 13.0
ls /usr/local/cuda/bin/nvcc && /usr/local/cuda/bin/nvcc --version   # expect: release 13.x
free -g                  # expect: ~119-128 GB total
df -h ~                  # need ~10 GB free for the code, environment and one data shard,
                         # plus ~5 GB per saved checkpoint
```

`nvcc` is required: `src/model.py` compiles the WKV CUDA kernel the first time it runs.

## Step 2: Protect the machine from running out of memory

Running out of GPU memory on a Spark can lock up the desktop and the terminal, because the GPU and
the operating system share the same 128 GB. Two habits help:

- Keep `nvidia-smi` or `htop` open in a second terminal while training, and stop the job with
  `Ctrl+C` if memory gets close to full.
- Community DGX Spark guides recommend turning swap off during training (`sudo swapoff -a`), since
  swapping is what turns an out-of-memory error into a frozen machine. It comes back on after a
  reboot, or with `sudo swapon -a`.

## Step 3: Get the code

```bash
cd ~
git clone -b claude/nice-mccarthy-r7o5ox https://github.com/ckeegan12/SpikeGPT.git
cd SpikeGPT
```

Always run the scripts **from this folder**. `src/model.py` finds the CUDA kernel source by the
relative path `cuda/`.

## Step 4: Point the build tools at CUDA 13

Add these lines to the end of `~/.bashrc`, then run `source ~/.bashrc`:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="12.1"     # build the WKV kernel for the GB10 GPU
```

## Step 5: Create the Python environment

```bash
cd ~/SpikeGPT
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# PyTorch built for CUDA 13 (ARM64 wheels come from this index)
pip install torch --index-url https://download.pytorch.org/whl/cu130

# everything else SpikeGPT needs
pip install numpy tqdm accelerate tokenizers transformers ninja matplotlib huggingface_hub
pip install cupy-cuda13x               # CUDA 13 CuPy, used by the spiking-neuron (LIF) kernel
```

`deepspeed` is optional and not needed. Without it the trainer prints
`DeepSpeed not found. Using torch optimizer instead` and uses PyTorch's Adam, which is fine.

PyTorch may print a warning at import that `sm_121` is not officially supported. Community reports
say the cu130 wheels still run on the GB10. Step 6 checks this.

Every time you open a new terminal, run `cd ~/SpikeGPT && source .venv/bin/activate` first.

## Step 6: Check that the GPU stack works

Run these three checks one at a time. Each must succeed before you continue.

```bash
# 1. PyTorch sees the GPU and can run on it
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); x = torch.randn(1024, 1024, device='cuda'); print('matmul ok', (x @ x).sum().item())"
```
Expect `NVIDIA GB10 (12, 1)` and `matmul ok`.

```bash
# 2. CuPy can compile and run a kernel (this is what the LIF neurons use)
python -c "import cupy as cp; k = cp.RawKernel(r'extern \"C\" __global__ void f(float* x){x[threadIdx.x] += 1;}', 'f'); a = cp.zeros(4, dtype=cp.float32); k((1,), (4,), (a,)); print('cupy ok', a)"
```
Expect `cupy ok [1. 1. 1. 1.]`.

```bash
# 3. The WKV kernel compiles and SpikeGPT's model file loads
python -c "import src.model; print('WKV kernel ok')"
```
The first run takes a minute or two while `nvcc` compiles `cuda/wkv_cuda.cu`. Expect
`WKV kernel ok` at the end.

## Step 7: Get training data

`train.py` reads a **binidx** corpus (a `.bin` + `.idx` pair) tokenized with the repo's
`20B_tokenizer.json`. The pre-tokenized Pile on Hugging Face is in this format. You don't need all
of it: one shard is enough for a test.

1. Open https://huggingface.co/datasets/RichardErkhov/RWKV-LM_pile_binidx_dataset in a browser
   and pick one `.bin` / `.idx` pair with the same name.
2. Download that pair (replace `<name>` with the file name without its extension):

   ```bash
   mkdir -p ~/SpikeGPT/data
   hf download RichardErkhov/RWKV-LM_pile_binidx_dataset --repo-type dataset \
       --include "<name>.bin" "<name>.idx" --local-dir ~/SpikeGPT/data
   ```

   On older `huggingface_hub` versions the command is `huggingface-cli download` with the same
   arguments.

3. Open `train.py` and set `datafile_train` to that path **without** the extension:

   ```python
   datafile_train = "data/<name>"
   ```

## Step 8: Short test run (about 10 minutes)

This uses the real 1B configuration. You only watch it long enough to confirm it trains and to
measure its speed.

```bash
cd ~/SpikeGPT && source .venv/bin/activate
python train.py 2>&1 | tee test_run.log
```

Check, in order:

1. `number of parameters: 1.242364e+09`. This confirms the 1B configuration.
2. A progress bar appears and starts moving:
   `mini-epoch 1 prog ... iter N: ppl ... loss ... lr ...`
3. The loss starts near **10.8** (= ln 50277, the value for an untrained model) and goes down over
   the first few hundred iterations.
4. GPU memory in `nvidia-smi` stays well below the 128 GB total.

Then press `Ctrl+C` to stop. Nothing is saved before the end of the first mini-epoch, which takes
1,250 iterations.

**Measure the speed.** The progress bar shows iterations per second (`it/s`), or seconds per
iteration (`s/it`) if it is slower than one per second. Then:

```
tokens per second = it/s × 8 (batch_size) × 1024 (ctx_len)
hours for the full run = 10,240,000,000 ÷ (tokens per second × 3600)
```

This number replaces the 1–2 month estimate above. It is also a useful number to put in an Empire AI
allocation request.

## Step 9: Full training run

Run it in the background so it keeps going if you close the terminal:

```bash
cd ~/SpikeGPT && source .venv/bin/activate
nohup python train.py > train.log 2>&1 &
tail -f train.log            # Ctrl+C stops the tail, not the training
```

- **Checkpoints:** about **5 GB each**, saved after mini-epochs 1, 11, 21, … and after the last one
  (`SpikeGPT-1B-1.pth`, `SpikeGPT-1B-11.pth`, …, `SpikeGPT-1B-1000.pth`). With the default `n_epoch = 1000` that is about 500 GB. Raise
  `epoch_save_frequency` in `train.py` (for example to 50, about 100 GB in total) if your disk is
  smaller.
- **Loss log:** one line per mini-epoch is appended to `wik8-0.01.txt`
  (mini-epoch, loss, perplexity, learning rate, time).
- **Stopping:** `pkill -f train.py`. `train.py` cannot resume a stopped run with its optimizer and
  learning-rate schedule intact, so a stopped run restarts from the beginning.

## Step 10: Generate text with a checkpoint

Open `run.py` and set `MODEL_NAME` to the checkpoint name **without** `.pth`, for example:

```python
MODEL_NAME = 'SpikeGPT-1B-11'
```

`n_layer = 19`, `n_embd = 2048` and `ctx_len = 1024` are already set for the 1B model. Change the
`context` variable to your own prompt, then run:

```bash
python run.py
```

It prints the prompt followed by the generated text. `NUM_TRIALS` (default 999) is how many samples
it generates, so press `Ctrl+C` when you have seen enough.

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `no kernel image is available for execution on the device` | This PyTorch build has no kernels that run on `sm_121` | Install the newest cu130 PyTorch (`pip install -U torch --index-url https://download.pytorch.org/whl/cu130`), or use a community PyTorch build for the GB10 |
| `libcudart.so.12: cannot open shared object file` | A CUDA 12 package is installed (often `cupy-cuda12x`) | `pip uninstall cupy-cuda12x` and install `cupy-cuda13x` |
| `CuPy is not installed!` | `cupy-cuda13x` is missing from the active environment | `source .venv/bin/activate`, then `pip install cupy-cuda13x` |
| `nvcc: not found` or `CUDA_HOME environment variable is not set` | Step 4 is not active in this terminal | `source ~/.bashrc` |
| `Unknown CUDA arch` | `TORCH_CUDA_ARCH_LIST` has a typo | It must be exactly `12.1` |
| `No such file or directory: 'cuda/wkv_op.cpp'` | The script was started from another folder | `cd ~/SpikeGPT` first |
| `FileNotFoundError ... .idx` | `datafile_train` is wrong | Use the path without `.bin` / `.idx` |
| `CUDA out of memory`, or the machine freezes | Batch too large for the shared 128 GB | Lower `batch_size` in `train.py` (try 4), and see step 2 |
| `AssertionError` in `WKV.forward` | `ctx_len` is above 1024 (the kernel's `T_MAX`) | Keep `ctx_len = 1024` |
