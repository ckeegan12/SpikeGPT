# Training SpikeGPT-216M on the TITAN machine

Step-by-step instructions for training SpikeGPT-216M (18 layers × 768, 215,399,424 parameters) on
a Linux machine with two Pascal GPUs: a **TITAN Xp** and a **TITAN X**, 12 GB each. This branch
(`spike216M`) already has the right settings, so you don't need to edit any code.

Run every block in order, from a fresh terminal.

---

## Step 1: Check you have admin rights

```bash
sudo -v && echo "sudo ok"
```

If this fails, ask whoever manages the machine to do steps 2 and 3 for you. Then continue from
step 4.

## Step 2: Install the NVIDIA driver, then reboot

```bash
sudo apt update
sudo ubuntu-drivers install
sudo reboot
```

Your remote connection will drop. Wait about 2 minutes, reconnect, and check:

```bash
nvidia-smi
```

You must see both `TITAN Xp` and `TITAN X` before continuing.

## Step 3: Install the CUDA 12.6 toolkit

This installs `nvcc`, which compiles SpikeGPT's GPU kernel. It has to be **12.6**, because
CUDA 13 does not support these GPUs.

```bash
UBU=ubuntu$(. /etc/os-release; echo $VERSION_ID | tr -d .)
wget https://developer.download.nvidia.com/compute/cuda/repos/$UBU/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-6
```

## Step 4: Get the code

```bash
cd ~
git clone -b spike216M https://github.com/ckeegan12/SpikeGPT.git SpikeGPT-216M
cd ~/SpikeGPT-216M
```

## Step 5: Set up Python

Add the CUDA settings to `~/.bashrc` so every new terminal has them:

```bash
cat >> ~/.bashrc <<'EOF'
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=/usr/local/cuda-12.6/bin:$PATH
export TORCH_CUDA_ARCH_LIST=6.1
EOF
source ~/.bashrc
```

Then create the environment. PyTorch 2.14 with CUDA 12.6 is the last PyTorch build that supports
these GPUs.

```bash
cd ~/SpikeGPT-216M
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install "torch==2.14.*" --index-url https://download.pytorch.org/whl/cu126
pip install numpy tqdm accelerate tokenizers transformers ninja matplotlib huggingface_hub cupy-cuda12x
```

## Step 6: Check that everything works

```bash
python -c "import torch; print(torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```
Expect `2` and both TITAN names.

```bash
python -c "import src.model; print('WKV kernel ok')"
```
The first time, this takes a minute or two while it compiles the kernel. Expect `WKV kernel ok`
at the end. The `SyntaxWarning` lines before it are harmless.

## Step 7: Download the training data

```bash
python download_data.py
```

This downloads one shard of the pre-tokenized Pile from Hugging Face (it can be many GB) and links
it as `data/pile`, which is where `train.py` looks. At the end it prints the shard's size and
approximate token count.

## Step 8: Start training on both GPUs

```bash
nohup accelerate launch --multi_gpu --num_processes 2 --mixed_precision no train.py > train.log 2>&1 &
tail -f train.log
```

Within a few minutes you should see:

1. `number of parameters: 2.153994e+08`
2. A progress bar: `mini-epoch 1 prog ... iter N: ppl ... loss ... lr ...`
3. A loss that starts near **10.8** and goes down.

Press `Ctrl+C` to stop watching. **Training keeps running** after you disconnect.

If you see `CUDA out of memory`, stop the run (`pkill -f train.py`), set `batch_size = 1` in
`train.py`, and start step 8 again.

## Step 9: Check on training after you reconnect

```bash
cd ~/SpikeGPT-216M
tr '\r' '\n' < train.log | tail -3     # latest progress line
cat wik8-0.01.txt                      # one line per finished mini-epoch: number, loss, perplexity, lr, time
nvidia-smi                             # GPU memory and use
pkill -f train.py                      # stop training
```

**Healthy training:** the loss keeps falling, quickly at first and then slowly.
**Stop and investigate** if you see `nan`, a loss stuck near 10.8, or a loss rising over several
mini-epochs.

**How long:** the run is 500 mini-epochs, about 5.1B tokens, the same as the published
SpikeGPT-216M. My rough estimate for these GPUs is 1–2 weeks. For the real figure, read `it/s`
from the progress bar:

```
hours = 5,120,000,000 ÷ (it/s × 2 (batch) × 1024 (tokens) × 2 (GPUs) × 3600)
```

**Checkpoints:** about 0.9 GB each, saved after mini-epochs 1, 11, 21, … 491 and 500
(`SpikeGPT-216M-1.pth`, `SpikeGPT-216M-11.pth`, …, `SpikeGPT-216M-500.pth`). If training stops,
it restarts from the beginning; it cannot resume.

## Step 10: Generate text

When training finishes, `run.py` is already set to load `SpikeGPT-216M-500`:

```bash
python run.py
```

To try an earlier checkpoint, change `MODEL_NAME` in `run.py` (for example to `SpikeGPT-216M-101`).
To use your own prompt, change the `context` variable in `run.py`. Press `Ctrl+C` when you have
seen enough samples.
