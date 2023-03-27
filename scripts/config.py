from pathlib import Path

import torch


ROOT_FOLDER = str(Path(__file__).parent.parent)

IS_CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device('cuda' if IS_CUDA_AVAILABLE else 'cpu')

LABEL_PAD_TOKEN_ID = -100
GENERATION_LEN = 512
N_BEAMS = 3
