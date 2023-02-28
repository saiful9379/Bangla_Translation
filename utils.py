import pickle
import torch
from config import config as cfg

class utils_cls:
    def __init__(self, device):
        self.device = device

    def load_bn_vocal(self, bn_vocal_path):
        file = open(bn_vocal_path, 'rb')
        bn_vocal = pickle.load(file)
        file.close()
        return bn_vocal
    
    def load_en_vocal(self, en_vocal_path):
        file = open(en_vocal_path, 'rb')
        en_vocal = pickle.load(file)
        file.close()
        return en_vocal

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones((sz, sz), device=self.device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

