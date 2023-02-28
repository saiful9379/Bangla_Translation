import os
import argparse
import warnings
warnings.filterwarnings('ignore')
import torch
from tqdm import tqdm
import sentencepiece as spm
from utils import utils_cls
from model import BanglaTransformer
from config import config as cfg
# very short
from nltk.translate.bleu_score import sentence_bleu
torch.manual_seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

uobj = utils_cls(device=device)

__MODULE__       = "Bangla Language Tranlation"
__MAIL__         = "saifulbrur79@gmail.com"
__MODIFICAIOTN__ = "28/03/2023"
__LICENSE__      = "MIT"


def read_data(data_path):
    with open(data_path, "r") as f:
        data = f.readlines()
    data = list(map(lambda x: [x.split("\t")[0], x.split("\t")[1].replace("\n", "")], data))
    return data

def load_tokenizer(tokenizer_path:str = "")->object:
    _tokenizer = spm.SentencePieceProcessor(model_file=tokenizer_path)
    return _tokenizer

def get_vocab(BN_VOCAL_PATH:str="", EN_VOCAL_PATH:str=""):
    bn_vocal, en_vocal = uobj.load_bn_vocal(BN_VOCAL_PATH), uobj.load_en_vocal(EN_VOCAL_PATH)
    return bn_vocal, en_vocal

def load_model(model_path:str = "", SRC_VOCAB_SIZE:int=0, TGT_VOCAB_SIZE:int=0):
    model = BanglaTransformer(
        cfg.NUM_ENCODER_LAYERS, cfg.NUM_DECODER_LAYERS, cfg.EMB_SIZE,  SRC_VOCAB_SIZE,
        TGT_VOCAB_SIZE, cfg.FFN_HID_DIM, nhead= cfg.NHEAD)
    model.to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def get_token(text, en_tokenizer):
    return en_tokenizer.encode_as_pieces(text)

def get_blue_score(gt, pt):
    score = sentence_bleu(gt, pt)
    return score

def greedy_decode(model, src, src_mask, max_len, start_symbol, eos_index):
    src = src.to(device)
    src_mask = src_mask.to(device)
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    for i in range(max_len-1):
        memory = memory.to(device)
        memory_mask = torch.zeros(ys.shape[0], memory.shape[0]).to(device).type(torch.bool)
        tgt_mask = (uobj.generate_square_subsequent_mask(ys.size(0))
                                    .type(torch.bool)).to(device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim = 1)
        next_word = next_word.item()
        ys = torch.cat([ys,torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == eos_index:
            break
    return ys

def translate(model, src, src_vocab, tgt_vocab, src_tokenizer):
    PAD_IDX, BOS_IDX, EOS_IDX= src_vocab['<pad>'], src_vocab['<bos>'], src_vocab['<eos>']
    tokens = [BOS_IDX] + [src_vocab.get_stoi()[tok] for tok in src_tokenizer.encode(src, out_type=str)]+ [EOS_IDX]
    num_tokens = len(tokens)
    src = (torch.LongTensor(tokens).reshape(num_tokens, 1) )
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX, eos_index= EOS_IDX).flatten()
    p_text = " ".join([tgt_vocab.get_itos()[tok] for tok in tgt_tokens]).replace("<bos>", "").replace("<eos>", "")
    pts = " ".join(list(map(lambda x : x , p_text.replace(" ", "").split("▁"))))
    return pts.strip()


def evaluation(data, model, bn_vocab, en_vocab, bn_tokenizer, en_tokenizer):
    # for i in tqdm
    score_list = []
    for i in tqdm(range(len(data)), desc ="Bangla and Englaish Sentence:"):
        text = data[i]
        pre = translate(model, text[0], bn_vocab, en_vocab, bn_tokenizer)
        gt = get_token(text[1], en_tokenizer)
        pt = get_token(pre, en_tokenizer)
        score = get_blue_score([gt], pt)
        score_list.append(score)
    print("BLUE SCORE : ", sum(score_list)/len(score_list))
    return score
    
# score = list(map(evaluation, data))
# print("BLUE SCORE : ", sum(score)/len(score))


if __name__ == "__main__":

    print(torch.cuda.get_device_name(0))

    """
    python evaluation.py --data process_data/merge_data.txt --bn_tokenizer ./model/bn_model.model --en_tokenizer ./model/en_model.model\
    --bn_vocab ./model/bn_vocab.pkl --en_vocab ./model/en_vocab.pkl --model ./model/model_checkpoint.pt
    
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--bn_tokenizer', type=str, default='./model/bn_model.model', help='tokenizer path(s)')
    parser.add_argument('--en_tokenizer', type=str, default='./model/en_model.model', help='tokenizer path(s)')
    parser.add_argument('--data', type=str, default='./process_data/merge_data.txt', help='data')  # file/folder, 0 for webcam
    parser.add_argument('--bn_vocab', type=str, default='./model/bn_vocab.pkl', help='data')  # file/folder, 0 for webcam
    parser.add_argument('--en_vocab', type=str, default='./model/en_vocab.pkl', help='data')  # file/folder, 0 for webcam
    parser.add_argument('--model', type=str, default='./model/model_checkpoint.pt', help='model path')
    
    opt = parser.parse_args()

    data = read_data(opt.data)
    print("Tokenizer Loading ...... : ", end="", flush=True)
    bn_tokenizer = load_tokenizer(tokenizer_path=opt.bn_tokenizer)
    en_tokenizer = load_tokenizer(tokenizer_path=opt.en_tokenizer)
    print("Done")
    print("Vocab Loading ......     : ", end="", flush=True)
    bn_vocab, en_vocab = get_vocab(BN_VOCAL_PATH=opt.bn_vocab, EN_VOCAL_PATH=opt.en_vocab)
    print("Done")
    print("Model Loading ......     : ", end="", flush=True)
    model = load_model(model_path=opt.model, SRC_VOCAB_SIZE=len(bn_vocab), TGT_VOCAB_SIZE=len(en_vocab))
    print("Done")

    evaluation(data[:100], model, bn_vocab, en_vocab, bn_tokenizer, en_tokenizer)
    
    # pre = translate(model, text, bn_vocab, en_vocab, bn_tokenizer)
    # print(f"input : {text}")
    # print(f"prediction: {pre}")

