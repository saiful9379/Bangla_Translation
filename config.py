class config:
    EMB_SIZE = 512
    NHEAD = 8
    FFN_HID_DIM = 512
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6
    NUM_EPOCHS = 300
    BN_TOKENIZER_PATH = "./model/bn_model.model"
    EN_TOKENIZER_PATH = "./model/en_model.model"
    BN_VOCAL_PATH = "./model/bn_vocab.pkl"
    EN_VOCAL_PATH = "./model/en_vocab.pkl"
    MODEL_PATH = "./model/model_checkpoint.pt"
