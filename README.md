# Bn_Translator
Transformer Based Bangla Machine Translator. Here we have used 195775 number of sentence pair Bangla to English sentence.

Using the Bangla dataset plus the dataset in the data folder, I was able to achieve a BLEU score of 0.39 on the test set (current SOTA is around 0.42), after 4/5 days of training on a single 8gb GPU. For more results see the tutorial again.

# Configuration

Additional parameters:
-epochs : 300
-batch_size : 1500
-n_layers : 
-heads : 
-no_cuda : 
-SGDR : 
-d_model : Dimension of embedding vector and layers (default=512)
-dropout' : Decide how big dropout will be (default=0.1)
-printevery : how many iterations run before printing (default=100)
-lr : learning rate (default=0.0001)
-load_weights : 
-max_strlen : 
-checkpoint : 
