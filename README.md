# Bangla_Translation
Transformer Based Bangla Machine Translator. Here we have used 195775 number of sentence pair Bangla to English sentence. We trained sentencepice tokenizer for both language and vocab size = 30000. 

Using the Bangla dataset plus the dataset in the data folder, I was able to achieve a BLEU score of 0.39 on the test set (current SOTA is around 0.42), after 4/5 days of training on a single 8gb GPU. For more results see the tutorial again.

# Requirements
```
numpy==1.19.5
sentencepiece==0.1.97
tokenizers==0.8.1rc1
torch==1.11.0+cu113
torchsummary==1.5.1
torchtext==0.12.0
torchvision==0.12.0+cu113
transformers==3.0.2
```

# Dataset
We have collected ```195775``` number of sentence  pair ```Bangla<\t>English```. we seperated bangla and english sentence using ```\t```.

Number of Bangla unique word :
Number of English unique word : 

```
তারা হলের প্রোভস্টের বাড়িতেও প্রবেশ করে	They enter the house of the provost of the hall 
তার ঘোষণা মতে ২ আগস্ট থেকে ক্লাস শুরুর কথা ছিল	According to his announcement, the class was to start from August 2 
তখন সকল পরীক্ষা স্থগিত ছিল	Then all the tests were suspended 
ঐ সময় ক্লাসে ছাত্রদের উপস্থিতি ছিল খুব কম	Attendance of students in the class was very low at that time 
প্রতিদিনই প্রায় কলাভবনে গ্রেনেড বিস্ফোরন হত	Grenades exploded in Kalabhavan almost every day 
রাউলিং বলেন  তাঁর সবসময়ই মনে হয় যে ডাম্বলডোর সমকামী	Rowling said he always thinks Dumbledore is gay 
তিনি গেলার্ট গ্রিন্ডেলওয়াল্ডের প্রেমে পড়েছিলেন	He fell in love with Gelart Grindelwald 
তিনি ব্রহ্মচর্য ও পুথিগত জীবনকেই বেছে নেন	He chose celibacy and bookish life 
```
# Configuration

Additional parameters:
```
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
```

# Pretrain Model
Chcek the training model [click here]()

# Inference

![image](assert/translation.png)
