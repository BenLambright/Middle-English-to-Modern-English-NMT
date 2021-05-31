# All of the following code is from these videos: https://www.youtube.com/watch?v=EoGUlvhRYpk
# and https://www.youtube.com/watch?v=sQUqQddQtB4.
# I have just made comments throughout so I can understand how to write this kind of code myself.

# imports
import torch # This is Pytorch
import torch.nn as nn
import torch.optim as optim
from torchtext import data, datasets
from torchtext.datasets import Multi30k # Multi30k is the German to English dataset
from torchtext.data import Field, BucketIterator
import numpy as np # other useful math-related libraries and modules
import spacy # this is where he gets his nlp datasets
import spacy.cli
import random
# additional functions Aladdin Persson wrote
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint

# comment out these downloads sometimes to make code faster
#spacy.cli.download('de') # importing the tokenizers, it didn't like these two parameters together
#spacy.cli.download('en')

#==PROBLEM IMPORTS==
# I might not need these anyways
#import tensorboard
#from torch.utils.tensorboard import SummaryWriter

# loading and constructing the tokenizers
spacy_ger = spacy.load('de') # This is the German tokenizer
spacy_eng = spacy.load('en') # This is the English tokenizer
# Essentially, he's taking every word (tok) in a given sentence (spacy_eng.tokenizer(text))
# and making it a string value in a list
def tokenizer_ger(text): # German tokenizer function
  return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenizer_eng(text): # English tokenizer function
  return [tok.text for tok in spacy_eng.tokenizer(text)]

# Constructing field and defining how the preprocessing will be done. Also defines the beginning and end of the field
# (sentence) and makes sure everything is in lowercase for the german and english variables. Also, we're building the
# things necessary for training data.
german = Field(tokenize=tokenizer_ger, lower=True,
               init_token='sos', eos_token='<eos>')

english = Field(tokenize=tokenizer_eng, lower=True,
               init_token='<sos>', eos_token='<eos>')

train_data, validation_data, test_data = Multi30k.splits(
    exts=(".de", ".en"), fields=(german, english)
)



#Multi30k.splits()

#print(train_data[0].__dict__.values())

print(type(train_data))

german.build_vocab(train_data, max_size=500, min_freq=2)
english.build_vocab(train_data, max_size=500, min_freq=2)

# Things to implement the model. Building the seq2seq and encoder decoder models
class Encoder(nn.Module): # first LSTM
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, p):
    # __init__ is like a java constructor
    # input_size is size of vocab
    # embeds each map in some dimensional space
    # num_layers of LSTM
        # basic object stuff
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # define all the modules are going to use
        # these are all specific machine learning modules (nn)
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
    # after the sentence is tokenized and mapped as an index corresponding to where it is in the vocabulary,
    # the vector is sent to the LSTM
    # NOTE TO SELF - explain LSTM and embedding
    def forward(self, x): # defines the vector of indices
        # x vector shape: (seq_length, N)
        embedding = self.dropout(self.embedding(x))
        # embedding vector shape: (seq_length, N, embedding_size)
        # each word is also mapped to an embedding size
        outputs, (hidden, cell) = self.rnn(embedding)
        # all we care about is the context vector, which is (hidden, cell)
        # which is why that's all we return
        return hidden, cell

class Decoder(nn.Module): # second LSTM, it's going to be pretty similar. I won't comment stuff I already explained.
  def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
      # output_size should be the same as input_size, because the size of the vocabulary should change. We have both for clarity.
      # hidden_size of the encoder and the decoder are the same
      super(Decoder, self).__init__()
      self.hidden_size = hidden_size
      self.num_layers = num_layers

      self.dropout = nn.Dropout(p)
      self.embedding = nn.Embedding(input_size, embedding_size)
      self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
      self.fc = nn.Linear(hidden_size, output_size) # fc stands for fully connected

  def forward(self, x, hidden, cell): # also takes in the context vector (hidden, cell)
      # shape of x: (N) but we want (1, N)
      # difference in shape is because we want to take in one word at a time, rather than sequence length words at a time like the encoder
      x = x.unsqueeze(0) #this adds another dimension

      # sending it to the embedding
      embedding = self.dropout(self.embedding(x))
      # embedding shape: (1, N, embedding_size)

      #sending it through the LSTM
      outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
      # we use the hidden and cell to predict the next word of the translation
      # outputs is what we think this next word should be
      # shape of the outputs: (1, N, hidden_size)

      predictions = self.fc(outputs)
      # shape of predictions: (1, N, length_of_vocab)
      # this will be sent to the loss function

      predictions = predictions.squeeze(0)
      # we remove the 1 dimension because we want this vector to have the prediction for the entire sentence, not word by word

      return predictions, hidden, cell

class Seq2Seq(nn.Module): # combines the encoder and decoder
  def __init__(self, encoder, decoder):
      super(Seq2Seq, self).__init__()
      self.encoder = encoder
      self.decoder = decoder

  def forward(self, source, target, teacher_force_ratio=0.5):
      # source is source language which is translated into the target [language]
      # we need to send the target back into the our input because it is used to predict the next word
      # teacher_force_ratio means that we take the predicted word 50% of the time and the target translated word the other 50%
      batch_size = source.shape[1]
      # batch_size should look like (trg_len, N)
      target_len = target.shape[0]
      target_vocab_size = len(english.vocab) #make sure to change this to pde

      outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
      # predicts one word at a time, but each word predicts an entire batch and every prediction is a vector of the entire vocabularly size

      # now we're going to run the encoder to get the hidden and the cell to input into the decoder
      hidden, cell = self.encoder(source)

      # grab start token (<sos>)
      x = target[0]

      for t in range(1, target_len):
          output, hidden, cell = self.decoder(x, hidden, cell)

          outputs[t] = output

          # the output will look like (N, english_vocab_size)
          best_guess = output.argmax(1)

          # what I mentioned before about teacher_force_ratio choosing which translation to use
          x = target[t] if random.random() < teacher_force_ratio else best_guess

      return outputs

# Model is setup, now it's time to train it

# hyperparameters for the training model
# you can play around with all of these numbers
# google what these mean
num_epochs = 2
learning_rate = 3e-4
batch_size = 32 # batch size must be smaller than total amount of data

# model hyperparameters
# google these too, review the papers Persson uses
save_model = True
load_model = True # so long as I am loading up old epochs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
# you can play around with all of these numbers
# look up what they mean
encoder_embedding_size = 50
decoder_embedding_size = 50
hidden_size = 400
num_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5

# writing iterators
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, validation_data, test_data), # make sure these are in same order as declared variables
    # the rest of this stuff uses examples that are similar in a batch to minimize padding and thus save on compute
    batch_size=batch_size,
    sort_within_batch = True,
    sort_key = lambda x: len(x.src),
    device=device
)

# running the encoder decoder models
encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout).to(device)
decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, output_size, num_layers, dec_dropout).to(device)

# putting it all together into the Seq2Seq and improving computing efficacy
model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = english.vocab.stoi['<pad>'] # stoi is string to index, so making the pad string an index
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx) # function to decrease computing cost by ignoring padding

# If we're loading up previous trained epochs
if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

# This is an example sentence. We can see how this translation can improve as each epoch goes on.
# See what happens if I get rid of the ''
sentence = (
    "Er sagte, 'Guten Morgen, mein Freund.'"
)

# this is the main stuff I'm running when training data
for epoch in range(num_epochs):
    print(f'Epoch [{epoch} / {num_epochs}]')

    # if we're saving the model, saves the state data and optimizer in the file
    if save_model:
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

    # some aspects of the model work differently when it is evaluating data instead of training, so this switches how
    # it's thinking
    model.eval()

    # plug in a sentence to translate and run it through the function from utils
    translated_sentence = translate_sentence(
        model, sentence, german, english, device, max_length=50
    )

    print(f"Translated example sentence: \n {translated_sentence}")

    # changing the model's mindset back to training mode
    model.train()

    for batch_idx, batch in enumerate(train_iterator):
        # Get input and targets and get to cuda
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        output = model(inp_data, target)
        # from Persson: "# Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshapin. While we're at it
        # Let's also remove the start token while we're at it"
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)

        # figure out what this means sometime
        loss.backward()

        # good thing to check but takes a lot of effort on the computer's part
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) # make sure gradients are in a healthy range

        # figure out what this means sometime
        optimizer.step()

# running on entire test data takes a while, so we'll take just some sample data (the more the better)
# even this takes a while though, in my test run it took 5 minutes with only 10 sample data
#score = bleu(test_data[1:10], model, german, english, device)
#print(f"Bleu score {score * 100:.2f}")

print("run complete")