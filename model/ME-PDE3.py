# Much of the following code is from this video tutorial: https://www.youtube.com/watch?v=EoGUlvhRYpk.

# I have commented notes on the open-sourced code explaining what it means as well as noting changes I have made to it, such as changing german and english to me and pde


# imports
import torch # This is Pytorch
import torch.nn as nn
import torch.optim as optim
from torchtext import data, datasets
from torchtext.datasets import Multi30k # Multi30k is the class that includes German to English dataset
from torchtext.data import Field, BucketIterator
import numpy as np # other useful math-related libraries and modules
import spacy # this is where he gets his nlp datasets
import spacy.cli
import random
# additional functions Aladdin Persson wrote
from utils3 import translate_sentence, bleu, save_checkpoint, load_checkpoint

# comment out these downloads sometimes to make code faster, I've already downloaded them
#spacy.cli.download('de') # importing the tokenizers, it didn't like these two parameters together
#spacy.cli.download('en')

#==PROBLEM IMPORTS==
# I found that I don't actually need these anyways, they just make the data look nicer once it has all trained
#import tensorboard
#from torch.utils.tensorboard import SummaryWriter

# loading and constructing the tokenizers
# Essentially, I'm taking every word (tok) in a given sentence (spacy_eng.tokenizer(text))
# and making it a string value in a list
spacy_eng = spacy.load('en') # This is the English tokenizer

def tokenizer_eng(text): # pde tokenizer function
  return [tok.text for tok in spacy_eng.tokenizer(text)]

# Constructing field (sentence) and defining the preprocessing. The model requires the sentences to be written as a
# field in order to train with them.
# NOTE: makes sure everything is in lowercase for the me and pde variables.
me = Field(tokenize=tokenizer_eng, lower=True,
               init_token='sos', eos_token='<eos>')

pde = Field(tokenize=tokenizer_eng, lower=True,
               init_token='<sos>', eos_token='<eos>')

train_data, validation_data, test_data = Multi30k.splits(
    exts=(".me", ".pde"), fields=(me, pde)
)

# NOTE on train data: For some reason "" are sometimes copied into me sentence but not the pde translation. These are
# not something I put in the translation, which means that it is a problem when running Multi30k.splits()

# making sure the data is what it's supposed to be
# print(train_data[0].__dict__.values())
# print(type(train_data))

# this includes all the vocab the model knows, I took out (, max_size=500, min_freq=1) because I don't need to limit
# data size right now
me.build_vocab(train_data)
pde.build_vocab(train_data)

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

      # sending it through the LSTM
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
      target_vocab_size = len(pde.vocab) #make sure to change this to pde

      outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
      # predicts one word at a time, but each word predicts an entire batch and every prediction is a vector of the entire vocabularly size

      # now we're going to run the encoder to get the hidden and the cell to input into the decoder
      hidden, cell = self.encoder(source)

      # grab start token (<sos>)
      x = target[0]

      for t in range(1, target_len):
          output, hidden, cell = self.decoder(x, hidden, cell)

          outputs[t] = output

          # the output will look like (N, pde_vocab_size)
          best_guess = output.argmax(1)

          # what I mentioned before about teacher_force_ratio choosing which translation to use
          x = target[t] if random.random() < teacher_force_ratio else best_guess

      return outputs

# Model is setup, now it's time to train it

# hyperparameters for the training model
# you can play around with all of these numbers
# I decided to include num_epochs as a parameter in a later function I made because I change that number so much
# num_epochs = 20
# learning rate attempts to minimize loss (measurement for bad predictions) in epochs
learning_rate = 3e-4
# batch size must be smaller than total amount of data, amount of data taken in when training
batch_size = 32

# turn off load model after changing data otherwise you'll get tensor size errors, turn it back on after saving
save_model = True
load_model = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size_encoder = len(me.vocab)
input_size_decoder = len(pde.vocab)
output_size = len(pde.vocab)
# you can play around with all of these numbers too
# embeddings can increase the amount of context/relationships words have with each other, each tok always embedded
encoder_embedding_size = 50
decoder_embedding_size = 50
# should be about 2/3 the size of the input layer
hidden_size = 400
# number of RNN layers
num_layers = 2
# more data that can be hidden
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

pad_idx = pde.vocab.stoi['<pad>'] # stoi is string to index, so making the pad string an index
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx) # function to decrease computing cost by ignoring padding

# If we're loading up previous trained epochs
if load_model:
    load_checkpoint(torch.load("my_checkpoint4.pth.tar"), model, optimizer)

# this is the main stuff I'm running when training data
# I put this all in a function "trainer(num_epochs)" so I only train the data when I want to
def trainer(num_epochs):
    for epoch in range(num_epochs):
        print(f'Epoch [{epoch + 1} / {num_epochs}]')

        # if we're saving the model, saves the state data and optimizer in the file
        if save_model:
            checkpoint = {
                'input_size': len(me.vocab),
                'output_size': len(me.vocab),
                'hidden_layers': 400,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)

        # some aspects of the model work differently when it is evaluating data instead of training, so this switches how
        # it's thinking
        model.eval()

        # plug in a sentence to translate and run it through the function from utils
        translated_sentence = translate_sentence(
            model, sentence, me, pde, device, max_length=50
        )
        # I got rid of Persson's print statement and made one I prefer
        print("Translated sentence '" + sentence + "': " + ' '.join(translated_sentence))

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

            # I copied that all in because I had some problems with tensor sizes at some points, so I thought that might
            # help me figure out why some tensors were the wrong size
            output = output[1:].reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            optimizer.zero_grad()
            loss = criterion(output, target)

            # accumulates the gradients of the parameters
            loss.backward()

            # good thing to check but takes a lot of effort on the computer's part
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) # make sure gradients are in a healthy range

            # updates the parameters based on a current stored gradient of the parameters, which should help to
            # optimize the parameters
            optimizer.step()


# I wrote this to state dict keys when loading data
# print("The state dict keys: \n\n", model.state_dict().keys())

# input sentences here to have them be translated
sentence = (
    "Ther was noon auditour koude on him wynne."
)

translated_sentence = translate_sentence(model, sentence, me, pde, device, max_length=50)
# print(f"Translated example sentence: \n {translated_sentence}")
# above is what Persson used to print the input, but I wrote a cleaner way to print the translation
print("Translated sentence '" + sentence + "': " + ' '.join(translated_sentence))

# running the trainer function to train the model
# trainer(100)

# running on entire test data takes a while, so we'll take just some sample data (the more the better)
# score = bleu(test_data[1:10], model, me, pde, device)
# print(f"Bleu score {score * 100:.2f}")

# print("run complete")
