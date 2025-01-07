# additional functions from Aladdin Persson's German to English translator, includes the following
# translate_sentence function, measure bleu score, and saving the memory of the AI to keep it trained

# imports
# might want to write notes on these later
import torch
import spacy
from torchtext.data.metrics import bleu_score
import logging
import sys

# function to actually translate a sentence from german to english
# this is almost entirely directly from Aladdin Persson, including comments
def translate_sentence(model, sentence, german, english, device, max_length=50):
    # print(sentence)

    # sys.exit()

    # Load english tokenizer
    spacy_ger = spacy.load("en")

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = [token.text.lower() for token in spacy_ger(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # print(tokens)

    # sys.exit()
    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)

    # Go through each german token and convert to an index
    text_to_indices = [german.vocab.stoi[token] for token in tokens]

    # Convert to Tensor
    # this would be an interesting thing to do more research on
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    # Build encoder hidden, cell state
    # encoding the sentence
    with torch.no_grad():
        hidden, cell = model.encoder(sentence_tensor)

    outputs = [english.vocab.stoi["<sos>"]]

    # code to reference previous word for context
    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hidden, cell = model.decoder(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == english.vocab.stoi["<eos>"]:
            break

    # list of the translated words
    translated_sentence = [english.vocab.itos[idx] for idx in outputs]

    # remove start token when returning the sentence
    return translated_sentence[1:-1]

# calculating the BLEU score
# this is almost entirely directly from Aladdin Persson, including comments
# I should probably just research how this works
def bleu(data, model, german, english, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)["src"]
        trg = vars(example)["trg"]

        prediction = translate_sentence(model, src, german, english, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)

# saving my AI
# this is almost entirely directly from Aladdin Persson, including comments
# probably will need to explain state data
def save_checkpoint(state, filename="my_checkpoint4.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

# loading up my AI
# this is almost entirely directly from Aladdin Persson, including comments
def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])