# Now we want to encode and decode the label
# Example:
# Encode: 7abc --> [7,10,11,12]  (IntTensor)
# Docode: [11,12,13] --> [b,c,d]

# Define alphabet in config.py
import torch

class LabelConverter(object):
    def __init__(self, alphabet, ignore_case = True):
        self.ignore_case = ignore_case
        if self.ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet
        self.dict = {}
        for index, c in enumerate(self.alphabet):
            self.dict[c] = index
    
    def encode(self, labels):
        # input: labels: list of labels name
        # return: [labels encoded], [labels length]
        length = [len(c) for c in labels]
        text = ''.join(labels)
        encoded = []
        for c in text:
            encoded.append(self.dict[c.lower()] )
        return torch.IntTensor(encoded), torch.IntTensor(length)
    
    def decode(self, text, length):
        # decode to strs, batch mode
        # text: list of encodings    length: list of length
        assert sum(length) == text.numel()  # .numel used to calculate number of elements in text
        decodings = []
        pos = 0
        for str_len in length:
            encode = text[pos:pos+str_len]
            decode = ''
            for digit in encode:
                decode += self.alphabet[digit-1]
            decodings.append(decode)
            pos+= str_len
        return decodings

