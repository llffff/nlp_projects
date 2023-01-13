
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class CoNLLData(Dataset):
    """ CoNLLData Dataset """
    def __init__(self, args, data):
        self.data = data
        self.length = len(data)
        self.max_len_word = args.max_len_word
        self.idx_pad_char = args.idx_pad_char

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        chars_padded = [x + [self.idx_pad_char]*(self.max_len_word - len(x)) for x in self.data[index]['chars']]
        return \
            torch.LongTensor(self.data[index]['words']), torch.LongTensor(chars_padded), \
            torch.LongTensor(self.data[index]['tags'])


def collate_fn(insts, args):
    """ Batch preprocess """
    words_batch, chars_batch, tags_batch = list(zip(*insts))

    # get sorted indices
    lens = torch.as_tensor([v.size(0) for v in words_batch])
    sorted_lens, sorted_indices = torch.sort(lens, descending=True)

    # sort data
    words_batch = pad_sequence(words_batch, batch_first=True, padding_value=args.idx_pad_word)
    words_batch = words_batch.index_select(0, sorted_indices)

    chars_batch = pad_sequence(chars_batch, batch_first=True, padding_value=args.idx_pad_char)
    chars_batch = chars_batch.index_select(0, sorted_indices)

    tags_batch = pad_sequence(tags_batch, batch_first=True, padding_value=args.idx_pad_tag)
    tags_batch = tags_batch.index_select(0, sorted_indices)

    return words_batch, chars_batch, tags_batch, sorted_lens
