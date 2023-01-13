import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np


def log_sum_exp(vec):
    """
    This function calculates the score explained above for the forward algorithm
    vec 2D: 1 * size_tag
    """
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def argmax(vec):
    """ This function returns the max index in a vector """
    _, idx = torch.max(vec, 1)
    return idx.view(-1).data.tolist()[0]


class BiLSTM_CRF(nn.Module):
    def __init__(self, args, word2idx, char2idx, tag2idx, glove_word=None):
        """
        Input parameters from args:

                args = Dictionary that maps NER tags to indices
                word2idx = Dimension of word embeddings (int)
                tag2idx = hidden state dimension
                char2idx = Dictionary that maps characters to indices
                glove_word = Numpy array which provides mapping from word embeddings to word indices
        """
        super(BiLSTM_CRF, self).__init__()


        # 参数
        self.device = args.device
        self.enable_crf = args.enable_crf
        self.idx_pad_tag = args.idx_pad_tag

        self.START_TAG = args.START_TAG
        self.STOP_TAG = args.STOP_TAG
        self.word2idx = word2idx
        self.char2idx = char2idx
        self.tag2idx = tag2idx
        self.n_word = len(word2idx)
        self.n_char = len(char2idx)
        self.n_tag = len(tag2idx)

        self.max_len_word = args.max_len_word
        self.idx_pad_char = args.idx_pad_char
        self.idx_pad_word = args.idx_pad_word

        self.dim_emb_char, self.dim_emb_word, self.dim_out_char,self.dim_out_word  = args.dims
        self.window_kernel = args.window_kernel

        self.mode_char = args.mode_char
        self.mode_word = args.mode_word


        # 定义架构
        # 嵌入层

        ## char 嵌入
        self.embedding_char = nn.Embedding(self.n_char+1, self.dim_emb_char, padding_idx=self.idx_pad_char)
        # 均匀采样初始化
        init_embedding(self.embedding_char)

        ## word嵌入    
        self.embedding_word = nn.Embedding.from_pretrained(
                torch.FloatTensor(glove_word), 
                freeze=False,
                padding_idx=self.idx_pad_word)
    

        self.dropout0 = nn.Dropout(args.dropout)
        # character encoder
        if self.mode_char == 'lstm':
            self.lstm_char = nn.LSTM(self.dim_emb_char, self.dim_out_char, num_layers=1, batch_first=True, bidirectional=True)
            init_lstm(self.lstm_char)
        elif self.mode_char == 'cnn':
            self.conv_char = nn.Conv2d(in_channels=1, out_channels=self.dim_out_char * 2,
                                       kernel_size=(3, self.dim_emb_char), padding=(2, 0))
            init_linear(self.conv_char)
            self.max_pool_char = nn.MaxPool2d((self.max_len_word + 2, 1))  # padding x 2 - kernel_size + 1
        else:
            # no char representation
            assert self.dim_out_char == 0
            #raise Exception('Character encoder mode unknown...')
        self.dropout1 = nn.Dropout(args.dropout)

        # word encoder
        self.dim_in_word = self.dim_emb_word + self.dim_out_char * 2

        print(self.mode_word)
        
        if self.mode_word == 'lstm':
            self.lstm_word = nn.LSTM(self.dim_in_word, self.dim_out_word, batch_first=True,
                                     bidirectional=True)
            init_lstm(self.lstm_word)

        elif self.mode_word == 'cnn':
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.dim_out_word * 2,
                                   kernel_size=(self.window_kernel, self.dim_in_word),
                                   padding=(self.window_kernel // 2, 0))
            self.conv2 = nn.Conv2d(in_channels=1, out_channels=self.dim_out_word * 2,
                                   kernel_size=(self.window_kernel, self.dim_out_word * 2),
                                   padding=(self.window_kernel // 2, 0))
            init_linear(self.conv1)
            init_linear(self.conv2)

        else:
            raise Exception('Word encoder mode '+self.mode_word+' unknown...')

        self.dropout2 = nn.Dropout(args.dropout)

        # predictor
        self.hidden2tag = nn.Linear(self.dim_out_word * 2, self.n_tag)
        init_linear(self.hidden2tag)
        if args.enable_crf:
            self.transitions = nn.Parameter(torch.zeros(self.n_tag, self.n_tag))
            self.transitions.data[self.tag2idx[self.START_TAG], :] = -10000
            self.transitions.data[:, self.tag2idx[self.STOP_TAG]] = -10000

    @staticmethod
    def reformat_conv2d(t):
        return t.squeeze(-1).transpose(-1, -2).unsqueeze(1)

    def forward(self, words_batch, chars_batch, lens_word):
        len_batch, len_sent = words_batch.shape

        # 字符embedding
        emb_chars = self.embedding_char(chars_batch)

        # 进入cnn之前进行一次dropout
        emb_chars = self.dropout0(emb_chars)

        # char encoding
        if self.mode_char == 'lstm':
            lens_char = (chars_batch != self.idx_pad_char).sum(dim=2)
            lens_char_covered = torch.where(lens_char == 0, 1, lens_char)
            
            # print(f"lens_char: {lens_char.shape}")
            # print(f"lens_char_covered: {lens_char_covered.shape}")
            # lens_char: torch.Size([128, 52]) b,w_s
            

            # 压缩填充字符
            packed_char = pack_padded_sequence(
                emb_chars.view(-1, self.max_len_word, self.dim_emb_char),
                lens_char_covered.view(-1).cpu(), batch_first=True, enforce_sorted=False)
            # lstm
            out_lstm_char, _ = self.lstm_char(packed_char)

            # 恢复填充字符
            output_char, _ = pad_packed_sequence(
                out_lstm_char, batch_first=True, 
                total_length=self.max_len_word)
            output_char = output_char * lens_char.view(-1, 1, 1).bool()
            output_char = output_char.reshape(len_batch, len_sent, self.max_len_word, self.dim_out_char*2)

            # print(f"output_char: {output_char.shape}")
            # torch.Size([128, 52, 61, 20])
            # b, w_s, c_s, dim_out_char*2

            output_char = torch.cat(
                (torch.stack(
                    [sample[torch.arange(len_sent).long(), lens-1, :self.dim_out_char]
                     for sample, lens in zip(output_char, lens_char)]),
                 torch.stack(
                     [sample[torch.arange(len_sent).long(), lens*0, self.dim_out_char:]
                      for sample, lens in zip(output_char, lens_char)]))
                , dim=-1)
            
            # print(f"output_char: {output_char.shape}")
            # torch.Size([128, 52, 20])
            # b, ws, dim_out_char*2 
        

        elif self.mode_char == 'cnn':
            enc_char = self.conv_char(emb_chars.unsqueeze(2).view(-1, 1, self.max_len_word, self.dim_emb_char))
            output_char = self.max_pool_char(enc_char).view(len_batch, len_sent, self.dim_out_char * 2)
        else:
            # no char representation
            output_char = torch.Tensor([]).to(emb_chars.device)
            # raise Exception('Unknown character encoder: '+self.mode_char+'...')


        # 词嵌入
        emb_words = self.embedding_word(words_batch)
        emb_words_chars = torch.cat((emb_words, output_char), dim=-1)

        # 进行词编码前dropout
        emb_words_chars = self.dropout1(emb_words_chars)

        # word encoding
        if self.mode_word == 'lstm':
            packed_word = pack_padded_sequence(emb_words_chars, lens_word.cpu(), batch_first=True)
            out_lstm_word, _ = self.lstm_word(packed_word)
            enc_word, _ = pad_packed_sequence(out_lstm_word, batch_first=True)
        elif self.mode_word == 'cnn':
            out_cnn_word = self.conv1(emb_words_chars.unsqueeze(1))
            out_cnn_word = self.reformat_conv2d(out_cnn_word)
            out_cnn_word = self.conv2(out_cnn_word)
            enc_word = self.reformat_conv2d(out_cnn_word).squeeze(1)
        else:
            raise Exception('Unknown word encoder: '+self.mode_word+'...')

        # 特征 dropout
        enc_word = self.dropout2(enc_word)
        

        # 得到#tags的表征
        outputs = self.hidden2tag(enc_word)
        
        return outputs


    ## encoder+crf
    def get_loss(self, words_batch, chars_batch, tags_batch, lens_batch):
        
        # encoder
        feats_batch = self.forward(words_batch, chars_batch, lens_batch)


        # crf decoder
        if self.enable_crf:
            loss_batch, pred_batch = [], []
            # iterate each sentence
            for i, (feats, tags, len_sent) in enumerate(zip(feats_batch, tags_batch, lens_batch)):
                feats, tags = feats[:len_sent, :], tags[:len_sent]
                forward_score = self.forward_alg(feats)
                gold_score = self.score_sentence(feats, tags)
                score, pred = self.viterbi_decode(feats)
                loss_batch.append(forward_score - gold_score)
                pred_batch.append(pred)

            return torch.stack(loss_batch).sum() / lens_batch.sum(), pred_batch
        
        else:

            loss = F.cross_entropy(feats_batch.view(-1, self.n_tag), tags_batch.view(-1), ignore_index=self.idx_pad_tag)

            _, pred_batch = torch.max(feats_batch.view(-1, self.n_tag)[:, :-1], 1)
            pred_batch = pred_batch.reshape(feats_batch.shape[0], feats_batch.shape[1])

            return loss, pred_batch.cpu().tolist()



    ## crf实现
    def forward_alg(self, feats):
        """
        This function performs the forward algorithm explained above
        """
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.zeros((1, self.n_tag)).fill_(-10000.)

        # START_TAG has all score.
        init_alphas[0][self.tag2idx[self.START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas.clone().to(feats.device)
        forward_var.requires_grad = True

        # Iterate through the sentence
        for feat in feats:
            # broadcast the emission score: it is the same regardless of the previous tag
            emit_score = feat.view(-1, 1)

            # the ith entry of trans_score is the score of transitioning to next_tag from i
            tag_var = forward_var + self.transitions + emit_score

            # The ith entry of next_tag_var is the value for the edge (i -> next_tag) before we do log-sum-exp
            max_tag_var, _ = torch.max(tag_var, dim=1)

            # The forward variable for this tag is log-sum-exp of all the scores.
            tag_var = tag_var - max_tag_var.view(-1, 1)

            # Compute log sum exp in a numerically stable way for the forward algorithm
            forward_var = max_tag_var + torch.log(torch.sum(torch.exp(tag_var), dim=1)).view(1, -1)  # ).view(1, -1)
        terminal_var = (forward_var + self.transitions[self.tag2idx[self.STOP_TAG]]).view(1, -1)
        alpha = log_sum_exp(terminal_var)
        # Z(x)
        return alpha

    def score_sentence(self, feats, tags):
        # remove padding word

        r = torch.tensor(range(feats.size()[0]), dtype=torch.long, device=feats.device)
        pad_start_tags = torch.cat(
            [torch.tensor([self.tag2idx[self.START_TAG]], 
                          dtype=torch.long, device=tags.device),tags])
        pad_stop_tags = torch.cat([tags,torch.tensor([self.tag2idx[self.STOP_TAG]], 
                                                     dtype=torch.long, device=tags.device)])
        score = torch.sum(self.transitions[pad_stop_tags, pad_start_tags]) + torch.sum(feats[r, tags])

        return score

    def viterbi_decode(self, feats):
        """ viterbi algorithm """
        backpointers = []
        # analogous to forward

        # Initialize the viterbi variables in log space
        init_vars = torch.zeros((1, self.n_tag), device=feats.device).fill_(-10000.)
        init_vars[0][self.tag2idx[self.START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vars.clone()
        forward_var.requires_grad = True
        for feat in feats:
            next_tag_var = forward_var.view(1, -1).expand(self.n_tag, self.n_tag) + self.transitions
            _, bptrs_t = torch.max(next_tag_var, dim=1)
            bptrs_t = bptrs_t.squeeze().data.cpu().numpy()  # holds the backpointers for this step
            next_tag_var = next_tag_var.data.cpu().numpy()
            viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t]  # holds the viterbi variables for this step
            viterbivars_t = torch.tensor(viterbivars_t, device=feats.device, requires_grad=True)

            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = viterbivars_t + feat
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag2idx[self.STOP_TAG]]
        terminal_var.data[self.tag2idx[self.STOP_TAG]] = -10000.
        terminal_var.data[self.tag2idx[self.START_TAG]] = -10000.
        best_tag_id = argmax(terminal_var.unsqueeze(0))
        path_score = terminal_var[best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag2idx[self.START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path



def init_embedding(layer):
    """ Initialize embedding """
    bias = np.sqrt(3.0 / layer.embedding_dim)
    nn.init.uniform_(layer.weight, -bias, bias)

def init_linear(input_linear):
    """ Initialize linear layer """
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform_(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()


def init_lstm(input_lstm):
    """
    Initialize lstm

    PyTorch weights parameters:

        weight_ih_l[k]: the learnable input-hidden weights of the k-th layer,
            of shape `(hidden_size * input_size)` for `k = 0`. Otherwise, the shape is
            `(hidden_size * hidden_size)`

        weight_hh_l[k]: the learnable hidden-hidden weights of the k-th layer,
            of shape `(hidden_size * hidden_size)`
    """

    # Weights init for forward layer
    for ind in range(0, input_lstm.num_layers):
        ## Gets the weights Tensor from our model, for the input-hidden weights in our current layer
        weight = eval('input_lstm.weight_ih_l' + str(ind))

        # Initialize the sampling range
        sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))

        # Randomly sample from our samping range using uniform distribution and apply it to our current layer
        nn.init.uniform_(weight, -sampling_range, sampling_range)

        # Similar to above but for the hidden-hidden weights of the current layer
        weight = eval('input_lstm.weight_hh_l' + str(ind))
        sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform_(weight, -sampling_range, sampling_range)

    # We do the above again, for the backward layer if we are using a bi-directional LSTM (our final model uses this)
    if input_lstm.bidirectional:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.weight_ih_l' + str(ind) + '_reverse')
            sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform_(weight, -sampling_range, sampling_range)
            weight = eval('input_lstm.weight_hh_l' + str(ind) + '_reverse')
            sampling_range = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform_(weight, -sampling_range, sampling_range)

    # Bias initialization steps

    # We initialize them to zero except for the forget gate bias, which is initialized to 1
    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            bias = eval('input_lstm.bias_ih_l' + str(ind))

            # Initializing to zero
            bias.data.zero_()

            # This is the range of indices for our forget gates for each LSTM cell
            bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

            # Similar for the hidden-hidden layer
            bias = eval('input_lstm.bias_hh_l' + str(ind))
            bias.data.zero_()
            bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

        # Similar to above, we do for backward layer if we are using a bi-directional LSTM
        if input_lstm.bidirectional:
            for ind in range(0, input_lstm.num_layers):
                bias = eval('input_lstm.bias_ih_l' + str(ind) + '_reverse')
                bias.data.zero_()
                bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
                bias = eval('input_lstm.bias_hh_l' + str(ind) + '_reverse')
                bias.data.zero_()
                bias.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1

