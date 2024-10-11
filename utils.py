import copy
import time

import torch
import itertools
import os
import pickle
import numpy as np
import nltk

import config
import DataUtil
from config import logger
from torch_geometric.nn import GCNConv, ChebConv, BatchNorm
from torch_geometric.utils import dropout_adj
from GNN_utils import KNNGraph
import matplotlib.pyplot as plt


class Vocab(object):

    PAD_TOKEN = '<pad>'  # padding token
    SOS_TOKEN = '<sos>'  # start of sequence
    EOS_TOKEN = '<eos>'  # end of sequence
    UNK_TOKEN = '<unk>'  # unknown token


    START_VOCAB = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

    def __init__(self, name, additional_special_symbols=None, ignore_case=False):
        """
        Initialization Definition.
        Args:
            name (str): vocabulary name
            additional_special_symbols (list): optional, list of custom special symbols
            ignore_case (bool): optional, ignore cases if True, default False
        """
        self.ignore_case = ignore_case
        self.special_symbols = Vocab.START_VOCAB.copy()
        if additional_special_symbols:
            self.add_special_symbols(additional_special_symbols)
        self.name = name
        self.trimmed = False
        self.word2index = {}  
        self.word2count = {}  
        self.index2word = {}  
        self.num_words = 0
        self.add_sentence(self.special_symbols)  

        self.origin_size = 0  # vocab size before trim

    def add_dataset(self, dataset):
        """
        Add a list of list of tokens.
        Args:
            dataset (list): a list object whose elements are all lists of str objects

        """
        for seq in dataset:
            for token in seq:
                self.add_word(token)

    def add_sentence(self, sentence):
        """
        Add a list of tokens.
        Args:
            sentence (list): a list object whose elements are all str objects

        """
        for word in sentence:
            self.add_word(word)

    def add_word(self, word):
        """
        Add a single word.
        Args:
            word (str): str object

        """
        if self.ignore_case:
            word = word.lower()
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    def trim(self, max_vocab_size):
        """
        
        Trim the vocabulary to the given size according to the frequency of the words.
            if the size is greater than the given size
        Args:
            max_vocab_size: max vocabulary size

        Returns:
            list:
                - words which is eliminated, list of tuples (word, count)
        """
        if self.trimmed:
            return None
        self.trimmed = True
        self.origin_size = self.num_words

        if self.num_words <= max_vocab_size:
            return None
        for special_symbol in self.special_symbols:
            self.word2count.pop(special_symbol)
        all_words = list(self.word2count.items())
        all_words = sorted(all_words, key=lambda item: item[1], reverse=True) 

        keep_words = all_words[:max_vocab_size - len(self.special_symbols)]  
        keep_words = self.special_symbols + [word for word, _ in keep_words]  

        trimmed_words = all_words[max_vocab_size - len(self.special_symbols):]  


        self.word2index.clear()
        self.word2count.clear()
        self.index2word.clear()
        self.num_words = 0
        self.add_sentence(keep_words)  

        return trimmed_words

    def add_special_symbols(self, symbols: list):
        assert isinstance(symbols, list)
        for symbol in symbols:
            assert isinstance(symbol, str)
            if symbol not in self.special_symbols:
                self.special_symbols.append(symbol)

    def get_index(self, word):
        """
        Return the index of given word, if the given word is not in the vocabulary, return the index of UNK token.
        Args:
            word (str): word in str

        Returns:
            int:
                - index of the given word, UNK if OOV
        """
        if self.ignore_case:
            word = word.lower()
        return self.word2index[word] if word in self.word2index else self.word2index[Vocab.UNK_TOKEN]

    def get_pad_index(self):
        return self.word2index[Vocab.PAD_TOKEN]

    def get_sos_index(self):
        return self.word2index[Vocab.SOS_TOKEN]

    def get_eos_index(self):
        return self.word2index[Vocab.EOS_TOKEN]

    def get_unk_index(self):
        return self.word2index[Vocab.UNK_TOKEN]

    def get_word(self, index):
        """
        Return the corresponding word of the given index, if not in the vocabulary, return '<unk>'.
        Args:
            index: given index

        Returns:
            str:
                - token of the given index
        """
        return self.index2word[index] if index in self.index2word else Vocab.UNK_TOKEN

    def save(self, vocab_dir, name=None):
        path = os.path.join(vocab_dir, '{}_vocab.pk'.format(self.name) if name is None else name)
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    def save_txt(self, vocab_dir, name=None):
        path = os.path.join(vocab_dir, '{}_vocab.txt'.format(self.name) if name is None else name)
        with open(path, 'w', encoding='utf-8') as file:
            for word, _ in self.word2index.items():
                file.write(word + '\n')

    def __len__(self):
        return self.num_words

    def __contains__(self, item):
        """
        Return True if the given word is in the vocab, else False.
        Args:
            item: word to query

        Returns:
            bool:
                - True if the given word is in the vocab, else False.
        """
        if self.ignore_case:
            item = item.lower()
        return item in self.word2index


def filter_data(sources, codes, asts_vocabs, nls, sliced_flatten_ast):
    """
    filter the data according to the rules
    :param sources: list of tokens of source codes
    :param codes: list of tokens of split source codes
    :param sliced_flatten_ast: list of tokens of sequence asts
    :param asts_vocabs: list of tokens of sequence asts_vocab
    :param nls: list of tokens of comments
    :return: filtered codes, asts and nls
    """
    assert len(sources) == len(codes)
    assert len(codes) == len(sliced_flatten_ast)
    assert len(sliced_flatten_ast) == len(nls)
    assert len(sliced_flatten_ast) == len(asts_vocabs)

    new_sources = []
    new_codes = []
    new_asts = []
    new_asts_vocab = []
    new_nls = []
    for i in range(len(codes)):
        source = sources[i]
        code = codes[i]
        ast = sliced_flatten_ast[i]
        nl = nls[i]
        asts_vocab = asts_vocabs[i]

        if len(code) > config.max_code_length or len(nl) > config.max_nl_length or len(nl) < config.min_nl_length:
            continue
        max_ast_len = config.max_ast_length
        max_subast_len = config.max_subast_len
        for temp_data in ast[0]:
            max_ast_len = max_ast_len - len(temp_data)
            if len(temp_data) > max_subast_len:
                max_subast_len = -1
        if max_ast_len < 0 or max_subast_len < 0:
            continue
        new_sources.append(source)
        new_codes.append(code)
        new_asts.append(ast)
        new_nls.append(nl)
        new_asts_vocab.append(asts_vocab)
    return new_sources, new_codes, new_asts_vocab, new_nls, new_asts


class Timer(object):
    """
    Computes elapsed time.
    """

    def __init__(self):
        self.running = True
        self.total = 0
        self.start = time.time()

    def reset(self):
        self.running = True
        self.total = 0
        self.start = time.time()
        return self

    def resume(self):
        if not self.running:
            self.running = True
            self.start = time.time()
        return self

    def stop(self):
        if self.running:
            self.running = False
            self.total += time.time() - self.start
        return self

    def time(self):
        if self.running:
            return self.total + time.time() - self.start
        return self.total


class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping(object):

    def __init__(self, patience, delta=0, high_record=False):
        """
        Initialize an EarlyStopping instance
        Args:
            patience: How long to wait after last time validation loss decreased
            delta: Minimum change in the monitored quantity to qualify as an improvement
            high_record: True if the improvement of the record is seen as the improvement of the performance,
                default False
        """
        self.patience = patience
        self.counter = 0
        self.record = None
        self.early_stop = False
        self.delta = delta
        self.high_record = high_record

        self.refreshed = False
        self.best_model = None
        self.best_epoch = -1

    def __call__(self, score, model, epoch):
        """
        Call this instance when get a new score
        Args:
            score (float): the new score
            model:
        """
        # first call
        if self.record is None:
            self.record = score
            self.refreshed = True
            self.best_model = model
            self.best_epoch = epoch
        # not hit the best
        elif (not self.high_record and score > self.record + self.delta) or \
                (self.high_record and score < self.record - self.delta):
            self.counter += 1
            self.refreshed = False
            logger.info('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info('Early stop')
        # hit the best
        else:
            self.record = score
            self.counter = 0
            self.refreshed = True
            self.best_model = model
            self.best_epoch = epoch



def build_word_vocab(dataset, vocab_name, ignore_case=False, max_vocab_size=None, special_symbols=None,
                     save_dir=None, save_name=None, save_txt_name=None):

    vocab = Vocab(name=vocab_name, ignore_case=ignore_case, additional_special_symbols=special_symbols)
    vocab.add_dataset(dataset)
    if max_vocab_size:
        vocab.trim(max_vocab_size)
    if save_dir:
        vocab.save(vocab_dir=save_dir, name=save_name)
        vocab.save_txt(vocab_dir=save_dir, name=save_txt_name)
    return vocab



def extend_indices_from_batch(source_batch: list, nl_batch: list, source_vocab: Vocab, nl_vocab: Vocab, raw_nl):

    extend_source_batch_indices = []  # [B, T]
    extend_nl_batch_indices = []
    batch_oovs = []  # list of list of oov words in sentences
    for source, nl in zip(source_batch, nl_batch):

        oovs = []
        extend_source_indices = []
        extend_nl_indices = []

        oov_temp_index = {}  # maps the oov word to temp index

        for word in source:
            if word not in source_vocab.word2index:  
                if word not in oovs:
                    oovs.append(word)
                oov_index = oovs.index(word)
                temp_index = len(source_vocab) + oov_index  
                extend_source_indices.append(temp_index)
                oov_temp_index[word] = temp_index
            else:
                extend_source_indices.append(source_vocab.word2index[word])
        extend_source_indices.append(source_vocab.get_eos_index())  

        if not raw_nl:
            for word in nl:
                if word not in nl_vocab.word2index:  
                    if word in oov_temp_index:  
                        temp_index = oov_temp_index[word]
                        extend_nl_indices.append(temp_index)  
                    else:
                        extend_nl_indices.append(nl_vocab.get_unk_index())  
                else:
                    extend_nl_indices.append(nl_vocab.word2index[word])
            extend_nl_indices.append(nl_vocab.get_eos_index())

        extend_source_batch_indices.append(extend_source_indices)
        extend_nl_batch_indices.append(extend_nl_indices)
        batch_oovs.append(oovs)

    return extend_source_batch_indices, extend_nl_batch_indices, batch_oovs


def indices_from_batch(batch: list, vocab: Vocab) -> list:

    indices = []
    for sentence in batch:
        indices_sentence = []
        for word in sentence:
            if word not in vocab.word2index:
                indices_sentence.append(vocab.get_unk_index())
            else:
                indices_sentence.append(vocab.word2index[word])
        indices_sentence.append(vocab.get_eos_index())
        indices.append(indices_sentence)
    return indices


def indices_from_sliced_flatten_ast(batch: list, vocab: Vocab) -> list:
    indices = []
    for word in batch:
        if word not in vocab.word2index:
            indices.append(vocab.get_unk_index())
        else:
            indices.append(vocab.word2index[word])
    return indices


def get_seq_lens(batch: list) -> list:
    """
    get sequence lengths of given batch
    :param batch: [B, T]
    :return: sequence lengths
    """
    seq_lens = []
    for seq in batch:
        seq_lens.append(len(seq))
    return seq_lens


def get_seq_lens_ast(batch: list) -> list:
    """
    get sequence lengths of given batch
    :param batch: [B, T]
    :return: sequence lengths
    """
    seq_lens = []
    for seq in batch:
        seq_lens.append(len(seq[0]))
    return seq_lens


def pad_one_batch(batch: list, vocab: Vocab) -> torch.Tensor:
    """

    pad batch using _PAD token and get the sequence lengths
    :param batch: one batch, [B, T]
    :param vocab: corresponding vocab
    :return:
    """
    batch = list(itertools.zip_longest(*batch, fillvalue=vocab.get_pad_index())) 
    batch = [list(b) for b in batch]  
    return torch.tensor(batch, device=config.device).long()


class Batch(object):

    def __init__(self, source_batch, source_seq_lens, code_batch, code_seq_lens,
                 ast_batch, ast_seq_lens, nl_batch, nl_seq_lens, sliced_flatten_ast_lens, sliced_flatten_ast):
        self.source_batch = source_batch
        self.source_seq_lens = source_seq_lens
        self.code_batch = code_batch
        self.code_seq_lens = code_seq_lens
        self.ast_batch = ast_batch
        self.ast_seq_lens = ast_seq_lens
        self.nl_batch = nl_batch
        self.nl_seq_lens = nl_seq_lens

        self.sliced_flatten_ast = sliced_flatten_ast
        self.sliced_flatten_ast_lens = sliced_flatten_ast_lens

        self.batch_size = len(source_seq_lens)


        self.extend_source_batch = None
        self.extend_nl_batch = None
        self.max_oov_num = None
        self.batch_oovs = None
        self.extra_zeros = None

    def get_sliced_flatten_ast(self):
        return self.sliced_flatten_ast, self.sliced_flatten_ast_lens

    def get_regular_input(self):
        return self.source_batch, self.source_seq_lens, self.code_batch, self.code_seq_lens, \
               self.ast_batch, self.ast_seq_lens, self.nl_batch, self.nl_seq_lens

    def config_point_gen(self, extend_source_batch_indices, extend_nl_batch_indices, batch_oovs,
                         source_vocab, nl_vocab, raw_nl):
        self.batch_oovs = batch_oovs
        self.max_oov_num = max([len(oovs) for oovs in self.batch_oovs])

        self.extend_source_batch = pad_one_batch(extend_source_batch_indices, source_vocab)  # [T, B]
        self.extend_source_batch = self.extend_source_batch.transpose(0, 1)

        # [T, B]
        if not raw_nl:
            self.extend_nl_batch = pad_one_batch(extend_nl_batch_indices, nl_vocab)

        if self.max_oov_num > 0:
            # [B, max_oov_num]
            self.extra_zeros = torch.zeros((self.batch_size, self.max_oov_num), device=config.device)

    def get_pointer_gen_input(self):
        return self.extend_source_batch, self.extend_nl_batch, self.extra_zeros


def init_decoder_inputs(batch_size, vocab: Vocab) -> torch.Tensor:
    """
    initialize the input of decoder
    :param batch_size:
    :param vocab:
    :return: initial decoder input, torch tensor, [batch_size]
    """
    return torch.tensor([vocab.get_sos_index()] * batch_size, device=config.device)



def collate_fn(batch, source_vocab, code_vocab, ast_vocab, nl_vocab, raw_nl=False):
    """
    process the batch without sorting  不
    :param batch: one batch, first dimension is batch, [B]
    :param source_vocab:
    :param code_vocab:
    :param ast_vocab: [B, T]
    :param nl_vocab: [B, T]
    :param raw_nl: True when test, nl_batch will not be translated and returns the raw data
    :return:
    """
    batch = batch[0]  
    source_batch = []
    code_batch = []
    ast_batch = []
    nl_batch = []
    sliced_flatten_ast_batch = []
    for b in batch:  
        source_batch.append(b[0])
        code_batch.append(b[1])
        ast_batch.append(b[2])  
        nl_batch.append(b[3])
        sliced_flatten_ast_batch.append(b[4])
    # transfer words to indices including oov words, and append EOS token to each sentence, list
    
    extend_source_batch_indices = None
    extend_nl_batch_indices = None
    batch_oovs = None
    if config.use_pointer_gen: 
        # if raw_nl, extend_nl_batch_indices is a empty list

        extend_source_batch_indices, extend_nl_batch_indices, batch_oovs = extend_indices_from_batch(source_batch,
                                                                                                     nl_batch,
                                                                                                     source_vocab,
                                                                                                     nl_vocab,
                                                                                                     raw_nl)
 
    source_batch = indices_from_batch(source_batch, source_vocab)
    code_batch = indices_from_batch(code_batch, code_vocab)  # [B, T]
    ast_batch = indices_from_batch(ast_batch, ast_vocab)  # [B, T]
    if not raw_nl:
        nl_batch = indices_from_batch(nl_batch, nl_vocab)  # [B, T]
    temp_a_list, temp_list = [], []
    for an_arb in sliced_flatten_ast_batch:
        for arb in an_arb[0]:
            code_index = []
            code_index += indices_from_sliced_flatten_ast(arb, ast_vocab)
            temp_a_list.append(code_index)
        temp_list.append((temp_a_list, an_arb[1]))
        temp_a_list = []
    sliced_flatten_ast_batch = copy.deepcopy(temp_list)


    source_seq_lens = get_seq_lens(source_batch)
    code_seq_lens = get_seq_lens(code_batch)
    ast_seq_lens = get_seq_lens(ast_batch)
    nl_seq_lens = get_seq_lens(nl_batch)

    temp_sliced_flatten_ast_lens = get_seq_lens_ast(sliced_flatten_ast_batch)
    ast_max_lens = max(temp_sliced_flatten_ast_lens)

    temp_a_list, temp_list = [], []
    for an_arb in sliced_flatten_ast_batch: 
        temp_a_list = an_arb
        for i in range(ast_max_lens-len(an_arb[0])):
            temp_a_list[0].append([ast_vocab.get_pad_index()])
        temp_list.append(temp_a_list)
 
    sliced_flatten_ast_batch = []
    temp_degree = []  
    for n in range(0, ast_max_lens):
        sliced_flatten_ast_batch.append([])
        for a_ast in temp_list:
            sliced_flatten_ast_batch[n].append(a_ast[0][n])  
    for a_ast in temp_list:
        temp_degree.append(a_ast[1])
    sliced_flatten_ast_batch = (sliced_flatten_ast_batch, temp_degree)
    del temp_list

    source_batch = pad_one_batch(source_batch, source_vocab)
    code_batch = pad_one_batch(code_batch, code_vocab)
    ast_batch = pad_one_batch(ast_batch, ast_vocab)
    if not raw_nl:
        nl_batch = pad_one_batch(nl_batch, nl_vocab)

    for n in range(0, ast_max_lens):
        sliced_flatten_ast_batch[0][n] = pad_one_batch(sliced_flatten_ast_batch[0][n], ast_vocab)

    sliced_flatten_ast_lens = []
 
    for index in range(0, len(sliced_flatten_ast_batch[0])):
        sliced_flatten_ast_lens.append(sliced_flatten_ast_batch[0][index].shape[0])

        if index == 0:
            subflatten_ast_batch = sliced_flatten_ast_batch[0][index]
        else:
            subflatten_ast_batch = torch.cat((subflatten_ast_batch, sliced_flatten_ast_batch[0][index]), dim=0)
    subflatten_ast_batch = (subflatten_ast_batch, sliced_flatten_ast_batch[1])


    batch = Batch(source_batch, source_seq_lens, code_batch, code_seq_lens,
                  ast_batch, ast_seq_lens, nl_batch, nl_seq_lens, sliced_flatten_ast_lens, subflatten_ast_batch)

    if config.use_pointer_gen:  
        batch.config_point_gen(extend_source_batch_indices,
                               extend_nl_batch_indices,
                               batch_oovs,
                               source_vocab,
                               nl_vocab,
                               raw_nl)

    return batch


def count_params(model):
    """
    Count the number of parameters of given model
    """
    return sum(p.numel() for p in model.parameters())


def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'),
                         ['', 'K', 'M', 'B', 'T'][magnitude])


def to_time(float_time):
    """
    translate float time to h, min, s and ms
    :param float_time: time in float
    :return: h, min, s, ms
    """
    time_s = int(float_time)
    time_ms = int((float_time - time_s) * 1000)
    time_h = time_s // 3600
    time_s = time_s % 3600
    time_min = time_s // 60
    time_s = time_s % 60
    return time_h, time_min, time_s, time_ms


def load_vocab(vocab_dir, name):
    with open(os.path.join(vocab_dir, '{}_vocab.pk'.format(name)), mode='rb') as f:
        obj = pickle.load(f)
    assert isinstance(obj, Vocab)
    return obj


def sentence_bleu_score(reference, candidate) -> float:
    """
    calculate the sentence level bleu score, 4-gram with weights(0.25, 0.25, 0.25, 0.25)
    :param reference: tokens of reference sentence
    :param candidate: tokens of sentence generated by model
    :return: sentence level bleu score
    """
    smoothing_function = nltk.translate.bleu_score.SmoothingFunction()
    return nltk.translate.bleu_score.sentence_bleu(references=[reference],
                                                   hypothesis=candidate,
                                                   smoothing_function=smoothing_function.method5)


def meteor_score(reference, candidate):
    """
    meteor score
    :param reference:
    :param candidate:
    :return:
    """
    # return nltk.translate.meteor_score.single_meteor_score(reference=' '.join(reference),
    #                                                        hypothesis=' '.join(candidate))

    return nltk.translate.meteor_score.single_meteor_score(reference=reference,
                                                           hypothesis=candidate)


def corpus_bleu_score(references, candidates) -> float:
    smoothing_function = nltk.translate.bleu_score.SmoothingFunction()
    return nltk.translate.bleu_score.corpus_bleu(list_of_references=[[reference] for reference in references],
                                                 hypotheses=[candidate for candidate in candidates],
                                                 smoothing_function=smoothing_function.method5)


def tune_up_decoder_input(index, vocab):
    """
    replace index with unk if index is out of vocab size
    :param index:
    :param vocab:
    :return:
    """
    if index >= len(vocab):
        index = vocab.get_unk_index()
    return index


def is_unk(word):
    if word == Vocab.UNK_TOKEN:
        return True
    return False


def is_special_symbol(word):
    if word in Vocab.START_VOCAB:
        return True
    else:
        return False

def measure(references, candidates) -> (float, float):
    """
    measures the top sentence model generated
    :param references: batch of references
    :param candidates: batch of sentences model generated
    :return: total sentence level bleu score, total meteor score
    """
    assert len(references) == len(candidates)
    batch_size = len(references)

    total_s_bleu = 0
    total_meteor = 0
    total_s_bleus = []
    total_meteors = []

    for reference, candidate in zip(references, candidates):

        # sentence level bleu score
        sentence_bleu = sentence_bleu_score(reference, candidate)
        if sentence_bleu*100 > 100:
            total_s_bleus.append(100)
        else:
            total_s_bleus.append(sentence_bleu*100)
        total_s_bleu += sentence_bleu

        # meteor score
        meteor = meteor_score(reference, candidate)
        total_meteor += meteor
        total_meteors.append(meteor)

    return total_s_bleus, total_s_bleu / batch_size, total_meteors, total_meteor / batch_size


def time2str(float_time):
    time_h, time_min, time_s, time_ms = to_time(float_time)
    return '{}h {}min {}s {}ms'.format(time_h, time_min, time_s, time_ms)


def plot_data(data, name):
    x = [i for i in range(1, len(data)+1)]
    plt.figure(1, figsize=(8, 4))
    plt.title(name)
    plt.plot(x, data, label='loss')
    plt.xlabel("batch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(config.plt_photo_dir + 'epochs_' + name + 'loss.jpg', bbox_inches='tight')

