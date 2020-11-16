import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pickle
import random

from collections import defaultdict
from tqdm.auto import trange


class CharLSTMTagger(nn.Module):
    """ Модель, которая кодирует каждое слово посимвольной LSTM,
    а затем пословной LSTM предсказывает метки каждого слова. """
    def __init__(
            self,
            word_embedding_dim, word_hidden_dim, word_vocab_size,
            char_emb_dim, char_hid_dim, char_voc_size,
            tagset_size
    ):
        super(CharLSTMTagger, self).__init__()

        self.char_embeddings = nn.Embedding(char_voc_size, char_emb_dim)
        self.char_lstm = nn.LSTM(char_emb_dim, char_hid_dim, bidirectional=True, batch_first=True)

        self.word_embeddings = nn.Embedding(word_vocab_size, word_embedding_dim)
        self.word_lstm = nn.LSTM(word_embedding_dim + char_hid_dim * 2, word_hidden_dim, bidirectional=True)

        self.hidden2tag = nn.Linear(word_hidden_dim * 2, tagset_size)

    def forward(self, sentence_by_words, sentence_by_chars):
        char_embeds = self.char_embeddings(sentence_by_chars)
        char_seq, _ = self.char_lstm(char_embeds)
        word_char_embeds = char_seq[:, -1, :]

        word_embeds = self.word_embeddings(sentence_by_words)
        joint_word_embeds = torch.cat([word_embeds, word_char_embeds], 1)

        lstm_out, _ = self.word_lstm(joint_word_embeds.view(len(sentence_by_words), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence_by_words), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


def to_cuda(x):
    if torch.cuda.is_available():
        return x.cuda()
    return x


def prepare_sequence(seq, to_ix, dropout=0.0, min_length=None, eos_bos=False):
    """Довольно скучная служебная функция для превращения последовательности слов в последовательность чисел.
    Что она делает:
    0) Каждый токен из последовательности seq заменяется на его индекс из словарика to_ix
    1) Токены заменяются на "unknown" c вероятностью dropout, чтобы усложнить задачу для нейросети.
    2) Опционально вставляются специальные токены для начала и конца последовательности.
    3) Слишком короткие последовательности дополняются справла ещё одним специальным токеном.
    """
    idxs = [to_ix[w] if w in to_ix and random.random() >= dropout else len(to_ix) for w in seq]
    if eos_bos:
        idxs = [len(to_ix) + 2] + idxs + [len(to_ix) + 3]
    if min_length is not None and len(idxs) < min_length:
        idxs.extend([len(idxs) + 1] * (min_length - len(idxs)))
    return to_cuda(torch.tensor(idxs, dtype=torch.long))


def prepare_sentence(seq, word_to_ix, char_to_ix, dropout=0.0):
    """ Кодируем предложение и слово соответствующими словариками. """
    word_idxs = prepare_sequence(seq, word_to_ix, dropout=dropout)
    max_len = max(len(w) for w in seq) + 2
    char_idxs = torch.stack([
        prepare_sequence(word, char_to_ix, dropout=dropout, eos_bos=True, min_length=max_len)
        for word in seq
    ])
    return word_idxs, char_idxs


class ModelWrapper:
    def __init__(self, model=None, word_to_ix=None, char_to_ix=None, tag_to_ix=None):
        self.model = model
        self.word_to_ix = word_to_ix
        self.char_to_ix = char_to_ix
        self.tag_to_ix = tag_to_ix
        self.idx_to_tag = {i: t for t, i in self.tag_to_ix.items()} if self.tag_to_ix is not None else None

    def build_vocab(self, training_data):
        self.word_to_ix = {}
        self.char_to_ix = {}
        for sent, tags in training_data:
            for word in sent:
                if word not in self.word_to_ix:
                    self.word_to_ix[word] = len(self.word_to_ix)
                for char in word:
                    if char not in self.char_to_ix:
                        self.char_to_ix[char] = len(self.char_to_ix)
        self.tag_to_ix = {tag: i for i, tag in enumerate(sorted({t for x, y in training_data for t in y}))}
        self.idx_to_tag = {i: t for t, i in self.tag_to_ix.items()}

    def init_model(self):
        self.model = CharLSTMTagger(
            32, 32, len(self.word_to_ix) + 2,
            32, 32, len(self.char_to_ix) + 4,
            # для букв мы используем специальные символы начала и конца слова, поэтому словарь чуть больше
            len(self.tag_to_ix)
        )

    def predict_slots(self, tokens):
        """ Принимаем на вход предложение, возвращаем словарик имя слота -> выделившиеся подстроки """
        slot_dict = defaultdict(list)
        w, c = prepare_sentence(tokens, self.word_to_ix, self.char_to_ix, dropout=0)
        with torch.no_grad():
            tag_scores = self.model(w, c)
        tag_indices = tag_scores.argmax(axis=1).detach().cpu().numpy()
        tags = [self.idx_to_tag[i] for i in tag_indices]
        prev_slot = 'O'
        for word, tag in zip(tokens, tags):
            if tag == 'O':
                prev_slot = 'O'
                continue
            bi, slot = tag.split('-', 1)
            if bi == 'B' or prev_slot != slot:
                slot_dict[slot].append(word)
            else:
                slot_dict[slot][-1] += (' ' + word)
            prev_slot = slot
        return dict(slot_dict)

    @classmethod
    def make_names(cls, file_prefix):
        return '{}_vocab.pkl'.format(file_prefix), '{}_weights.pkl'.format(file_prefix)

    def dump(self, file_prefix):
        neural_vocab = {
            'word_to_ix': self.word_to_ix,
            'chat_to_ix': self.char_to_ix,
            'tag_to_ix': self.tag_to_ix,
        }
        vocab_name, weight_name = self.make_names(file_prefix)
        with open(vocab_name, 'wb') as f:
            pickle.dump(neural_vocab, f)
        torch.save(self.model.state_dict(), weight_name)

    @classmethod
    def load(cls, file_prefix, map_location='cpu'):
        vocab_name, weight_name = cls.make_names(file_prefix)
        with open(vocab_name, 'rb') as f:
            vocab = pickle.load(f)
        result = cls(**vocab)
        result.init_model()
        result.model.load_state_dict(torch.load(weight_name, map_location=map_location))

    def prepare_sentence(self, sentence, dropout=0.0):
        return prepare_sentence(sentence, self.word_to_ix, self.char_to_ix, dropout=dropout)


def train_neural_tagger(model_wrapper, training_data, test_data, epochs=20, rate=1e-3):
    model = model_wrapper.model
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=rate)

    if torch.cuda.is_available():
        model.cuda()

    for epoch in trange(epochs):
        loss_sum = 0
        acc_sum = 0
        test_acc_sum = 0
        random.shuffle(training_data)
        for sentence, tags in training_data:
            model.zero_grad()

            word_input, char_inputs = model_wrapper.prepare_sentence(sentence, dropout=0.3)
            targets = prepare_sequence(tags, model_wrapper.tag_to_ix)

            tag_scores = model(word_input, char_inputs)

            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            acc_sum += (tag_scores.argmax(axis=1) == targets).detach().cpu().numpy().mean()
        with torch.no_grad():
            for sentence, tags in test_data:
                word_input, char_inputs = model_wrapper.prepare_sentence(sentence, dropout=0)
                targets = prepare_sequence(tags, model_wrapper.tag_to_ix)
                tag_scores = model(word_input, char_inputs)
                test_acc_sum += (tag_scores.argmax(axis=1) == targets).detach().cpu().numpy().mean()
        print('loss: {}, train accuracy: {}, test accuracy: {}'.format(
            loss_sum / len(training_data),
            acc_sum / len(training_data),
            test_acc_sum / len(test_data))
        )
