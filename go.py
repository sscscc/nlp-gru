print("importing modules")
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
import jieba
from nltk.tokenize import word_tokenize
from tqdm import tqdm

print("modules imported")


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
    pairs = [line.split("\t")[:2] for line in lines]
    return pairs


def tokenize_sentence(text, language):
    if language == "en":
        return word_tokenize(text.lower())
    elif language == "cn":
        return jieba.lcut(text)
    else:
        return None


# 创建词汇表
def create_vocab(pairs):
    en_vocab = set()
    cn_vocab = set()
    for pair in pairs:
        eng, cn = pair
        en_vocab.update(tokenize_sentence(eng, "en"))
        cn_vocab.update(tokenize_sentence(cn, "cn"))
    en_vocab = sorted(en_vocab)
    cn_vocab = sorted(cn_vocab)
    en_word_to_index = {word: index for index, word in enumerate(en_vocab, 1)}
    en_word_to_index["<PAD>"] = 0
    cn_word_to_index = {word: index for index, word in enumerate(cn_vocab, 1)}
    cn_word_to_index["<PAD>"] = 0
    return en_word_to_index, cn_word_to_index


# 将句子转换为向量
def sentence_to_vector(sentence, vocab, max_length, language):
    vector = [vocab[word] for word in tokenize_sentence(sentence, language)][
        :max_length
    ]
    vector += [vocab["<PAD>"]] * (max_length - len(vector))
    return vector


# 定义数据集类
class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, trg_vocab, max_length):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_length = max_length
        self.tokenize_pairs = [
            (
                sentence_to_vector(src_sentence, src_vocab, max_length, "en"),
                sentence_to_vector(trg_sentence, trg_vocab, max_length, "cn"),
            )
            for src_sentence, trg_sentence in pairs
        ]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        src_vector, trg_vector = self.tokenize_pairs[index]
        return torch.tensor(src_vector), torch.tensor(trg_vector)


# 定义模型
class GRUTranslator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRUTranslator, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden=None):
        embedded = self.embedding(input_seq)
        outputs, hidden = self.gru(embedded, hidden)
        predictions = self.out(outputs)
        return predictions, hidden


# 设置超参数
HIDDEN_SIZE = 256
NUM_LAYERS = 2
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
MAX_LENGTH = 24

# 加载数据并进行预处理
pairs = load_data("cmn.txt")
src_vocab, trg_vocab = create_vocab(pairs)
input_size = len(src_vocab)
output_size = len(trg_vocab)

# 创建数据集和数据加载器
full_dataset = TranslationDataset(pairs, src_vocab, trg_vocab, MAX_LENGTH)
train_data_len = int(0.8 * len(full_dataset))
val_data_len = len(full_dataset) - train_data_len
train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_data_len, val_data_len]
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 初始化模型、损失函数和优化器
model = GRUTranslator(input_size, HIDDEN_SIZE, output_size, NUM_LAYERS)
loss_function = nn.CrossEntropyLoss(ignore_index=src_vocab["<PAD>"])
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

# 将模型转移到GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 训练模型
epoch_bar = tqdm(range(EPOCHS))
for epoch in epoch_bar:
    model.train()
    total_loss = 0
    train_bar = tqdm(train_loader, leave=False)
    for src_batch, trg_batch in train_bar:
        src_batch, trg_batch = src_batch.to(device), trg_batch.to(device)
        optimizer.zero_grad()
        output, _ = model(src_batch)
        loss = loss_function(output.view(-1, output_size), trg_batch.view(-1))
        # loss = loss_function(output, trg_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        train_bar.set_postfix(loss=loss.item())
    epoch_bar.set_description(
        f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(train_loader)}"
    )

    # 验证模型
    model.eval()
    val_loss = 0
    val_bar = tqdm(val_loader, leave=False)
    with torch.no_grad():
        for src_batch, trg_batch in val_bar:
            src_batch, trg_batch = src_batch.to(device), trg_batch.to(device)
            output, _ = model(src_batch)
            loss = loss_function(output.view(-1, output_size), trg_batch.view(-1))
            val_loss += loss.item()
            val_bar.set_postfix(loss=loss.item())
        epoch_bar.set_description(f"Validation Loss: {val_loss/len(val_loader)}")
