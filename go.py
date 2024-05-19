print("importing modules")
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import torch.nn.functional as F
import nni
import jieba
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from opencc import OpenCC
from datasets import load_dataset
from torch.cuda.amp import autocast
import os

print("modules imported")

params = {
    "hidden_size": 512,
    "num_layers": 1,
    "batch_size": 128,
    "learning_rate": 0.001,
    "epochs": 10,
    "max_length": 24,
}
params.update(nni.get_next_parameter())


def load_data_manythings(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
    pairs = [line.split("\t")[:2] for line in lines]
    cc = OpenCC("t2s")
    for pair in pairs:
        pair[1] = cc.convert(pair[1])
        if SRC_LANG == "cn":
            pair[0], pair[1] = pair[1], pair[0]
    return pairs


def load_data_wmt():
    dataset = load_dataset("wmt/wmt19", "zh-en")
    if os.path.exists(f"pairs_{DATASET}.pt"):
        pairs = torch.load(f"pairs_{DATASET}.pt")
    else:
        pairs = []
        bar = tqdm(dataset["train"])
        for example in bar:
            en, zh = example["translation"]["en"], example["translation"]["zh"]
            if (
                len(tokenize_sentence(en, "en")) < 128
                and len(tokenize_sentence(zh, "cn")) < 128
            ):
                pairs.append(
                    (
                        example["translation"]["en"],
                        example["translation"]["zh"],
                    )
                )
                pair_num = len(pairs)
                bar.set_description(f"pairs: {pair_num}")
                if pair_num >= 1000000:
                    break
        torch.save(pairs, f"pairs_{DATASET}.pt")
    return pairs


def tokenize_sentence(text, lang):
    if lang == "en":
        return word_tokenize(text.lower())
    elif lang == "cn":
        return jieba.lcut(text)
    else:
        return None


# 创建词汇表
def create_vocab(pairs):
    src_vocab = {}
    trg_vocab = {}
    if os.path.exists(f"vocab_raw_{DATASET}.pt"):
        src_vocab, trg_vocab = torch.load(f"vocab_raw_{DATASET}.pt")
    else:
        for pair in tqdm(pairs):
            src, trg = pair
            for token in tokenize_sentence(src, SRC_LANG):
                if token not in src_vocab:
                    src_vocab[token] = 1
                else:
                    src_vocab[token] += 1
            for token in tokenize_sentence(trg, TRG_LANG):
                if token not in trg_vocab:
                    trg_vocab[token] = 1
                else:
                    trg_vocab[token] += 1
        torch.save((src_vocab, trg_vocab), f"vocab_raw_{DATASET}.pt")

    # write vocab count to csv
    with open("src_vocab.csv", "w", encoding="utf-8") as f:
        for _, count in src_vocab.items():
            f.write(f"{count}\n")
    with open("trg_vocab.csv", "w", encoding="utf-8") as f:
        for _, count in trg_vocab.items():
            f.write(f"{count}\n")

    # remove words that appear <= 2 times
    src_vocab = {word: count for word, count in src_vocab.items() if count > 2}
    trg_vocab = {word: count for word, count in trg_vocab.items() if count > 2}
    src_vocab = ["<PAD>", "<UNK>"] + list(sorted(src_vocab.keys()))
    trg_vocab = ["<PAD>", "<UNK>"] + list(sorted(trg_vocab.keys()))
    src_word_to_index = {word: index for index, word in enumerate(src_vocab)}
    trg_word_to_index = {word: index for index, word in enumerate(trg_vocab)}
    return src_vocab, trg_vocab, src_word_to_index, trg_word_to_index


# 将句子转换为向量
def sentence_to_vector(sentence, vocab, max_length, lang):
    vector = []
    for word in tokenize_sentence(sentence, lang)[:max_length]:
        if word in vocab:
            vector.append(vocab[word])
        else:
            vector.append(vocab["<UNK>"])
    vector += [vocab["<PAD>"]] * (max_length - len(vector))
    return vector


def vector_to_sentence(vector, vocab, lang):
    if lang == "cn":
        sentence = "".join([vocab[id] for id in vector]).split("<PAD>")[0]
    elif lang == "en":
        sentence = " ".join([vocab[id] for id in vector]).split("<PAD>")[0]
    else:
        sentence = None
    return sentence


# 定义数据集类
class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, trg_vocab, max_length):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_length = max_length
        if os.path.exists(f"tokenize_pairs_{DATASET}.pt"):
            self.tokenize_pairs = torch.load(f"tokenize_pairs_{DATASET}.pt")
        else:
            self.tokenize_pairs = [
                (
                    sentence_to_vector(src_sentence, src_vocab, max_length, SRC_LANG),
                    sentence_to_vector(trg_sentence, trg_vocab, max_length, TRG_LANG),
                )
                for src_sentence, trg_sentence in tqdm(pairs)
            ]
            torch.save(self.tokenize_pairs, f"tokenize_pairs_{DATASET}.pt")

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


def test_model():
    model.eval()
    # 任取100个样本进行测试
    test_loader = DataLoader(val_dataset, batch_size=100, shuffle=True)
    src_batch, trg_batch = next(iter(test_loader))
    src_batch, trg_batch = src_batch.to(device), trg_batch.to(device)
    with torch.no_grad():
        output, _ = model(src_batch)
        output = output.argmax(dim=-1)
    for i in range(100):
        src_sentence = vector_to_sentence(src_batch[i], src_id2word, SRC_LANG)
        trg_sentence = vector_to_sentence(trg_batch[i], trg_id2word, TRG_LANG)
        pred_sentence = vector_to_sentence(output[i], trg_id2word, TRG_LANG)
        print(f"src: {src_sentence}")
        print(f"trg: {trg_sentence}")
        print(f"pred: {pred_sentence}")
        print()


torch.manual_seed(0)

# 设置超参数
HIDDEN_SIZE = int(params["hidden_size"])
NUM_LAYERS = int(params["num_layers"])
BATCH_SIZE = params["batch_size"]
LEARNING_RATE = params["learning_rate"]
EPOCHS = params["epochs"]
MAX_LENGTH = params["max_length"]

SRC_LANG = "en"
TRG_LANG = "cn"

DATASET = "manythings"

# 加载数据并进行预处理
print("loading data")
if DATASET == "manythings":
    pairs = load_data_manythings("cmn.txt")
elif DATASET == "wmt":
    pairs = load_data_wmt()
print("creating vocab")
src_id2word, trg_id2word, src_word2id, trg_word2id = create_vocab(pairs)
input_size = len(src_id2word)
output_size = len(trg_id2word)
print("input size: ", input_size)
print("output size: ", output_size)

# 创建数据集和数据加载器
print("creating dataset and dataloader")
full_dataset = TranslationDataset(pairs, src_word2id, trg_word2id, MAX_LENGTH)
train_data_len = int(0.8 * len(full_dataset))
val_data_len = len(full_dataset) - train_data_len
train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_data_len, val_data_len]
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 初始化模型、损失函数和优化器
model = GRUTranslator(input_size, HIDDEN_SIZE, output_size, NUM_LAYERS)
loss_function = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

# 将模型转移到GPU
device = "cuda"
model.to(device)

# 训练模型
print("training model")
worse_count = 0
best_loss = float("inf")
last_loss = float("inf")
epoch_bar = tqdm(range(EPOCHS), leave=True)
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
        val_loss /= len(val_loader)
        epoch_bar.set_description(
            f"Train Loss: {total_loss/len(train_loader) :.2f}, Val Loss: {val_loss :.2f}"
        )
    if val_loss < best_loss:
        best_loss = val_loss
        best_model_state = model.state_dict().copy()
        # torch.save(best_model_state, "best_model.pt")

    if val_loss > last_loss:
        worse_count += 1

    last_loss = val_loss

    if worse_count >= 2:
        break

# 测试模型
model.load_state_dict(best_model_state)
test_model()
print("best val loss: ", best_loss)
nni.report_final_result(best_loss)
