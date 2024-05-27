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
import random
import os
from torchtext.data.metrics import bleu_score

print("modules imported")

params = {
    "hidden_size": 768,
    "embd_size": 768,
    "num_layers": 1,
    "batch_size": 128,
    "learning_rate": 0.001,
    "epochs": 20,
    "max_length": 24,
}
params.update(nni.get_next_parameter())


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
    pairs = [line.split("\t") for line in lines]
    for pair in pairs:
        if SRC_LANG == "cn":
            pair[0], pair[1] = pair[1], pair[0]
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
    if os.path.exists(f"vocab_raw.pt"):
        src_vocab, trg_vocab = torch.load(f"vocab_raw.pt")
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
        torch.save((src_vocab, trg_vocab), f"vocab_raw.pt")

    # write vocab count to csv
    with open("src_vocab.csv", "w", encoding="utf-8") as f:
        for _, count in src_vocab.items():
            f.write(f"{count}\n")
    with open("trg_vocab.csv", "w", encoding="utf-8") as f:
        for _, count in trg_vocab.items():
            f.write(f"{count}\n")

    print(f"raw src vocab size: {len(src_vocab)}")
    print(f"raw trg vocab size: {len(trg_vocab)}")

    # remove words that appear <= 2 times
    src_vocab = {word: count for word, count in src_vocab.items() if count > 3}
    trg_vocab = {word: count for word, count in trg_vocab.items() if count > 3}
    src_vocab = ["<SOS>", "<EOS>", "<PAD>"] + list(sorted(src_vocab.keys()))
    trg_vocab = ["<SOS>", "<EOS>", "<PAD>"] + list(sorted(trg_vocab.keys()))

    print(f"src vocab size: {len(src_vocab)}")
    print(f"trg vocab size: {len(trg_vocab)}")

    src_word_to_index = {word: index for index, word in enumerate(src_vocab)}
    trg_word_to_index = {word: index for index, word in enumerate(trg_vocab)}
    return src_vocab, trg_vocab, src_word_to_index, trg_word_to_index


# 将句子转换为向量
def tokens_to_vector(tokens, vocab, max_length, lang):
    vector = []
    vector.append(vocab["<SOS>"])
    for word in tokens[: max_length - 2]:
        if word in vocab:
            vector.append(vocab[word])
        else:
            vector.append(vocab["<UNK>"])
    vector.append(vocab["<EOS>"])
    vector += [vocab["<PAD>"]] * (max_length - len(vector))
    return vector


def vector_to_sentence(vector, vocab, lang):
    if lang == "cn":
        sentence = (
            "".join([vocab[id] for id in vector[1:]])
            .split("<PAD>")[0]
            .split("<EOS>")[0]
        )
    elif lang == "en":
        sentence = (
            " ".join([vocab[id] for id in vector[1:]])
            .split("<PAD>")[0]
            .split("<EOS>")[0]
        )
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
        if os.path.exists(f"tokenize_pairs.pt"):
            self.tokenize_pairs = torch.load(f"tokenize_pairs.pt")
        else:
            self.tokenize_pairs = []
            for src_sentence, trg_sentence in tqdm(pairs):
                src_tokens = tokenize_sentence(src_sentence, SRC_LANG)
                trg_tokens = tokenize_sentence(trg_sentence, TRG_LANG)

                # remove pairs with unknown words in trg
                has_unk = False
                for trg_token in trg_tokens:
                    if trg_token not in trg_vocab:
                        has_unk = True
                        break
                for src_token in src_tokens:
                    if src_token not in src_vocab:
                        has_unk = True
                        break
                if has_unk:
                    continue

                src_vector = tokens_to_vector(
                    src_tokens, src_vocab, max_length, SRC_LANG
                )
                trg_vector = tokens_to_vector(
                    trg_tokens, trg_vocab, max_length, TRG_LANG
                )
                self.tokenize_pairs.append((src_vector, trg_vector))

            torch.save(self.tokenize_pairs, f"tokenize_pairs.pt")

    def __len__(self):
        return len(self.tokenize_pairs)

    def __getitem__(self, index):
        src_vector, trg_vector = self.tokenize_pairs[index]
        return torch.tensor(src_vector), torch.tensor(trg_vector)


# 定义模型
class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, hidden_size, dropout=0.5):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(src_vocab_size, embed_size, padding_idx=2)
        self.gnu = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src [batch seq_len]
        x_embeding = self.dropout(self.embedding(src))  # [batch, seq_len, embed_size]
        _, h_n = self.gnu(x_embeding)
        return h_n


class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, hidden_size, dropout=0.5):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(trg_vocab_size, embed_size, padding_idx=2)

        self.gnu = nn.GRU(embed_size + hidden_size, hidden_size, batch_first=True)
        self.classify = nn.Linear(embed_size + hidden_size * 2, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg_i, context, h_n):
        # trg_i为某一时间步词的输入，[bacth_size]
        # context为原始上下文向量，[bacth_size,1,hidden_size]
        # h_n为上一时间布的隐状态[1，batch_size，hidden_size]
        trg_i = trg_i.unsqueeze(1)
        # trg_i[bacth_size,1]
        trg_i_embed = self.dropout(self.embedding(trg_i))
        # trg_i_embed [bacth_size,1,embed_size]

        # 输入rnn模块的不仅仅只有词嵌入和上一时间步的隐状态，还有原始上下向量
        input = torch.cat((trg_i_embed, context), dim=2)
        # input[bacth_size,1,embed_size+hidden_size]
        output, h_n = self.gnu(input, h_n)
        # output[batch_size,1,hidden_size]
        # h_n[1,batch_size,hidden_size]

        # 原本rnn模型的输入直接带入线性分类层映射到英语空间中，这里新添原始词嵌入和原始上下文向量，即上面的input
        input = input.squeeze()
        output = output.squeeze()
        # input[bacth_size embed_size+hidden_size]
        # output[batch_szie hidden_size]
        input = torch.cat((input, output), dim=1)
        output = self.classify(input)
        # output[bacth trg_vocab_size]
        return output, h_n


class GRUTranslator(nn.Module):
    def __init__(self, src_vocab_size, embed_size, hidden_size, trg_vocab_size):
        super(GRUTranslator, self).__init__()
        self.encoder = Encoder(src_vocab_size, embed_size, hidden_size)
        self.decoder = Decoder(trg_vocab_size, embed_size, hidden_size)
        self.trg_vocab_size = trg_vocab_size

    def forward(self, src, trg, teach_threshold=0.5):
        # src[batch seq_len]
        # trg[bacth seq_len]

        trg_seq_len = src.shape[1]

        batch_size = src.shape[0]

        outputs_save = torch.zeros(batch_size, trg_seq_len, self.trg_vocab_size).cuda()

        h_n = self.encoder(src)  # [1,batch_size,hidden_size]
        context = h_n.permute(1, 0, 2)  # [batch_size,1,hidden_size]
        input = trg[:, 0]

        for t in range(1, trg_seq_len):
            output, h_n = self.decoder(input, context, h_n)
            outputs_save[:, t, :] = output
            probability = random.random()
            # 是否采用强制教学
            if probability < teach_threshold:
                input = trg[:, t]
            else:
                input = output.argmax(1)
        return outputs_save


def test_model():
    model.eval()
    # 任取100个样本进行测试
    test_loader = DataLoader(val_dataset, batch_size=100, shuffle=True)
    src_batch, trg_batch = next(iter(test_loader))
    src_batch, trg_batch = src_batch.to(device), trg_batch.to(device)
    with torch.no_grad():
        output = model(src_batch, trg_batch, 0)
        output = output.argmax(dim=-1)
    for i in range(15):
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
EMBD_SIZE = int(params["embd_size"])
NUM_LAYERS = int(params["num_layers"])
BATCH_SIZE = params["batch_size"]
LEARNING_RATE = params["learning_rate"]
EPOCHS = params["epochs"]
MAX_LENGTH = params["max_length"]

SRC_LANG = "cn"
TRG_LANG = "en"

# 加载数据并进行预处理
print("loading data")
pairs = load_data("cmn_zhsim.txt")
print("creating vocab")
src_id2word, trg_id2word, src_word2id, trg_word2id = create_vocab(pairs)
input_size = len(src_id2word)
output_size = len(trg_id2word)
print("input size: ", input_size)
print("output size: ", output_size)

# 创建数据集和数据加载器
print("creating dataset and dataloader")
full_dataset = TranslationDataset(pairs, src_word2id, trg_word2id, MAX_LENGTH)
print("dataset size: ", len(full_dataset))
train_data_len = int(0.8 * len(full_dataset))
val_data_len = len(full_dataset) - train_data_len
train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_data_len, val_data_len]
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 初始化模型、损失函数和优化器
model = GRUTranslator(input_size, EMBD_SIZE, HIDDEN_SIZE, output_size)
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

        output = model(src_batch, trg_batch)
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
            output = model(src_batch, trg_batch, 0)
            loss = loss_function(output.view(-1, output_size), trg_batch.view(-1))
            val_loss += loss.item()
            val_bar.set_postfix(loss=loss.item())

            # bleu score:
            batch_size = src_batch.shape[0]
            output = output.argmax(dim=-1)
            bleu_sum = 0
            for i in range(batch_size):
                trg_sentence = vector_to_sentence(
                    trg_batch[i], trg_id2word, TRG_LANG
                ).split()
                pred_sentence = vector_to_sentence(
                    output[i], trg_id2word, TRG_LANG
                ).split()
                bleu = bleu_score([pred_sentence], [[trg_sentence]])
                bleu_sum += bleu
            bleu = bleu_sum / batch_size

        val_loss /= len(val_loader)
        epoch_bar.set_description(
            f"Train Loss: {total_loss/len(train_loader) :.3f}, Val Loss: {val_loss :.3f}, Bleu: {bleu:.3f}"
        )
    if val_loss < best_loss:
        best_loss = val_loss
        best_model_state = model.state_dict().copy()
        # torch.save(best_model_state, "best_model.pt")

    if val_loss > last_loss:
        worse_count += 1
    else:
        worse_count = 0

    last_loss = val_loss

    if worse_count >= 2:
        break

# 测试模型
model.load_state_dict(best_model_state)
test_model()
print("best val loss: ", best_loss)
nni.report_final_result(best_loss)
