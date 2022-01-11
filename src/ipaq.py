import tez
import torch
from torch import nn as nn
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn import metrics
from sklearn.model_selection import train_test_split
import ast
import pandas as pd


class BERTDataset:
    def __init__(self, texts, targets, max_len=250):
        self.texts = texts
        self.targets = targets
        self.tokenizers = transformers.BertTokenizer.from_pretrained("bert-base-uncased",
                                                                           do_lower_case=False)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        inputs = self.tokenizers.encode_plus(text,
                                             add_special_tokens=True,
                                             max_length=self.max_len,
                                             padding='max_length',
                                             truncation=True
                                             )
        resp = {
            "ids": torch.tensor(inputs["input_ids"], dtype=torch.long),
            "mask": torch.tensor(inputs["attention_mask"], dtype=torch.long),
            "token_type_ids": torch.tensor(inputs["token_type_ids"], dtype=torch.long),
            "targets": torch.tensor(self.targets[idx], dtype=torch.float)
        }

        return resp


class TextModel(tez.Model):
    def __init__(self, num_classes, num_train_steps):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained("bert-base-uncased", return_dict=False)
        self.bert_drop = nn.Dropout(0.2)
        self.out = nn.Linear(768, num_classes)
        self.num_train_steps = num_train_steps

    def fetch_optimizer(self):
        opt = AdamW(self.parameters(), lr=1e-4)
        return opt

    def fetch_scheduler(self):
        sch = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=self.num_train_steps
        )
        return sch

    def loss(self, outputs, targets):
        return nn.BCEWithLogitsLoss()(outputs, targets.view(-1,1))

    def monitor_metrics(self, outputs, targets):
        outputs = torch.simgoid(outputs, dim=1).cpu().detach().numpy().tolist()

        output = []
        for h in outputs:
            tmp_arr =[]
            for j in h:
                if j > 0.5:
                    tmp_arr.append(1)
                else:
                    tmp_arr.append(0)
            output.append(tmp_arr)

        targets = targets.cpu().detach().numpy().tolist()
        return {"accuracy": metrics.accuracy_score(targets, output)}


    def forward(self, ids, mask, token_type_ids, targets=None):
        _, x = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        x = self.bert_drop(x)
        x = self.out(x)

        if targets is not None:
            loss = self.loss(x, targets)
            met = self.monitor_metrics(x, targets)
            return x, loss, met
        return x, None, {}

def train_model():

    df = pd.read_csv("data/text_data.csv")
    print(df.head(5))
    lab_dict = {'ham':0, 'spam':1}
    df["labels"] = df.Category.apply(lambda x:lab_dict[x])
    print(df)

    train_df, test_df = train_test_split(df, test_size=0.20, random_state=42)

    train_dataset = BERTDataset( train_df.Message.values, train_df.labels.values.tolist())
    valid_dataset = BERTDataset( train_df.Message.values, train_df.labels.values.tolist())

    TRAIN_BS = 8
    EPOCHS = 2

    n_train_steps = int(len(train_df) / TRAIN_BS * EPOCHS)
    model = TextModel(num_classes=1, num_train_steps=n_train_steps)
    model.fit(train_dataset, valid_dataset, device="cpu", epochs=10, n_jobs=1, train_bs=8)


if __name__=="__main__":
    train_model()
