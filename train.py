import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import transformers
from types import Union
from transformers import AutoModel, BertTokenizerFast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from utils.preprocessor import Preprocessor
from utils.bag_of_words import BOW
from sklearn.model_selection import train_test_split
from model.model import classifier
from sklearn.metrics import f1_score
from tqdm.notebook import tqdm
from torch.optim import lr_scheduler

class Train():

    def __init__(self,
                 df_train: pd.DataFrame,
                 df_test: pd.DataFrame):
        self.df_train = df_train
        self.df_test = df_test

        self.bert = AutoModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.bow = BOW(df_train,
                       df_test)
        self.history = pd.DataFrame()

    def create_data_loader_train(self,
                           train: pd.DataFrame,
                           batch_size: int = 128):
        tokens_train = self.tokenizer.batch_encode_plus(
            train.text.tolist(),
            max_length=15,
            pad_to_max_length=True,
            truncation=True
        )

        train_seq = torch.tensor(tokens_train['input_ids'])
        train_mask = torch.tensor(tokens_train['attention_mask'])
        train_y = torch.tensor(train.target.tolist())

        train_keyword = torch.tensor(self.bow.get_matrix((train['new_keyword'])))

        train_data = TensorDataset(train_seq, train_mask, train_keyword, train_y)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        return train_dataloader

    def create_test_data(self,
                         test: pd.DataFrame):
        tokens_test = self.tokenizer.batch_encode_plus(
            test.text.tolist(),
            max_length=15,
            pad_to_max_length=True,
        )

        test_seq = torch.tensor(tokens_test['input_ids'])
        test_mask = torch.tensor(tokens_test['attention_mask'])
        test_keyword = torch.tensor(self.bow.get_matrix(test['new_keyword']))
        test_y = torch.tensor(test.target.tolist())

        return test_seq, test_mask, test_keyword, test_y

    def score(self,
              model,
              data,
              labels,
              mask,
              keyword) -> float:

        keyword = keyword.long()
        model.eval()
        f1 = 0
        with torch.no_grad():
            pred = model(data, mask, keyword)
            pred = torch.argmax(pred, dim=1).cpu().detach().numpy()
            f1 = f1_score(pred, labels.cpu().detach().numpy(), average='macro')
        return f1

    def training(self,
                 device=torch.device('cuda'),
                 EPOCHS: int = 15):

        for param in self.bert.parameters():
            param.requires_grad = False

        train, test = train_test_split(self.df_train, random_state=241, test_size=0.15)
        train_dataloader = self.create_data_loader_train(train)
        test_seq, test_mask, test_keyword, test_y = self.create_test_data(test)
        model = classifier(self.bert, n_tokens=self.bow.get_len_tokes()).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.12)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=180, gamma=0.9)
        f1 = 0
        f1_test = []
        loss_score = []
        params_model = 0
        x = []
        count = 0
        for epoch in range(EPOCHS):
            print(f"epoch: {epoch}")
            model.train()
            for i, batch in enumerate(tqdm(train_dataloader)):
                count += 1
                x.append(count)
                batch = [r.to(device) for r in batch]
                sent_id, mask, keyword, labels = batch
                optimizer.zero_grad()
                pred = model(sent_id, mask, keyword)
                labels = labels.long()
                loss = criterion(pred, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                f1_test.append(self.score(model, test_seq.to(device), test_y, test_mask.to(device), test_keyword.to(device)))
                loss_score.append(loss.item())
                if f1 < f1_test[-1]:
                    params_model = model.state_dict()
                    f1 = f1_test[-1]
                i += 1
                if i % 20 == 0:
                    print('loss = %f' % loss.item())
                    print('f1 score test = {}'.format(f1))

        self.history = pd.DataFrame({'step': x, 'f1_test': f1_test, 'loss': loss_score})
        return model, params_model

    def get_history(self):

        return self.history

    def create_submit(self,
                      model,
                      name: str,
                      device=torch.device('cuda')):
        tokens_test_sub = self.tokenizer.batch_encode_plus(
            self.df_test.text.tolist(),
            max_length=15,
            pad_to_max_length=True,
            truncation=True
        )
        test_seq = torch.tensor(tokens_test_sub['input_ids'])
        test_mask = torch.tensor(tokens_test_sub['attention_mask'])
        test_keyword = torch.tensor(self.bow.get_matrix(self.df_test['new_keyword']))

        model.eval()

        predicted = model(test_seq.to(device), test_mask.to(device), test_keyword.to(device))

        df_sub = pd.read_csv('data/sample_submission.csv')
        df_sub.target = torch.argmax(predicted, dim=1).cpu().detach().numpy()

        df_sub.to_csv('data/'+name+'.csv', index=False)

if __name__ == '__main__':
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')

    preprocessor = Preprocessor(df_train, df_test)

    df_train, df_test = preprocessor.preprocessing()

    train_loop = Train(df_train, df_test)

    model, best_params = train_loop.training()

    model.load_state_dict(best_params)

    train_loop.create_submit(model, 'first_sub')




