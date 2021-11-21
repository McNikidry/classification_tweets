import torch
import torch.nn as nn

class classifier(nn.Module):

    def __init__(self, bert,
                 n_tokens: int,
                 emb_size: int = 30,
                 n_features: int = 768,
                 hid_size: int = 512,
                 n_class: int = 2):
        super(classifier, self).__init__()

        self.bert = bert

        self.linear_bert = nn.Sequential(nn.Linear(n_features, hid_size * 2),
                                         nn.LeakyReLU(inplace=True),
                                         nn.Dropout(p=0.2),
                                         nn.Linear(hid_size * 2, hid_size * emb_size),
                                         nn.LeakyReLU(inplace=True),
                                         nn.Dropout(p=0.2)
                                         )

        self.linear_keyword = nn.Sequential(nn.Embedding(num_embeddings=n_tokens, embedding_dim=emb_size),
                                            nn.Linear(emb_size, hid_size * 2),
                                            nn.LeakyReLU(inplace=True),
                                            nn.Dropout(p=0.2),
                                            nn.Linear(hid_size * 2, hid_size),
                                            nn.LeakyReLU(inplace=True),
                                            nn.Dropout(p=0.2))

        self.classifier = nn.Sequential(nn.Linear(15872, hid_size),
                                        nn.LeakyReLU(inplace=True),
                                        nn.Dropout(p=0.2),
                                        nn.Linear(hid_size, n_class))

    def forward(self, sent_id, mask, keyword):
        bert_linear = self.linear_bert(self.bert(sent_id, attention_mask=mask)['pooler_output'])
        keyword_linear = self.linear_keyword(keyword)
        keyword_linear = keyword_linear.view(keyword_linear.shape[0], keyword_linear.shape[1] * keyword_linear.shape[2])
        out = torch.cat([bert_linear, keyword_linear], 1)
        return self.classifier(out)