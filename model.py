import os
import torch
from tqdm import tqdm
from torch import nn, optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from transformers import RobertaModel
from utils import Averager, Recorder, metrics
from scl_loss import SupConLoss
from typing import List, Tuple

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout):
        super(MLP, self).__init__()
        layers = list()
        curr_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, input):
        return self.mlp(input)

class MaskAttention(nn.Module):
    def __init__(self, input_dim):
        super(MaskAttention, self).__init__()
        self.attention_layer = nn.Linear(input_dim, 1)

    def forward(self, input, mask):
        score = self.attention_layer(input).squeeze()
        score = score.masked_fill(mask == 0, float('-inf'))
        score = torch.softmax(score, dim=-1).unsqueeze(1)
        output = torch.matmul(score, input).squeeze(1)
        return output, score

class CNNExtractor(nn.Module):
    def __init__(self, feature_kernel, input_dim):
        super(CNNExtractor, self).__init__()
        self.convs = nn.ModuleList([nn.Conv1d(input_dim, feature_num, kernel_size) for kernel_size, feature_num in feature_kernel.items()])

    def forward(self, input):
        input = input.permute(0, 2, 1)
        feature = [conv(input) for conv in self.convs]
        feature = [torch.max_pool1d(f, f.shape[-1]).squeeze() for f in feature]
        feature = torch.cat(feature, dim=1)
        return feature

class ConvFeatureExtractionModel(nn.Module):

    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        conv_dropout: float = 0.0,
        conv_bias: bool = False,
    ):
        super().__init__()

        def block(n_in, n_out, k, stride=1, conv_bias=False):
            padding = k // 2
            return nn.Sequential(
                nn.Conv1d(in_channels=n_in, out_channels=n_out, kernel_size=k, stride=stride, padding=padding, bias=conv_bias),
                nn.Dropout(conv_dropout),
                nn.ReLU()
            )

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for _, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(in_d, dim, k, stride=stride, conv_bias=conv_bias))
            in_d = dim

    def forward(self, x):
        # x = x.unsqueeze(1)
        for conv in self.conv_layers:
            x = conv(x)
        return x

class Model(nn.Module):
    def __init__(self, n_family=4, emb_dim=768, hidden_dims=[256], dropout=0.2, feature_kernel={1: 64, 2: 64, 3: 64, 5: 64, 10: 64}):
        super(Model, self).__init__()
        self.n_family = n_family

        self.n_feat = 3
        feature_enc_layers = [(64, 5, 1)] + [(128, 3, 1)] * 3 + [(64, 3, 1)]
        self.conv = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            conv_dropout=0.0,
            conv_bias=False,
        )
        embedding_size = self.n_feat * 64

        self.encoder_layer = TransformerEncoderLayer(
            d_model=embedding_size,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer=self.encoder_layer,
                                            num_layers=2)
        seq_len = 256
        self.position_encoding = torch.zeros((seq_len, embedding_size))
        for pos in range(seq_len):
            for i in range(0, embedding_size, 2):
                self.position_encoding[pos, i] = torch.sin(
                    torch.tensor(pos / (10000**((2 * i) / embedding_size))))
                self.position_encoding[pos, i + 1] = torch.cos(
                    torch.tensor(pos / (10000**((2 *
                                                 (i + 1)) / embedding_size))))
        self.norm = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(0.1)

        self.attention = MaskAttention(emb_dim)
        self.family_feature_extractor = MLP(embedding_size, hidden_dims, 128, dropout)
        self.family_classifier = nn.Linear(128, n_family)
        self.expert = nn.ModuleList([CNNExtractor(feature_kernel, embedding_size) for i in range(n_family)])
        mlp_input_shape = sum([feature_num for _, feature_num in feature_kernel.items()])
        self.binary_classifier = MLP(mlp_input_shape, hidden_dims, 1, dropout)

    def conv_feat_extract(self, x):
        out = self.conv(x)
        out = out.transpose(1, 2)
        return out

    def forward(self, prob_feature, feature, mask):
        prob_feature = torch.cat([self.conv_feat_extract(prob_feature[:, i:i+1, :]) for i in range(self.n_feat)], dim=2)  # (batch_size, seq_len, embedding_size)
        prob_feature = prob_feature + self.position_encoding.cuda()
        prob_feature = self.norm(prob_feature)
        prob_feature = self.encoder(prob_feature)
        prob_feature = self.dropout(prob_feature)  # (bs, seq_len, embedding_size)
        prob_feature_mean = torch.mean(prob_feature, dim=1)  # (bs, embedding_size)

        # attention_feature, _ = self.attention(feature, mask)

        family_feature = self.family_feature_extractor(prob_feature_mean)  # (bs, 128)
        pred_family = self.family_classifier(family_feature)
        gate = torch.softmax(pred_family, dim=1)

        shared_feature = sum([self.expert[i](prob_feature) * gate[:, i].unsqueeze(1) for i in range(self.n_family)])
        pred_binary = self.binary_classifier(shared_feature)
        pred_binary = torch.sigmoid(pred_binary).squeeze()

        return pred_binary, pred_family, family_feature

class Trainer:
    def __init__(self, device, pretrain_model, train_dataloader, val_dataloader, test_dataloader, epoch, lr, early_stop, model_save_path, n_family, is_cl, is_binary):
        self.device = device
        self.epoch = epoch
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.early_stop = early_stop
        self.n_family = n_family
        self.pretrain = RobertaModel.from_pretrained(pretrain_model).to(device)
        self.model_save_path = model_save_path
        self.model = Model(n_family=n_family).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.is_clLoss = is_cl
        self.is_binary = is_binary

    def get_loss(self, batch):
        ll_tokens_list = batch['ll_tokens_list'].to(self.device)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        feature = self.pretrain(input_ids, attention_mask).last_hidden_state.detach()
        label_family = batch['label_family'].to(self.device)
        label_binary = batch['label_binary'].to(self.device)

        pred_binary, pred_family, family_feature = self.model(ll_tokens_list, feature, attention_mask)
        if  self.is_clLoss:
            loss = nn.BCELoss()(pred_binary, label_binary.float()) \
                    + nn.CrossEntropyLoss()(pred_family, label_family) \
                    + SupConLoss(temperature=0.1)(family_feature.unsqueeze(dim=-1), label_family)
        else:
            loss = nn.BCELoss()(pred_binary, label_binary.float()) \
                    + nn.CrossEntropyLoss()(pred_family, label_family)
        return loss

    def get_output(self, batch):
        if self.is_binary:
            ll_tokens_list = batch['ll_tokens_list'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            feature = self.pretrain(input_ids, attention_mask).last_hidden_state.detach()
            with torch.no_grad():
                output, _, _ = self.model(ll_tokens_list, feature, attention_mask)
            return output
        else:
            ll_tokens_list = batch['ll_tokens_list'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            feature = self.pretrain(input_ids, attention_mask).last_hidden_state.detach()
            with torch.no_grad():
                output, pred_family, _ = self.model(ll_tokens_list, feature, attention_mask)
            return output, pred_family



    def train(self):
        recorder = Recorder(self.early_stop)
        for epoch in range(self.epoch):
            print('----epoch %d----' % (epoch+1))
            self.model.train()
            avg_loss = Averager()
            for i, batch in enumerate(tqdm(self.train_dataloader)):
                self.optimizer.zero_grad()
                loss = self.get_loss(batch)
                loss.backward()
                self.optimizer.step()
                avg_loss.add(loss.item())

            results = self.test(self.val_dataloader)
            print('epoch %d: loss = %.4f, acc = %.4f, macf1 = %.4f, auc_ovo = %.4f' % (epoch+1, avg_loss.get(), results['accuracy'], results['macf1'], results['auc_ovo']))

            # early stop
            decision = recorder.update(results['macf1'])
            if decision == 'save':
                torch.save(self.model.state_dict(), self.model_save_path)
            elif decision == 'stop':
                break
            elif decision == 'continue':
                continue
            torch.save(self.model.state_dict(), self.model_save_path)

        # load best model
        self.model.load_state_dict(torch.load(self.model_save_path))
        print('----test----')
        results = self.test(self.test_dataloader)
        print('test: acc = %.4f, macf1 = %.4f, auc_ovo = %.4f' % (results['accuracy'], results['macf1'], results['auc_ovo']))
        print(results)

    def test(self, dataloader):
        self.model.eval()
        y_true = torch.empty(0)
        y_score = torch.empty(0)
        if self.is_binary:
            for i, batch in enumerate(tqdm(dataloader)):
                output = self.get_output(batch).cpu()
                y_score = torch.cat((y_score, output))
                y_true = torch.cat((y_true, batch['label_binary']))

            results = metrics(y_true, y_score)
            return results
        else:
            y_true_family = torch.empty(0)
            y_score_family = torch.empty((0, 5))
            for i, batch in enumerate(tqdm(dataloader)):
                output, pred_family = self.get_output(batch)
                output = output.cpu()
                pred_family = pred_family.cpu()
                y_score = torch.cat((y_score, output))
                y_true = torch.cat((y_true, batch['label_binary']))
                y_score_family = torch.cat((y_score_family, pred_family))
                y_true_family = torch.cat((y_true_family, batch['label_family']))

            results = metrics(y_true_family, y_score_family)
            return results
        
            # inference with binary logits
    def inference(self, dataloader, threshold=0.5):
        predictions = []
        texts = []
        label_binarys = []
        probs_list = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader)):
               batch_texts = batch['text']
               batch_label_binary = batch['label_binary'].to(self.device)
               output = self.get_output(batch)
               # 转换为CPU并应用阈值
               pred_probs = output.cpu().numpy()
               batch_predictions = ['generated' if prob >= threshold else 'human' for prob in pred_probs]
               probs_list.extend(pred_probs.tolist())
               predictions.extend(batch_predictions)
               texts.extend(batch_texts)
               label_binarys.extend(batch_label_binary.cpu().tolist())
            
        return texts, predictions, label_binarys, probs_list

       
