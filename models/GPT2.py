from transformers import GPT2Model
import torch.nn as nn
import torch
from peft import LoraConfig, get_peft_model
import math
from einops import rearrange
from embed import PositionalEmbedding, TimeSeriesEmbedding


class GPT4TS(nn.Module):
    def __init__(self, args, device="cuda:0"):
        super(GPT4TS, self).__init__()
        print(args.lora, args.lstm)
        # GPT2 Model
        self.gpt2 = GPT2Model.from_pretrained(
            "./llm/gpt2", output_attentions=True, output_hidden_states=True
        )
        self.gpt2.h = self.gpt2.h[:args.gpt_layers]
        print("GPT2 Loaded with {} layers".format(args.gpt_layers))

        # LoRA 配置
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["attn.c_attn", "attn.c_proj"],
            lora_dropout=0.1,
            bias="none",
        )

        if args.lora == True:
            self.gpt2 = get_peft_model(self.gpt2, lora_config)
        for name, param in self.gpt2.named_parameters():
            if "ln" in name or "wpe" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.pos_encoder = PositionalEmbedding(self.gpt2.config.n_embd)

        self.patch_size = args.patch_size
        self.stride = args.stride
        self.patch_num = (args.seq_len - self.patch_size) // self.stride + 1
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1

        self.d_model = args.d_model
        self.d_LLM = args.d_LLM
        self.patch = args.patch
        
        self.isconv = args.isconv

        self.start_linear = nn.Linear(self.patch_size, args.d_model)
        if args.patch:
            self.TSEmbedding = TimeSeriesEmbedding(args.window, args.cycle_len, args.d_model, args.d_LLM, device)
        else:
            self.TSEmbedding = TimeSeriesEmbedding(args.window, args.cycle_len, args.seq_len, args.d_LLM, device)
        # LSTM 和分类器
        if args.patch:
            self.lstm = nn.LSTM(
                input_size=args.d_LLM*self.patch_num,
                hidden_size=256,
                num_layers=2,
                bidirectional=True,
                batch_first=True,
                dropout=0.3
            )
        else:
            self.lstm = nn.LSTM(
                input_size=args.d_LLM,
                hidden_size=256,
                num_layers=2,
                bidirectional=True,
                batch_first=True,
                dropout=0.3
            )
        if args.lstm == True:
            self.classifier = nn.Linear(256*2, args.num_classes)
        else:
            self.classifier = nn.Linear(
                self.gpt2.config.n_embd, args.num_classes)
        self.embedding_layer = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Conv1d(in_channels=64, out_channels=args.d_LLM,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(args.d_LLM),
            nn.ReLU(),
            nn.Dropout(0.2), 
        )

    def forward(self, x, lstm=True):
        if self.isconv:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True,
                            unbiased=False) + 1e-5).detach()
            x /= stdev
            
            x = x.permute(0, 2, 1)
            x = self.embedding_layer(x)
            x = x.permute(0, 2, 1)
            
            output = self.gpt2(inputs_embeds=x).last_hidden_state
            output = output * stdev
            output = output + means
            
            if lstm:
                lstm_output, _ = self.lstm(output)
                # print(lstm_output.shape)
                cls_embedding = lstm_output[:, 0, :]
            else:
                cls_embedding = output[:, 0, :]
            logits = self.classifier(cls_embedding)
            return logits
        else:    
            B, L, M = x.shape  # (B, 100, input_dim)
            B_ori = B
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True,
                            unbiased=False) + 1e-5).detach()
            x /= stdev
            x = rearrange(x, 'b l m -> b m l')
            if self.patch:
                x = self.padding_patch_layer(x)
                x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
                x = rearrange(x, 'b m n p -> (b m) n p')
            B, N, P = x.shape
            if self.patch:
                x = x.reshape(B * N, P)
                x = self.start_linear(x)
                x = x.reshape(B, N, self.d_model)  # b,n,256

            x = self.TSEmbedding(x)
            # x = x.unsqueeze(1)

            # 位置编码 + GPT2
            # x_embed = self.pos_encoder(x)
            # print(x.shape)
            output = self.gpt2(inputs_embeds=x).last_hidden_state
            output = output.reshape(B, -1)  # (b m) (n 768)
            output = output.reshape(B_ori, -1, M)
            output = output * stdev
            output = output + means  # b n*768 m

            output = rearrange(output, 'b l m -> (b m) l')
            output = output.unsqueeze(1)
            # LSTM 或直接分类
            if lstm:
                lstm_output, _ = self.lstm(output)
                # print(lstm_output.shape)
                cls_embedding = lstm_output[:, 0, :]
            else:
                cls_embedding = output[:, 0, :]
            logits = self.classifier(cls_embedding)
            return logits
