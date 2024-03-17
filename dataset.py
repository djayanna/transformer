import torch
import torch.nn as nn

from torch.utils.data import  Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_source, tokenizer_target, lang_source, lang_target, seq_len):
        self.ds = ds
        self.tokenizer_source = tokenizer_source
        self.tokenizer_target = tokenizer_target
        self.lang_source = lang_source
        self.lang_target = lang_target
        self.seq_len = seq_len

        print(tokenizer_target.token_to_id("[SOS]"))

        self.sos_token = torch.tensor([tokenizer_target.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_target.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_target.token_to_id('[PAD]')], dtype=torch.int64)


    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        source_target_pair = self.ds[index]

        source_text = source_target_pair['translation'][self.lang_source]
        target_text = source_target_pair['translation'][self.lang_target]

        # encode the source and target text
        source_encoded = self.tokenizer_source.encode(source_text).ids
        target_encoded = self.tokenizer_target.encode(target_text).ids

        encoder_num_padding_tokens = self.seq_len - len(source_encoded) - 2 # -2 for [SOS] and [EOS]
        decoder_num_padding_tokens = self.seq_len - len(target_encoded) - 1 # -1 for [EOS]

        if encoder_num_padding_tokens < 0 or decoder_num_padding_tokens < 0:
            raise ValueError("sentence is too long")


        # Add SOS and EOS tokens to the encoder input
        encoder_input = torch.cat(
            [
                self.sos_token, 
                torch.tensor(source_encoded, dtype=torch.int64),
                self.eos_token, 
                torch.tensor([self.pad_token] * encoder_num_padding_tokens, dtype=torch.int64)

            ]
        )

        # Add SOS token to the decoder input
        decoder_input = torch.cat(
            [
                self.sos_token, 
                torch.tensor(target_encoded, dtype=torch.int64),
                torch.tensor([self.pad_token] * decoder_num_padding_tokens, dtype=torch.int64)

            ]
        )

        # Add EOS token to the label (what we expect the model to predict)
        labels = torch.cat(
            [
                torch.tensor(target_encoded, dtype=torch.int64),
                self.eos_token, 
                torch.tensor([self.pad_token] * decoder_num_padding_tokens, dtype=torch.int64)

            ]
        )

        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert labels.size(0) == self.seq_len

        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            "encoder_padding_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_padding_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            'labels': labels, 
            "source_text": source_text,
            "target_text": target_text,
        }

def causal_mask(length):
    mask = torch.ones(1, length, length)
    mask = torch.triu(mask, diagonal=1).type(torch.int)
    return mask == 0 # 1s where the mask is 0, 0s where the mask is 1

    