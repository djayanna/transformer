import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from datasets import load_dataset
from model import build_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
from tqdm import tqdm

from dataset import BilingualDataset, causal_mask

from config import get_weights_file_path, get_config

from torch.utils.tensorboard import SummaryWriter

import warnings



def get_or_build_tokenizer(config, ds, lang):
    
    tokenizer_path = Path(config['tokenizer_file'].format(lang)) # Path to the tokenizer file for the language
    if tokenizer_path.exists():
        print(f"loading tokenizer from {tokenizer_path}")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else: 
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]")) # replace unknown tokens with [UNK]
        tokenizer.pre_tokenizer = Whitespace() # split by space
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[SOS]", "[EOS]", "[PAD]"], min_frequency=2) # build trainer
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer = trainer)
        tokenizer.save(str(tokenizer_path))

    return tokenizer

def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang] 

def get_ds(config):
    ds_raw = load_dataset("opus_books", f'{config["lang_source"]}-{config["lang_target"]}', split="train")

    # build tokenizers
    tokenizer_source = get_or_build_tokenizer(config, ds_raw, config['lang_source'])
    tokenizer_target = get_or_build_tokenizer(config, ds_raw, config['lang_target'])

    # 90% train, 10% test/validation
    train_ds_size = int(len(ds_raw) * 0.9)
    validation_ds_size = len(ds_raw) - train_ds_size

    ds_train, ds_validation = random_split(ds_raw, [train_ds_size, validation_ds_size])

    train_ds = BilingualDataset(ds_train, tokenizer_source, tokenizer_target, config['lang_source'], config['lang_target'], config['seq_len'])
    validation_ds = BilingualDataset(ds_validation, tokenizer_source, tokenizer_target, config['lang_source'], config['lang_target'], config['seq_len'])

    max_length_source = 0
    max_length_target = 0

    for item in ds_raw:
        source_ids = tokenizer_source.encode(item['translation'][config['lang_source']]).ids
        target_ids = tokenizer_target.encode(item['translation'][config['lang_target']]).ids
        max_length_source = max(max_length_source, len(source_ids))
        max_length_target = max(max_length_target, len(target_ids))
    
    print(f"max length of source sentence: {max_length_source}")
    print(f"max length of target sentence: {max_length_target}")

    train_data_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    validation_data_loader = DataLoader(validation_ds, batch_size=1, shuffle=True)

    return train_data_loader, validation_data_loader, tokenizer_source, tokenizer_target

def get_model(config, vocab_src_len, vocab_target_len):
    model = build_model(vocab_src_len, vocab_target_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model


def train_model(config):
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")


    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_data_loader, validation_data_loader, tokenizer_source, tokenizer_target = get_ds(config)
    model = get_model(config, tokenizer_source.get_vocab_size(), tokenizer_target.get_vocab_size()).to(device)

    # Tensorboard

    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], eps=1e-9)


    inital_epoch = 0
    global_step = 0

    if config['preload']:
        model_filename = get_weights_file_path(config, config['preload'])
        print(f"loading model from {model_filename}")
        state = torch.load(model_filename)
        inital_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
      
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_target.token_to_id("[PAD]"), label_smoothing=0.1).to(device)

    for epoch in range(inital_epoch, config['num_epochs']):
        print(f"epoch: {epoch}")
        model.train()
        batch_iterator = tqdm(train_data_loader, desc=f"processing epoch: {epoch}")

        for batch in batch_iterator:
            
            encoder_input = batch['encoder_input'].to(device) # batch, seq_len
            decoder_input = batch['decoder_input'].to(device) # batch, seq_len
            encoder_mask = batch['encoder_padding_mask'].to(device) # batch, 1, 1, seq_len
            decoder_mask = batch['decoder_padding_mask'].to(device) # batch, 1, seq_len, seq_len - different mask


            encoder_output = model.encode(encoder_input, encoder_mask) # batch, seq_len, d_model
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # batch, seq_len, d_model

            projection = model.projection(decoder_output) # batch, seq_len, target_vocab_size - map to the target vocab

            label = batch['labels'].to(device) # batch, seq_len

            # # batch, seq_len, target_vocab_size -> batch * seq_len, target_vocab_size
            loss = loss_fn(projection.view(-1, tokenizer_target.get_vocab_size()), label.view(-1))

            batch_iterator.set_postfix({f'loss': f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('training loss', loss.item(), global_step)
            writer.flush()

            # backprop the loss
            loss.backward()

            # update the weights
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1  

        # save model
        model_filename = get_weights_file_path(config, f'{epoch:02d}')

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(), # weights of model
            'optimizer_state_dict': optimizer.state_dict(), # weights of optimizer
            'global_step': global_step # global step
        }, model_filename)
            

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)