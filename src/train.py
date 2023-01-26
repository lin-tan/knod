import sys
import os
import time
import math
import torch
import torch.nn as nn

SRC_DIR = os.path.abspath(__file__)
SRC_DIR = SRC_DIR[: SRC_DIR.rfind('/') + 1]
sys.path.append(SRC_DIR + '/dataset/')
sys.path.append(SRC_DIR + '/models/')

from vocabulary import NodeVocabulary, EdgeVocabulary
from graph_transformer_dataset import GraphTransformerDataset
from graph_transformer import GraphTransformer, Config


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps,
                                    num_cycles=0.5, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def validate_step(epoch, valid_loader, model, save_dir=None):
    print('start validation')
    start_time = time.time()
    valid_loss = []
    valid_father_rule_loss, valid_father_loss = [], []
    valid_edge_rule_loss, valid_edge_loss = [], []
    valid_node_loss = []
    valid_oom, valid_nan = 0, 0

    model.zero_grad()
    model.eval()
    with torch.no_grad():
        for batch in valid_loader:
            batch = {
                k: v.squeeze(0).to(devices_id[0]) for (k, v) in batch.items()
            }
            try:
                father_rule_loss, father_loss, edge_rule_loss, edge_loss, node_loss = model(batch)
                loss = (father_rule_loss + father_loss) * 0.1 + (edge_rule_loss + edge_loss) * 0.1 + node_loss * 0.9
                valid_father_rule_loss.append(father_rule_loss.mean().item())
                valid_father_loss.append(father_loss.mean().item())
                valid_edge_rule_loss.append(edge_rule_loss.mean().item())
                valid_edge_loss.append(edge_loss.mean().item())
                valid_node_loss.append(node_loss.mean().item())
                valid_loss.append(loss.mean().item())
                del batch, father_rule_loss, father_loss, edge_rule_loss, edge_loss, node_loss, loss
            except Exception as e:
                print(str(e))
                if 'out of memory' in str(e):
                    valid_oom += 1
                if 'loss is nan' in str(e):
                    valid_nan += 1
                torch.cuda.empty_cache()

        info = "val_loss: {} ({}, {}, {}, {}, {}), oom: {}, nan:{}, time: {}s".format(
            round(float(sum(valid_loss) / len(valid_loss)), 5),
            round(float(sum(valid_father_rule_loss) / len(valid_father_rule_loss)), 5),
            round(float(sum(valid_father_loss) / len(valid_father_loss)), 5),
            round(float(sum(valid_edge_rule_loss) / len(valid_edge_rule_loss)), 5),
            round(float(sum(valid_edge_loss) / len(valid_edge_loss)), 5),
            round(float(sum(valid_node_loss) / len(valid_node_loss)), 5),
            valid_oom, valid_nan, int(time.time() - start_time)
        )
        print(info)
        if save_dir:
            checkpoint = {
                'model': model.module.state_dict(),
                'config': model.module.config.to_dict(),
                'val_loss': round(float(sum(valid_loss) / len(valid_loss)), 5)
            }
            torch.save(checkpoint, save_dir + '/' + str(epoch + 1) + '.pt')
            print('saved')
    model.train()


def train(model, train_dataset, valid_dataset, epochs, save_dir):
    train_sampler = torch.utils.data.SequentialSampler(train_dataset,)
    valid_sampler = torch.utils.data.SequentialSampler(valid_dataset)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True, sampler=train_sampler
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset, batch_size=1, shuffle=False,
        num_workers=0, pin_memory=True, sampler=valid_sampler
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=2e3,
                                                num_training_steps=len(train_loader) * epochs)

    print('model parameters:', sum(param.numel() for param in model.parameters()))
    model = nn.DataParallel(model, device_ids=devices_id).to(devices_id[0])
    for epoch in range(epochs):
        model.train()
        train_loss = []
        train_father_rule_loss, train_father_loss = [], []
        train_edge_rule_loss, train_edge_loss = [], []
        train_node_loss = []
        oom, nan = 0, 0
        start_time = time.time()
        for i, batch in enumerate(train_loader):
            batch = {
                k: v.squeeze(0).to(devices_id[0]) for (k, v) in batch.items()
            }
            try:
                model.zero_grad()
                optimizer.zero_grad()
                father_rule_loss, father_loss, edge_rule_loss, edge_loss, node_loss = model(batch)
                loss = (father_rule_loss + father_loss) * 0.1 + \
                       (edge_rule_loss + edge_loss) * 0.1 + node_loss * 0.8
                loss.mean().backward()
                nn.utils.clip_grad_value_(model.parameters(), 0.3)
                optimizer.step()
                scheduler.step()
                train_father_rule_loss.append(father_rule_loss.mean().item())
                train_father_loss.append(father_loss.mean().item())
                train_edge_rule_loss.append(edge_rule_loss.mean().item())
                train_edge_loss.append(edge_loss.mean().item())
                train_node_loss.append(node_loss.mean().item())
                train_loss.append(loss.mean().item())
                del batch, father_rule_loss, father_loss, edge_rule_loss, edge_loss, node_loss, loss
            except Exception as e:
                if 'out of memory' not in str(e):
                    print(i, str(int(time.time() - start_time)) + 's', str(e))
                if 'out of memory' in str(e):
                    oom += 1
                if 'loss is nan' in str(e):
                    nan += 1
                model.zero_grad()
                optimizer.zero_grad()
                del batch
                try:
                    del father_rule_loss, father_loss, edge_rule_loss, edge_loss, node_loss, loss
                except Exception as e:
                    pass
            if i % 1000 == 0 and i > 0:
                info = "epoch: {}, step: {}/{}, loss: {} ({}, {}, {}, {}, {}), lr: {}, " \
                       "oom: {}, nan: {}, time: {}s, ".format(
                    epoch + 1, i, len(train_loader),
                    round(float(sum(train_loss) / len(train_loss)), 4),
                    round(float(sum(train_father_rule_loss) / len(train_father_rule_loss)), 4),
                    round(float(sum(train_father_loss) / len(train_father_loss)), 4),
                    round(float(sum(train_edge_rule_loss) / len(train_edge_rule_loss)), 4),
                    round(float(sum(train_edge_loss) / len(train_edge_loss)), 4),
                    round(float(sum(train_node_loss) / len(train_node_loss)), 4),
                    round(scheduler.get_last_lr()[0], 6),
                    oom, nan, int(time.time() - start_time),
                )
                print(info)
                start_time = time.time()
                oom = 0

                if i % 10000 == 0 and i > 0:
                    validate_step(epoch, valid_loader, model, save_dir)
        validate_step(epoch, valid_loader, model, save_dir)


def trainer_main(training_data_path, validating_data_path, save_dir, epochs=2):
    node_vocabulary = NodeVocabulary(
        nonterminal_file=SRC_DIR + '../data/vocabulary/nodes_nonterminal.txt',
        terminal_file=SRC_DIR + '../data/vocabulary/nodes_terminal.txt',
        abstraction_file=SRC_DIR + '../data/vocabulary/abstractions.txt',
        idiom_file=SRC_DIR + '../data/vocabulary/idioms.txt',
        nonidentifier_file=SRC_DIR + '../data/vocabulary/nonidentifiers.txt'
    )
    edge_vocabulary = EdgeVocabulary(SRC_DIR + '../data/vocabulary/specified_edges.txt')
    print('finish loading vocabulary, node vocabulary:', len(node_vocabulary),
          ', edge vocabulary:', len(edge_vocabulary))

    train_dataset = GraphTransformerDataset(
        ast_files=[training_data_path],
        node_vocabulary=node_vocabulary,
        edge_vocabulary=edge_vocabulary,
        batch_size=3 * len(devices_id),
        shuffle=True,
        gpu_num=len(devices_id)
    )
    print('finish loading training data:', train_dataset.total_size)
    valid_dataset = GraphTransformerDataset(
        ast_files=[validating_data_path],
        node_vocabulary=node_vocabulary,
        edge_vocabulary=edge_vocabulary,
        batch_size=10 * len(devices_id),
        shuffle=False,
        gpu_num=len(devices_id)
    )
    print('finish loading validating data:', valid_dataset.total_size)
    config = Config(
        node_vocabulary=node_vocabulary,
        edge_vocabulary=edge_vocabulary,
        hidden_dim=256, edge_dim=64, num_head=8,
        num_encoder_layer=6,
        num_father_layer=2,
        num_edge_layer=2,
        num_node_layer=4,
        dropout=0.1
    )
    model = GraphTransformer(config)
    train(model, train_dataset, valid_dataset, epochs=epochs, save_dir=save_dir)


if __name__ == "__main__":
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        devices_id = [0, 1, 2, 3]

    training_data_path = SRC_DIR + '../data/general_training_ast.json'
    validating_data_path = SRC_DIR + '../data/general_validation_ast.json'
    save_dir = SRC_DIR + '../data/models/'
    trainer_main(training_data_path, validating_data_path, save_dir, epochs=10)
