import argparse
import copy
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import dgl
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from model import MLP, MLPLinear, CorrectAndSmooth
from dgl.nn.pytorch.conv import SGConv


def evaluate(y_pred, y_true, idx, evaluator):
    return evaluator.eval({
        'y_true': y_true[idx],
        'y_pred': y_pred[idx]
    })['acc']


def main():
    # check cuda
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu'
    # load data
    dataset = DglNodePropPredDataset(name=args.dataset)
    evaluator = Evaluator(name=args.dataset)

    split_idx = dataset.get_idx_split()
    g, labels = dataset[0] # graph: DGLGraph object, label: torch tensor of shape (num_nodes, num_tasks)
    
    deg_in = g.in_degrees().float().clamp(min=1)
    deg_out = g.out_degrees().float().clamp(min=1)
    if args.dataset == 'ogbn-arxiv':

        g = dgl.to_bidirected(g, copy_ndata=True)
        
        feat = g.ndata['feat']
        feat = (feat - feat.mean(0)) / feat.std(0)
        g.ndata['feat'] = feat

    g = g.to(device)
    feats = g.ndata['feat']
    labels = labels.to(device)

    # load masks for train / validation / test
    train_idx = split_idx["train"].to(device)
    valid_idx = split_idx["valid"].to(device)
    test_idx = split_idx["test"].to(device)

    n_features = feats.size()[-1]
    n_classes = dataset.num_classes
    
    # load model
    if args.model == 'mlp':
        model = MLP(n_features, args.hid_dim, n_classes, args.num_layers, args.dropout)
    elif args.model == 'linear':
        model = MLPLinear(n_features, n_classes)
    else:
        raise NotImplementedError(f'Model {args.model} is not supported.')

    model = model.to(device)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters())}')

    if args.pretrain:
        print('---------- Before ----------')
        model = torch.load(f'base/{args.dataset}-{args.model}.pt')
        model.eval()

        y_soft = model(feats).exp()

        y_pred = y_soft.argmax(dim=-1, keepdim=True)
        valid_acc = evaluate(y_pred, labels, valid_idx, evaluator)
        test_acc = evaluate(y_pred, labels, test_idx, evaluator)
        print(f'Valid acc: {valid_acc:.4f} | Test acc: {test_acc:.4f}')

        print('---------- Correct & Smoothing ----------')
        cs = CorrectAndSmooth(num_correction_layers=args.num_correction_layers,
                              correction_alpha=args.correction_alpha,
                              correction_adj=args.correction_adj,
                              num_smoothing_layers=args.num_smoothing_layers,
                              smoothing_alpha=args.smoothing_alpha,
                              smoothing_adj=args.smoothing_adj,
                              autoscale=args.autoscale,
                              scale=args.scale)
        
        mask_idx = train_idx#torch.cat([train_idx, valid_idx])
        if True:
            steps = [y_soft]
            if args.aggregator == 'LightGBM':
                propagations = [3,6]
            else:
                propagations = [3]
            input = y_soft.clone()
            for prop in propagations:
                correct = cs.correct(g, input, labels[mask_idx], mask_idx, num_layers=prop, y_soft2=y_soft)
                smooth = cs.smooth(g, correct, labels[mask_idx], mask_idx, use_labels=False, num_layers=prop)
                steps += [correct]
                steps += [smooth]
                input = torch.softmax(smooth, dim=0)

            step_features = torch.cat(steps, dim=1)

            features = torch.cat([step_features,deg_in.view(-1,1), deg_out.view(-1,1)], dim=1)
            
    
            if args.aggregator == 'LightGBM':
                prediction = cs.lightgbm(g, features, labels[mask_idx], mask_idx)
            else:
                features = (features - features.mean(0))/features.std(0)
                prediction = cs.linear(g, feats,  features, labels[mask_idx], mask_idx, model)
            smooth_prediction = cs.smooth(g, prediction, labels[mask_idx], mask_idx, use_labels=True, num_layers=2)



        y_pred = smooth_prediction.argmax(dim=-1, keepdim=True)
        valid_acc = evaluate(y_pred, labels, valid_idx, evaluator)
        test_acc = evaluate(y_pred, labels, test_idx, evaluator)
        print(f'Valid acc: {valid_acc:.4f} | Test acc: {test_acc:.4f}')
    else:
        opt = optim.Adam(model.parameters(), lr=args.lr)

        best_acc = 0
        best_model = copy.deepcopy(model)

        # training
        print('---------- Training ----------')
        for i in range(args.epochs):

            model.train()
            opt.zero_grad()

            logits = model(feats)
            
            train_loss = F.nll_loss(logits[train_idx], labels.squeeze(1)[train_idx])
            train_loss.backward()

            opt.step()
            
            model.eval()
            with torch.no_grad():
                logits = model(feats)
                
                y_pred = logits.argmax(dim=-1, keepdim=True)

                train_acc = evaluate(y_pred, labels, train_idx, evaluator)
                valid_acc = evaluate(y_pred, labels, valid_idx, evaluator)

                print(f'Epoch {i} | Train loss: {train_loss.item():.4f} | Train acc: {train_acc:.4f} | Valid acc {valid_acc:.4f}')

                if valid_acc > best_acc:
                    best_acc = valid_acc
                    best_model = copy.deepcopy(model)
        
        # testing & saving model
        print('---------- Testing ----------')
        best_model.eval()
        
        logits = best_model(feats)
        
        y_pred = logits.argmax(dim=-1, keepdim=True)
        test_acc = evaluate(y_pred, labels, test_idx, evaluator)
        print(f'Test acc: {test_acc:.4f}')

        if not os.path.exists('base'):
            os.makedirs('base')

        torch.save(best_model, f'base/{args.dataset}-{args.model}.pt')


if __name__ == '__main__':
    """
    Hyperparameters
    """
    parser = argparse.ArgumentParser(description='Base predictor(C&S)')

    # Dataset
    parser.add_argument('--gpu', type=int, default=0, help='-1 for cpu')
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv', choices=['ogbn-arxiv', 'ogbn-products'])
    # Base predictor
    parser.add_argument('--model', type=str, default='mlp', choices=['mlp', 'linear'])
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--hid-dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=300)
    # extra options for gat
    parser.add_argument('--n-heads', type=int, default=3)
    parser.add_argument('--attn_drop', type=float, default=0.05)
    # C & S
    parser.add_argument('--pretrain', action='store_true', help='Whether to perform C & S')
    parser.add_argument('--correction-alpha', type=float, default=0.979)
    parser.add_argument('--correction-adj', type=str, default='DAD')
    parser.add_argument('--num-correction-layers', type=int, default=3)
    parser.add_argument('--num-smoothing-layers', type=int, default=3)
    parser.add_argument('--smoothing-alpha', type=float, default=0.756)
    parser.add_argument('--smoothing-adj', type=str, default='DAD')
    parser.add_argument('--autoscale', action='store_true')
    parser.add_argument('--scale', type=float, default=20.)
    parser.add_argument('--aggregator', type=str, default='LightGBM')

    args = parser.parse_args()
    print(args)

    main()