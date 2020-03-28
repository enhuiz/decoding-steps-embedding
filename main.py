import argparse
import tqdm
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence
from collections import defaultdict


parser = argparse.ArgumentParser()
parser.add_argument('--bs', default=16)
parser.add_argument('--dim', default=128)
parser.add_argument('--iterations', default=2000)
parser.add_argument('--lr', default=1e-4)
parser.add_argument('--maxl', default=10)
parser.add_argument('--decay', default=0.95)
parser.add_argument('--self-critic', action='store_true')
parser.add_argument('--baseline-decay', default=0.95)
parser.add_argument('--meter-decay', default=0.95)
args = parser.parse_args()


def batch():
    return torch.randint(1, args.maxl, (args.bs, ))


class Model(nn.Module):
    def __init__(self, dim, maxl):
        super().__init__()
        self.maxl = maxl
        self.embed = nn.Embedding(maxl, dim)
        self.cell = nn.GRU(dim, dim)
        self.fc = nn.Linear(dim, 1)
        self.i0 = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, l, determinstic=False):
        maxl = self.maxl
        bs = len(l)

        bt = torch.arange(bs)
        it = self.i0.repeat(1, bs, 1)
        ht = self.embed(l)[None]
        lt = torch.ones(bs)

        lg = torch.zeros(maxl, bs)
        pred_l = torch.ones(bs).long()

        for t in range(maxl):
            ot, ht = self.cell(it, ht)

            lgt = self.fc(ot).flatten()
            lg[t, bt] = lgt

            if determinstic:
                pred_l[bt] += lgt.sigmoid() > 0.5
            else:
                dist = torch.distributions.Bernoulli(logits=lgt)
                pred_l[bt] += dist.sample().long()

            running = t + 1 < pred_l[bt]

            [it, ht] = \
                map(lambda x: x[:, running],
                    [ot, ht])

            [bt] = \
                map(lambda x: x[running],
                    [bt])

            if len(bt) == 0:
                break

        lg = [lgi[:li] for lgi, li in zip(lg.transpose(0, 1), pred_l)]

        return lg, pred_l


class EMA():
    def __init__(self, decay, init=0):
        self.decay = decay
        self.value = init

    def __call__(self, new=None):
        if new is not None:
            self.value = self.decay * self.value \
                + (1 - self.decay) * new
        return self.value


def action_nll(lg, eps=1e-7):
    y = torch.ones_like(lg)
    y[-1] = 0
    nll = F.binary_cross_entropy_with_logits(lg, y)
    return nll


model = Model(args.dim, args.maxl)
optimizer = torch.optim.Adam(model.parameters(), args.lr)

baseline_meter = EMA(args.baseline_decay)
accuracy_meter = EMA(args.meter_decay)
accuracies = []

pbar = tqdm.tqdm(range(args.iterations))

try:
    for i in pbar:
        l = batch()

        lg, pred_l = model(l)
        nll_mean = torch.stack([action_nll(lgi).mean() for lgi in lg])
        reward = -(pred_l.float() - l).abs()

        if args.self_critic:
            baseline_pred_l = model(l, True)[1]
            baseline_reward = -(baseline_pred_l.float() - l).abs()
        else:
            baseline_reward = baseline_meter(reward.mean())

        # reduce variance
        reward -= baseline_reward

        # R * logp
        loss = (reward * nll_mean).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accuracy = (pred_l == l).sum().float() / len(l)
        accuracies.append(accuracy_meter(accuracy).item())

        pbar.set_description(f'Accuracy: {accuracy_meter():.3g}, '
                             f'loss: {loss.item():.3g}, '
                             f'reward: {reward.mean():.3g}, '
                             f'lengths: {pred_l[:5]}')
finally:
    try:
        with open('history.json') as f:
            history = json.load(f)
    except:
        history = defaultdict(list)

    history['self_critic' if args.self_critic else 'ema'] = accuracies

    with open('history.json', 'w') as f:
        json.dump(history, f)

    print('Plotting')
    import matplotlib.pyplot as plt
    for mode in ['ema', 'self_critic']:
        plt.plot(history[mode], label=mode)
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.savefig('accuracy.png')
    print('Done.')
