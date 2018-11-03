###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
#
###############################################################################

import argparse

import torch
from torch.autograd import Variable

import data

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')

parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')

parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')

parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')

parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')

parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')

parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)
model.eval()

if args.cuda:
    model.cuda()
else:
    model.cpu()

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)

hidden = model.init_hidden(1)
input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)

if args.cuda:
    input.data = input.data.cuda()

sentence1 = ''
sentence2 = ''
sentence3 = ''
with open(args.outf, 'w') as outf:
    for i in range(args.words):
        # model(input, hidden) --> forward(self, input, hidden):
        output, hidden = model(input, hidden)
        print(output.shape) #[1, 1, vocab_size]

        # squeeze() -- Returns a Tensor with all the dimensions of input of size 1 removed.
        # exp() 求指数，默认是以e（2.718）为底。
        word_weights = output.squeeze().data.div(args.temperature).exp().cpu()

        # Returns a Tensor where each row contains num_samples indices sampled from the multinomial probability distribution located in the corresponding row of Tensor input.
        word_idx = torch.multinomial(word_weights, 1)[0]

        # 全部填充为word_idx值，也就是下一次输出的word_idx
        input.data.fill_(word_idx)
        word = corpus.dictionary.idx2word[word_idx]
        sentence1 += word

        outf.write(word + ('\n' if i % 20 == 19 else ' '))

        output_softmax = torch.softmax(output)
        word_idx = output.softmax(dim=2).item()
        sentence2 += corpus.dictionary.idx2word[word_idx]

        if i % args.log_interval == 0:
            print('| Generated {}/{} words'.format(i, args.words))

print('sentence1: %s' % sentence1)
print('sentence2: %s' % sentence2)
print('sentence3: %s' % sentence3)