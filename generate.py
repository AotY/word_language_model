###############################################################################
# Language Modeling on Penn Tree Bank
#
# This file generates new sentences sampled from the language model
#
#
###############################################################################

import argparse

import torch

import data
from queue import Queue

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

parser.add_argument('--device', type=str,
                    help='use CUDA or CPU')

parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')

parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

device = torch.device(args.device)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f)

model.eval()

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)

hidden = model.init_hidden(1)
input = torch.rand(1, 1).mul(ntokens).long().to(device)

if args.cuda:
    input.data = input.data.cuda()

sentence1 = ''
sentence2 = ''

with torch.no_grad():
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

            output_softmax = torch.softmax(output, dim=2)
            word_idx = output.softmax(dim=2).item()
            sentence2 += corpus.dictionary.idx2word[word_idx]

            if i % args.log_interval == 0:
                print('| Generated {}/{} words'.format(i, args.words))

print('sentence1: %s' % sentence1)
print('greedy sentence2: %s' % sentence2)


print("""beam search""")
class BeamNode:
    def __init__(self):
        self.sentence = ''
        self.log_prob = 0.0

    def push(self, word_idx, log_prob):
        self.sentence += str(word_idx) + ','
        self.log_prob += log_prob

    def get_ids(self):
        return [int(item) for item in self.sentence[:-1].split(',')]



beam_width = 64
best_n = 5
node_queue = Queue()

with torch.no_grad():
    # init
    output, hidden = model(input, hidden)
    output_softmax = torch.softmax(output, dim=2)
    log_probs, next_inputs = output_softmax.topk(2, beam_width) #[1, 1, beam_width]
    next_inputs = next_inputs.squeeze(0)
    next_hiddens = hidden.repeat(1, beam_width, 1) #[layers, beam_width, hidden_size]

    node_list = []
    for word_idx, log_prob in zip(next_inputs.view(-1).tolist(), log_probs.view(-1).tolist()):
        node = BeamNode()
        node.push(word_idx, log_prob)
        node_list.append(node)

    node_queue.put(node_queue)

    for i in range(args.words):
        outputs, hiddens = model(next_inputs, next_hiddens)
        outputs_softmax = torch.softmax(outputs, dim=2) #[1, beam_width, vocab_size]
        log_probs, next_inputs = output_softmax.view(-1).topk(0, beam_width)

        last_node_list = node_queue.get()
        cur_node_list = []
        for log_prob, index in zip(log_probs, next_inputs):
            last_i = index // outputs.size(2)
            word_idx = index % outputs.size(2)

            node = BeamNode()
            node.push(last_node_list[last_i].sentence + str(word_idx), log_prob)

            cur_node_list.append(node)

        node_queue.put(cur_node_list)
        del last_node_list


final_node_list = node_queue.get()

tmp_sentences = []
for node in final_node_list:
    ids = node.get_ids()
    score = node.log_prob / len(ids)
    tmp_sentences.append((score, ids))

sentences = sorted(tmp_sentences, key=lambda item: item[0], reverse=True)
sentences = [ids for _, ids in sentences]

for ids in sentences:
    sentence = ' '.join([corpus.dictionary.idx2word[id] for id in ids])
    print(sentence)





