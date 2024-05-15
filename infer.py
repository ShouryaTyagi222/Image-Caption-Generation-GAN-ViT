import torch

from utils import vocab as uvoc
from utils import memory_usage
from pprint import pprint
import os
import sys
import torch
import torch.optim as optim
import utils.explorer_helper as exh
import utils.vocab as uvoc

from metrics.scores import bleu_score, prepare_references
from models.wgan import WGAN
from utils import check_args, fix_seed



img_path = ''

def main(args):
    print(torch.backends.cudnn.benchmark)
    torch.backends.cudnn.deterministic = True
    # Get configuration
    config = exh.load_json(args.CONFIG)

    # Global initialization
    torch.cuda.init()
    device = torch.device(config['cuda']['device'] if (torch.cuda.is_available() and config['cuda']['ngpu'] > 0) else "cpu")
    seed = fix_seed(config['seed'])

    # Load vocabulary
    vocab = exh.load_json(config['data']['vocab'])

    # Prepare references
    references = exh.read_file(config['data']['test']['captions'])
    references = prepare_references(references)
    weights = None
    if len(config['model']['embeddings']) > 0:
        weights = uvoc.init_weights(vocab, config['model']['emb_dim'])
        uvoc.glove_weights(weights, config['model']['embeddings'], vocab)

    model = WGAN(len(vocab['token_list']), config['model'], weights)

    model.reset_parameters()
    print("The state dict keys: \n\n", model.state_dict().keys())

    model.load_state_dict(torch.load(config['load_dict']))
    for param in list(model.parameters()):
        param.requires_grad = False

    c = torch.load(config['load_dict'])
    for x in model.state_dict():
        if len(model.state_dict()[x].shape) == 1:
            model.state_dict()[x][:] = c[x]
        elif len(model.state_dict()[x].shape) == 2:
            model.state_dict()[x][:,:] = c[x]

    model.to(device)

    fix_seed(config['seed'] + 1)


    model.train(False)
    torch.set_grad_enabled(False)
    model.eval()

    model.G.emb.weight.data = c['G.emb.weight']






    img = PIL_Image.open(img_path)
    if img.mode != 'RGB':
        img = img.convert(mode='RGB')
    feature_extractor = ImageEncoder()
    _transforms = []
    if resize is not None:
        _transforms.append(transforms.Resize(resize))
    if crop is not None:
        _transforms.append(transforms.CenterCrop(crop))
    _transforms.append(transforms.ToTensor())

    transform = transforms.Compose(_transforms)
    img = transform(img).cuda()
    img = feature_extractor.get(img.unsqueeze(0))

    features = img.squeeze()

    generator = model.G
    n_vocab = len(vocab['token_list'])
    results = []

    # features = model.encode(batch)

    sentences= batch['tokenized']
    h = generator.f_init(features)
    prob = torch.zeros(sentences.shape[1], n_vocab, device=device)
    y_t = generator.emb(sentences[0])
    tokens = torch.zeros(max_len, batch.size, device=device)

    for tstep in range(max_len):
        prob, h = generator.f_next(features, y_t, prob, h)
        y_t = torch.argmax(prob,dim=1)
        tokens[tstep] = y_t
        y_t = generator.emb(y_t)


tokens = tokens.to('cpu')
tokens = tokens[:,range(batch.size)].t().tolist()
results.extend(tokens)

sentences = []
    for row in results:
    sentences.append(uvoc.tokens2words(row, vocab))