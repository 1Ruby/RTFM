wordlist = ['wall', 'agent', 'goal', 'gate', 'A', 'B', 'C', 'yellow', 'green', 'purple', 'water', 'coin', 'empty',
            'action', 'extra', 'step', 'stay', 'forward', 'block', 'left', 'right', 'pass', 'inverse', 'normal',
            ':', ' ', ';', '.']


def read_things(vocab, thing, max_num: int, eos='pad', pad='pad'):
    # read a word or a list of words, add eos and paddings at the end
    # return sentance with size=max_num, length = words till eos
    sentance = []
    if isinstance(thing, str):
        sentance = [vocab.word2index(thing), vocab.word2index(eos)]
    elif isinstance(thing, list):
        for word in thing:
            sentance.append(vocab.word2index(word))
        sentance.append(vocab.word2index(eos))
    else:
        print('thing:', thing)
        raise ValueError('cannot read such above.')
    length = len(sentance)
    if length > max_num:
        print(str(max_num) + '-word is not enough for', thing)
        raise NotImplementedError('Ask scm to build it.')
    sentance += [vocab.word2index(pad)] * (max_num - length)
    return sentance, length