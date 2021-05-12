import numpy as np
from textblob import TextBlob


def alter_sentence(sentence, changeable_words):
    '''
    Alter a given sentence by changing words.
    '''
    blob = TextBlob(sentence)
    new_sent_tokens = [ changeable_words[np.random.randint(len(changeable_words))] if (word in changeable_words) else word for (word,tag) in blob.tags ]
    
    return " ".join(new_sent_tokens)



def sentence_diff(sentences, alt_sentences, changeable_words):
    changes = []
    sent_wit_changes = []
    for i_s in range(len(sentences)):
        s_toks, sa_toks = sentences[i_s].split(' '), alt_sentences[i_s].split(' ')
        change = ''
        sent_wit_change = ''
        for j_s in range(min(len(s_toks), len(sa_toks))):
            if (s_toks[j_s] != sa_toks[j_s] and sa_toks[j_s] in changeable_words):
                change += ' ( ' + s_toks[j_s] + ' -> ' + sa_toks[j_s] + ' )___  '
                sent_wit_change += ' ( ' + s_toks[j_s] + ' -> ' + sa_toks[j_s] + ' ) '
            else:
                sent_wit_change += s_toks[j_s] + ' '
        changes.append(change)
        sent_wit_changes.append(sent_wit_change)
    return sent_wit_changes