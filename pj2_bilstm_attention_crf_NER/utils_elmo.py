import re
import codecs


def num_to_zero(s):
    """
    Replace every digit in a string by a zero.
    """
    return re.sub(r'\d', r'0', s)


def load_sentences(path, zeros):
    """
    Load sentences. 空格分割
    """
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf8'):
        line = num_to_zero(line.rstrip()) if zeros else line.rstrip()  # delete tail character

        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                elif not zeros: # print
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            
            assert len(word) >= 2
            sentence.append(word)

    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
        elif not zeros: # print
            sentences.append(sentence)
        

    return sentences




def update_tag_scheme(sentences, tag_scheme):
    '''
    预处理标签
    '''
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]  # remove other labels
        # Check that tags are given in the BIO format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in BIO format! ' +'Please check sentence %i:\n%s' % (i, s_str))

        if tag_scheme == "BIO":
            new_tags = tags
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Wrong tagging scheme!')


def iob2(tags):
    """
    判断标签的格式: O, I-X
    根据前一个word的标签: 将O, I-X, I-X 转换为 O, B-X, I-X
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        # convert IOB1 to IOB2
        elif i == 0 or tags[i - 1] == 'O':
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:
            tags[i] = 'B' + tag[1:]
    return True


def lower_case(x, lower=False):
    """
    Use lowercase for all letters.
    """
    if lower:
        return x.lower()
    else:
        return x


def cap_feature(s):
    """
    Capitalization feature:
    0 = low caps
    1 = all caps
    2 = first letter caps
    3 = one capital (not first letter)
    """
    if s.lower() == s:
        return 0
    elif s.upper() == s:
        return 1
    elif s[0].upper() == s[0]:
        return 2
    else:
        return 3


def prepare_dataset(sentences, word_to_id, char_to_id, tag_to_id, lower=False):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    """
    data = []
 
    for s in sentences:

        str_words = [w[0] for w in s]
        
        # words使用小写
        words = [word_to_id[lower_case(w, lower) if lower_case(w, lower) in word_to_id else '<UNK>']
                 for w in str_words]
        
        # chars 不小写
        chars = [[char_to_id[c] for c in w if c in char_to_id]
                 for w in str_words]
 
        tags = [tag_to_id[w[-1]] for w in s]

        caps = [cap_feature(w) for w in str_words] # char中已经区分了大小写
        data.append({
            'str_words': str_words,
            'words': words,
            'chars': chars,
            'caps': caps,
            'tags': tags,
        })

    return data
