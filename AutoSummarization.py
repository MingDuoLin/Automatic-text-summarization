import os
from gensim.models import Word2Vec
import re
import pickle
import jieba
from functools import partial
from scipy.spatial.distance import cosine
import numpy as np
from pyltp import SentenceSplitter


MODEL_DIR = './models'  # 模型目录的路径
w2v_model_path = os.path.join(MODEL_DIR, 'news.50.model')  # 词向量模型路径
fre_dict_path = os.path.join(MODEL_DIR, 'frequence.dic')   # 词频率


def cut(string): return ' '.join(jieba.cut(string))


def preprocess(text):
    text = text.replace(u'\r\n', u' ')
    text = text.replace(u'\u3000', u' ')
    text = text.replace(u'\\r\\n', u' ')
    text = text.replace(u'\\u3000', u' ')
    return text


def sentence_split_by_pytlp(content):
    sentences = SentenceSplitter.split(content)
    return [s.strip() for s in sentences if len(s) != 0]


def split_sentence(sentence):
    pattern = re.compile('[。，,.]：')
    split = pattern.sub(' ', sentence).split()
    return split


class AutoSummarization:
    """
    新闻自动摘要
    """

    def __init__(self):
        """
        初始化模型
        """
        self.model = Word2Vec.load(w2v_model_path)
        with open(fre_dict_path, "rb") as f:
            self.frequence = pickle.load(f)

    def sentence_embedding(self, sentence):
        # weight = alpah/(alpah + p)
        # alpha is a parameter, 1e-3 ~ 1e-5
        alpha = 1e-4

        max_fre = max(self.frequence.values())

        words = cut(sentence).split()

        sentence_vec = np.zeros_like(self.model.wv['测试'])

        words = [w for w in words if w in self.model]

        for w in words:
            weight = alpha / (alpha + self.frequence.get(w, max_fre))
            sentence_vec += weight * self.model.wv[w]

        sentence_vec /= len(words)
        # Skip the PCA
        return sentence_vec

    def get_corrlations(self, text):
        if isinstance(text, list): text = ' '.join(text)

        sub_sentences = sentence_split_by_pytlp(text)
        sentence_vector = self.sentence_embedding(text)

        correlations = {}

        for sub_sentence in sub_sentences:
            sub_sen_vec = self.sentence_embedding(sub_sentence)
            correlation = cosine(sentence_vector, sub_sen_vec)
            correlations[sub_sentence] = correlation

        return sorted(correlations.items(), key=lambda x: x[1], reverse=True)

    def get_summarization_simple(self, text, score_fn, constraint=200):
        sub_sentence = sentence_split_by_pytlp(text)

        ranking_sentence = score_fn(sub_sentence)
        selected_text = set()
        current_text = ''

        for sen, _ in ranking_sentence:
            if len(current_text) < constraint:
                current_text += sen
                selected_text.add(sen)
            else:
                break

        summarized = []
        for sen in sub_sentence:
            if sen in selected_text:
                summarized.append(sen)

        return summarized

    def get_summarization_simple_by_sen_embedding(self, text, constraint=200):
        text = preprocess(text)
        return ''.join(self.get_summarization_simple(text, self.get_corrlations, constraint))


if __name__ == '__main__':
    test_string = '''
    虽然至今夏普智能手机在市场上无法排得上号，已经完全没落，并于 2013 年退出中国市场，但是今年 3 月份官方突然宣布回归中国，预示着很快就有夏普新机在中国登场了。
    那么，第一款夏普手机什么时候登陆中国呢？又会是怎么样的手机呢？\r\n近日，一款型号为 FS8016 的夏普神秘新机悄然出现在 GeekBench 的跑分库上。从其中相关信息了解到，
    这款机子并非旗舰定位，所搭载的是高通骁龙 660 处理器，配备有 4GB 的内存。骁龙 660 是高通今年最受瞩目的芯片之一，采用 14 纳米工艺，八个 Kryo 260 核心设计，
    集成 Adreno 512 GPU 和 X12 LTE 调制解调器。\r\n当前市面上只有一款机子采用了骁龙 660 处理器，那就是已经上市销售的 OPPO R11。骁龙 660 尽管并非旗舰芯片，
    但在多核新能上比去年骁龙 820 强，单核改进也很明显，所以放在今年仍可以让很多手机变成高端机。不过，由于 OPPO 与高通签署了排他性协议，可以独占两三个月时间。
    \r\n考虑到夏普既然开始测试新机了，说明只要等独占时期一过，夏普就能发布骁龙 660 新品了。按照之前被曝光的渲染图了解，夏普的新机核心竞争优势还是全面屏，因为从 2013 
    年推出全球首款全面屏手机 EDGEST 302SH 至今，夏普手机推出了多达 28 款的全面屏手机。\r\n在 5 月份的媒体沟通会上，惠普罗忠生表示：我敢打赌，12 个月之后，在座的各位手机都会换掉。
    因为全面屏时代的到来，我们怀揣的手机都将成为传统手机。\r\n
    '''


    auto_sum = AutoSummarization()
    result = auto_sum.get_summarization_simple_by_sen_embedding(test_string)
    print(result)

