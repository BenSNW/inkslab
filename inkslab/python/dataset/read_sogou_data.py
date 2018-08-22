# -*- coding: utf-8 -*-

import os
from collections import Counter
import argparse
import csv, random

SPLIT = (0.8, 0.1, 0.1)
out_words = ['\u3000', '\n', '叠', '矗', '…', '猓', '措', '辏', 'ū', '梗', '髡', '螅', 'ぃ', '恚', '」', 'ǖ',
 '埽', 'ǎ', '欤', '菹', '硎', '氖', '觯', 'ü', '蛱', '煜', '纾', 'ā', '鳎', '穑', '⒉', '菔', '鋈', '┦', '耙', '∈', '冢',
 'ぷ', '刂', 'ㄔ', 'ㄖ', '⒐', 'ツ', '獯', '⑾', '⒁', '『', '⑸', '〕', '｝', '∥', 'ǜ', '窬', '阒', '攵', '樯',
 '崾', '⒆', '〗', '∽', '≡', '鍪', '∩', '∪', '詈', '笾', '〖', '「', '芏', '钅', '⌒', 'ノ', '』', '⒏', 'γ',
 'と', '⒌', 'ㄊ', '≌', '肀', '∏', '〔', '觳', '嗾', '丶', '⒎', '⒄', '嘌', '魇', 'ㄉ', '缃', '罱', '⑿',
 'ㄐ', '焐', '刈', '袢', '嫠', '庋', '⊥', '⒍', 'Γ', '椤', '炱', '〈', '∠', '叻', '⊙', '↓', '⒅', '⑹', '┮',
 '⒈', '鄹', 'ば', '畲', '葜', '⒒', 'Ω', '诔', 'Ю', '∧', '∑', '±', 'ㄎ', '┍', 'ど', '∫', '⑷', '氲', '蠹',
 'ǘ', '┐', '涞', 'Φ', '┑', '曛', '乇', '∷', '⑼', '┯', '⒀', '虺', 'ぜ', '∮', 'さ', '谠', '⒑', '龀', '匦',
 '跸', '醯', '瘛', '÷', '苁', '⑽', 'ㄈ', '⒂', '√', '⒚', '┗', 'ǔ', '＿', 'ê', '┕', 'т', 'サ', '⒔', 'し',
 '蠖', '闶', '萑', '⒗', '甏', '臁', '罴', 'な', '卣', 'ú', '⒋', '庀', 'ひ', '⒊', 'ㄌ', 'ナ', 'г', '⒕', 'и',
 '⒓', 'ぁ', '鼗', '鞴', 'て', '┲', '亍', 'そ', '┤', 'ǚ', '⑻', 'ㄓ', '┛', '菝', '厣', 'ィ', 'ㄆ', 'ド', '匚',
 'ざ', '鼙', '┪', '⒘', '⒖', 'ㄋ', 'だ', '厶', '鹾', 'ぞ', '┓', '┌', '蟪', '┏', 'β', 'ò', 'ご', 'Ψ', '⑵',
 '┟', '⑺', 'δ', '亓', '⒃', '⑶', 'ㄇ', 'τ', '┥', 'κ', 'に', 'げ', '┰', '┩', '┒', 'м', '＆', '艹', '⒛',
 'Ъ', 'ù', '饣', 'ν', '宀', 'ぱ', 'シ', 'ぐ', 'ッ', '嘏', 'ぢ', 'н', 'ι', 'ψ', 'П', 'せ', '砉', '┳', 'ね', '⒙',
 'Ч', '劢', 'ぶ', '凵', '‰', '┢', '窳', '┎', '┨', '└', '虿', '⑴', '┠', '廴', 'д', '┖', '┝', '┘', 'Τ',
 '┧', 'Э', 'じ', '┬', '醭', '嬲', 'ㄕ', 'ァ', 'ぴ', '罨', '罾', '辶', 'け', 'к', '酃', '°', 'す', '┙', '┫', '┚',
 '乜', 'ス', 'つ', 'α', '薹', 'ザ', 'Ы', 'θ', '癯', 'й', '橥', 'ふ', '庹', 'の', '敫', '扌', 'た', '髟', 'ち', 'б',
 '├', 'っ', 'ず', 'で', 'Ц', 'ㄅ', 'ё', 'び', '♀', 'Ⅱ', 'チ', 'は', 'Σ', 'ρ', 'Щ', 'π', '┱', '夂', 'セ', '攴',
 'ソ', 'Ф', 'Ь', '镄', 'コ', '~', '雠', '榱', 'ハ', 'ゲ', 'σ', 'φ', '┭', 'ь', 'ケ', 'ゼ', '|', 'パ', 'ヂ', 'Π',
 '｜', 'ダ', 'Я', '┞', 'ブ', 'ξ', 'Ⅲ', '衤', 'ヒ', 'ズ', 'プ', '┣', 'ゴ', 'η', 'ぬ', 'ヅ', 'づ', 'ε', '＼', 'タ',
 'п', ']', 'μ', '湔', 'Ш', '¨', 'ジ', 'Ⅳ', '┡', '→', 'ゾ', 'ヌ', 'ピ', 'バ', '○', 'ζ', 'フ', 'ネ', '廾', '△',
 'ビ', '仂', 'デ', 'л', '●', 'テ', '`', '≥', 'я', 'グ', '＾', '{', '①', '\\', '¤', 'ц', '♂', '≤', 'К']

raw_url2label_dict = {"http://auto.sohu.com/": "汽车",
            "http://business.sohu.com/": "财经",
            "http://it.sohu.com/": "科技",
            "http://health.sohu.com/": "健康",
            "http://sports.sohu.com/": "体育",
            "http://travel.sohu.com/": "旅游",
            "http://learning.sohu.com/": "教育",
            "http://career.sohu.com/": "招聘",
            "http://cul.sohu.com/": "文化",
            "http://mil.news.sohu.com/": "军事",
            "http://news.sohu.com/": "社会",
            "http://house.sohu.com/": "房产",
            "http://yule.sohu.com/": "娱乐",
            "http://women.sohu.com/": "时尚",
            "http://media.sohu.com/": "传媒",
            "http://gongyi.sohu.com/": "公益",
            "http://auto.sina.com.cn/": "汽车",
            "http://finance.sina.com.cn/": "财经",
            "http://tech.sina.com.cn/it/": "科技",
            "http://sina.kangq.com/": "健康",
            "http://sports.sina.com.cn/": "体育",
            "http://tour.sina.com.cn/": "旅游",
            "http://edu.sina.com.cn/": "教育",
            "http://edu.sina.com.cn/j/": "招聘",
            "http://cul.book.sina.com.cn/": "文化",
            "http://mil.news.sina.com.cn/": "军事",
            "http://news.sina.com.cn/society/": "社会",
            "http://news.sina.com.cn/china/": "国内",
            "http://news.sina.com.cn/world/": "国际",
            "http://house.sina.com.cn/": "房产",
            "http://ent.sina.com.cn/": "娱乐",
            "http://eladies.sina.com.cn/": "时尚",
            "http://news.sina.com.cn/media/": "传媒",
            "http://tech.sina.com.cn/": "科技",
            "http://auto.163.com/": "汽车",
            "http://money.163.com/": "财经",
            "http://sports.163.com/": "体育",
            "http://war.163.com/": "军事",
            "http://news.163.com/shehui/": "社会",
            "http://news.163.com/domestic/": "国内",
            "http://news.163.com/world/": "国际",
            "http://house.163.com/": "房产",
            "http://ent.163.com/": "娱乐",
            "http://lady.163.com/": "时尚",
            "http://gongyi.163.com/": "公益",
            "http://media.163.com/": "传媒",
            "http://tech.163.com/": "科技",
            "http://edu.163.com/": "教育",
            "http://news.qq.com/": "社会",
            "http://mil.qq.com/": "军事",
            "http://auto.qq.com/": "汽车",
            "http://finance.qq.com/": "财经",
            "http://tech.qq.com/": "科技",
            "http://sports.qq.com/": "体育",
            "http://edu.qq.com/": "教育",
            "http://cul.qq.com/": "文化",
            "http://luxury.qq.com/": "时尚",
            "http://ent.qq.com/": "娱乐",
            "http://gongyi.qq.com/": "公益",
            "http://house.qq.com/": "房产",
            "http://lady.qq.com/": "时尚",
            "http://finance.ifeng.com/": "财经",
            "http://ent.ifeng.com/": "娱乐",
            "http://news.ifeng.com/sports/": "体育",
            "http://fashion.ifeng.com/health/": "健康",
            "http://auto.ifeng.com/": "汽车",
            "http://house.ifeng.com/": "房产",
            "http://tech.ifeng.com/": "科技",
            "http://fashion.ifeng.com/travel/": "旅游",
            "http://fashion.ifeng.com/": "时尚",
            "http://edu.ifeng.com/": "教育",
            "http://gongyi.ifeng.com/": "公益",
            "http://culture.ifeng.com/": "文化",
            "http://news.ifeng.com/mil/": "军事",
             "http://auto": "汽车",
             "http://1688.autos": "汽车",
             "http://business": "财经",
             "http://finance": "财经",
             "http://money": "财经",
             "http://biz": "财经",
             "http://1688.tech": "科技",
             "http://taobao.finance": "财经",
             "http://alibaba.finance": "财经",
             "http://it": "科技",
             "http://tech": "科技",
             "http://taobao.diji": "科技",
             "http://health": "健康",
             "http://tour": "旅游",
             "http://travel": "旅游",
             "http://sports": "体育",
             "http://yundong": "体育",
             "http://taobao.sports": "体育",
             "http://learning": "教育",
             "http://edu": "教育",
             "http://career": "招聘",
             "http://cul": "文化",
             "http://art": "文化",
             "http://mil": "军事",
             "http://war": "军事",
             "http://society": "社会",
             "http://news.sina.com.cn/": "社会",
             "http://news.163.com/": "社会",
             "http://house": "房产",
             "http://yule": "娱乐",
             "http://ent": "娱乐",
             "http://taobao.ent": "娱乐",
             "http://media": "传媒",
             "http://gongyi": "公益",
             "http://women": "时尚",
             "http://eladies": "时尚",
             "http://lady": "时尚",
             "http://luxury": "时尚",
             "http://fashion": "时尚",
             "http://taobao.lady": "时尚",
}

url2label_dict = {"http://auto.sohu.com/": "汽车",
            "http://business.sohu.com/": "财经",
            "http://it.sohu.com/": "科技",
            "http://sports.sohu.com/": "体育",
            "http://mil.news.sohu.com/": "军事",
            "http://women.sohu.com/": "时尚",
            "http://auto.sina.com.cn/": "汽车",
            "http://finance.sina.com.cn/": "财经",
            "http://tech.sina.com.cn/it/": "科技",
            "http://sports.sina.com.cn/": "体育",
            "http://cul.book.sina.com.cn/": "文化",
            "http://news.sina.com.cn/society/": "社会",
            "http://ent.sina.com.cn/": "娱乐",
            "http://eladies.sina.com.cn/": "时尚",
            "http://tech.sina.com.cn/": "科技",
            "http://auto.163.com/": "汽车",
            "http://money.163.com/": "财经",
            "http://sports.163.com/": "体育",
            "http://news.163.com/shehui/": "社会",
            "http://ent.163.com/": "娱乐",
            "http://lady.163.com/": "时尚",
            "http://tech.163.com/": "科技",
            "http://news.qq.com/": "社会",
            "http://auto.qq.com/": "汽车",
            "http://finance.qq.com/": "财经",
            "http://tech.qq.com/": "科技",
            "http://sports.qq.com/": "体育",
            "http://cul.qq.com/": "文化",
            "http://luxury.qq.com/": "时尚",
            "http://ent.qq.com/": "娱乐",
            "http://lady.qq.com/": "时尚",
            "http://finance.ifeng.com/": "财经",
            "http://ent.ifeng.com/": "娱乐",
            "http://news.ifeng.com/sports/": "体育",
            "http://auto.ifeng.com/": "汽车",
            "http://house.ifeng.com/": "房产",
            "http://tech.ifeng.com/": "科技",
            "http://fashion.ifeng.com/": "时尚",
            "http://culture.ifeng.com/": "文化",
            "http://auto": "汽车",
            "http://1688.autos": "汽车",
            "http://business": "财经",
            "http://finance": "财经",
            "http://money": "财经",
            "http://biz": "财经",
            "http://1688.tech": "科技",
            "http://taobao.finance": "财经",
            "http://alibaba.finance": "财经",
            "http://it": "科技",
            "http://tech": "科技",
            "http://taobao.diji": "科技",
            "http://sports": "体育",
            "http://yundong": "体育",
            "http://taobao.sports": "体育",
            "http://learning": "教育",
            "http://cul": "文化",
            "http://art": "文化",
             "http://society": "社会",
             "http://news.sina.com.cn/": "社会",
             "http://news.163.com/": "社会",
             "http://yule": "娱乐",
             "http://ent": "娱乐",
             "http://taobao.ent": "娱乐",
             "http://women": "时尚",
             "http://eladies": "时尚",
             "http://lady": "时尚",
             "http://luxury": "时尚",
             "http://fashion": "时尚",
             "http://taobao.lady": "时尚",
}


def url2label(url):
    for u, label in url2label_dict.items():
        if url.startswith(u):
            if label in ['社会', '体育', '娱乐', '财经', '汽车', '文化', '时尚', '科技']:
                return label
    return None


def get_replace(string):
    for o in out_words:
        string = string.replace(o, '')
    return string


# 把数据做成
def read_data(argv):
    dir = '/home/transwarp/projects/text_classification/data/raw_data/news.dat'
    with open(dir, 'r', encoding='u8') as f:
        lines = f.readlines()
        # store the values respectively
        new_lines = []

        title = None
        content = None
        url = None
        label = None
        label_freq = Counter()
        skip_num = 0
        skip_url_list = []
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('</doc>'):
                if label is None:
                    skip_url_list.append(url)
                    skip_num += 1
                    continue
                else:
                    new_lines.append((label, title, content))
                    label_freq.update([label])
                title = None
                content = None
                url = None
                label = None
            elif line.startswith('<doc>'):
                continue
            elif line.startswith('<url>') and line.endswith('</url>'):
                try:
                    url = line[5:-6]
                    label = url2label(url)
                    # label_freq.update([label])
                except:
                    pass
            elif line.startswith('<contenttitle>') and line.endswith('</contenttitle>'):
                try:
                    title = line[14:-15]
                    title = get_replace(title)
                    # title_word_freq.update(title)
                    # title_content_word_freq.update(title)
                except:
                    pass
            elif line.startswith('<content>') and line.endswith('</content>'):
                try:
                    content = line[9:-10]
                    content = get_replace(content)
                    # content_word_freq.update(content)
                    # title_content_word_freq.update(content)
                except:
                    pass
    f.close()

    with open('/home/transwarp/projects/text_classification/data/raw_data/urls.txt', 'w') as f:
        for url in skip_url_list:
            f.write(url)
            f.write('\n')
    f.close()

    print('The number of new_lines is {}'.format(len(new_lines)))
    print('The number of no label sample is {}'.format(skip_num))
    print(label_freq)
    # save_new_line(argv, new_lines)


def save_new_line(argv, new_lines):
    title_label_list = []
    context_label_list = []
    for i, val in enumerate(new_lines):
        label, title, context = val
        title_label_list.append((title, label))
        context_label_list.append((context, label))
    save_text(lines=title_label_list, path=os.path.join(argv.train_dir, 'sogou_title_label'))
    save_text(lines=context_label_list, path=os.path.join(argv.train_dir, 'sogou_context_label'))


def save_text(lines, path):
    random.shuffle(lines)
    volume = len(lines)
    train_val_split = int(volume * (SPLIT[0] / float(sum(SPLIT))))
    val_test_split = int(volume * (SPLIT[0] + SPLIT[1]) / float(sum(SPLIT)))
    train_lines = lines[:train_val_split]
    val_lines = lines[train_val_split:val_test_split]
    test_lines = lines[val_test_split:]
    with open(path + '_train.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'label'])
        for line in train_lines:
            if line[0] is not None and line[1] is not None:
                writer.writerow(line)
    f.close()
    with open(path + '_val.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'label'])
        for line in val_lines:
            if line[0] is not None and line[1] is not None:
                writer.writerow(line)
    f.close()
    with open(path + '_test.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'label'])
        for line in test_lines:
            if line[0] is not None and line[1] is not None:
                writer.writerow(line)
    f.close()


def save_new_text(argv, lines):
    output_file = os.path.join(argv.train_dir, 'output_new.txt')
    with open(output_file, 'w') as f:
        for label, title, content in lines:
            f.write(label)
            f.write('\t')
            f.write(title)
            f.write('\t')
            f.write(content)
            f.write('\n')
    f.close()


def read_new_text(argv, name):
    text = []
    file_name = os.path.join(argv.train_dir, name)
    with open(file_name, 'r', encoding='u8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            label, title, content = line.split('\t')
            text.append((label, title, content))
    f.close()
    return text


def save_word_freq(argv, name, word_freq):
    word_freq_path = os.path.join(argv.train_dir, name)
    with open(word_freq_path, 'w') as file:
        for word, freq in word_freq:
            file.write(word)
            file.write(' ')
            file.write(str(freq))
            file.write('\n')
    file.close()


def read_word_freq(argv, name):
    word_freq_dict = dict()
    word_freq_path = os.path.join(argv.train_dir, name)
    with open(word_freq_path, 'r') as file:
        for line in file.readlines():
            line = line.strip()
            word, freq = line.split(' ')
            word_freq_dict[word] = freq
    file.close()
    return word_freq_dict


def word_freq_filter(argv, word_freq, unk=True):
    word2id = dict()
    index = 0
    if unk:
        word2id['_PAD'] = 0
        word2id['_UNK'] = 1
        index = 2
    for word, freq in word_freq:
        if freq >= argv.min_count:
            word2id[word] = index
            index += 1
    return word2id


def save_word2id(argv, name, word2id):
    word_freq_path = os.path.join(argv.train_dir, name)
    with open(word_freq_path, 'w') as file:
        for word, id in word2id.items():
            file.write(word)
            file.write(' ')
            file.write(str(id))
            file.write('\n')
    file.close()


def data_tansform(argv):
    label_freq = read_word_freq(argv, 'label_freq.txt')
    title_word_freq = read_word_freq(argv, 'title_word_freq.txt')
    content_word_freq = read_word_freq(argv, 'content_word_freq.txt')
    title_content_word_freq = read_word_freq(argv, 'title_content_word_freq.txt')

    label2id = word_freq_filter(argv, label_freq, False)
    titleword2id = word_freq_filter(argv, title_word_freq)
    contentword2id = word_freq_filter(argv, content_word_freq)
    titlecontent2id = word_freq_filter(argv, title_content_word_freq)
    text = read_new_text(argv, 'output_new.txt')


def main(argv):
    lines = read_data(argv)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--train_dir",
        type=str,
        default="raw_data",
        help="The local path or hdfs path of result files"
    )

    parser.add_argument(
        "--min_count",
        type=int,
        default=4,
        help="The min word count"
    )
    args = parser.parse_args()

    main(args)

