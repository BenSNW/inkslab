# coding:utf-8

class Node(object):
    def __init__(self, text='', is_root=False, is_word=False):
        self.next_p = {}
        self.fail = None
        self.is_root = is_root
        self.is_word = is_word
        self.str = text
        self.parent = None
        self.branchlist = []

    def append(self, keyword):
        assert len(keyword) > 0
        _buff = self
        for k in keyword[:-1]:
            _buff[k] = Node(k)
            _buff = _buff[k]
        else:
            _buff[keyword[-1]] = Node(keyword[-1], is_word=True)

    def __iter__(self):
        return iter(self.next_p.keys())

    def __getitem__(self, item):
        return self.next_p[item]

    def __setitem__(self, key, value):
        _u = self.next_p.setdefault(key, value)
        _u.is_word = _u.is_word or value.is_word
        _u.parent = self


class AhoCorasick(object):
    def __init__(self, *words):
        self.words = words
        self._root = Node(is_root=True)
        map(self._root.append, self.words)
        self._make()

    def _get_all_parentnode(self, node, root_start=True):
        _u = []
        while node != self._root:
            node = node.parent
            if node != self._root:
                _u.append(node)
        if root_start:
            _u.reverse()
        return _u

    def _make(self):
        _endnodelist = []

        def _handlesun(node):
            for i in node:
                if not node[i].next_p.keys():
                    _endnodelist.append(node[i])
                if node == self._root:
                    pass
                else:
                    if i in node.fail.next_p:
                        node[i].fail = node.fail.next_p[i]
                    else:
                        if i in self._root.next_p:
                            node[i].fail = self._root[i]
                        else:
                            node[i].fail = self._root
                    parentlist = self._get_all_parentnode(node[i])[1:]
                    for index, j in enumerate(parentlist):
                        if j.str in self._root.next_p:
                            try:
                                _startnode = self._root
                                for _l in parentlist[index:] + [node[i]]:
                                    _startnode = _startnode.__getitem__(_l.str)
                                assert _startnode.is_word
                                node[i].branchlist.append(_startnode)
                            except Exception as e:
                                pass
                        else:
                            pass
                _handlesun(node[i])

        self._root.fail = self._root
        for i in self._root:
            self._root[i].fail = self._root
            _handlesun(self._root[i])
        for i in _endnodelist:
            if i.str == i.parent.str and i.parent != self._root and i.fail.parent == self._root and i.parent.fail.fail != self._root \
                    and i.str in i.parent.fail.fail.next_p.keys():
                i.fail = i.parent.fail.fail[i.str]

    def search(self, content, with_index=False):
        result = set()
        node = self._root

        def match_case(node, current_index=None):
            if current_index == None:
                current_index = index
            else:
                pass
            parent_times = 0
            string = ''
            _len = -1
            while node != self._root:
                string = node.str + string
                _len += 1
                node = node.parent
                if node.is_word:
                    parent_times += 1
                    match_case(node, current_index=current_index - parent_times - 1)
            if not with_index:
                result.add(string)
            else:
                result.add((string, (current_index - _len, current_index + 1)))

        index = 0
        for i in content:
            while 1:
                if node == self._root:
                    if i not in node.next_p:
                        break
                    else:
                        node = self._root[i]
                        if node.is_word:
                            if with_index:
                                result.add((i, (index, index + 1)))
                            else:
                                result.add(i)
                        break
                else:
                    if node.next_p.has_key(i):
                        node = node.next_p[i]
                        if node.is_word:
                            match_case(node, current_index=index)
                            parentnode = [node] + self._get_all_parentnode(node, False)
                            for _, m in enumerate(parentnode):
                                for n in m.branchlist:
                                    match_case(n, current_index=index - _)
                        break
                    else:
                        parentnode = [node] + self._get_all_parentnode(node, False)
                        for _, m in enumerate(parentnode):
                            for n in m.branchlist:
                                match_case(n, current_index=index - _)
                        node = node.fail
                        continue
            index += 1
        return result


if __name__ == '__main__':
    tree = AhoCorasick("test", "book", "oo", "ok")
    print(list(tree.search("book")))


class node:
    def __init__(self, ch):
        self.ch = ch  # 结点值
        self.fail = None  # Fail指针
        self.tail = 0  # 尾标志：标志为 i 表示第 i 个模式串串尾
        self.child = []  # 子结点
        self.childvalue = []  # 子结点的值


# AC自动机类
class acmation:
    def __init__(self):
        self.root = node("")  # 初始化根结点
        self.count = 0  # 模式串个数

    # 第一步：模式串建树
    def insert(self, strkey):
        self.count += 1  # 插入模式串，模式串数量加一
        p = self.root
        for i in strkey:
            if i not in p.childvalue:  # 若字符不存在，添加子结点
                child = node(i)
                p.child.append(child)
                p.childvalue.append(i)
                p = child
            else:  # 否则，转到子结点
                p = p.child[p.childvalue.index(i)]

        p.tail = self.count  # 修改尾标志

    # 第二步：修改Fail指针
    def ac_automation(self):
        queuelist = [self.root]  # 用列表代替队列
        while len(queuelist):  # BFS遍历字典树
            temp = queuelist[0]
            queuelist.remove(temp)  # 取出队首元素
            for i in temp.child:
                if temp == self.root:  # 根的子结点Fail指向根自己
                    i.fail = self.root
                else:
                    p = temp.fail  # 转到Fail指针
                    while p:
                        if i.ch in p.childvalue:  # 若结点值在该结点的子结点中，则将Fail指向该结点的对应子结点
                            i.fail = p.child[p.childvalue.index(i.ch)]
                            break
                        p = p.fail  # 否则，转到Fail指针继续回溯
                    if not p:  # 若p==None，表示当前结点值在之前都没出现过，则其Fail指向根结点
                        i.fail = self.root
                queuelist.append(i)  # 将当前结点的所有子结点加到队列中

    # 第三步：模式匹配
    def runkmp(self, strmode):
        p = self.root
        cnt = {}  # 使用字典记录成功匹配的状态
        for i in strmode:  # 遍历目标串
            while i not in p.childvalue and p is not self.root:
                p = p.fail
            if i in p.childvalue:  # 若找到匹配成功的字符结点，则指向那个结点，否则指向根结点
                p = p.child[p.childvalue.index(i)]
            else:
                p = self.root
            temp = p
            while temp is not self.root:
                if temp.tail:  # 尾标志为0不处理
                    if temp.tail not in cnt:
                        cnt.setdefault(temp.tail)
                        cnt[temp.tail] = 1
                    else:
                        cnt[temp.tail] += 1
                temp = temp.fail
        return cnt  # 返回匹配状态
        # 如果只需要知道是否匹配成功，则return bool(cnt)即可
        # 如果需要知道成功匹配的模式串种数，则return len(cnt)即可


key = ["殷俊", "王志青", "dahai", "id"]  # 创建模式串
acp = acmation()

for i in key:
    acp.insert(i)  # 添加模式串

acp.ac_automation()

d = acp.runkmp('ad 王志青 dahai dahaidahaidahai')  # 运行自动机
print(d)

for i in d.keys():
        print(key[i-1], d[i])
