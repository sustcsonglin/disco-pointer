import pdb


class Token:
    # Leaf of a tree
    header = None

    def __init__(self, token, i, features=None):
        self.token = token
        self.features = features  # Only used for POS tags for now which should be self.features[0]
        self.i = i
        self.parent = None
        self.span_sorted = [i]
        self.visited_idx = -1
        self.sibling = None


    def get_tag(self):
        if len(self.features) > 0:
            return self.features[0]
        return None

    def set_tag(self, tag):
        self.features[0] = tag

    def is_leaf(self):
        return True

    def is_continuous(self):
        return True

    def get_span(self):
        return {self.i}

    @property
    def span_repr(self):
        return [self.i, self.i + 1]

    def return_arc(self):
        return []

    def convert(self):
        pass

    def __str__(self):
        return "({} {}={})".format(self.features[0], self.i, self.token)

    def return_continuous(self):
        return [[self.i, self.i + 1]]

    def return_child_span(self):
        return []

    def return_arc(self):
        return []

    def return_arcs(self):
        return []

    def convert(self):
        return []

    def dis(self):
        return []

    def cont(self):
        return []


class Tree:
    def __init__(self, label, children):
        self.label = label
        self.children = children
        self.update_child()
        self.index = -1
        self.span = {i for c in self.children for i in c.get_span()}
        span = list(self.span)
        span.sort()
        self.span_sorted = span
        self.parent = None
        for c in self.children:
            c.parent = self

        self.sibling = None

    def update_child(self):
        self.children = sorted(self.children, key=lambda x: min(x.get_span()))
        self.index = -1
        self.span = {i for c in self.children for i in c.get_span()}
        span = list(self.span)
        span.sort()
        self.span_sorted = span
        for c in self.children:
            c.parent = self




    def dis(self):
        dis = []
        if not self.is_continuous():
            dis.append([self.return_continuous(), self.label])
        for child in self.children:
            dis.extend(child.dis())
        return dis


    def generate_dis_point(self):
        dis = []
        if not self.is_continuous():
            dis.append([self.return_point()])
        for child in self.children:
            dis.extend(child.dis())
        return dis




    def cont(self):

        cont = []
        if self.is_continuous():
            cont.append(self.span_repr)
        for child in self.children:
            cont.extend(child.cont())

        return cont

    def return_child_span(self):
        children = []
        for child in self.children:
            children.extend(child.return_continuous())
        return children

    def convert(self):
        s = self.return_arcs()
        for child in self.children:
            s.extend(child.convert())
        return s


    def return_arcs(self):
        arcs = []
        for child in self.children:
            arcs.extend(child.return_arc())
        return arcs

    def generate_point(self):
        pointer = []

        for c in self.children:
            pointer += c.generate_point()

        if self.is_continuous():
            pointer += [self.span_sorted[-1]+1, self.span_sorted[0], self.label, 0]

        else:
            any_discontinuous = False
            for  c in self.children:
                if not c.is_continuous():
                    any_discontinuous = True
                    break

            # all continuous
            if not any_discontinuous:
                pass
            else:
                pass

                # for i in range(len(self.children) - 2):
                #     pointer += [self.children[i+1].span_sorted[-1]+1, self.children[i].span_sorted[-1]+1, 'NULL']





    def return_continuous(self):
        if self.is_continuous():
            return [self.span_repr]

        else:
            span = []
            for child in self.children:
                span.extend(child.return_continuous())

            final_span = []
            span = sorted(span, key=lambda x: x[0])

            while len(span) != 0:
                start = span[0][0]
                end = span[0][1]
                span.pop(0)
                while len(span) != 0 and span[0][0] == end:
                    # start = end
                    end = span[0][1]
                    span.pop(0)

                final_span.append([start, end])

            # final_span.append(self.label)
            return final_span

    @property
    def span_repr(self):
        if self.is_continuous():
            return [self.span_sorted[0], self.span_sorted[-1] + 1, self.label]

        else:
            ## ..
            raise ValueError
            start = self.span_sorted[0]
            i = 1
            while self.span_sorted[i] == start + 1:
                start = self.span_sorted[i]
                i += 1
            first_part = [self.span_sorted[0], self.span_sorted[i - 1] + 1]
            second_part = [self.span_sorted[i], self.span_sorted[-1] + 1]

            assert first_part[-1] < second_part[0]
            assert self.is_contiguous(self.span_sorted[:i])
            assert self.is_contiguous(self.span_sorted[i:])
            return [first_part, second_part]

    def is_contiguous(self, array):
        prev = array[0]
        for i in range(1, len(array)):
            if array[i] != prev + 1:
                return False
            prev = array[i]
        return True

    def __len__(self):
        return len(self.span)

    def __le__(self, other):
        return len(self) < len(other)

    def is_continuous(self):
        span = self.span_sorted
        is_continuous = (span[0] + len(span) == span[-1] + 1)
        if is_continuous:
            return True
        else:
            return False

    def is_leaf(self):
        assert (self.children != [])
        return False

    def get_span(self):
        return self.span

    def get_yield(self, tokens):
        # Updates list of tokens
        for c in self.children:
            if c.is_leaf():
                tokens.append(c)
            else:
                c.get_yield(tokens)



    def merge_adjacent_tokens(self):
        i = 0
        to_add_child = []

        has_nonleaf = False
        for c in self.children:
            if isinstance(c, Tree):
                has_nonleaf = Tree
                break

        if has_nonleaf is False and self.is_continuous():
            return

        while True:
            if isinstance(self.children[i], Token):
                to_merge = [ self.children.pop(i)]
                assert isinstance(to_merge[-1], Token)
                if i >= len(self.children):
                    to_add_child.append(Tree(children=to_merge, label='NULL'))
                    break
                while isinstance(self.children[i], Token) and self.children[i].i == to_merge[-1].i + 1:
                    to_merge.append(self.children.pop(i))
                    assert isinstance(to_merge[-1], Token)
                    if i >= len(self.children):
                        # to_add_child.append(Tree(children=to_merge, label='NULL'))
                        break
                to_add_child.append(Tree(children=to_merge, label='NULL'))
            else:
                self.children[i].merge_adjacent_tokens()
                i+=1
            if i >= len(self.children):
                break

        self.children += to_add_child
        self.update_child()

        if not self.is_continuous():
            # # merge continuous span.
            # if len(self.children) == 2:
            #     return
            #
            # children = []
            # prev_dis = True
            #
            # while len(self.children) > 0:
            #     if not self.children[0].is_continuous():
            #         prev_dis  = True
            #         children.append([self.children.pop(0)])
            #         any_disc = True
            #     else:
            #         if prev_dis:
            #             children.append([self.children.pop(0)])
            #             prev_dis = False
            #         else:
            #             if self.children[0].span_sorted[0] == children[-1][-1].span_sorted[-1] + 1:
            #                 children[-1].append(self.children.pop(0))
            #             else:
            #                 children.append([self.children.pop(0)])
            #
            # new_children = []
            # for c in children:
            #     if len(c) == 1:
            #         new_children.append(c[0])
            #     else:
            #         new_children.append( Tree(children=c, label='NULL'))
            #
            # self.children += new_children
            # self.children = sorted(self.children, key=lambda x: min(x.get_span()))
            # self.update_child()
            #
            if len(self.children) == 2:
                return
            else:
                new = Tree(children=[self.children.pop(-1), self.children.pop(-1)], label='NULL')
                while len(self.children) > 1:
                    new = Tree(children=[new, self.children.pop(-1)], label='NULL')
                self.children = [new, self.children.pop(0)]
                self.update_child()

    def merge_unaries(self):
        # Collapse unary nodes
        for c in self.children:
            if not c.is_leaf():
                c.merge_unaries()


        if len(self.children) == 1 and not self.children[0].is_leaf():
            c = self.children[0]
            self.label = "{}@{}".format(self.label, c.label)
            self.children = c.children
            for c in self.children:
                c.parent = self

    def expand_unaries(self):
        # Cancel unary node collapse
        for c in self.children:
            if not c.is_leaf():
                c.expand_unaries()

        if "@" in self.label:
            split_labels = self.label.split("@")
            t = Tree(split_labels[-1], self.children)
            for l in reversed(split_labels[1:-1]):
                t = Tree(l, [t])
            self.label = split_labels[0]
            self.children = [t]
            t.parent = self

    def get_boundary(self):
        boundary = []
        boundary.append(self.span_sorted[0])
        boundary.append(self.span_sorted[-1]+1)
        for i in range(0, len(self.span_sorted)-1):
            if self.span_sorted[i] != self.span_sorted[i+1] -1 :
                boundary.append(self.span_sorted[i]+1)
                boundary.append(self.span_sorted[i+1])
        return list(set(boundary))

    def get_point_action(self, cursor_position):
        if self.is_continuous():
            has_nonleaf = False

            for c in self.children:
                if isinstance(c, Tree):
                    has_nonleaf = True
                    break

            # 向右的pointing
            if not has_nonleaf:
                cursor_position[self.span_sorted[0]].append([ self.span_sorted[-1]+1, self.label, 0, self.span_sorted, self.get_boundary(), self])
                return
            else:
                #向左的pointing.
                for child in self.children:
                    try:
                        child.get_point_action(cursor_position)
                    except:
                        print("????")
                        pdb.set_trace()
                cursor_position[self.span_sorted[-1]+1].append([self.span_sorted[0], self.label, 0, self.span_sorted, self.get_boundary(), self])

        #discontinuous
        else:
            try:
                assert len(self.children) == 2
            except:
                pdb.set_trace()

            for child in self.children:
                child.get_point_action(cursor_position)

            child_1_end = self.children[0].span_sorted[-1] + 1
            child_2_end = self.children[1].span_sorted[-1] + 1
            big = max(child_2_end, child_1_end)
            small = min(child_2_end, child_1_end)

            if child_1_end == big:
                self.sibling = self.children[1]
            else:
                self.sibling = self.children[0]

            cursor_position[big].append([small, self.label, 1, self.span_sorted, self.get_boundary(), self])

    def get_constituents(self, constituents):
        # Update set of constituents
        constituents.add((self.label, tuple(sorted(self.span))))
        for c in self.children:
            if not c.is_leaf():
                c.get_constituents(constituents)

    def __str__(self):
        return "({} {})".format(self.label, " ".join([str(c) for c in self.children]))

    def get_words(self):
        leaves = []
        self.get_yield(leaves)
        leaves = sorted(leaves, key=lambda x: x.i)
        pos = []
        word = []
        for leaf in leaves:
            pos.append(leaf.features)
            word.append(leaf.token)
        return word, pos


def get_yield(tree):
    # Returns list of tokens in the tree (in surface order)
    tokens = []
    tree.get_yield(tokens)
    return sorted(tokens, key=lambda x: min(x.get_span()))


def get_constituents(tree, filter_root=False):
    # Returns a set of constituents in the tree
    # Ignores root labels (from PTB, Negra, and Tiger corpora) if filter_root
    constituents = set()
    tree.get_constituents(constituents)
    if filter_root:
        constituents = {(c, i) for c, i in constituents if c not in {'ROOT', 'VROOT', 'TOP'}}
    return constituents



