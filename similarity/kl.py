import pandas as pd
import time


def score(a, b):
    x1 = a.split(" ")
    x2 = b.split(" ")
    total1 = 0
    total2 = 0
    matrix1 = []
    for i in range(len(x1)):
        matrix1.append(0)
    matrix2 = []
    for i in range(len(x2)):
        matrix2.append(0)

    for i in range(len(x1)):
        index = 0
        score = 0
        for j in range(i, len(x1)):
            for k in range(index, len(x2)):
                if x1[j] in x2[k] or x2[k] in x1[j]:
                    grade = (len(x1) - j) * (len(x2) - k)
                    score += grade
                    index = k + 1
                    matrix1[j] = matrix1[j] + len(x1) - i
                    break
        total1 += score

    for i in range(len(x2)):
        index = 0
        score = 0

        for j in range(i, len(x2)):
            for k in range(index, len(x1)):
                if x2[j] in x1[k] or x1[k] in x2[j]:
                    grade = (len(x2) - j) * (len(x1) - k)
                    score += grade
                    index = k + 1
                    matrix2[j] = matrix2[j] + len(x2) - i
                    break

        total2 += score

    argument1 = []
    argument2 = []
    if total1 == total2:
        if total1 != 0:
            temp = [] + x1
            if temp.reverse() == x2 or x1 == x2:
                argument1 = x1
                argument2 = x1
            else:
                index = 0
                for j in range(len(x1)):
                    for k in range(index, len(x2)):
                        if x1[j] == x2[k] or x2[k] in x1[j]:
                            argument1.append(x1[j])
                            index = k + 1
                            break

                temp = []

                for i in range(len(matrix1)):
                    temp.append([matrix1[i], x1[i]])

                temp.sort()

                for i in temp:
                    if i[0] != 0:
                        argument2.append(i[1])
        else:
            argument1 = x1
            argument2 = x1

    if total1 > total2:
        index = 0
        for j in range(len(x1)):
            for k in range(index, len(x2)):
                if x1[j] == x2[k] or x2[k] in x1[j]:
                    argument1.append(x1[j])
                    index = k + 1
                    break

        temp = []

        for i in range(len(matrix1)):
            temp.append([matrix1[i], x1[i]])

        temp.sort()

        for i in temp:
            if i[0] != 0:
                argument2.append(i[1])

    if total1 < total2:
        index = 0
        for j in range(len(x2)):
            for k in range(index, len(x1)):
                if x2[j] == x1[k] or x1[k] in x2[j]:
                    argument1.append(x2[j])
                    index = k + 1
                    break

        temp = []

        for i in range(len(matrix2)):
            temp.append([matrix2[i], x2[i]])

        temp.sort()

        for i in temp:
            if i[0] != 0:
                argument2.append(i[1])
    final = 0
    if total1 > total2:
        final = total2 / total1
    else:
        final = total1 / total2
    print("KL learning score similarity: ", [final, total1, total2, " ".join(argument1), " ".join(argument2)])
    return [final, total1, total2, " ".join(argument1), " ".join(argument2)]


class KEDOLearning:
    def __init__(self, for_all=False, top_selection=10):
        self.raw_key_list = []
        self.raw_value_list = []
        self.key_index_class = {}
        self.value_index_class = {}
        self.index_key_class = {}
        self.index_value_class = {}
        self.key_value_mapping = {}
        self.key_class_index = {}
        self.value_class_index = {}
        self.key_value_class_index_mapping = pd.DataFrame()
        self.block_class_index = {}
        self.for_all = False
        self.block_relationship_rank = pd.DataFrame()
        self.block_index_id_list = {}
        self.top_selection = top_selection

        self.a_hit = []
        self.h_hit = []
        self.t_hit = []

    def key_value(self, key, value):
        find_key_class = ""
        find_value_class = ""
        find_key_class_index = 0
        find_value_class_index = 0
        if key in self.raw_key_list:
            find_key_class_index = self.key_index_class[key]
            find_key_class = self.index_key_class[find_key_class_index]
            if value in self.raw_value_list:
                find_value_class_index = self.value_index_class[value]
                find_value_class = self.index_value_class[find_value_class_index]
            else:
                stop = True
                if len(value.split(" ")) > 1:
                    for i in range(len(self.raw_value_list)):
                        s = score(value, self.raw_value_list[i])
                        if float(s[0]) >= float(s[1]) and s[0] != 0 and len(s[3].split(" ")) > 1:
                            if float(s[0]) == float(s[1]):
                                self.raw_value_list.insert(i + 1, value)
                            else:
                                self.raw_value_list.insert(i, value)
                            find_value_class = s[3]
                            if find_value_class not in self.value_class_index.keys():
                                find_value_class_index = len(self.value_class_index)
                                self.value_index_class.update({value: find_value_class_index})
                                self.index_value_class.update({find_value_class_index: find_value_class})
                                self.value_class_index.update({find_value_class: find_value_class_index})
                            else:
                                find_value_class_index = self.value_class_index[find_value_class]
                                self.value_index_class.update({value: find_value_class_index})
                            stop = False
                            break
                if stop:
                    find_value_class = value
                    find_value_class_index = len(self.value_class_index)
                    self.value_index_class.update({value: find_value_class_index})
                    self.index_value_class.update({find_value_class_index: find_value_class})
                    self.value_class_index.update({find_value_class: find_value_class_index})
        else:
            stop = True
            if len(key.split(" ")) > 1:
                for i in range(len(self.raw_key_list)):
                    s = score(key, self.raw_key_list[i])
                    if float(s[0]) >= float(s[1]) and s[0] != 0 and len(s[3].split(" ")) > 1:
                        if float(s[0]) == float(s[1]):
                            self.raw_key_list.insert(i + 1, key)
                        else:
                            self.raw_key_list.insert(i, key)
                        find_key_class = s[3]
                        if find_key_class not in self.key_class_index.keys():
                            find_key_class_index = len(self.key_class_index)
                            self.key_index_class.update({key: find_key_class_index})
                            self.index_key_class.update({find_key_class_index: find_key_class})
                            self.key_class_index.update({find_key_class: find_key_class_index})
                        else:
                            find_key_class_index = self.key_class_index[find_key_class]
                            self.key_index_class.update({key: find_key_class_index})
                        stop = False
                        break
            if stop:
                find_key_class = key
                find_key_class_index = len(self.key_class_index)
                self.key_index_class.update({key: find_key_class_index})
                self.index_key_class.update({find_key_class_index: find_key_class})
                self.key_class_index.update({find_key_class: find_key_class_index})
            if value in self.raw_value_list:
                find_value_class_index = self.value_index_class[value]
                find_value_class = self.index_value_class[find_value_class_index]
            else:
                stop = True
                if len(value.split(" ")) > 1:
                    for i in range(len(self.raw_value_list)):
                        s = score(value, self.raw_value_list[i])
                        if float(s[0]) >= float(s[1]) and s[0] != 0 and len(s[3].split(" ")) > 1:
                            if float(s[0]) == float(s[1]):
                                self.raw_value_list.insert(i + 1, value)
                            else:
                                self.raw_value_list.insert(i, value)
                            find_value_class = s[3]
                            if find_value_class not in self.value_class_index.keys():
                                find_value_class_index = len(self.value_class_index)
                                self.value_index_class.update({value: find_value_class_index})
                                self.index_value_class.update({find_value_class_index: find_value_class})
                                self.value_class_index.update({find_value_class: find_value_class_index})
                            else:
                                find_value_class_index = self.value_class_index[find_value_class]
                                self.value_index_class.update({value: find_value_class_index})
                            stop = False
                            break
                if stop:
                    find_value_class = value
                    find_value_class_index = len(self.value_class_index)
                    self.value_index_class.update({value: find_value_class_index})
                    self.index_value_class.update({find_value_class_index: find_value_class})
                    self.value_class_index.update({find_value_class: find_value_class_index})
        if find_key_class_index in self.key_value_class_index_mapping.index and find_value_class_index in self.key_value_class_index_mapping.columns:
            if str(self.key_value_class_index_mapping.loc[find_key_class_index, find_value_class_index]) == "nan":
                self.key_value_class_index_mapping.loc[find_key_class_index, find_value_class_index] = 1
            else:
                self.key_value_class_index_mapping.loc[find_key_class_index, find_value_class_index] += 1
        else:
            self.key_value_class_index_mapping.loc[find_key_class_index, find_value_class_index] = 1
        if key in self.key_value_mapping.keys():
            if value not in self.key_value_mapping[key]:
                self.key_value_mapping[key].append(value)
        else:
            self.key_value_mapping.update({key: [value]})
        if key not in self.raw_key_list:
            self.raw_key_list.append(key)
        if value not in self.raw_value_list:
            self.raw_value_list.append(value)
        return [find_key_class_index, find_value_class_index]

    def block(self, block, ID):
        block1_class = []
        for i in block:
            temp_input = i.split(":")
            temp = self.key_value(temp_input[0], temp_input[1])
            temp_output = [temp[0], temp[1]]
            block1_class.append(temp_output)
        stop = True
        temp_class_list = list(self.block_class_index.values())
        for i in range(len(temp_class_list)):
            if self.for_all:
                if all(x in temp_class_list[i] for x in block1_class) and len(block1_class) == len(temp_class_list[i]):
                    block_index = list(self.block_class_index.keys())[i]
                    ID = self.block_index_id_list[block_index]
                    stop = False
                    break
            else:
                if all(x in temp_class_list[i] for x in block1_class):
                    block_index = list(self.block_class_index.keys())[i]
                    ID = self.block_index_id_list[block_index]
                    stop = False
                    break
        if stop:
            block_index = len(self.block_class_index)
            self.block_class_index.update({block_index: block1_class})
            self.block_index_id_list.update({block_index: ID})
        return [block1_class, block_index, stop, ID]

    def rank(self, pairs, relationship):
        block_relationship_rank_copy = self.block_relationship_rank.copy()
        if len(block_relationship_rank_copy) == 0:
            self.update_rank(pairs, relationship)
            self.a_hit.append("Unknown")
            self.h_hit.append("Unknown")
            self.t_hit.append("Unknown")
            return relationship
        else:
            if all(x in list(block_relationship_rank_copy.keys()) for x in
                   pairs) and relationship in block_relationship_rank_copy.index:
                result = block_relationship_rank_copy.loc[:, pairs[0]]
                if len(pairs) > 1:
                    for i in pairs[1:]:
                        result += block_relationship_rank_copy.loc[:, i]
                result = result.sort_values(ascending=False)
                count = 0
                top = []
                temp_number = 10000000
                for m in result.index:
                    if result[m] < temp_number:
                        temp_number = result[m]
                        count += 1
                        if count == (self.top_selection + 1):
                            break
                        else:
                            top.append(m)
                    else:
                        top.append(m)

                if relationship in top:
                    # 2 pairs
                    pair1 = block_relationship_rank_copy.loc[relationship, pairs[0]]
                    pair2 = block_relationship_rank_copy.loc[relationship, pairs[1]]
                    check = False
                    if pair1 == 0:
                        self.h_hit.append("False")
                        check = True
                    else:
                        xresult = block_relationship_rank_copy.loc[:, pairs[0]]
                        xresult = xresult.sort_values(ascending=False)

                        count = 0
                        top = []
                        temp_number = 10000000
                        for m in xresult.index:
                            if xresult[m] < temp_number:
                                temp_number = xresult[m]
                                count += 1
                                if count == (self.top_selection + 1):
                                    break
                                else:
                                    top.append(m)
                            else:
                                top.append(m)
                        if relationship not in top:
                            check = True
                            self.h_hit.append("False")
                        else:
                            self.h_hit.append("True")

                    if pair2 == 0:
                        check = True
                        self.t_hit.append("False")
                    else:
                        xresult = block_relationship_rank_copy.loc[:, pairs[1]]
                        xresult = xresult.sort_values(ascending=False)

                        count = 0
                        top = []
                        temp_number = 10000000
                        for m in xresult.index:
                            if xresult[m] < temp_number:
                                temp_number = xresult[m]
                                count += 1
                                if count == (self.top_selection + 1):
                                    break
                                else:
                                    top.append(m)
                            else:
                                top.append(m)
                        if relationship not in top:
                            check = True
                            self.t_hit.append("False")
                        else:
                            self.t_hit.append("True")

                    if check:
                        self.a_hit.append("False")
                    else:
                        self.a_hit.append("True")

                    self.update_rank(pairs, relationship)
                    return relationship
                else:
                    # 2 pairs
                    pair1 = block_relationship_rank_copy.loc[relationship, pairs[0]]
                    pair2 = block_relationship_rank_copy.loc[relationship, pairs[1]]
                    check = False
                    if pair1 == 0:
                        self.h_hit.append("False")
                        check = True
                    else:
                        xresult = block_relationship_rank_copy.loc[:, pairs[0]]
                        xresult = xresult.sort_values(ascending=False)

                        count = 0
                        top = []
                        temp_number = 10000000
                        for m in xresult.index:
                            if xresult[m] < temp_number:
                                temp_number = xresult[m]
                                count += 1
                                if count == (self.top_selection + 1):
                                    break
                                else:
                                    top.append(m)
                            else:
                                top.append(m)
                        if relationship not in top:
                            check = True
                            self.h_hit.append("False")
                        else:
                            self.h_hit.append("True")

                    if pair2 == 0:
                        check = True
                        self.t_hit.append("False")
                    else:
                        xresult = block_relationship_rank_copy.loc[:, pairs[1]]
                        xresult = xresult.sort_values(ascending=False)

                        count = 0
                        top = []
                        temp_number = 10000000
                        for m in xresult.index:
                            if xresult[m] < temp_number:
                                temp_number = xresult[m]
                                count += 1
                                if count == (self.top_selection + 1):
                                    break
                                else:
                                    top.append(m)
                            else:
                                top.append(m)
                        if relationship not in top:
                            check = True
                            self.t_hit.append("False")
                        else:
                            self.t_hit.append("True")

                    if check:
                        self.a_hit.append("False")
                    else:
                        self.a_hit.append("True")

                    self.update_rank(pairs, relationship)
                    return result.keys()[0]
            else:
                self.a_hit.append("Unknown")
                self.h_hit.append("Unknown")
                self.t_hit.append("Unknown")
                self.update_rank(pairs, relationship)
                return relationship

    def update_rank(self, pairs, relationship):
        rank_score = 1 / len(pairs)
        block_relationship_rank_copy = self.block_relationship_rank.copy()
        if len(block_relationship_rank_copy) == 0:
            for i in pairs:
                block_relationship_rank_copy.loc[relationship, i] = rank_score
        elif relationship not in block_relationship_rank_copy.index:
            for i in pairs:
                block_relationship_rank_copy.loc[relationship, i] = rank_score
        else:
            for i in pairs:
                if i not in block_relationship_rank_copy.columns:
                    block_relationship_rank_copy.loc[relationship, i] = rank_score
                else:
                    block_relationship_rank_copy.loc[relationship, i] += rank_score
        self.block_relationship_rank = block_relationship_rank_copy.fillna(0)

    def build_block_relationship(self, block1, id1, block2, id2, relationship):
        a = self.block(block1, id1)
        b = self.block(block2, id2)
        block1_index_result = a[1]
        block2_index_result = b[1]
        block1_ID = a[3]
        block2_ID = b[3]
        block1_class_list = a[0]
        block2_class_list = b[0]
        temp_list = []

        for i in block1_class_list:
            temp_pair = str(i[0]) + "," + str(i[1])
            if temp_pair not in temp_list:
                temp_list.append(temp_pair)
        for j in block2_class_list:
            temp_pair = str(j[0]) + "," + str(j[1])
            if temp_pair not in temp_list:
                temp_list.append(temp_pair)
        output = self.rank(temp_list, relationship)
        return output
