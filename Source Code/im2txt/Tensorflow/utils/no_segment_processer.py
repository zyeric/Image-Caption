def copy_value(list):
    ret = []
    for e in list:
        ret.append(e)
    return ret


in_file = 'D:/train.txt'
out_dic_file = 'D:/dictionary.txt'
out_vec_file = 'D:/train_vector.txt'

sentences = []

content = []
for line in open('D:/train.txt'):
    line = line.split('\n')[0]
    if line.isdigit():
        if len(content) > 0:
            sentences.append(copy_value(content))
            content.clear()
    else:
        content.append(line)
sentences.append(copy_value(content))

dic = {}
dic["<S>"] = 1
dic["</S>"] = 2

dic_size = 3

for st in sentences:
    for sent in st:
        for i in range(len(sent)):
            if sent[i] in dic:
                continue
            else:
                dic[sent[i]] = dic_size
                dic_size += 1

dic_size -= 1

#print(dic['大'])

#output map
with open(out_dic_file, 'w') as f:
    id2word = {}
    for k, v in dic.items():
        id2word[v] = k
    for i in range(dic_size):
        f.write(id2word[i+1] + " " + repr(i+1) + '\n')


#output vector
with open(out_vec_file, 'w') as f:
    for index in range(len(sentences)):
        f.write(repr(index+1) + "\n")
        for sent in sentences[index]:
            f.write("1 ")
            for i in range(len(sent)):
                #print(sent[i])
                #print(dic[sent[i]])
                f.write(repr(dic[sent[i]]) + " ")
            f.write("2\n")
