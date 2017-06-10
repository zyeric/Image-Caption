def copy_value(list):
    ret = []
    for e in list:
        ret.append(e)
    return ret


in_file = 'D:/train.txt'
in2_file = 'D:/valid.txt'
out_dic_file = 'D:/valid_dictionary.txt'
out_vec_file = 'D:/valid_train_vector.txt'
out_vec2_file = 'D:/valid_valid_vector.txt'

sentences = []
valid_sentences = []

content = []
for line in open(in_file):
    line = line.split('\n')[0]
    if line.isdigit():
        if len(content) > 0:
            sentences.append(copy_value(content))
            content.clear()
    else:
        content.append(line)
sentences.append(copy_value(content))

content.clear()
for line in open(in2_file):
    line = line.split('\n')[0]
    if line.isdigit():
        if len(content) > 0:
            valid_sentences.append(copy_value(content))
            content.clear()
    else:
        content.append(line)
valid_sentences.append(copy_value(content))

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

for st in valid_sentences:
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

with open(out_vec2_file, 'w') as f:
    for index in range(len(valid_sentences)):
        f.write(repr(index+8001) + "\n")
        for sent in valid_sentences[index]:
            f.write("1 ")
            for i in range(len(sent)):
                #print(sent[i])
                #print(dic[sent[i]])
                f.write(repr(dic[sent[i]]) + " ")
            f.write("2\n")
