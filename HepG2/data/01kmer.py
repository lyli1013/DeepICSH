from Bio import SeqIO
import numpy as np
import hickle as hk

acgt2num = {'A': 0,'C': 1,'G': 2,'T': 3,'N': 0}
complement = {'A': 'T','C': 'G','G': 'C','T': 'A'}
num2acgt = {0:'A',1:'C',2:'G',3:'T'}
def seq2mat(seq):
    seq = seq.upper()
    h = 4
    w = len(seq)   # 200
    mat = np.zeros((h, w), dtype=bool)  # True or false in mat
    for i in range(w):
        mat[acgt2num[seq[i]], i] = 1.
        # print(np.shape(mat))
    return mat.reshape((np.shape(mat)[0], np.shape(mat)[1], 1))


def process_fasta(filename):
    nums = []
    for seq_record in SeqIO.parse(filename, "fasta"):
        seq = seq_record.seq
        # print(seq)
        nums.append(seq2mat(seq))
    return nums
# def seq_to_kspec(seq, K=6):
#     encoding_matrix = {'a':0, 'A':0, 'c':1, 'C':1, 'g':2, 'G':2, 't':3, 'T':3, 'n':0, 'N':0}
#     kspec_vec = np.zeros((4**K,1))
#     for i in range(len(seq)-K+1):
#         sub_seq = seq[i:(i+K)]
#         index = 0
#         for j in range(K):
#             index += encoding_matrix[sub_seq[j]]*(4**(K-j-1))
#         kspec_vec[index] += 1
#     return kspec_vec

def seq_to_kspec(seq, K=6):
    encoding_matrix = {'a':0, 'A':0, 'c':1, 'C':1, 'g':2, 'G':2, 't':3, 'T':3, 'n':0, 'N':0}
    kspec_vec = np.zeros((4**K,1))
    for i in range(len(seq)-K+1):
        sub_seq = seq[i:(i+K)]
        index = 0
        for j in range(K):
            index += encoding_matrix[sub_seq[j]]*(4**(K-j-1))
        weight = (i+K)/len(seq)  # 线性权重函数
        kspec_vec[index] += weight
        # print(kspec_vec)
    return kspec_vec / np.sum(kspec_vec)  # 对kspec_vec进行归一化，使得它的和为1
    #return kspec_vec     # 对kspec_vec进行归一化，使得它的和为1


pos = '/data/lyli/Silencer/HepG2/datasets/1679/12D/data_n=20/Silencer/0HepG2_200bp.fa'
neg = '/data/lyli/Silencer/HepG2/datasets/1679/12D/data_n=20/NS/HepG2_negative_fasta1679_200bp.fa'
# print(process_fasta(pos))
pos_train_vec = np.array(process_fasta(pos))
neg_trian_vec = np.array(process_fasta(neg))
print(np.shape(pos_train_vec))
print(np.shape(neg_trian_vec))

train_data = np.concatenate((pos_train_vec, neg_trian_vec), axis=0)
print(np.shape(train_data))
hk.dump(train_data, "/data/lyli/Silencer/HepG2/datasets/1679/12D/data_n=20/HepG2_data.hkl")

train_data = train_data.reshape(-1, 4, 200, 1)
num2acgt = {0:'A',1:'C',2:'G',3:'T'}
#K1 = 5
#K1 = 6
#K1 = 8
#K1 = 4
#K1 = 3
K1 = 5
train_data_kmer = []
for ind in range(train_data.shape[0]):
    seq = ''
    for i in np.argmax(train_data[ind].reshape(4, 200), axis=0):
        seq += num2acgt[i]
    print(seq)
    train_data_kmer.append(seq_to_kspec(seq, K=K1))
train_data_kmer = np.array(train_data_kmer).reshape(-1, 4 ** K1)
print(np.shape(train_data_kmer))
print(type(train_data_kmer))
# for i in train_data_kmer:
#     print(type(i))
#     print(np.shape(i))
#     print(i)
np.savetxt("/data/lyli/Silencer/HepG2/datasets/1679/12D/data_n=20/HepG2_5kmerline_data.txt", train_data_kmer, encoding='utf_8_sig', fmt='%f', delimiter=' ')