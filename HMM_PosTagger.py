#!/usr/bin/env python
from collections import defaultdict
import math
import random
import sys
import csv

# return nested-list [ntrain][0:words,1:tags][words or tags]. NO <s>
def read_train(filename):
#{{{
    res=[]
    words=[]
    tags=[]
    with open(filename, "r") as f:
        for line in f:
            if (len(line.strip('\n'))==0):
                res.append([words, tags])
                words=[]
                tags=[]
            else:
                strs=line.strip('\n').split('\t')
                words.append(strs[1])
                tags.append(strs[2])
        if(len(words)>0):
            res.append([words, tags])
    return res
#}}}

def write_train(filename, train_data):
#{{{
    is1st=True
    with open(filename, "w") as f:
        for data in train_data:
            if(is1st):
                is1st=False
            else:
                f.write("\n")
            if(data[0][0] == "<s>"):
                for i in range(1,len(data[0])):
                    f.write("%d\t%s\t%s\n" %(i, data[0][i], data[1][i]))
            else:
                for i in range(len(data[0])):
                    f.write("%d\t%s\t%s\n" %(i+1, data[0][i], data[1][i]))
#}}}

# read test_file (for real prediction). similar format with train, but no tag column. 
# in-memory structure of test_data is consistent with train_data, i.e. [ntest][0:words][words]
def read_test(filename): 
#{{{
    res=[]
    words=[]
    with open(filename, "r") as f:
        for line in f:
            if (len(line.strip('\n'))==0):
                res.append([words])
                words=[]
            else:
                strs=line.strip('\n').split('\t')
                words.append(strs[1])
        res.append([words])
    return res
#}}}
    

#add <s>:<s> for each sentence in train_data
def add_SentenseStart(train_data):
#{{{
    for data in train_data:
        if(data[0][0] != "<s>"):
            data[0].insert(0, "<s>")
            if(len(data)>1):
                data[1].insert(0, "<s>")
    return train_data
#}}}

# return a dict: word -> max-frequent-tag. Ties are break in aribitrary
def get_words_MostFreqTag(train_data):
#{{{
    model=dict()
    tag_counter=dict()
    for data in train_data:
        for w,t in zip(data[0], data[1]):
            if(w not in model):
                model[w]=dict()
            if(t not in model[w]):
                model[w][t]=1
            else:
                model[w][t]+=1
    res_model=dict()
    for w in model:
        max_count=-1
        for t in model[w]:
            if(model[w][t]>max_count):
                max_count=model[w][t]
                max_tag=t
        res_model[w]=max_tag
    return res_model
#}}}

# return dict: word -> count
def get_words_count(train_data):
#{{{
    word_counter=dict()
    for data in train_data:
        for w in data[0]:
            if(w not in word_counter):
                word_counter[w]=1
            else:
                word_counter[w]+=1
    return word_counter
#}}}

# for all low_frequent_words (<= freq_limit, e.g. 1), find the most likely tag (i.e. used for unknown word most-likelihood tag for baseline model)
def get_LowFreqWord_tag(word_counter, word_MFT, freq_limit):
#{{{
    dct=dict()
    for w in word_counter:
        if(word_counter[w] <= freq_limit):
            t=word_MFT[w]
            if(t not in dct):
                dct[t]=1
            else:
                dct[t]+=1
    max_cnt=-1
    for t in dct:
        if(dct[t]>max_cnt):
            max_cnt=dct[t]
            max_tag=t
    return max_tag
#}}}


# baseline_model = return value of get_words_MostFreqTag, output. unknown_tag can be some 'UNK' or some other known tag 
# return format, similar to train data format, but [n_sentence][0:word, 1:tag, 2:if_known][word/tag/if_known seq]
def baseline_test(baseline_model, test_data, unknown_tag):
#{{{
    results=[]
    for idx, data in enumerate(test_data):
        tags=[]
        known_flags=[]
        for jdx,w in enumerate(data[0]):
            if(w in baseline_model):
                tags.append(baseline_model[w])
                known_flags.append(True)
            else:
                tags.append(unknown_tag)
                known_flags.append(False)
        results.append([data[0],tags, known_flags])
    return results
#}}}
   
# return train_data format, containing only unknown word (indicated by unknown_index_list)
def get_unknown_wordtags(train_data, unknown_index_list):
#{{{
    unknown_wordtags = []
    last_idx=-1
    for unkidx in unknown_index_list:
        if(unkidx[0] != last_idx):
            if(last_idx>0):
                unknown_wordtags.append([sentOfWords, sentOfTags])
            sentOfWords=[]
            sentOfTags=[]
            last_idx = unkidx[0]
        sentOfWords.append(train_data[unkidx[0]][0][unkidx[1]])
        sentOfTags.append(train_data[unkidx[0]][1][unkidx[1]])
    unknown_wordtags.append([sentOfWords, sentOfTags])
    return (unknown_wordtags)
#}}}
    
#
class NFolderPartitioner:  
#{{{
    def __init__(self, nfolder, data): #data must has its outer-most dimension representing the number-of-samples
        self.nfolder=nfolder
        self.data=data
        self.n_data=len(data)
        self.n_eachfolder = int(self.n_data / nfolder)

    def getFolder(self, ifolder): #return train/test, ifolder in [0,..,nfolder-1]
        istart_test = (self.nfolder - 1 - ifolder) * self.n_eachfolder
        iend_test = self.n_data if ifolder == 0 else (self.nfolder - ifolder) * self.n_eachfolder
        test_data = self.data[istart_test:iend_test]
        train_idx = list(range(0,istart_test))
        train_idx.extend(list(range(iend_test,self.n_data)))
        train_data = [self.data[x] for x in train_idx]
        return (train_data, test_data)
#}}}        


#pred_data/gold_data, both format of train_data
def evaluate_simple(pred_data, gold_data):
#{{{
    count_correct=0
    count_all=0
    for sentPred, sentGold in zip(pred_data, gold_data):
        for tPred, tGold in zip(sentPred[1], sentGold[1]):
            if(tGold != "<s>"):
                count_all+=1
                if(tGold == tPred):
                    count_correct+=1
    rate_correct = float(count_correct) / count_all
    return (count_all, count_correct, rate_correct)
#}}}

#count_known, count_unknown, rate_known_corr, rate_unk_corr
def evaluate_known_unknown(pred_data, gold_data):
#{{{
    count_known=0
    count_known_corr=0
    count_unk=0
    count_unk_corr=0
    for sentPred, sentGold in zip(pred_data, gold_data):
        for tPred, tGold, known in zip(sentPred[1], sentGold[1], sentPred[2]):
            if(tGold != "<s>"):
                if(known):
                    count_known+=1
                    if(tGold == tPred):
                        count_known_corr+=1
                else:
                    count_unk+=1
                    if(tGold == tPred):
                        count_unk_corr+=1
    rate_known_corr = float(count_known_corr) / count_known
    if(count_unk > 0):
        rate_unk_corr = float(count_unk_corr) / count_unk
    else:
        rate_unk_corr = None
    return (count_known, count_unk, rate_known_corr, rate_unk_corr)
#}}}
    
   
#return a word correct_rate (dict)
def evaluate_byword(pred_data, gold_data):
#{{{
    word_count=defaultdict(lambda: 0)
    word_corr_count=defaultdict(lambda: 0)
    for sentPred, sentGold in zip(pred_data, gold_data):
        for i in range(0,len(sentGold[0])):
            if(sentGold[1][i] != "<s>"):
                word_count[sentGold[0][i]] +=1
                if(sentGold[1][i] == sentPred[1][i]):
                    word_corr_count[sentGold[0][i]] +=1
    word_corr_rate=defaultdict()
    for w in word_count:
        word_corr_rate[w] = word_corr_count[w] / word_count[w]
    return word_corr_rate
#}}}



##----Viterbi----
class Viterbi_Decoder:

    def __init__(self):
        self.logspace=True

    def train(self, train_data):
        self.scan_tags(train_data)
        self.setupTransCountMatrix(train_data)
        self.setupTransProbMatrix(smooth_option=1, smooth_para=(0.25,))
        self.setupEmissCountMatrix(train_data)
        self.setupEmissProbMatrix(unk_option=2, unk_para=(1e-6,))
    
    # return the pred_data [n_sentence][0:word, 1:predTag][:]
    def test(self, test_data):
        pred_data=[]
        for data in test_data:
            (predTags, prob) = self.decode(data[0])
            known_flags = []
            for w in data[0]:
                if(w in self.words):
                    known_flags.append(True)
                else:
                    known_flags.append(False)
            pred_data.append([data[0], predTags, known_flags])
        return pred_data
        

    # self.tags, self.tag_counts, self.n_tag (both <s>, '.' are considered as tags)
    def scan_tags(self, train_data):
    #{{{
        self.tags=set()
        self.words=set()
        self.tag_counts=defaultdict(lambda: 0)
        self.word_counts=defaultdict(lambda: 0)
        for data in train_data:
            for (w,t) in zip(data[0],data[1]):
                self.tags.add(t)
                self.tag_counts[t] += 1
                self.words.add(w)
                self.word_counts[w] += 1
        self.n_tag = len(self.tags)
    #}}}
        
    # self.trans_count_matrix, actually, nested dict (defaultdict)
    def setupTransCountMatrix(self, train_data):
    #{{{
        self.trans_count_matrix=defaultdict(lambda: defaultdict(lambda: 0))
        for data in train_data:
            for i in range(len(data[1])-1):
                tag1=data[1][i]
                tag2=data[1][i+1]
                self.trans_count_matrix[tag1][tag2] += 1
    #}}}

    # self.trains_prob_matrix: in log space
    def setupTransProbMatrix(self, smooth_option=0, smooth_para=0):
    #{{{
        self.trans_prob_matrix=defaultdict(lambda: defaultdict(lambda: 0))
        if(smooth_option == 0):  #no smooth
            for tag1 in self.tags:
                for tag2 in self.tags:
                    value=float(self.trans_count_matrix[tag1][tag2]) / self.tag_counts[tag1]
                    self.trans_prob_matrix[tag1][tag2] = math.log(value) if self.logspace else value 
        elif (smooth_option == 1): # add-k smoothing, k is smooth_para[0]
            k=smooth_para[0]
            for tag1 in self.tags:
                for tag2 in self.tags:
                    value = (float(self.trans_count_matrix[tag1][tag2]) + k) / (self.tag_counts[tag1] + k*self.n_tag)
                    self.trans_prob_matrix[tag1][tag2] = math.log(value) if self.logspace else value 
    #}}}
            
    # self.emiss_count_matrix, actually, nested dict (defaultdict)
    def setupEmissCountMatrix(self, train_data):
    #{{{
        self.emiss_count_matrix=defaultdict(lambda: defaultdict(lambda: 0))
        for data in train_data:
            for (w,t) in zip(data[0], data[1]):
                self.emiss_count_matrix[t][w] +=1
    #}}}

    # return tag_lowfrq_ratio)
    def getEmissionLowFrqCount(self):
        tag_lowfrq_ratio=dict()
        for t in self.tags:
            cnt=0
            for w in self.emiss_count_matrix[t]:
#                if(self.emiss_count_matrix[t][w]==1):
                if(self.word_counts[w] == 1):
                    cnt+=1
            tag_lowfrq_ratio[t] = float(cnt)/self.tag_counts[t]
        self.tag_lowfrq_ratio=tag_lowfrq_ratio
       # print(tag_lowfrq_ratio)
        return tag_lowfrq_ratio

    # for unk_option==2, use
    def initEmissProbMatrix(self, factor):
        self.emiss_prob_matrix=dict()
        for t in self.tags:
            if(self.tag_lowfrq_ratio[t]==0): 
                self.tag_lowfrq_ratio[t]=1e-6
            self.emiss_prob_matrix[t] = defaultdict(lambda v=self.tag_lowfrq_ratio[t]*factor: math.log(v))
    
    # unk_option=1, use a fix value for unknown words, fix value = unk_para[0]; 
    # unk_option=2, call initEmissProbMatrix (which makes use of tag_lowfrq_ratio), i.e. different tags have different emission default prob. unk_para[0]: factor
    def setupEmissProbMatrix(self, unk_option=1, unk_para=None):
        if(unk_option == 1):
            if(self.logspace):
                self.emiss_prob_matrix=defaultdict(lambda : defaultdict(lambda : math.log(unk_para[0])))
            else:
                self.getEmissionLowFrqCount()
                self.emiss_prob_matrix=defaultdict(lambda : defaultdict(lambda v=unk_para[0]: v))  ## no improvement
        elif(unk_option== 2):
            self.getEmissionLowFrqCount()
            self.initEmissProbMatrix(unk_para[0])
        for t in self.tags:
            for w in self.emiss_count_matrix[t]:
                value=float(self.emiss_count_matrix[t][w]) /self.tag_counts[t]
                self.emiss_prob_matrix[t][w] = math.log(value)  if self.logspace else value

    def decode(self,a_sentence):
        if(self.logspace):
            return self.decode_log(a_sentence)
        else:
            return self.decode_nonlog(a_sentence)
    
    # use _log version (recommend)
    #input, a_sentence is list of word, begins with <s>, end with '.'
    def decode_nonlog(self, a_sentence): #{{{
        viterbi=defaultdict(lambda: []) # tag -> list, viterbi[tag][i] -> prob until now
        backpoint=defaultdict(lambda: []) # backpoint[tag][i] -> previous tag (for i-1)
        xtags=set(self.tags) #effective tags
        xtags.remove("<s>")
        #xtags.remove(".")
        for tag in xtags:
            viterbi[tag].append(1)  #viterbi[tag][0] == 1, i.e. the prob until <s>
            viterbi[tag].append( self.trans_prob_matrix["<s>"][tag] * self.emiss_prob_matrix[tag][a_sentence[1]] )
            backpoint[tag].append("BPEND")
            backpoint[tag].append("<s>")
        for iword in range(2, len(a_sentence)):
            w=a_sentence[iword]
            for tag in xtags:
                maxp=-1
                for ptag in xtags:
                    thisp=viterbi[ptag][iword-1] * self.trans_prob_matrix[ptag][tag] * self.emiss_prob_matrix[tag][w]
                    if(thisp > maxp):
                        maxp = thisp
                        bp = ptag
                viterbi[tag].append(maxp)
                backpoint[tag].append(bp)
        bestpathprob = viterbi["."][len(a_sentence)-1]
        bestpathpointer = "."
        bestpath=[]
        cnt=len(a_sentence)-1
        while(cnt>=0):
            bestpath.append(bestpathpointer)
            if(bestpathpointer=="<s>"):
                break
            bestpathpointer = backpoint[bestpathpointer][cnt]
            cnt-=1
        sentOfTags=list(reversed(bestpath))
        return (sentOfTags, bestpathprob) 
    #}}}
       
    def decode_log(self, a_sentence):
        viterbi=defaultdict(lambda: []) # tag -> list, viterbi[tag][i] -> prob until now
        backpoint=defaultdict(lambda: []) # backpoint[tag][i] -> previous tag (for i-1)
        xtags=set(self.tags) #effective tags
        xtags.remove("<s>")
        #xtags.remove(".")
        for tag in xtags:
            viterbi[tag].append(0)  #viterbi[tag][0] == 1, i.e. the prob until <s>
            viterbi[tag].append( self.trans_prob_matrix["<s>"][tag] + self.emiss_prob_matrix[tag][a_sentence[1]] )
            backpoint[tag].append("BPEND")
            backpoint[tag].append("<s>")
        for iword in range(2, len(a_sentence)):
            w=a_sentence[iword]
            for tag in xtags:
                maxp=-math.inf
               # if(w not in self.words):
               #     print((w,tag,self.emiss_prob_matrix[tag][w]), math.exp(self.emiss_prob_matrix[tag][w]))
                for ptag in xtags:
                    thisp=viterbi[ptag][iword-1] + self.trans_prob_matrix[ptag][tag] + self.emiss_prob_matrix[tag][w]
                    if(thisp > maxp):
                        maxp = thisp
                        bp = ptag
                viterbi[tag].append(maxp)
                backpoint[tag].append(bp)
        bestpathprob = math.exp(viterbi["."][len(a_sentence)-1])
        bestpathpointer = "."
        bestpath=[]
        cnt=len(a_sentence)-1
        while(cnt>=0):
            bestpath.append(bestpathpointer)
            if(bestpathpointer=="<s>"):
                break
            bestpathpointer = backpoint[bestpathpointer][cnt]
            cnt-=1
        sentOfTags=list(reversed(bestpath))
        return (sentOfTags, bestpathprob) 
            
def count_unique_sents(train_data):
#{{{
    sents=set()
    for data in train_data:
        sent=" ".join(data[0])
        sents.add(sent)
    return len(sents)
#}}}

#generate confusion matrix
def evaluate_confusionmatrix(pred_data, gold_data, tags, conf): 
#{{{
    for (dataP, dataG) in zip(pred_data, gold_data):
        for (tagP, tagG) in zip(dataP[1], dataG[1]):
            if(tagP == "<s>" or tagP == "."):
                continue
            conf[tagG][tagP] += 1
#}}}

def output_confusionmatrix_csv(confmtrx, csvfile):
#{{{
    tags=sorted(list(confmtrx.keys()))
    with open(csvfile, "w") as f:
        writer=csv.writer(f)
        line=["gold"]
        line.extend(tags)
        writer.writerow(line)
        writer.writerow(["pred"])
        for tagP in tags:
            line=[tagP]
            for tagG in tags:
                line.append(confmtrx[tagG][tagP])
            writer.writerow(line)
#}}}
    
#return unknown words
def find_unknown(pred_data):
#{{{
    unk_words=list()
    for data in pred_data:
        for (w,known) in zip(data[0],data[2]):
            if(known == False):
                unk_words.append(w)
    return unk_words
#}}}

def help():
    print('''
<Usage>:
#0. (no parameter): train model with default training data, then tag for user stdin sentences.
#1. -shuffle <output file shuffled>
#2. -traindev [baseline | viterbi] <input of train data> 
#3. -predict <input of train data> <input of test data> <output of pred data>
''')
    exit(0)


##functions:
#1. -shuffle <output file shuffled>
#2. -traindev [baseline | viterbi] <input of train data> 
#3. -predict <input of train data> <input of test data> <output of pred data>

if __name__=="__main__":
    argvs=sys.argv
    #argvs=["shen-si-assgn2.py", "-traindev", "viterbi", "berp-POS-training_shuffled.txt"]

    if (len(argvs) > 1 and argvs[1] == "-h"):
        help()
    if (len(argvs)==3 and argvs[1] == "-shuffle"):  #do shuffle
        train_all = read_train("berp-POS-training.txt")
        random.shuffle(train_all)
        write_train(argvs[2], train_all)
        exit(0)
    elif (len(argvs)==5 and argvs[1] == "-predict"):  #do predict
        train_file = argvs[2]
        test_file = argvs[3]
        output_pred_data= argvs[4]
        train_data = add_SentenseStart(read_train(train_file))
        test_data= add_SentenseStart(read_test(test_file))
        ## to train, predict, and output
        Vb_decoder=Viterbi_Decoder()
        Vb_decoder.train(train_data)
        pred_data = Vb_decoder.test(test_data)
        write_train(output_pred_data, pred_data)
        print("Summary")
        unkwords=find_unknown(pred_data)
        print("unknown words (tokens): %d", len(unkwords))
        print(unkwords)
    elif (len(argvs)==4 and argvs[1] == "-traindev"):  #do 5folder train/test
        train_data_all = read_train(argvs[3])
        train_data_all = add_SentenseStart(read_train(argvs[3]))
        #print((len(train_data_all), count_unique_sents(train_data_all)))
        ##randomize
        #random.shuffle(train_data_all)
        all_scores=[] 
        all_knw_scores=[]
        all_unk_scores=[]
        confmtrix=defaultdict(lambda: defaultdict(lambda: 0))
        folder5 = NFolderPartitioner(5, train_data_all)
        for ifolder in range(5):
            train_data, test_data = folder5.getFolder(ifolder)  #split data into train + test
            ##Baseline Model
            if(argvs[2] == "baseline"):
                baseline_model = get_words_MostFreqTag(train_data)
                print(len(baseline_model.keys()))
                #--  method 1, use 
                word_counts = get_words_count(train_data)
                unk_tag = get_LowFreqWord_tag(word_counts, baseline_model, 1)
                print("find LowFreqWord tag: "+ unk_tag)
                pred_baseline1 = baseline_test(baseline_model, test_data, unk_tag)  #predict
                evals=evaluate_simple(pred_baseline1, test_data)  #evaluate simple
                print(evals)
                evals_knw_unk = evaluate_known_unknown(pred_baseline1, test_data) #evaluate by known & unknown
                print(evals_knw_unk)
                all_scores.append(evals[2])
                all_unk_scores.append(evals_knw_unk[3])
                all_knw_scores.append(evals_knw_unk[2])
            else: 
                ##   Viterbi Model ------
                # test Viterbi
                Vb_decoder = Viterbi_Decoder()
                Vb_decoder.train(train_data)
                #tag_emisslowfrq_ratio = Vb_decoder.getEmissionLowFrqCount()
                #print(tag_emisslowfrq_ratio)
                # batch test
                pred_data = Vb_decoder.test(test_data)  #predict
                evals=evaluate_simple(pred_data, test_data) # evaluate_simple
                print(evals)
                evals_knw_unk = evaluate_known_unknown(pred_data, test_data) #evaluate by known & unknown
                print(evals_knw_unk)
                all_scores.append(evals[2])
                all_unk_scores.append(evals_knw_unk[3])
                all_knw_scores.append(evals_knw_unk[2])
                evaluate_confusionmatrix(pred_data, test_data, Vb_decoder.tags, confmtrix)
        print(sum(all_scores)/5)
        print(sum(all_knw_scores)/5)
        print(sum(all_unk_scores)/5)
        output_confusionmatrix_csv(confmtrix, "conf_matrix.csv")
    else:
        train_file = "berp-POS-training_shuffled.txt"
        print("read training file: %s" %train_file)
        train_data = add_SentenseStart(read_train(train_file))
        #test_data= add_SentenseStart(read_test(test_file))
        ## to train, predict, and output
        print("start training HMM model")
        Vb_decoder=Viterbi_Decoder()
        Vb_decoder.train(train_data)
        import re
        print("training done, please write sentences to decode: (ctrl+c) to exit")
        for line in sys.stdin:
            line=re.sub(r"(\.|\?|!)"," .",line)
            words = line.strip().lower().split()
            words.insert(0, "<s>")
            out_ws, out_ts, out_in = Vb_decoder.test([[words]])[0]  #out_ws: output_words; out_ts: output tags; out_in: if word has been met in training data
            out_word_tag_pairs =[ (w, t) for (w,t) in zip(out_ws,out_ts) if w != "<s>" and w != "." ]
            print(out_word_tag_pairs)

        exit(0)
        
