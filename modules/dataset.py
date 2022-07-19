import os
import re
from gensim import corpora
from gensim.utils import simple_preprocess
from nltk import tokenize
import pickle
from pathlib import Path
from modules.document import document
import datetime

# some function I need

def sent_to_words(sentences):
	for sentence in sentences:
		yield(simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

map_unicode = {
	"“": "\"",
  	"”": "\"",
	" ̶" : "-",
	" ̕" : "\'"
}

'''
This class is able to deal with a textual dataset (clean, store, load)
There is also a set of filtering tools (work in progress)
'''

class dataset():
	
    def __init__(self, name, filename):
        self.name = name
        self.filename = filename
        self.corpus_filename = os.path.join("datasets", "corpus", self.filename + ".pkl")
        self.corpus_filename_light = os.path.join("datasets", "corpus", self.filename + "-light.pkl")

    def set_generate(self, generate=False, k=100):
        self.k = k
        self.generate = generate

	#def set_num_topics(self, k=100):
	#	self.k = k

    def number_docs(self):
        return self.ndocs

    def printff(self, n=10):
        #print(self.data_words[0:n])
        print(self.corpus[0:n])

	# read a directory with .txt files or a single .txt file
	# (one text per line)
    def read_file_txt(self, lang="english", min_len_doc=5):
        file_name_path = os.path.join("datasets", self.filename)
        lines = []
        sources = []
        ids = []

        # folder with .txt files
        if os.path.isdir(file_name_path):
            for file in os.listdir(file_name_path):
                if Path(file).suffix == '.txt':
                    file_path = os.path.join("datasets", self.filename, file)
                    with open(file_path) as f:
                        t = [line.strip() for line in f.readlines()]
                        lines += t
                        sources += len(t)*[file]
                        ids += [i for i in range(len(t))]
        else: # single .txt file
            with open(file_name_path) as f:
                if f.suffix == '.txt':
                    t = [line.strip() for line in f.readlines()]
                    lines += t
                    sources += len(t)*[self.filename]
                    ids+= [i for i in range(len(t))]
      	# create the doc list
        self.ndocs = 0
        self.docs = []
        for i in range(len(lines)):
            text = lines[i]
            #if type(text) != "str":
           # 	raise(Exception("error " + str(text)))
            doc_set = tokenize.sent_tokenize(text) 			
            for t in list(sent_to_words(doc_set)):
            	if (len(t) > min_len_doc):
                	doc = document(self.ndocs, t, lang=lang,
                    	id_src = sources[i], id_line = ids[i])
                	self.docs.append(doc) 
                	self.ndocs += 1

    def read_file_csv(self, lang="english", min_len_doc=5):
        file_name_path = os.path.join("datasets", self.filename + '.csv')
        lines = []
        sources = []
        ids = []
        authors = []
        dates = []
        with open(file_name_path) as f:
            for i, line in enumerate(f.readlines()):
                sp = line.split("\t")
                lines += [sp[3].strip()]
                sources += [sp[0].strip()]
                authors += [sp[2].strip()]
                dates += [sp[4].strip()]
                ids += [i]
      	# create the doc list
        self.ndocs = 0
        self.docs = []
        for i in range(len(lines)):
            text = lines[i]
            #if type(text) != "str":
           # 	raise(Exception("error " + str(text)))
            doc_set = tokenize.sent_tokenize(text) 			
            for t in list(sent_to_words(doc_set)):
            	if (len(t) > min_len_doc):
                	doc = document(self.ndocs, t, lang=lang,
                    	id_src = sources[i], id_line = ids[i],
                    	author = authors[i], date = dates[i])
                	self.docs.append(doc) 
                	self.ndocs += 1

    def clean(self, unicode=map_unicode, remove_sw=True):
        for d in self.docs:
            d.clean(unicode=unicode, remove_sw=remove_sw)

    def filter_words(self, word):
        l = []
        for d in self.docs:
            if word in d.tokens:
                l.append(d)
        return l

    def filter_author(self, name):
        l = []
        for d in self.docs:
            if d.author.lower() == name.lower():
                l.append(d)
        return l
    
    def filter_date(self, begin, end):
        l = []
        date_begin = datetime.datetime.strptime(begin, "%d-%m-%Y")
        date_end = datetime.datetime.strptime(end, "%d-%m-%Y")
        for d in self.docs:
            if (d.get_date() != None) and (d.get_date() >= date_begin) and (d.get_date() <= date_end):
                l.append(d)
        return l        

		# to filter which doc is used for the model
		#data_words_nostops = [d for d in data_words_nostops if len(d)>=min_len_doc]

	#def cut_into_sentences(self):
	# fonction qui génère les listes de mots (token) à partir des textes
	#	doc_set = []
#		for d in self.lines_cleaned:
			#doc_set += tokenize.sent_tokenize(d)  
		## on construit le corpus
		#self.data_words = list(sent_to_words(doc_set))

    def build_corpus(self, min_tf_word=5): #, stopwords="english"):
		# on retire les mots-outils
		#self.ndocs = len(data_words_nostops)
		# création du dictionnaire
		#self.dico = corpora.Dictionary(data_words_nostops)
        texts = [d.tokens for d in self.docs]
        self.dico = corpora.Dictionary(texts)
		# ce qui permet par ex. de filtrer le vocabulaire
        self.dico.filter_extremes(no_below=min_tf_word)
		# create Corpus
		#texts = data_words_nostops
		# matrice Term Document Frequency
        self.corpus = [self.dico.doc2bow(text) for text in texts]

    def load_corpus(self):
        with open(self.corpus_filename, "rb") as f:	
            [self.name, self.filename, self.ndocs, self.dico, self.corpus, self.docs] = pickle.load(f)

    def load_corpus_light(self):
        with open(self.corpus_filename_light, "rb") as f: 
            [self.name, self.filename, self.ndocs, self.dico] = pickle.load(f)

    def save_corpus(self):
        with open(self.corpus_filename, "wb") as f:	
            pickle.dump([self.name, self.filename, self.ndocs, self.dico, self.corpus, self.docs], f)	

    # save the essentiel information of a corpus, but not the documents themselves
    def save_corpus_light(self):
        with open(self.corpus_filename_light, "wb") as f: 
            pickle.dump([self.name, self.filename, self.ndocs, self.dico], f)   

    def is_corpus(self):
        return os.path.isfile(self.corpus_filename)

    def is_corpus_light(self):
        return os.path.isfile(self.corpus_filename_light)

