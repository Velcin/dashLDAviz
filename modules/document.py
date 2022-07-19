#from nltk import tokenize
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
import datetime

#def remove_stopwords(texts, lang="english"):
#    stop_words = stopwords.words(lang)
#    return [ for doc in texts]

def clean_unicode(map_unicode, s):
	cl = s
	for c in map_unicode:
		cl = re.sub(c, map_unicode[c], cl)
	return cl

# encode a document

class document():

	def __init__(self, id, text, lang="english", id_src=None,
			id_line=0, author=None, date=None):
		self.id = id
		self.text = text
		self.id_src = id_src
		self.id_line = id_line
		self.lang = lang
		self.author = author
		if date is not None and (len(date)>0):
			self.date = datetime.datetime.strptime(date, "%d-%m-%Y")
		else:
			self.date = None   

	def clean(self, unicode=None, remove_sw=True):
		if remove_sw:
			stop_words = stopwords.words(self.lang)
		else:
			stop_words = []
		#list_tokens = simple_preprocess(self.text)
		#self.tokens = [t for t in tokens if t not in stop_words and unicode.keys()]
		self.tokens = [t for t in self.text if t not in stop_words and unicode.keys()]

	def get_date(self):
		return self.date
		

