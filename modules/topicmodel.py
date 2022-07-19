import numpy as np
import pickle
import networkx as nx
from modules.gradient_color import linear_gradient
from modules.tools import rescale
from networkx import degree_centrality, eigenvector_centrality, betweenness_centrality, pagerank, katz_centrality, clustering
import os

'''
This class encapsulates a basic LDA topic model
It builds the topic-topic correlation graph and compute basic graph measures for the nodes (e.g., centrality)
'''

class topicmodel():
	
	def build_matrix_pzd(self):
		self.pzd = np.zeros((self.k, self.data.ndocs)) 
		for i, row in enumerate(self.lda[self.data.corpus]):
			for (topic_num, prop_topic) in row:
				self.pzd[topic_num, i] = prop_topic
		
	def compute_pz_pw(self):
		pwz = self.get_topics()
		self.pz = np.sum(self.pzd, axis=1)
		self.pz = self.pz / np.sum(self.pz)		
		self.pw = list(np.matmul(np.transpose(pwz), np.transpose(self.pz)))

	def __init__(self, data, filename, lda=None, k=100):
		self.k = k
		self.data = data
		# all 4 files we need
		self.file_lda = os.path.join("models", filename + ".pkl")
		self.file_pzd = os.path.join("models", filename + "_pzd.pkl")
		self.file_pz = os.path.join("models", filename + "_pz.pkl")
		self.file_triplets = os.path.join("models", filename + "_triplets.csv")
		###
		if data.generate:
			assert lda is not None			
			self.lda = lda
			with open(self.file_lda, "wb") as f:
				pickle.dump(self.lda, f)
			with open(self.file_pzd, "wb") as f:        
				self.build_matrix_pzd()
				pickle.dump(self.pzd, f)
		self.hidden_topics = []

	def load_model(self, light=False):
		with open(self.file_lda, "rb") as f:
			self.lda = pickle.load(f)
		if not light:
			with open(self.file_pzd, "rb") as f:        
				self.pzd = pickle.load(f)

	# return a *copy* of the matrix p(z|d)
	# list_ids allows to restrict to a limited number of d ids
	def get_pzd(self, list_ids=None):
		copy_pzd = np.copy(self.pzd)
		if (list_ids is None):
			return copy_pzd
		else:
		    return copy_pzd[:, list_ids]

	def save_pz(self):
		with open(self.file_pz, "wb") as f:
			pickle.dump(self.pz, f)

	def load_pz(self):
		with open(self.file_pz, "rb") as f:
			self.pz = pickle.load(f)

	# return the top n words from the vocabulary (based on model)
	def top_w(self, n=10):
		cc = {self.data.dico[i]: self.pw[i] for i in range(len(self.pw))}
		cc = sorted(cc.items(), key=lambda x: x[1], reverse=True)
		return cc[0:n]

	def get_pw(self, w):
		return self.pw[self.data.dico.token2id[w]]

	def compute_pzw(self):
		pwz = self.get_topics()
		

	def get_pzw(self, w):

		# ici ça ne va pas
		# -> il faut une mult de matrice pour tous les mots

		pwz_v = self.get_topics()[:, self.data.dico.token2id[w]]
		pzw_v = np.multiply(pwz_v, self.pz)
		pzw_v_norm = pzw_v / (np.sum(pzw_v))
		return pzw_v_norm


    # normalize p(z|d)
	def normalize_pzd(self):
		if self.pz is None:
			self.compute_pz_pw()
		self.pzd_norm = np.zeros((self.k, self.data.ndocs))
		for i in range(self.k):
			self.pzd_norm[i,:] = self.pzd[i,:]/self.pz[i]
		self.mat_zz_docs_norm = np.matmul(self.pzd_norm, np.transpose(self.pzd_norm))

	def get_node_name(self, i):
		return "z" + str(i).zfill(len(str(self.k-1)))

	def compute_edge_triplets(self):
		assert self.mat_zz_docs_norm is not None
		t = []
		for i in [t_i for t_i in range(self.k) if t_i not in self.hidden_topics]:
			for j in [t_j for t_j in range(self.k) if t_j not in self.hidden_topics]:
				if (i != j):
					t.append((self.get_node_name(i), self.get_node_name(j), float(self.mat_zz_docs_norm[i, j])))
		self.triplets = {(i, j): k for (i, j, k) in t}
		self.triplets = sorted(self.triplets.items(), key=lambda x: x[1], reverse=True)
		#return self.triplets

	def save_edge_triplets(self):
		st_p = ""
		for (i,j),k in self.triplets:
			st_p += i + "\t" + j + "\t" + str(k) + "\n"
		with open(self.file_triplets, "w") as f:
			f.write(st_p)		

	def load_edge_triplets(self):
		self.triplets = []
		with open(self.file_triplets, "r") as f:
			for row in f:
				row_sp = row.split("\t")
				self.triplets.append(((row_sp[0], row_sp[1]), row_sp[2]))

	def set_hidden_topics(self, h):
		self.hidden_topics = h

	def get_hidden_topics(self):
		return self.hidden_topics

	def compute_topwords(self, num_words=10):
		self.top_words = {}
		for i in range(self.k):
			#self.top_words["z"+str(i).zfill(2)] = [t for (t, w) in self.lda.show_topics(num_topics=self.k,formatted=False, num_words=num_words)[i][1]]
			self.top_words[self.get_node_name(i)] = self.lda.show_topics(num_topics=self.k,formatted=False, num_words=num_words)[i][1]

	def build_graph(self, num_top_edges = 300):
		assert self.triplets is not None
		self.num_top_edges = num_top_edges
		sorted_triplets_top = self.triplets[:self.num_top_edges]
		self.topic_graph = nx.empty_graph(0)
		self.node_list = set()
		for (i,j),k in sorted_triplets_top:
			if i not in self.node_list:
				self.node_list.add(i)
				self.topic_graph.add_node(i, size=1, title=i)
			if j not in self.node_list:
				self.node_list.add(j)        
				self.topic_graph.add_node(j, size=1, title=j)
		for (i,j),k in sorted_triplets_top:
			self.topic_graph.add_edge(i, j, weight=k)

	#def copy_graph(self):
	#	self.triplets

	def compute_graph_measures(self, dico_measures_names, num_values = 10):
	# num_values = number of buckets for the values (actually -1)
		# color gradient for the nodes
		gradient_color = linear_gradient("#553300", finish_hex="#FF3300", n=num_values+1)["hex"]
		self.topic_graph_m = {}
		self.m_rescaled = {}
		for measure in dico_measures_names.keys():
			try:
				if measure.__name__ == "pagerank":
					m = measure(self.topic_graph, alpha=0.9)
				else:
					m = measure(self.topic_graph)
				self.m_rescaled[measure] = rescale(m, num_values)
			except:
				m = {k:1 for k in self.node_list}
				self.m_rescaled[measure] = m
			self.topic_graph_m[measure] = self.topic_graph.copy()
			for k,v in self.m_rescaled[measure].items():
				self.topic_graph_m[measure].add_node(k, size=v, weight=v, value=v, color=gradient_color[v])
		
	def get_graphs(self):
		return self.topic_graph_m

	def get_measures_values(self, m):
		return self.m_rescaled[m]

	def get_topics(self):
		return self.lda.get_topics()

	def show_topics(self, formatted=False):
		return self.lda.show_topics(num_topics=self.k, formatted=formatted)

	def get_first_nothiddentopic(self):
		for t in range(self.k):
			if t not in self.hidden_topics:
				return self.get_node_name(t)
