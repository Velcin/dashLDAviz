import pandas as pd
import os
import modules.topicmodel

'''
A set of models, metadata in a dataframe stored in a simple csv file
'''

class models():

	def __init__(self, filename, data):
		self.models = {}
		self.dataset = data
		self.filename = os.path.join("models", filename + ".csv")
		if os.path.isfile(self.filename):
			self.df = pd.read_csv(self.filename, sep="\t")
		else:
			self.df = pd.DataFrame(columns=["id", "data", "k", "filename"])
			self.df.to_string(index=False)
			self.df.index.name = "model"
			self.save_df()

	def save_df(self):
		self.df.to_csv(self.filename, sep="\t", index=False)

	def get_new_filename(self, dataname, k):
		new_index = self.df.shape[0] + 1
		return "model_" + dataname + "_k" + str(k) + "_" + str(new_index)

	def add_model(self, tm):
		new_index = self.df.shape[0] + 1
		new_row = {"id": new_index, "data": tm.data.name, "k": tm.k, "filename": self.get_new_filename(tm.data.name, tm.k)}
		self.df = self.df.append(new_row, ignore_index = True)
		self.save_df()

	def get_all_models(self):
		return self.df

	def get_ids_models(self):
		return list(self.df["id"])

	def get_loaded_models(self):
		return self.models.keys()

	def get_model(self, id):
		if id in self.models.keys():
			return self.models[id]
		else:
			return None

	# load the models associated to the list of ids
	def load_models(self, ids, light=False):
		for id in ids:
			if id in list(self.df["id"]):
				d = self.df.loc[self.df["id"]==id, "data"].item()
				k = self.df.loc[self.df["id"]==id, "k"].item()
				filename = self.df.loc[self.df["id"]==id, "filename"].item()
				# we're loading a model so we must switch off the generation process
				self.dataset[d].set_generate(generate=False, k=k)			
				self.models[id] = modules.topicmodel.topicmodel(self.dataset[d], filename, lda=None, k=k)
				self.models[id].load_model(light=light)
			else:
				print(str(id) + " non trouvé")

	def add_measure(self, id, measure, value):
		if id in list(self.df["id"]):
			self.df.loc[self.df["id"] == id, measure] = value
		self.save_df()
		#try:
	#		self.df.loc[id, measure] = value
	#	except:
#			if id in list(self.df["id"]):
#				print("model exists but not the measure, we add a column")
#				nrows = self.df.shape[0]
#				self.df[measure] = [-1 for i in range(nrows)]
#				self.df.loc[id, measure] = value
#			else:
#				print("model doesn't exist, do nothing")
