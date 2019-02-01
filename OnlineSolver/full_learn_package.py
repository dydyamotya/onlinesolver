
import os
import main_cut_package as mcp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def get_gases(path, list_of_folders):
	gases = set()
	for folder in list_of_folders:
		for gas in os.listdir(path+folder):
			gases.add(gas)
	return gases
def get_concs(path, gas, list_of_folders):
	concs = set()
	for folder in list_of_folders:
		for gas_ in os.listdir(path+folder):
			if gas_ == gas:
				for conc in os.listdir(path+folder+'/'+gas):
					concs.add(conc)
	concs.add('0')
	return concs

def gases_mapping(x, gases_trial=None):
	gases_dict = dict(zip(gases_trial, range(len(gases_trial))))
	return gases_dict[x]
def check_mkdir(path):
	if not os.path.exists(path):
		os.mkdir(path)
def copy_walk(path_1, path_2):
	if not os.path.exists(path_1):
		os.mkdir(path_1)
	if not os.path.exists(path_2):
		os.mkdir(path_2)
	for path_main, folders, files in os.walk(path_1):
		temp_path = path_2 + path_main[path_main.index(path_1)+len(path_1):]
		for folder in folders:
			check_mkdir(temp_path+'/'+folder)
def book_convert(array, indexes=None, scale=True):
	'''Convert array to bookstein coordinates with baseline'''
	book_coords = []
	array = np.hstack([np.array(range(len(array)))[:, np.newaxis], array[:, np.newaxis]])
	if indexes:
		i1, i2 = indexes
		try:
			x1, y1 = array[i1]
			x2, y2 = array[i2]
		except:
			print('Bad indexes')
			raise Exception()
	else:
		x1, y1 = array[0]
		x2, y2 = array[-1]
	D2 = (x2-x1)**2+(y2-y1)**2
	for dot in array:
		xj, yj = dot
		if scale:
			v = ((x2-x1)*(yj-y1)-(y2-y1)*(xj-x1))/D2
		else:
			v = (x2-x1)*(yj-y1)-(y2-y1)*(xj-x1)
		book_coords.append(v)
	return np.array(book_coords)
def plot_hist(itog, num=1, ax=None):
	step = (np.max(itog[:, num]) - np.min(itog[:, num])) / 100
	bins = [i for i in np.arange(np.min(itog[:, num])-step, np.max(itog[:, num])+step,  step)]
	if ax:
		ax.hist(itog[:, num], bins=bins)
	else:
		plt.hist(itog[:, num], bins=bins)
def make_list(smth):
	if isinstance(smth, list):
		return smth
	if isinstance(smth, str):
		return [smth]
def create_data(path, sensor_num, to_pop_task=2, log=False, start_from=0, start_from_test=0, num_of_exp=None, start_cycle_from=0, get_last_list=None, shuffle=False, rt=False):
	'''Return: x_train, x_test, y_train, y_test'''
	sensors = ['sens'+str(i) for i in range(1,5)]
	if rt:
		fizvel = 'S'
		multiplier = 2
	else:
		fizvel = 'S'
		multiplier = 1
	list_of_folders = [elem+'/' for elem in os.listdir(path) if os.path.isdir(path+elem) and not elem.startswith('sens')]
	if isinstance(sensor_num, list):
		chosen_sens = [sensors[i-1] for i in sensor_num if ((i < 5) and (i > 0))]
	else:
		chosen_sens = make_list(sensors[sensor_num-1])
	gases_dict_train = dict(zip(get_gases(path, list_of_folders),[pd.DataFrame()]*len(get_gases(path, list_of_folders))))
	gases_dict_test = dict(zip(get_gases(path, list_of_folders),[pd.DataFrame()]*len(get_gases(path, list_of_folders))))
	#Check what you are popping!!!!
	#There we can choose the number of experiments. Usefull, when you wanna get only restricted set of experiments.
	#Also you have another variant to pop choosen experiments.
	#You can collect all experiments in one file, using num_of_exp == full number of experiments.
	if num_of_exp == None:
		if isinstance(to_pop_task, list):
			list_of_test = make_list([list_of_folders.pop(idx_pop) for idx_pop in sorted(to_pop_task, reverse = True)])
			list_of_train = make_list(list_of_folders)
		else:
			list_of_train, list_of_test = make_list(list_of_folders), make_list(list_of_folders.pop(to_pop_task))
		print(list_of_train, list_of_test)
	else:
		if num_of_exp > len(list_of_folders):
			raise Exception
		elif num_of_exp == len(list_of_folders):
			list_of_train = list_of_folders
			list_of_test = []
		else:
			list_of_train = list_of_folders[:num_of_exp-1]
			list_of_test = [list_of_folders[num_of_exp-1]]
		print(list_of_train, list_of_test)
		
	for sens in chosen_sens:
		temp_dict_train = dict(zip(get_gases(path, list_of_folders),[pd.DataFrame()]*len(get_gases(path, list_of_folders))))
		temp_dict_test = dict(zip(get_gases(path, list_of_folders),[pd.DataFrame()]*len(get_gases(path, list_of_folders))))
		for folder in list_of_train:
			for gas in os.listdir(path+folder):
				for conc in os.listdir(path+folder+gas+'/'):
					if get_last_list:
						if folder in get_last_list:
							print(folder+'in')
							for file in [file for file in os.listdir(path+folder+gas+'/'+conc) if file.startswith(fizvel+sens[-1]+'_')][start_cycle_from:]:
								temp_dict_train[gas] = pd.concat([temp_dict_train[gas], pd.read_csv(path+folder+gas+'/'+conc+'/'+file, header=None).iloc[multiplier*start_from:]],
													  ignore_index=True)
						else:
							print(folder+'out')
							file_array = [file for file in os.listdir(path+folder+gas+'/'+conc) if file.startswith(fizvel+sens[-1]+'_')]
							for file in file_array[:-1]:
								temp_dict_train[gas] = pd.concat([temp_dict_train[gas], pd.read_csv(path+folder+gas+'/'+conc+'/'+file, header=None).iloc[multiplier*start_from:]],
													  ignore_index=True)
							for file in file_array[-1:]:
								temp_dict_test[gas] = pd.concat([temp_dict_test[gas], pd.read_csv(path+folder+gas+'/'+conc+'/'+file, header=None).iloc[multiplier*start_from_test:]],
													 ignore_index=True)
					else:
						for file in [file for file in os.listdir(path+folder+gas+'/'+conc) if file.startswith(fizvel+sens[-1]+'_')][start_cycle_from:]:
							temp_dict_train[gas] = pd.concat([temp_dict_train[gas], pd.read_csv(path+folder+gas+'/'+conc+'/'+file, header=None).iloc[multiplier*start_from:]],
													ignore_index=True)
				
		for folder in list_of_test:
			for gas in os.listdir(path+folder):
				for conc in os.listdir(path+folder+gas+'/'):
					if get_last_list:
						if folder in get_last_list:
							print(folder+'in')
							for file in [file for file in os.listdir(path+folder+gas+'/'+conc) if file.startswith(fizvel+sens[-1]+'_')][start_cycle_from:]:
								temp_dict_test[gas] = pd.concat([temp_dict_test[gas], pd.read_csv(path+folder+gas+'/'+conc+'/'+file, header=None).iloc[multiplier*start_from_test:]],
													 ignore_index=True)
						else:
							print(folder+'out')
							file_array = [file for file in os.listdir(path+folder+gas+'/'+conc) if file.startswith(fizvel+sens[-1]+'_')]
							for file in file_array[-1:]:
								temp_dict_test[gas] = pd.concat([temp_dict_test[gas], pd.read_csv(path+folder+gas+'/'+conc+'/'+file, header=None).iloc[multiplier*start_from_test:]],
													 ignore_index=True)
							for file in file_array[:-1]:
								temp_dict_train[gas] = pd.concat([temp_dict_train[gas], pd.read_csv(path+folder+gas+'/'+conc+'/'+file, header=None).iloc[multiplier*start_from:]],
													  ignore_index=True)
					else:	
						for file in [file for file in os.listdir(path+folder+gas+'/'+conc) if file.startswith(fizvel+sens[-1]+'_')][start_cycle_from:]:
							temp_dict_test[gas] = pd.concat([temp_dict_test[gas], pd.read_csv(path+folder+gas+'/'+conc+'/'+file, header=None).iloc[multiplier*start_from_test:]],
													ignore_index=True)
		for fr_key in gases_dict_train:
			gases_dict_train[fr_key] = pd.concat([gases_dict_train[fr_key], temp_dict_train[fr_key]], axis=1, ignore_index=True)
		for fr_key in gases_dict_test:
			gases_dict_test[fr_key] = pd.concat([gases_dict_test[fr_key], temp_dict_test[fr_key]], axis=1, ignore_index=True)
		
	for fr_key in gases_dict_test:
		if log:
			gases_dict_test[fr_key] = gases_dict_test[fr_key].apply(np.log10)
		gases_dict_test[fr_key]['y'] = fr_key
	for fr_key in gases_dict_train:
		if log:
			gases_dict_train[fr_key] = gases_dict_train[fr_key].apply(np.log10)
		gases_dict_train[fr_key]['y'] = fr_key
		
	#Sample не нужен, если перемешивание производится в самом fit'е. Но я все равно оставил.
	df_train = pd.concat(gases_dict_train.values(), ignore_index=True)
	df_test = pd.concat(gases_dict_test.values(), ignore_index=True)
	if rt:
		df_train = df_train.iloc[1::2]
		df_test = df_test.iloc[1::2]
	if shuffle:
		df_train = df_train.sample(frac=1.0)
		df_test = df_test.sample(frac=1.0)
	df_train = df_train.dropna()
	df_test = df_test.dropna()
	#to_pop = df_train.columns[-1]
	x_train, y_train = df_train, df_train.pop('y')
	x_test, y_test = df_test, df_test.pop('y')
	return x_train, x_test, y_train, y_test
def create_regr_data(path, sensor_num, gas=None, to_pop_task=2, log=False, start_from=0, start_from_test=0, part_of_zeros=6, num_of_exp=None):
	'''Outdated'''
	print('Warning, function is outdated!!!!!!!')
	list_of_folders = [elem+'/' for elem in os.listdir(path) if os.path.isdir(path+elem) and elem.startswith('Task')]
	concs = get_concs(path, gas)
	concs_dict_train = dict(zip(concs, [pd.DataFrame()]*len(concs)))
	concs_dict_test = dict(zip(concs, [pd.DataFrame()]*len(concs)))
	if num_of_exp == None:
		if isinstance(to_pop_task, list):
			list_of_test = make_list([list_of_folders.pop(idx_pop) for idx_pop in sorted(to_pop_task, reverse = True)])
			list_of_train = make_list(list_of_folders)
		else:
			list_of_train, list_of_test = make_list(list_of_folders), make_list(list_of_folders.pop(to_pop_task))
		print(list_of_train, list_of_test)
	else:
		if num_of_exp > len(list_of_folders):
			raise Exception
		list_of_train = list_of_folders[:num_of_exp-1]
		list_of_test = [list_of_folders[num_of_exp-1]]
		print(list_of_train, list_of_test)
	#Creating training data
	for folder in list_of_train:
		for conc in os.listdir(path+folder+gas+'/'):
			for file in os.listdir(path+folder+gas+'/'+conc):
				if file.startswith('R'+str(sensor_num)+'_'):
					concs_dict_train[conc] = pd.concat([concs_dict_train[conc], pd.read_csv(path+folder+gas+'/'+conc+'/'+file, header=None).iloc[start_from:]], 
													  ignore_index=True)
		for file in os.listdir(path+folder+'Air/0/')[::part_of_zeros]:
			if file.startswith('R'+str(sensor_num)+'_'):
				concs_dict_train['0'] = pd.concat([concs_dict_train['0'], pd.read_csv(path+folder+'Air/0/'+file, header=None).iloc[start_from:]], 
													ignore_index=True)
	for fr_key in concs_dict_train:
		if log:
			concs_dict_train[fr_key] = concs_dict_train[fr_key].apply(np.log10)
		concs_dict_train[fr_key]['y'] = float(fr_key)
	#Creating test data
	for folder in list_of_test:
		for conc in os.listdir(path+folder+gas+'/'):
			for file in os.listdir(path+folder+gas+'/'+conc):
				if file.startswith('R'+str(sensor_num)+'_'):
					concs_dict_test[conc] = pd.concat([concs_dict_test[conc], pd.read_csv(path+folder+gas+'/'+conc+'/'+file, header=None).iloc[start_from_test:]], 
													 ignore_index=True)
		for file in os.listdir(path+folder+'Air/0/')[::part_of_zeros]:
			if file.startswith('R'+str(sensor_num)+'_'):
				concs_dict_test['0'] = pd.concat([concs_dict_test['0'], pd.read_csv(path+folder+'Air/0/'+file, header=None).iloc[start_from:]], 
													ignore_index=True)
	for fr_key in concs_dict_test:
		if log:
			concs_dict_test[fr_key] = concs_dict_test[fr_key].apply(np.log10)
		concs_dict_test[fr_key]['y'] = float(fr_key)
	df_train = pd.concat(concs_dict_train.values(), ignore_index=True).sample(frac=1.0)
	df_test = pd.concat(concs_dict_test.values(), ignore_index=True).sample(frac=1.0)
	#to_pop = df_train.columns[-1]
	x_train, y_train = df_train, df_train.pop('y')
	x_test, y_test = df_test, df_test.pop('y')
	return x_train, x_test, y_train, y_test
	
def conf_matrix(y_true, y_pred, type_='light', percent=False):
	'''Return the confusion matrix
		If percent = True, => return the percent answer.
		type argument introduced for future works.
	'''
	if y_true.shape != y_pred.shape:
		raise Exception
	if type_ == 'light':
		result = np.zeros((y_pred.shape[1], y_true.shape[1]))
		for i in range(y_true.shape[0]):
			result[np.argmax(y_true[i]), np.argmax(y_pred[i])] += 1
		if percent:
			return result/np.sum(result, axis=1)[:, np.newaxis]
		return result