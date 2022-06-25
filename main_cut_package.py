import pandas as pd
import numpy as np
import os


def listdir(path):
	if path.endswith('/'):
		result = [path+file for file in os.listdir(path)]
	else:
		result = [path+'/'+file for file in os.listdir(path)]
	return result
	
def find_neighbourhood(reper, data):
	intercept = 3
	limit = 2.5e6
	indexes = find_indexes(reper, data, limit)
	idx = check_decrease(indexes, data, intercept)
	if idx != -1:
		return idx
	else:
		return -1
		
def find_indexes(reper, data, limit):
	abs_diff = np.array(abs(data - reper))
	intercept_check = abs_diff < limit
	result = []
	for idx, state in enumerate(intercept_check):
		if state:
			if len(result) > 0:
				if idx-result[-1] == 1:
					result[-1] = idx
				else:
					result.append(idx)
			else:
				result.append(idx)
	#print(result)
	return result
	
def check_decrease(idxs, data, intercept):
	len_data = len(data)
	result = []
	for idx in idxs:
		if idx > (len_data - intercept) or idx < (intercept):
			break
		check_data = data[idx-intercept:idx+intercept]
		#print(np.diff(check_data))
		if (np.diff(check_data) <= 0).all():
			result.append(idx)
	if len(result) == 1:
		return result[0]
	else:
		#print('Wrong definition of smoothing\n', 'Find {} results'.format(len(result)))
		return -1

def align(data, reper=1e7):
	if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
		data = np.array(data)
	idx = find_neighbourhood(reper, data)
	if idx == -1:
		return data
	data[idx] = (data[idx+1] - data[idx-1])/2 + data[idx-1]
	return data
	
def moving(data, reper_temp=230, block_size=102, intercept=7):
	'''Принимает данные в формате "температура-сопротивление" и, расширяя массив на несколько точек
		влево-вправо, двигает его так, чтобы температура, совпадающяя с репером, была на фиксированной точке.
		Возвращает сдвинутые и обрезанные данные.
		Желательно брать точку, которая находится где-нибудь посередине'''
	idx_place = block_size - intercept
	data = np.array(data)
	if data.shape[0] > data.shape[1]:
		data = data.T
	idx_border = len(data[0])*3//4
	idx = np.argmin(abs(data[0, idx_border:] - reper_temp))
	needed_idx = idx+idx_border
	temp_array = np.zeros((2, block_size))
	idx_inter_place = idx_place - needed_idx
	temp_array[:, idx_inter_place:idx_inter_place+len(data[0])] = data
	return temp_array
def cut_zeros(data):
	'''Отрезает с краев все нули. Входящие данные - DataFrame'''
	drop = True
	iterator = (column for column in data.columns)
	while drop:
		column = iterator.__next__()
		if any(data[column] == 0.0):
			data.drop(labels=column, axis=1, inplace=True)
		else:
			drop= False
	drop = True
	iterator = (column for column in reversed(data.columns))
	while drop:
		column = iterator.__next__()
		if any(data[column] == 0.0):
			data.drop(labels=column, axis=1, inplace=True)
		else:
			drop= False
	return data
	
def taking_mean(data, step=4, restrict=24):
	'''Takes one array with temperature (or time) and resistance columns.
	Return new meaned array, with step mean and restricted to given number'''
	data = np.array(data)
	if data.shape[1] > data.shape[0]:
		data = data.T
	return np.vstack([np.mean(data[i:i+step], axis=0) for i in range(0, len(data), step)])[:restrict]

def book_convert(data, indexes=None, scale=True, max_min=False, metric_use=False):
	'''Convert array to bookstein coordinates with baseline in
	(0, 0) and (1, 0) or (-1, 0) and (0, 0)
	'''
	book_coords = []
	data = np.array(data)
	if data.shape[1] > data.shape[0]:
		data = data.T
	array = data
	if indexes:
		i1, i2 = indexes
		try:
			x1, y1 = array[i1]
			x2, y2 = array[i2]
		except:
			print('Bad indexes')
			raise Exception()
	else:
		if max_min:
			if metric_use:
				dist_matrix = np.zeros((array.shape[0], array.shape[0]))
				for idx1, dot1 in enumerate(array):
					xi, yi = dot1
					for idx2, dot2 in enumerate(array):
						xj, yj = dot2
						dist_matrix[idx1, idx2] = np.sqrt((xi-xj)**2 + (yi-yj)**2)
			else:
				idx_max = np.argmax(array[:, 1])
				idx_min = np.argmin(array[:, 1])
				x1, y1 = array[idx_min]
				x2, y2 = array[idx_max]
		else:
			x1, y1 = array[0]
			x2, y2 = array[-1]
	if scale:
		D2 = (x2-x1)**2+(y2-y1)**2
	for dot in array:
		xj, yj = dot
		if scale:
			u = ((x2-x1)*(xj-x1)-(y2-y1)*(yj-y1))/D2
			v = ((x2-x1)*(yj-y1)-(y2-y1)*(xj-x1))/D2
		else:
			u = (x2-x1)*(xj-x1)-(y2-y1)*(yj-y1)
			v = (x2-x1)*(yj-y1)-(y2-y1)*(xj-x1)
		book_coords.append((u,v))
	return np.array(book_coords)