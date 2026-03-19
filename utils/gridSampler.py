import numpy as np

def gridSampler(width: int, height: int, n_points: int):
	"""Designed for image pixel sampling"""
	n = int(np.sqrt(n_points))
	row_step = height / (n-1)
	col_step = width / (n-1)

	datapoints = []

	r = 0
	while r < height:
	    c = 0
	    while c < width:
	        datapoints.append(int(r * width + c))
	        c += col_step
	    r += row_step


	# print("Sampling done!")
	datapoints = np.array(datapoints)
	filtered_datapoints = datapoints[datapoints < (width*height)]

	return filtered_datapoints