import numpy as np
import os
import datetime

np.random.seed(42)

dataset = [['Date', 'Time', 'Type', 'GT', 'Magnitude', 'M', 'Latitude', 'Longitude', 'Depth', 'Q', 'EVID', 'NPH', 'NGRM']]
year = 1932
while year <= 2019:
	with open('data/SCEC_DC/%s.catalog' % year) as f:
		content = f.readlines()
		content = [x.strip() for x in content[10:-2]] 
		for line in content:
			features = line.split()
			# fix date formatting
			date = features[0]
			date = datetime.datetime.strptime(date, '%Y/%m/%d').strftime('%m/%d/%Y')
			features[0] = date
			# remove the milliseconds
			time = features[1]
			time = time[:-3]
			features[1] = time
			dataset.append(features)
	f.close()
	year += 1

import csv

with open('data/database2.csv', 'w') as csvf:
	writer = csv.writer(csvf)
	writer.writerows(dataset)

csvf.close()

