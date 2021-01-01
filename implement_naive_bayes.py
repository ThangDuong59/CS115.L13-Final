import numpy as np 
import matplotlib.pyplot as plt 

# Trích data từ file .csv
dataset = np.genfromtxt('heart.csv', delimiter=',', skip_header=1) 

# Lấy các indepentdent variables và dependent variable
X = dataset[:, : -1]
y = dataset[:, -1]

# Tạo thêm một cột chứa tham số cho b
intercept = np.ones((X.shape[0], 1))
# Nối intercept vào X
X = np.concatenate((intercept, X), axis=1)
## Xáo trộn bộ dataset
# Tạo một array chứa các chỉ điểm của bộ dữ liệu
index = np.arange(X.shape[0])
np.random.shuffle(index) # trộn lẫn phần tử trong array
# Shuffle dữ liệu bằng cách gán
X = X[index]
y = y[index]