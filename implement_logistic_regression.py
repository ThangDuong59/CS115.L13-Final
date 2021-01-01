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

# Khai báo các hàm cần thiết
def sigmoid_function(z):
    return 1 / (1 + np.exp(-z)) # Công thức hàm sigmoid

def predict(x, theta):
    return sigmoid_function(x.dot(theta.T)) # Truyền vào tích vô hướng của X và theta

def loss_function(y_hat, label):
    return (-label.T * np.log(y_hat) - (1 - label.T) * np.log(1 - y_hat)).mean() # Công thức hàm loss binary cross entropy

def compute_gradient(x, y_hat, label):
    return np.dot(x.T, (y_hat - label)) / label.size # Tính gradient descent (đạo hàm hàm loss trên theo theta)

# Khởi tạo các chỉ số cần thiết
learning_rate = 0.01 # Chỉ số learning rate
theta = np.array([1.0, 2.3, -1.2, -0.25, 2.2, 3.0, 1.0, -0.24, -0.23, 1.2, -4.5, 1.2 , 0.23, 1.2]) # Các trọng số tương ứng với số indepentdent features
epochs = 100 # Số lần model được học dataset
losses = [] # Chứa loss của mỗi lần predict
accuracies = [] # Chứa độ chính xác của model sau mỗi epoch
batch_size = 50 # Số samples mà model sẽ học cùng một lúc

# Bắt đầu quá trình training
print("TRAINING START")
for epoch in range(epochs):
    print(f"EPOCH {epoch}: \n")
    accuracy = np.array([]) # Tạo một array rỗng làm vector chứa kết quả dự đoán
    for i in range(0, X.shape[0], batch_size):
        # Trích data theo batch size
        X_i = X[i : i + batch_size]
        y_i = y[i : i + batch_size]
        # Predict X_i
        y_hat = predict(X_i, theta)
        # Tính loss
        loss = loss_function(y_hat, y_i)
        losses.append(loss)
        # Tính gradient descent với kết quả predict vừa đạt được
        gradient = compute_gradient(X_i, y_hat, y_i)
        # Cập nhật lại theta theo kết quả gradient
        theta -= learning_rate * gradient
        accuracy = np.concatenate((accuracy, y_hat.round() == y_i))
        print('predicted value: \n', y_hat)
        print('loss: \n', loss)
        print('gradient: \n', gradient)
        print('updated theta: \n', theta)
        print('\n')
    accuracies.append(accuracy.mean())

plt.figure(figsize=(9, 3))
plt.plot(accuracies)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()
