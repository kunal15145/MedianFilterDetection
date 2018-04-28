import read as read
import cnn as cnn

data = read.readdata(3)
data = read.preprecessing_MFR(data)
# splitted_data = read.data_split(data)

cnnmodel = cnn.traincnn(data)
accuracy = cnn.testcnn(cnnmodel)
