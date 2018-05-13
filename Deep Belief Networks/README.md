# Deep Belief Networks
*[Hinton06]* cho thấy RBMs có thể xếp chồng lên nhau và được đào tạo một cách tham lam để tạo thành cái gọi là Deep Belief Networks(DBN). DBNs là graphical models học cách trích xuất một đại diện phân cấp sâu của training data. They model phân bố chung giữa observed vector x and the l hidden layers h^k như sau:</br>
![](https://latex.codecogs.com/gif.latex?P%28x%2C%20h%5E1%2C%20%5Cldots%2C%20h%5E%7B%5Cell%7D%29%20%3D%20%5Cleft%28%5Cprod_%7Bk%3D0%7D%5E%7B%5Cell-2%7D%20P%28h%5Ek%7Ch%5E%7Bk&plus;1%7D%29%5Cright%29%20P%28h%5E%7B%5Cell-1%7D%2Ch%5E%7B%5Cell%7D%29)
</br>
where ![](https://latex.codecogs.com/gif.latex?x%3Dh%5E0%2C%20P%28h%5E%7Bk-1%7D%20%7C%20h%5Ek%29) là phân phối có điều kiện chó các đợi vị hiển thị được điều chỉnh trên hidden units của RBM tại level k, and ![](https://latex.codecogs.com/gif.latex?P%28h%5E%7B%5Cell-1%7D%2C%20h%5E%7B%5Cell%7D%29) là phân phối chung có thể nhìn thấy được ẩn trong top-level RBM. Điều này được minh họa trong hình bên dưới.</br>
![](https://github.com/bigkizd/Deep_Learning/blob/master/Deep%20Belief%20Networks/Images/DBN3.png)</br>
Nguyên tắc đào tạo không được giám sát theo tham lam có thể được áp dụng cho các DBNs với RBMs như các khối xây dựng cho mỗi lớp*[Hinton06], [Bengio07]*. Qúa trình này như sau:</br>
1. Train the first layer như RBM that models the raw input ![](https://latex.codecogs.com/gif.latex?x%20%3D%20h%5E%7B%280%29%7D) as its visibel layer.
2. Sử dụng lớp đầu tiên đó để có được một biểu diễn đầu vào sẽ được sử dụng làm sạch dữ liệu cho lớp thứ 2. Two common solutions exist. Biếu diễn này có thể được chọn làm mean activations ![](https://latex.codecogs.com/gif.latex?p%28h%5E%7B%281%29%7D%3D1%7Ch%5E%7B%280%29%7D%29) or samples của ![](https://latex.codecogs.com/gif.latex?p%28h%5E%7B%281%29%7D%7Ch%5E%7B%280%29%7D%29).
3. Train the second layer dưới dạng RBM, lấy dữ liệu được chuyển đổi(samples or mean activations) as training examples(for the visible layer ò that RBM).
4. Lặp lại(2 and 3) cho số layers mong muốn, mỗi lần truyền lên hoặc là samples hoặc là mean values.
5. Tinh chỉnh tất cả các thông số của kến trúc sâu này đối với proxy for the DBN log-likehood, or liên quan đến một tiêu chí đào tạo được giám sát(after adding extra learning machinery để chuyển đổi biểu diễn đã học thành các dự đoán được giám sát, e.g. a linear classifier.</br>
</br>
Trong hướng dẫn này, chúng tôi tập trung vào tinh chỉnh thông qua supervised gradient descent. Cụ thể, chúng tôi sử dụng a logistic regression classifier để phân loại đầu vào x  dựa trên đầu ra của lớp ẩn cuối cùng của DBN. Tinh chỉnh sau đó được thực hiện thông qua supervised descent của negative log-likelihood cost function. Vì the supervised gradient chỉ không có giá trị cho trọng số và hiden layer biases of each layer(i.e. null for the visible biases of each RBM), this proceduce tương đương với việc khởi tạo các tham số của a deep MLP với trọng số và hidden layer biases thu được với chiến lược huấn luyện không giám sát.
# Justifyting Greedy-Layer Wire Pre-Training
why does such an algorithm work? Lấy ví dụ a 2-layer DBN with hidden layers ![](https://latex.codecogs.com/gif.latex?h%5E%7B%281%29%7D) and  ![](https://latex.codecogs.com/gif.latex?h%5E%7B%282%29%7D) (với tham số trọng số tương ứng ![](https://latex.codecogs.com/gif.latex?W%5E%7B%281%29%7D) and ![](https://latex.codecogs.com/gif.latex?W%5E%7B%282%29%7D), * [Hinton06] * established  (see also Bengio09]_ for a detailed derivation) that ![] (https://latex.codecogs.com/gif.latex?%5Clog%20p%28x%29) có thể !được viết lại như: </br>
![](https://latex.codecogs.com/gif.latex?%5Clog%20p%28x%29%20%3D%20%26KL%28Q%28h%5E%7B%281%29%7D%7Cx%29%7C%7Cp%28h%5E%7B%281%29%7D%7Cx%29%29%20&plus;%20H_%7BQ%28h%5E%7B%281%29%7D%7Cx%29%7D%20&plus;%20%5C%5C%20%26%5Csum_h%20Q%28h%5E%7B%281%29%7D%7Cx%29%28%5Clog%20p%28h%5E%7B%281%29%7D%29%20&plus;%20%5Clog%20p%28x%7Ch%5E%7B%281%29%7D%29%29.)


# Implementation
Để thực hiện DBNs trong Theano, chúng ta sẽ thực hiện lớp được địng nghĩa trong hướng dẫn ![Restricted Boltzmann Machines (RBM)](http://deeplearning.net/tutorial/rbm.html). Ta có thể quan sát rằng code cho DBN rất giống với code cho Sda, bời vì cả hai đều liên quan đến nguyên tắc đào tạo không giám sát không ngoan theo sau bởi tinh chỉnh được giám sát dưới dạng MLP sâu. Sự khác biệt chính là chúng tôi sử dụng lớp RBM thay vì Da.</br>
Chúng ta bắt đầu bằng cách định nghĩa lớp DBN sẽ lưu trữ lớp của MLP, cùng với các RBM liên quan của chúng. Vì chúng tôi lấy quan điểm sử dụng RBM để khởi tạo MLP, code sẽ phản ánh điều này bằng cách phân tách càng nhiều càng tốt các RBM được sử dụng để khởi tạo mạng  và MLP được sử dụng để phân loại.</br>
``` py
class DBN(object):
    """Deep Belief Network

    A deep belief network is obtained by stacking several RBMs on top of each
    other. The hidden layer of the RBM at layer `i` becomes the input of the
    RBM at layer `i+1`. The first layer RBM gets as input the input of the
    network, and the hidden layer of the last RBM represents the output. When
    used for classification, the DBN is treated as a MLP, by adding a logistic
    regression layer on top.
    """

    def __init__(self, numpy_rng, theano_rng=None, n_ins=784,
                 hidden_layers_sizes=[500, 500], n_outs=10):
        """This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the DBN

        :type hidden_layers_sizes: list of ints
        :param hidden_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network
        """

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = MRG_RandomStreams(numpy_rng.randint(2 ** 30))

        # allocate symbolic variables for the data

        # the data is presented as rasterized images
        self.x = T.matrix('x')

        # the labels are presented as 1D vector of [int] labels
        self.y = T.ivector('y')
```

