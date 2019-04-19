#ifndef MNIST_DATA_SET_H_
#define MNIST_DATA_SET_H_

#include <torch/data/datasets/base.h>
#include <torch/data/example.h>
#include <torch/types.h>

#include <cstddef>
#include <string>

namespace torch {
namespace data {
namespace datasets {

/*!
        实现下面两个虚函数即可：
        virtual ExampleType get(size_t index) = 0
        virtual optional<size_t> size() const = 0
*/
/// The MNIST dataset.
class Mnist : public Dataset<Mnist> {
public:
  /// The mode in which the dataset is loaded.
  enum class Mode { kTrain, kTest };

  /// Loads the MNIST dataset from the `root` path.
  ///
  /// The supplied `root` path should contain the *content* of the unzipped
  /// MNIST dataset, available from http://yann.lecun.com/exdb/mnist.
  explicit Mnist(const std::string &root, Mode mode = Mode::kTrain);

  /// Returns the `Example` at the given `index`.
  Example<> get(size_t index) override;

  /// Returns the size of the dataset.
  optional<size_t> size() const override;

  /// Returns true if this is the training subset of MNIST.
  bool is_train() const noexcept;

  /// Returns all images stacked into a single tensor.
  const Tensor &images() const;

  /// Returns all targets stacked into a single tensor.
  const Tensor &targets() const;

private:
  Tensor images_, targets_;
};
} // end namespace datasets
} // end namespace data
} // end namespace torch

// Auxiliary
namespace aux {

#ifdef OPENCV_CORE_MAT_HPP
static torch::Tensor GrayToTensor(const cv::Mat &gray) {
  if (gray.empty() || gray.channels() != 1) {
    fprintf(stderr, "grayToTorch gray is null image!");
    exit(0);
  }

  //定义一个等于输入灰度图大小的张量并拷贝输入灰度图的内存数据至张量
  auto tensor = torch::empty({1, 1, gray.rows, gray.cols}, torch::kByte);
  std::memcpy(tensor.data_ptr(), gray.data,
              tensor.numel() /*1x1xgray.rowsx*gray.cols*/);
  return tensor.to(torch::kFloat32).div_(255);
}

static torch::Tensor GrayToTensor(const char *path) {
  return GrayToTensor(cv::imread(path, 0));
}

static torch::Tensor MatToTensor(const cv::Mat &img) {
  if (img.empty()) {
    fprintf(stderr, "grayToTorch gray is null image!\r\n");
    exit(0);
  }

  // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  // cv::resize(img, img, { 1,1 });
  // img.convertTo(img, CV_32F, 1.0 / 255.0);

  torch::Tensor tensor = torch::from_blob(
      img.data, {1, img.rows, img.cols, img.channels()}, torch::kByte);
  tensor = tensor.permute({0, 3, 1, 2}); // Opencv H x W x C => Tensor C x H x W

  // mean std
  // tensor[0][0] = tensor[0][0].sub_(0.485).div_(0.229);
  // tensor[0][1] = tensor[0][1].sub_(0.456).div_(0.224);
  // tensor[0][2] = tensor[0][2].sub_(0.406).div_(0.225);

  return tensor.to(torch::kFloat32).div_(255);
}

static torch::Tensor MatToTensor(const char *path) {
  return MatToTensor(cv::imread(path));
}

static int LoadMnistData(const char *path,
                         std::vector<std::vector<unsigned char>> &set,
                         int retturn_number = 10, bool isTrain = true) {
  std::fstream fs;
  fs.open(path, std::ios::in | std::ios::binary);
  if (!fs.is_open()) {
    fprintf(stderr, "open %s file failed!\r\n", path);
    return 0;
  }

  // raed 4 + 4 + 4 + 4
  // http://yann.lecun.com/exdb/mnist/index.html
  fs.read(reinterpret_cast<char *>(stderr), 16);
  int mnist_size = isTrain ? 60000 : 10000;
  for (int i = 0; i < mnist_size; ++i) {
    if (i > retturn_number)
      break;
    std::vector<unsigned char> receive(784);
    fs.read(reinterpret_cast<char *>(receive.data()), 784);
    set.push_back(receive);
  }
  fs.close();
  return set.size();
}

static cv::Mat MnistMemoryToMat(std::vector<unsigned char> &data) {
  return cv::Mat(28, 28, CV_8UC1, data.data());
}

static void DisplayMnistImage(std::vector<unsigned char> &data) {
  cv::imshow("mnist", MnistMemoryToMat(data));
  cv::waitKey(0);
}

static void SaveMnistImage(const char *path, cv::Mat &data) {
  cv::imwrite(path, data);
}

static void load_labels(std::string label_f, std::vector<string> &labels) {
  std::ifstream inf(label_f);
  std::string line;
  while (getline(inf, line)) {
    labels.push_back(line);
  }
}
#endif

} // namespace aux
#endif