#include "lenet.hpp"
#include <iostream>
#include <torch/torch.h>

constexpr const char *data_mnist_path = "./datasets/mnist";
constexpr const char *data_fashion_mnist_path = "./datasets/fashion-mnist"

    int
    main() {

  std::cout << "load data path:" << std::string(data_mnist_path) << std::endl;

  // Create a new Net.
  auto net = std::make_shared<LeNet>();

  // Create a multi-threaded data loader for the MNIST dataset.
  // 为MNIST数据集创建多线程数据加载器
  auto train_dataset =
      torch::data::datasets::MNIST(data_mnist_path)
          .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
          .map(torch::data::transforms::Stack<>());
  auto train_loader =
      torch::data::make_data_loader(train_dataset, /*batch_size=*/64);

  // 加载测试集
  auto test_dataset =
      torch::data::datasets::MNIST(data_mnist_path,
                                   torch::data::datasets::MNIST::Mode::kTest)
          .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
          .map(torch::data::transforms::Stack<>());
  auto test_loader = torch::data::make_data_loader(
      test_dataset, /*batch_size=*/1000); // 10 x 1000 = 10000

  // Instantiate an SGD optimization algorithm to update our Net's parameters.
  // 实例化SGD优化算法以更新我们的Net参数。
  torch::optim::SGD optimizer(
      net->parameters(), torch::optim::SGDOptions(/*lr=*/0.01).momentum(0.5));

  for (size_t epoch = 1; epoch <= 10; ++epoch) {
    size_t batch_index = 0;
    // Iterate the data loader to yield batches from the dataset.
    // 迭代数据加载器以从数据集中生成批次。
    net->train(); //启用训练模式
    for (auto &batch : *train_loader) {
      // Reset gradients.
      // 重置渐变。
      optimizer.zero_grad();
      // Execute the model on the input data.
      // 在输入数据上执行模型。
      torch::Tensor prediction = net->forward(batch.data);

      // Compute a loss value to judge the prediction of our model.
      // 计算损失值以判断我们模型的预测。
      torch::Tensor loss = torch::nll_loss(prediction, batch.target);
      // Compute gradients of the loss w.r.t. the parameters of our model.
      // 计算损失的梯度w.r.t. 我们模型的参数。
      loss.backward();
      // Update the parameters based on the calculated gradients.
      // 根据计算的梯度更新参数。
      optimizer.step();
      // Output the loss and checkpoint every 100 batches.
      // 每100批输出损失和检查点。
      if (++batch_index % 100 == 0) {

        std::printf("Train Epoch: %ld [%5ld/%5ld] Loss: %.4f\r\n", epoch,
                    batch_index * batch.data.size(0),
                    train_dataset.size().value(), loss.item<float>());
        // Serialize your model periodically as a checkpoint.
        // 定期将模型序列化为检查点。
        torch::save(net, "net.pt");
      }
    }

    net->eval(); //禁用训练模式
    //测试数据
    torch::NoGradGuard no_grad;
    double test_loss = 0.0;
    int32_t correct = 0;
    for (auto &batch : *test_loader) {
      auto data = batch.data;      //获取测试图像数据
      auto targets = batch.target; //获取测数标签数据
      auto output = net->forward(data);

      test_loss +=
          torch::nll_loss(output, targets, /*weight*/ {}, Reduction::Sum)
              .item<float>();
      auto pred = output.argmax(1);
      correct += pred.eq(targets).sum().item().toLong(); //判断结果是否正确
    }

    test_loss /= test_dataset.size().value();
    std::printf("Test set: Average loss: %.4f | Accuracy: %.3f\r\n", test_loss,
                static_cast<double>(correct) / test_dataset.size().value());
  }
}