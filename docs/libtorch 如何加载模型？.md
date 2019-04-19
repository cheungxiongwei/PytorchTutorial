# libtorch 如何加载模型?

```c++
#include <torch/torch.h>
#include <iostream>
#include <memory>

int main()
{
    //实例化一个训练该模型时一致的网络
    auto net = std::make_shared<Net>();

    //已灰度的模式读取该图片
    cv::Mat img = cv::imread("7.png",0);

    //创建一个空张量
    auto tensor = torch::emty({1,1,28,28},torch::kByte);

    //填充该张量
    std::memcpy(tensor.data_ptr(),img.data,1*1*28*28);

    //灰度 0~255 转 0~1 浮点类型
    torch::Tensor x = tensor.to(torch::kFloat32).divi_(255);

    torch::Tensor result = net->forward(x);

    //排序后获取前5个结果集
    auto top_set = result.topk(5);
    std::cout << std::get<0>(top_set) << std::endl;
    std::cout << std::get<1>(top_set) << std::endl;//该张量和 labels 一一对应，也就是预测的结果
    return 0;
}
```

R-CNN (Selective Search + CNN + SVM)
SPP-NET (ROI Pooling)
Fast R-CNN (Selective Search  + CNN + ROI)
Faster R-CNN (RPN + CNN + ROI)

yolo
ssd