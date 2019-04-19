#ifndef LENET_HPP_
#define LENET_HPP_
#include <torch/nn/modules.h>

struct LeNet : torch::nn::Module {
	LeNet() {
		conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 6, /*kernel_size*/{ 5,5 }).padding(/*28->32*/{ 2,2 })));
		conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(6, 16, /*kernel_size*/{ 5,5 })));
		conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(16, 120, /*kernel_size*/{ 5,5 })));
		fc1 = register_module("fc1", torch::nn::Linear(torch::nn::LinearOptions(120, 84)));
		fc2 = register_module("fc2", torch::nn::Linear(torch::nn::LinearOptions(84, 10)));
	}

	// Implement the Net's algorithm.
	torch::Tensor forward(torch::Tensor x) {
		x = conv1->forward(x);//6@28x28
		x = torch::max_pool2d(x, { 2,2 }, { 2,2 });//6@14x14
		x = conv2->forward(x);//16@10x10
		x = torch::max_pool2d(x, { 2,2 }, { 2,2 });//16@10x10

		x = conv3->forward(x);//120@1x1
		x = x.view({ x.size(0),-1 });//
		x = fc1->forward(x);//120->84
		x = fc2->forward(x);//84->10
		x = torch::log_softmax(x,/*dim=*/1);
		return x;
	}

	// Use one of many "standard library" modules.
	torch::nn::Conv2d conv1{ nullptr };
	torch::nn::Conv2d conv2{ nullptr };
	torch::nn::Conv2d conv3{ nullptr };
	torch::nn::Linear fc1{ nullptr };
	torch::nn::Linear fc2{ nullptr };
};
#endif