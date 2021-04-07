#include <torch/script.h> 
#include <iostream>
#include <memory>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <cmath>
#include <matplotlibcpp.h>
#include <Python.h>
namespace plt = matplotlibcpp;
int test() {
    plt::figure(1);
  //plt::plot({1,3,2,4});
    plt::show(plt::plot({1,3,2,4}));
    return 1;

}
int main(int argc, const char* argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }
  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }
  std::cout << "ok\n";
std::vector<torch::jit::IValue> inputs;
cv::Mat image = cv::imread("/home/annotated_image/as6.j.png");
cv::resize(image,image, cv::Size(224,224));
torch::Tensor img_tensor = torch::from_blob(image.data, {1, image.rows, image.cols, 3}, torch::kByte);
img_tensor = img_tensor.permute({0, 3, 1, 2});
img_tensor = img_tensor.toType(torch::kFloat);
img_tensor = img_tensor.div(255);
inputs.push_back(img_tensor);
at::Tensor output = module.forward(inputs).toTensor();
std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/11) << '\n';
auto max_result = output.max(1, true);
auto max_index = std::get<1>(max_result).item<float>();
std::cout << max_index << std::endl;
std::cout<<test<<std::endl;
std::cout<<" Successful!"<<std::endl;
}
 