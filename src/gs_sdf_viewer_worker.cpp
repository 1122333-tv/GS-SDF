#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

#include "neural_gaussian/neural_gaussian.h"
#include "params/params.h"

namespace {

constexpr int max_image_edge = 1280;

struct Request {
  sensor::Cameras camera{};
  std::array<float, 12> pose{};
};

Request parse_request(const std::string &line) {
  Request request;
  std::string command;
  std::istringstream input(line);
  auto &camera = request.camera;
  input >> command >> camera.width >> camera.height >> camera.fx >> camera.fy >>
      camera.cx >> camera.cy;
  for (float &value : request.pose) {
    input >> value;
  }
  if (!input || command != "RENDER" || input >> command ||
      camera.width <= 0 || camera.width > max_image_edge ||
      camera.height <= 0 || camera.height > max_image_edge ||
      !std::isfinite(camera.fx) || !std::isfinite(camera.fy) ||
      !std::isfinite(camera.cx) || !std::isfinite(camera.cy) ||
      camera.fx <= 0 || camera.fy <= 0 ||
      !std::all_of(request.pose.begin(), request.pose.end(),
                   [](float value) { return std::isfinite(value); })) {
    throw std::runtime_error("Invalid RENDER request");
  }
  return request;
}

void flush(FILE *stream) {
  if (std::fflush(stream) != 0) {
    throw std::runtime_error("Viewer pipe closed");
  }
}

}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 8) {
      throw std::runtime_error(
          "Usage: gs_sdf_viewer_worker gs.ply sh_degree background near far "
          "use_absgrad center_reg");
    }

    // The renderer writes diagnostic messages to stdout. Keep those on stderr
    // and reserve the original stdout pipe for frames and protocol headers.
    const int protocol_fd = dup(STDOUT_FILENO);
    if (protocol_fd < 0) {
      throw std::runtime_error("Could not open viewer protocol pipe");
    }
    FILE *stream = fdopen(protocol_fd, "wb");
    if (!stream) {
      close(protocol_fd);
      throw std::runtime_error("Could not open viewer protocol stream");
    }
    std::unique_ptr<FILE, decltype(&std::fclose)> protocol(stream, std::fclose);
    if (dup2(STDERR_FILENO, STDOUT_FILENO) < 0) {
      throw std::runtime_error("Could not redirect renderer diagnostics");
    }

    TORCH_CHECK(torch::cuda::is_available(), "A CUDA GPU is required");
    k_device = torch::kCUDA;
    k_sh_degree = std::stoi(argv[2]);
    k_bck_color = std::stoi(argv[3]);
    k_near = std::stof(argv[4]);
    k_far = std::stof(argv[5]);
    k_use_absgrad = std::stoi(argv[6]) != 0;
    k_center_reg = std::stoi(argv[7]) != 0;
    k_render_mode = false;
    TORCH_CHECK(k_sh_degree >= 0 && k_sh_degree <= 4 &&
                    k_bck_color >= 0 && k_bck_color <= 2 &&
                    std::isfinite(k_near) && std::isfinite(k_far) &&
                    0 < k_near && k_near < k_far,
                "Invalid renderer configuration");
    torch::NoGradGuard no_grad;
    NeuralGS model(nullptr, std::filesystem::path(argv[1]));
    model.key_for_gradient = "gradient_2dgs";
    std::fprintf(stream, "READY %lld\n",
                 static_cast<long long>(model.get_xyz().size(0)));
    flush(stream);

    std::string line;
    while (std::getline(std::cin, line)) {
      auto request = parse_request(line);
      const auto pose = torch::from_blob(request.pose.data(), {3, 4},
                                         torch::kFloat32);
      auto rgb = model.render(pose, request.camera, false, k_bck_color)
                     .at("color");
      TORCH_CHECK(rgb.dim() == 3 && rgb.size(0) == request.camera.height &&
                      rgb.size(1) == request.camera.width && rgb.size(2) == 3,
                  "Renderer returned an invalid RGB shape");
      rgb = (rgb.clamp(0.0f, 1.0f) * 255).to(torch::kUInt8).cpu().contiguous();
      const auto bytes = static_cast<size_t>(rgb.numel());
      std::fprintf(stream, "FRAME %d %d %zu\n", request.camera.width,
                   request.camera.height, bytes);
      if (std::fwrite(rgb.data_ptr<uint8_t>(), 1, bytes, stream) != bytes) {
        throw std::runtime_error("Could not send rendered RGB");
      }
      flush(stream);
    }
  } catch (const std::exception &error) {
    std::cerr << "Viewer worker: " << error.what() << '\n';
    return 1;
  }
  return 0;
}
