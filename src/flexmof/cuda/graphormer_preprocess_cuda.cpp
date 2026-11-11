#include "graphormer_preprocess.hpp"

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")

std::vector<torch::Tensor> floyd_warshall(const torch::Tensor adj, size_t max_dist) {
  CHECK_CUDA(adj);
  return floyd_warshall_cuda(adj, max_dist);
}

void floyd_warshall_batch(const torch::Tensor adj, const torch::Tensor n_node, const size_t max_n_node, const size_t max_dist, torch::Tensor output_dist, torch::Tensor output_floyd_pred) {
  CHECK_CUDA(adj);
  CHECK_CUDA(n_node);
  return floyd_warshall_batch_cuda(adj, n_node, max_n_node, max_dist, output_dist, output_floyd_pred);
}

void gen_edge_input(
  const size_t max_dist,
  const torch::Tensor pred,
  const torch::Tensor dist,
  const size_t num_edge_features,
  const torch::Tensor edge_features,
  torch::Tensor output_edge_features) {
  CHECK_CUDA(pred);
  CHECK_CUDA(dist);
  CHECK_CUDA(edge_features);
  CHECK_CUDA(output_edge_features);
  gen_edge_input_cuda(max_dist, pred, dist, num_edge_features, edge_features, output_edge_features);
}

void gen_edge_input_batch(
  const size_t max_dist,
  const size_t max_n_node,
  const size_t num_edge_features,
  const size_t multi_hop_max_dist,
  const torch::Tensor n_node,
  const torch::Tensor pred,
  const torch::Tensor dist,
  const torch::Tensor edge_features,
  torch::Tensor output_edge_features) {
  CHECK_CUDA(n_node);
  CHECK_CUDA(pred);
  CHECK_CUDA(dist);
  CHECK_CUDA(edge_features);
  CHECK_CUDA(output_edge_features);
  gen_edge_input_batch_cuda(max_dist, max_n_node, num_edge_features, multi_hop_max_dist, n_node, pred, dist, edge_features, output_edge_features);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("floyd_warshall", &floyd_warshall, "Floyd Warshall (CUDA)");
  m.def("gen_edge_input", &gen_edge_input, "Get Edge Input (CUDA)");
  m.def("floyd_warshall_batch", &floyd_warshall_batch, "Floyd Warshall Batch (CUDA)");
  m.def("gen_edge_input_batch", &gen_edge_input_batch, "Get Edge Input Batch (CUDA)");
}
