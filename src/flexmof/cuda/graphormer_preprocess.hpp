#ifndef GRAPHORMER_PREPROCESS_HPP_
#define GRAPHORMER_PREPROCESS_HPP_

#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> floyd_warshall_cuda(const torch::Tensor adj, const size_t max_dist);

void floyd_warshall_batch_cuda(
  const torch::Tensor adj, const torch::Tensor n_node, const size_t max_n_node, const size_t max_dist,
  torch::Tensor output_dist, torch::Tensor output_floyd_pred);

void gen_edge_input_cuda(
  const size_t max_dist,
  const torch::Tensor pred,
  const torch::Tensor dist,
  const size_t num_edge_features,
  const torch::Tensor edge_features,
  torch::Tensor output_edge_features);

void gen_edge_input_batch_cuda(
  const size_t max_dist,
  const size_t max_n_node,
  const size_t num_edge_features,
  const size_t multi_hop_max_dist,
  const torch::Tensor n_node,
  const torch::Tensor pred,
  const torch::Tensor dist,
  const torch::Tensor edge_features,
  torch::Tensor output_edge_features);

#endif  // GRAPHORMER_PREPROCESS_HPP_
