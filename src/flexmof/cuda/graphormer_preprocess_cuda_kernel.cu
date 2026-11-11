#include "graphormer_preprocess.hpp"

#define FLOYD_WARSHALL_BLOCK_DIM_X (16)
#define FLOYD_WARSHALL_BLOCK_DIM_Y (16)

template <typename int_t>
__global__ void floyd_warshall_set_init_kernel(
  int_t num_nodes,
  int_t max_dist,
  const torch::PackedTensorAccessor64<int_t, 2> adj,
  torch::PackedTensorAccessor64<int_t, 2> output_dist) {
  const int_t i = static_cast<int_t>(threadIdx.x + blockIdx.x * blockDim.x);
  const int_t j = static_cast<int_t>(threadIdx.y + blockIdx.y * blockDim.y);
  if (i < num_nodes && j < num_nodes) {
    if (i == j) {
      output_dist[i][j] = 0;
    } else {
      if (adj[i][j]) {
        output_dist[i][j] = 1;
      } else {
        output_dist[i][j] = max_dist;
      }
    }
  }
}

template <typename int_t>
__global__ void floyd_warshall_cuda_one_iter_kernel(
  int_t num_nodes,
  torch::PackedTensorAccessor64<int_t, 2> output_dist,
  torch::PackedTensorAccessor64<int_t, 2> output_pred,
  int_t k) {
  const int_t i = static_cast<int_t>(threadIdx.x + blockIdx.x * blockDim.x);
  const int_t j = static_cast<int_t>(threadIdx.y + blockIdx.y * blockDim.y);
  if (i < num_nodes && j < num_nodes && i != k && j != k && i != j) {
    const int_t new_dist = output_dist[i][k] + output_dist[k][j];
    const int_t old_dist = output_dist[i][j];
    if (new_dist < old_dist) {
      output_dist[i][j] = new_dist;
      output_pred[i][j] = k;
    }
  }
}

std::vector<torch::Tensor> floyd_warshall_cuda(const torch::Tensor adj, const size_t max_dist) {
  const size_t num_nodes = adj.size(0);
  const size_t grid_dim_x = (num_nodes + FLOYD_WARSHALL_BLOCK_DIM_X - 1) / FLOYD_WARSHALL_BLOCK_DIM_X;
  const size_t grid_dim_y = (num_nodes + FLOYD_WARSHALL_BLOCK_DIM_Y - 1) / FLOYD_WARSHALL_BLOCK_DIM_Y;
  dim3 grid_dim(grid_dim_x, grid_dim_y), block_dim(FLOYD_WARSHALL_BLOCK_DIM_X, FLOYD_WARSHALL_BLOCK_DIM_Y);
  auto output_dist = torch::zeros_like(adj);
  auto output_pred = torch::zeros_like(adj);
  AT_DISPATCH_INTEGRAL_TYPES(
    adj.scalar_type(), "floyd_warshall_set_init_kernel", ([&] {
      floyd_warshall_set_init_kernel<scalar_t><<<grid_dim, block_dim>>>(
        num_nodes,
        max_dist,
        adj.packed_accessor64<scalar_t, 2>(),
        output_dist.packed_accessor64<scalar_t, 2>());
  }));
  for (size_t k = 0; k < num_nodes; ++k) {
    AT_DISPATCH_INTEGRAL_TYPES(
      adj.scalar_type(), "floyd_warshall_cuda_one_iter_kernel", ([&] {
        floyd_warshall_cuda_one_iter_kernel<scalar_t><<<grid_dim, block_dim>>>(
          num_nodes,
          output_dist.packed_accessor64<scalar_t, 2>(),
          output_pred.packed_accessor64<scalar_t, 2>(),
          k);
    }));
  }
  return {output_dist, output_pred};
}

template <typename int_t>
__global__ void floyd_warshall_batch_kernel(
  int_t max_dist,
  const torch::PackedTensorAccessor64<int_t, 1> n_node_batch,
  const torch::PackedTensorAccessor64<int_t, 3> adj_batch,
  torch::PackedTensorAccessor64<int_t, 3> output_dist_batch,
  torch::PackedTensorAccessor64<int_t, 3> output_pred_batch) {
  const int_t n_node = n_node_batch[blockIdx.x];
  auto output_dist = output_dist_batch[blockIdx.x];
  auto output_pred = output_pred_batch[blockIdx.x];
  auto adj = adj_batch[blockIdx.x];
  for (int_t i = static_cast<int_t>(threadIdx.x); i < n_node; i += static_cast<int_t>(blockDim.x)) {
    for (int_t j = static_cast<int_t>(threadIdx.y); j < n_node; j += static_cast<int_t>(blockDim.y)) {
      if (i < n_node && j < n_node) {
        if (i == j) {
          output_dist[i][j] = 0;
        } else {
          if (adj[i][j]) {
            output_dist[i][j] = 1;
          } else {
            output_dist[i][j] = max_dist;
          }
        }
      }
    }
  }
  __syncthreads();
  for (int_t k = 0; k < n_node; ++k) {
    for (int_t i = static_cast<int_t>(threadIdx.x); i < n_node; i += static_cast<int_t>(blockDim.x)) {
      for (int_t j = static_cast<int_t>(threadIdx.y); j < n_node; j += static_cast<int_t>(blockDim.y)) {
        if (i < n_node && j < n_node && i != k && j != k && i != j) {
          const int_t new_dist = output_dist[i][k] + output_dist[k][j];
          const int_t old_dist = output_dist[i][j];
          if (new_dist < old_dist) {
            output_dist[i][j] = new_dist;
            output_pred[i][j] = k;
          }
        }
      }
    }
    __syncthreads();
  }
  for (int_t i = static_cast<int_t>(threadIdx.x); i < n_node; i += static_cast<int_t>(blockDim.x)) {
    for (int_t j = static_cast<int_t>(threadIdx.y); j < n_node; j += static_cast<int_t>(blockDim.y)) {
      if (i < n_node && j < n_node) {
        output_dist[i][j] += 1;
      }
    }
  }
}

void floyd_warshall_batch_cuda(const torch::Tensor adj, const torch::Tensor n_node, const size_t max_n_node, const size_t max_dist, torch::Tensor output_dist, torch::Tensor output_floyd_pred) {
  const size_t num_graphs = adj.size(0);
  int64_t max_n_node_aligned = 1;
  int64_t max_n_node_shift = static_cast<int64_t>(max_n_node) - 1;
  while (max_n_node_shift > 0) {
    max_n_node_shift >>= 1;
    max_n_node_aligned <<= 1;
  }
  const int block_dim_x_y = max_n_node_aligned < 32 ? max_n_node_aligned : 32;
  dim3 block_dim(block_dim_x_y, block_dim_x_y);
  AT_DISPATCH_INTEGRAL_TYPES(
    adj.scalar_type(), "floyd_warshall_batch_kernel", ([&] {
      floyd_warshall_batch_kernel<scalar_t><<<num_graphs, block_dim>>>(
        max_dist,
        n_node.packed_accessor64<scalar_t, 1>(),
        adj.packed_accessor64<scalar_t, 3>(),
        output_dist.packed_accessor64<scalar_t, 3>(),
        output_floyd_pred.packed_accessor64<scalar_t, 3>());
  }));
}

template <typename int_t, typename accessor_t, size_t MAX_DIST, int64_t offset>
__device__ void get_path(
  const int_t num_nodes,
  const int_t max_dist,
  const int_t i,
  const int_t j,
  const accessor_t pred,
  const accessor_t dist,
  int_t* out_path) {
  int_t stk[MAX_DIST * 2 + 10];
  int64_t stk_ptr = 0;
  stk[0] = i;
  stk[1] = j;
  stk[2] = max_dist;
  stk[3] = max_dist;
  while (stk_ptr >= 0) {
    const int_t stk_i = stk[2 * stk_ptr];
    const int_t stk_j = stk[2 * stk_ptr + 1];
    const int_t next_flag = stk[2 * stk_ptr + 2];

    const int_t stk_k = pred[stk_i][stk_j];
    const int_t stk_i_k_dist = dist[stk_i][stk_k] - static_cast<int_t>(offset);

    if (next_flag == max_dist) {
      // push first segment
      out_path[stk_i_k_dist - 1] = stk_k;

      ++stk_ptr;
      stk[2 * stk_ptr] = stk_i;
      stk[2 * stk_ptr + 1] = stk_k;
      if (stk_i_k_dist > 1) {
        stk[2 * (stk_ptr + 1)] = max_dist;
        stk[2 * (stk_ptr + 1) + 1] = max_dist;
      } else {
        --stk_ptr;
      }
    } else if (next_flag == stk_i) {
      // push second segment
      const int_t stk_k_j_dist = dist[stk_k][stk_j] - static_cast<int_t>(offset);
      ++stk_ptr;
      stk[2 * stk_ptr] = stk_k;
      stk[2 * stk_ptr + 1] = stk_j;
      if (stk_k_j_dist > 1) {
        stk[2 * (stk_ptr + 1)] = max_dist;
        stk[2 * (stk_ptr + 1) + 1] = max_dist;
      } else {
        --stk_ptr;
      }
      out_path += stk_i_k_dist;
    } else if (next_flag == stk_k) {
      // callback
      out_path -= stk_i_k_dist;
      --stk_ptr;
    }
  }
}

template <typename int_t, size_t MAX_DIST>
__global__ void gen_edge_input_kernel(
  const int_t num_nodes,
  const int_t max_dist,
  const torch::PackedTensorAccessor64<int_t, 2> pred,
  const torch::PackedTensorAccessor64<int_t, 2> dist,
  const int_t num_edge_features,
  const torch::PackedTensorAccessor64<int_t, 3> edge_features,
  torch::PackedTensorAccessor64<int_t, 4> output_edge_features) {
  const int i = static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x);
  const int j = static_cast<int>(threadIdx.y + blockIdx.y * blockDim.y);
  if (i < num_nodes && j < num_nodes) {
    int_t path[MAX_DIST];
    const int_t i_j_dist = dist[i][j];
    if (i_j_dist > 1 && i_j_dist != max_dist) {
      get_path<int_t, torch::PackedTensorAccessor64<int_t, 2>, MAX_DIST, 0>(num_nodes, max_dist, i, j, pred, dist, path);
    }
    if (i_j_dist != max_dist) {
      int start = i;
      for (int e = 0; e < i_j_dist; ++e) {
        const int end = (e == i_j_dist - 1 ? j : path[e]);
        for (int feature_index = 0; feature_index < num_edge_features; ++feature_index) {
          output_edge_features[i][j][e][feature_index] = edge_features[start][end][feature_index];
        }
        start = end;
      }
    }
  }
}

void gen_edge_input_cuda(
  const size_t max_dist,
  const torch::Tensor pred,
  const torch::Tensor dist,
  const size_t num_edge_features,
  const torch::Tensor edge_features,
  torch::Tensor output_edge_features) {
  const size_t num_nodes = pred.size(0);
  const size_t grid_dim_x = (num_nodes + FLOYD_WARSHALL_BLOCK_DIM_X - 1) / FLOYD_WARSHALL_BLOCK_DIM_X;
  const size_t grid_dim_y = (num_nodes + FLOYD_WARSHALL_BLOCK_DIM_Y - 1) / FLOYD_WARSHALL_BLOCK_DIM_Y;
  dim3 grid_dim(grid_dim_x, grid_dim_y), block_dim(FLOYD_WARSHALL_BLOCK_DIM_X, FLOYD_WARSHALL_BLOCK_DIM_Y);
  AT_DISPATCH_INTEGRAL_TYPES(pred.scalar_type(), "gen_edge_input_kernel", ([&] {
    auto dist_accessor = dist.packed_accessor64<scalar_t, 2>();
    auto pred_accessor = pred.packed_accessor64<scalar_t, 2>();
    auto edge_features_accessor = edge_features.packed_accessor64<scalar_t, 3>();
    auto output_edge_features_accessor = output_edge_features.packed_accessor64<scalar_t, 4>();
    if (max_dist <= 16) {
      gen_edge_input_kernel<scalar_t, 16><<<grid_dim, block_dim>>>(
        num_nodes, max_dist, pred_accessor, dist_accessor, num_edge_features, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist <= 32) {
      gen_edge_input_kernel<scalar_t, 32><<<grid_dim, block_dim>>>(
        num_nodes, max_dist, pred_accessor, dist_accessor, num_edge_features, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist <= 64) {
      gen_edge_input_kernel<scalar_t, 64><<<grid_dim, block_dim>>>(
        num_nodes, max_dist, pred_accessor, dist_accessor, num_edge_features, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist <= 128) {
      gen_edge_input_kernel<scalar_t, 128><<<grid_dim, block_dim>>>(
        num_nodes, max_dist, pred_accessor, dist_accessor, num_edge_features, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist <= 256) {
      gen_edge_input_kernel<scalar_t, 256><<<grid_dim, block_dim>>>(
        num_nodes, max_dist, pred_accessor, dist_accessor, num_edge_features, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist <= 512) {
      gen_edge_input_kernel<scalar_t, 512><<<grid_dim, block_dim>>>(
        num_nodes, max_dist, pred_accessor, dist_accessor, num_edge_features, edge_features_accessor, output_edge_features_accessor);
    } else {
      std::cout << "error, max_dist = " << max_dist << " is not supported by gen_edge_input." << std::endl;
      exit(-1);
    }
  }));
}

template <typename int_t, typename feat_int_t, size_t MAX_DIST>
__global__ void gen_edge_input_batch_kernel(
  const int_t max_dist,
  const int_t max_n_node,
  const int_t num_edge_features,
  const int_t multi_hop_max_dist,
  const torch::PackedTensorAccessor64<int_t, 1> n_node_batch,
  const torch::PackedTensorAccessor64<int_t, 3> pred_batch,
  const torch::PackedTensorAccessor64<int_t, 3> dist_batch,
  const torch::PackedTensorAccessor64<feat_int_t, 4> edge_features_batch,
  torch::PackedTensorAccessor64<feat_int_t, 5> output_edge_features_batch) {

  const int_t n_node = n_node_batch[blockIdx.x];
  auto dist = dist_batch[blockIdx.x];
  auto pred = pred_batch[blockIdx.x];
  auto edge_features = edge_features_batch[blockIdx.x];
  auto output_edge_features = output_edge_features_batch[blockIdx.x];
  int_t path[MAX_DIST];
  for (int_t i = static_cast<int_t>(threadIdx.x); i < n_node; i += static_cast<int_t>(blockDim.x)) {
    for (int_t j = static_cast<int_t>(threadIdx.y); j < n_node; j += static_cast<int_t>(blockDim.y)) {
      if (i == j) continue;
      const int_t i_j_dist = dist[i][j] - 1;
      if (i_j_dist > 1 && i_j_dist != max_dist) {
        get_path<int_t, torch::TensorAccessor<int_t, 2>, MAX_DIST, 1>(n_node, max_dist, i, j, pred, dist, path);
      }
      if (i_j_dist != max_dist) {
        int start = i;
        for (int e = 0; e < i_j_dist && e < static_cast<int>(multi_hop_max_dist); ++e) {
          const int end = (e == i_j_dist - 1 ? j : path[e]);
          for (int feature_index = 0; feature_index < num_edge_features; ++feature_index) {
            output_edge_features[i][j][e][feature_index] = edge_features[start][end][feature_index] + 1;
          }
          start = end;
        }
      }
    }
  }
}

#include <iostream>

void gen_edge_input_batch_cuda(
  const size_t max_dist,
  const size_t max_n_node,
  const size_t num_edge_features,
  const size_t multi_hop_max_dist,
  const torch::Tensor n_node,
  const torch::Tensor pred,
  const torch::Tensor dist,
  const torch::Tensor edge_features,
  torch::Tensor output_edge_features) {
  const size_t num_graphs = n_node.size(0);
  int64_t max_n_node_aligned = 1;
  int64_t max_n_node_shift = static_cast<int64_t>(max_n_node) - 1;
  while (max_n_node_shift > 0) {
    max_n_node_shift >>= 1;
    max_n_node_aligned <<= 1;
  }
  const int block_dim_x_y = max_n_node_aligned < 16 ? max_n_node_aligned : 16;
  dim3 block_dim(block_dim_x_y, block_dim_x_y);
  AT_DISPATCH_INTEGRAL_TYPES(pred.scalar_type(), "gen_edge_input_batch_kernel", ([&] {
    auto n_node_accessor = n_node.packed_accessor64<scalar_t, 1>();
    auto dist_accessor = dist.packed_accessor64<scalar_t, 3>();
    auto pred_accessor = pred.packed_accessor64<scalar_t, 3>();
    auto edge_features_accessor = edge_features.packed_accessor64<uint8_t, 4>();
    auto output_edge_features_accessor = output_edge_features.packed_accessor64<uint8_t, 5>();
    size_t max_dist_true = max_n_node + 1;
    if (max_dist_true <= 16) {
      gen_edge_input_batch_kernel<scalar_t, uint8_t, 16><<<num_graphs, block_dim>>>(
        max_dist, max_n_node, num_edge_features, multi_hop_max_dist, n_node_accessor, pred_accessor, dist_accessor, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist_true <= 32) {
      gen_edge_input_batch_kernel<scalar_t, uint8_t, 32><<<num_graphs, block_dim>>>(
        max_dist, max_n_node, num_edge_features, multi_hop_max_dist, n_node_accessor, pred_accessor, dist_accessor, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist_true <= 64) {
      gen_edge_input_batch_kernel<scalar_t, uint8_t, 64><<<num_graphs, block_dim>>>(
        max_dist, max_n_node, num_edge_features, multi_hop_max_dist, n_node_accessor, pred_accessor, dist_accessor, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist_true <= 128) {
      gen_edge_input_batch_kernel<scalar_t, uint8_t, 128><<<num_graphs, block_dim>>>(
        max_dist, max_n_node, num_edge_features, multi_hop_max_dist, n_node_accessor, pred_accessor, dist_accessor, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist_true <= 256) {
      gen_edge_input_batch_kernel<scalar_t, uint8_t, 256><<<num_graphs, block_dim>>>(
        max_dist, max_n_node, num_edge_features, multi_hop_max_dist, n_node_accessor, pred_accessor, dist_accessor, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist_true <= 512) {
      gen_edge_input_batch_kernel<scalar_t, uint8_t, 512><<<num_graphs, block_dim>>>(
        max_dist, max_n_node, num_edge_features, multi_hop_max_dist, n_node_accessor, pred_accessor, dist_accessor, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist_true <= 1024) {
      gen_edge_input_batch_kernel<scalar_t, uint8_t, 1024><<<num_graphs, block_dim>>>(
        max_dist, max_n_node, num_edge_features, multi_hop_max_dist, n_node_accessor, pred_accessor, dist_accessor, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist_true <= 1536) {
      gen_edge_input_batch_kernel<scalar_t, uint8_t, 1536><<<num_graphs, block_dim>>>(
        max_dist, max_n_node, num_edge_features, multi_hop_max_dist, n_node_accessor, pred_accessor, dist_accessor, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist_true <= 2048) {
      gen_edge_input_batch_kernel<scalar_t, uint8_t, 2048><<<num_graphs, block_dim>>>(
        max_dist, max_n_node, num_edge_features, multi_hop_max_dist, n_node_accessor, pred_accessor, dist_accessor, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist_true <= 2560) {
      gen_edge_input_batch_kernel<scalar_t, uint8_t, 2560><<<num_graphs, block_dim>>>(
        max_dist, max_n_node, num_edge_features, multi_hop_max_dist, n_node_accessor, pred_accessor, dist_accessor, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist_true <= 3072) {
      gen_edge_input_batch_kernel<scalar_t, uint8_t, 3072><<<num_graphs, block_dim>>>(
        max_dist, max_n_node, num_edge_features, multi_hop_max_dist, n_node_accessor, pred_accessor, dist_accessor, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist_true <= 3584) {
      gen_edge_input_batch_kernel<scalar_t, uint8_t, 3584><<<num_graphs, block_dim>>>(
        max_dist, max_n_node, num_edge_features, multi_hop_max_dist, n_node_accessor, pred_accessor, dist_accessor, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist_true <= 4096) {
      gen_edge_input_batch_kernel<scalar_t, uint8_t, 4096><<<num_graphs, block_dim>>>(
        max_dist, max_n_node, num_edge_features, multi_hop_max_dist, n_node_accessor, pred_accessor, dist_accessor, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist_true <= 4608) {
      gen_edge_input_batch_kernel<scalar_t, uint8_t, 4608><<<num_graphs, block_dim>>>(
        max_dist, max_n_node, num_edge_features, multi_hop_max_dist, n_node_accessor, pred_accessor, dist_accessor, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist_true <= 5120) {
      gen_edge_input_batch_kernel<scalar_t, uint8_t, 5120><<<num_graphs, block_dim>>>(
        max_dist, max_n_node, num_edge_features, multi_hop_max_dist, n_node_accessor, pred_accessor, dist_accessor, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist_true <= 5632) {
      gen_edge_input_batch_kernel<scalar_t, uint8_t, 5632><<<num_graphs, block_dim>>>(
        max_dist, max_n_node, num_edge_features, multi_hop_max_dist, n_node_accessor, pred_accessor, dist_accessor, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist_true <= 6144) {
      gen_edge_input_batch_kernel<scalar_t, uint8_t, 6144><<<num_graphs, block_dim>>>(
        max_dist, max_n_node, num_edge_features, multi_hop_max_dist, n_node_accessor, pred_accessor, dist_accessor, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist_true <= 6656) {
      gen_edge_input_batch_kernel<scalar_t, uint8_t, 6656><<<num_graphs, block_dim>>>(
        max_dist, max_n_node, num_edge_features, multi_hop_max_dist, n_node_accessor, pred_accessor, dist_accessor, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist_true <= 7168) {
      gen_edge_input_batch_kernel<scalar_t, uint8_t, 7168><<<num_graphs, block_dim>>>(
        max_dist, max_n_node, num_edge_features, multi_hop_max_dist, n_node_accessor, pred_accessor, dist_accessor, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist_true <= 7680) {
      gen_edge_input_batch_kernel<scalar_t, uint8_t, 7680><<<num_graphs, block_dim>>>(
        max_dist, max_n_node, num_edge_features, multi_hop_max_dist, n_node_accessor, pred_accessor, dist_accessor, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist_true <= 8192) {
      gen_edge_input_batch_kernel<scalar_t, uint8_t, 8192><<<num_graphs, block_dim>>>(
        max_dist, max_n_node, num_edge_features, multi_hop_max_dist, n_node_accessor, pred_accessor, dist_accessor, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist_true <= 8704) {
      gen_edge_input_batch_kernel<scalar_t, uint8_t, 8704><<<num_graphs, block_dim>>>(
        max_dist, max_n_node, num_edge_features, multi_hop_max_dist, n_node_accessor, pred_accessor, dist_accessor, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist_true <= 9216) {
      gen_edge_input_batch_kernel<scalar_t, uint8_t, 9216><<<num_graphs, block_dim>>>(
        max_dist, max_n_node, num_edge_features, multi_hop_max_dist, n_node_accessor, pred_accessor, dist_accessor, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist_true <= 10240) {
      gen_edge_input_batch_kernel<scalar_t, uint8_t, 10240><<<num_graphs, block_dim>>>(
        max_dist, max_n_node, num_edge_features, multi_hop_max_dist, n_node_accessor, pred_accessor, dist_accessor, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist_true <= 10752) {
      gen_edge_input_batch_kernel<scalar_t, uint8_t, 10752><<<num_graphs, block_dim>>>(
        max_dist, max_n_node, num_edge_features, multi_hop_max_dist, n_node_accessor, pred_accessor, dist_accessor, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist_true <= 11264) {
      gen_edge_input_batch_kernel<scalar_t, uint8_t, 11264><<<num_graphs, block_dim>>>(
        max_dist, max_n_node, num_edge_features, multi_hop_max_dist, n_node_accessor, pred_accessor, dist_accessor, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist_true <= 11776) {
      gen_edge_input_batch_kernel<scalar_t, uint8_t, 11776><<<num_graphs, block_dim>>>(
        max_dist, max_n_node, num_edge_features, multi_hop_max_dist, n_node_accessor, pred_accessor, dist_accessor, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist_true <= 12288) {
      gen_edge_input_batch_kernel<scalar_t, uint8_t, 12288><<<num_graphs, block_dim>>>(
        max_dist, max_n_node, num_edge_features, multi_hop_max_dist, n_node_accessor, pred_accessor, dist_accessor, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist_true <= 12800) {
      gen_edge_input_batch_kernel<scalar_t, uint8_t, 12800><<<num_graphs, block_dim>>>(
        max_dist, max_n_node, num_edge_features, multi_hop_max_dist, n_node_accessor, pred_accessor, dist_accessor, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist_true <= 13312) {
      gen_edge_input_batch_kernel<scalar_t, uint8_t, 13312><<<num_graphs, block_dim>>>(
        max_dist, max_n_node, num_edge_features, multi_hop_max_dist, n_node_accessor, pred_accessor, dist_accessor, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist_true <= 13824) {
      gen_edge_input_batch_kernel<scalar_t, uint8_t, 13824><<<num_graphs, block_dim>>>(
        max_dist, max_n_node, num_edge_features, multi_hop_max_dist, n_node_accessor, pred_accessor, dist_accessor, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist_true <= 14336) {
      gen_edge_input_batch_kernel<scalar_t, uint8_t, 14336><<<num_graphs, block_dim>>>(
        max_dist, max_n_node, num_edge_features, multi_hop_max_dist, n_node_accessor, pred_accessor, dist_accessor, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist_true <= 14848) {
      gen_edge_input_batch_kernel<scalar_t, uint8_t, 14848><<<num_graphs, block_dim>>>(
        max_dist, max_n_node, num_edge_features, multi_hop_max_dist, n_node_accessor, pred_accessor, dist_accessor, edge_features_accessor, output_edge_features_accessor);
    } else if (max_dist_true <= 15360) {
      gen_edge_input_batch_kernel<scalar_t, uint8_t, 15360><<<num_graphs, block_dim>>>(
        max_dist, max_n_node, num_edge_features, multi_hop_max_dist, n_node_accessor, pred_accessor, dist_accessor, edge_features_accessor, output_edge_features_accessor);
    } else {
      std::cout << "error, max_dist = " << max_dist_true << " is not supported by gen_edge_input." << std::endl;
      exit(-1);
    }
  }));
}
