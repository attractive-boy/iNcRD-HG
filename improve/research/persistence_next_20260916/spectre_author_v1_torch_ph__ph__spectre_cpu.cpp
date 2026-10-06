#include "ATen/core/function_schema.h"
#include "unionfind.hh"
#include <ATen/Parallel.h>
#include <algorithm>
#include <iostream>
#include <torch/extension.h>
#pragma omp parallel num_threads(4)

int max_size_eigenvalues;

using namespace torch::indexing;

torch::Tensor uf_find(torch::Tensor parents, int u) {
    //std::cout << "Entering uf_find with u=" << u << std::endl;
    auto out = torch::empty(1, parents.options());
    AT_DISPATCH_INTEGRAL_TYPES(parents.scalar_type(), "uf_find", ([&] {
        out[0] = UnionFind<scalar_t>::find(
            parents.accessor<scalar_t, 1>(), static_cast<scalar_t>(u));
    }));
    ////std::cout << "Exiting uf_find with result=" << out[0].item<int>() << std::endl;
    return out[0];
}

void uf_merge(torch::Tensor parents, int u, int v) {
    ////std::cout << "Entering uf_merge with u=" << u << ", v=" << v << std::endl;
    AT_DISPATCH_INTEGRAL_TYPES(parents.scalar_type(), "uf_merge", ([&] {
        UnionFind<scalar_t>::merge(
            parents.accessor<scalar_t, 1>(), static_cast<scalar_t>(u), static_cast<scalar_t>(v));
    }));
    ////std::cout << "Exiting uf_merge" << std::endl;
}

template <typename float_t, typename int_t>
void compute_spectre_raw(
    torch::TensorAccessor<float_t, 1> filtered_v,
    torch::TensorAccessor<float_t, 1> filtered_e,
    torch::TensorAccessor<float_t, 2> death_eigvals,
    torch::TensorAccessor<float_t, 2> birth_eigvals,
    torch::TensorAccessor<int_t, 2> edge_index,
    torch::TensorAccessor<int_t, 1> parents,
    torch::TensorAccessor<int_t, 1> sorting_space,
    torch::TensorAccessor<int_t, 2> pers_indices,
    torch::TensorAccessor<int_t, 2> pers1_indices,
    int_t vertex_begin, int_t vertex_end, int_t edge_begin, int_t edge_end) {

    ////std::cout << "Starting compute_rephine_raw" << std::endl;
    ////std::cout << "vertex_begin=" << vertex_begin << ", vertex_end=" << vertex_end
              //<< ", edge_begin=" << edge_begin << ", edge_end=" << edge_end << std::endl;

    auto n_vertices = vertex_end - vertex_begin;
    auto n_edges = edge_end - edge_begin;

    std::map<int, std::vector<int>> adjacencies;

    std::vector<std::pair<int, bool>> draw_v_e; // bool : is edge


    int_t* sorting_begin = sorting_space.data() + edge_begin;
    int_t* sorting_end = sorting_space.data() + edge_end;
    std::stable_sort(sorting_begin, sorting_end, [&filtered_e](int_t i, int_t j) {
        return filtered_e[i] < filtered_e[j];
    });

    auto process_component = [&](int_t root,
                            torch::TensorAccessor<float_t, 2>& eigvals_storage, int_t storage_index) {
        // Collect component nodes
        std::vector<int_t> component_nodes;
        for (int_t v = vertex_begin; v < vertex_end; ++v) {
            if (UnionFind<int_t>::find(parents, v) == root) {
                component_nodes.push_back(v);
            }
        }

        // Collect component edges
        std::vector<std::pair<int_t, int_t>> component_edges;
        for (auto e: adjacencies[root]) {
            int_t u = edge_index[e][0];
            int_t v = edge_index[e][1];

            component_edges.emplace_back(u, v);
        }


        // Create adjacency mapping
        std::unordered_map<int_t, int_t> node_to_index;
        for (size_t i = 0; i < component_nodes.size(); ++i) {
            node_to_index[component_nodes[i]] = i;
        }

        // Build adjacency matrix
        const size_t n = component_nodes.size();
        std::vector<std::vector<int_t>> adj(n, std::vector<int_t>(n, 0));
        for (const auto& edge : component_edges) {
            int_t u = node_to_index[edge.first];
            int_t v = node_to_index[edge.second];
            adj[u][v] = adj[v][u] = 1;
        }

        // Compute Laplacian matrix
        std::vector<std::vector<int_t>> lap(n, std::vector<int_t>(n, 0));
        for (size_t i = 0; i < n; ++i) {
            int_t degree = std::accumulate(adj[i].begin(), adj[i].end(), 0);
            lap[i][i] = degree;
            for (size_t j = 0; j < n; ++j) {
                if (i != j) lap[i][j] = -adj[i][j];
            }
        }

        // Convert to tensor and compute eigenvalues
        torch::Tensor L = torch::zeros({(int_t)n, (int_t)n}, torch::kFloat32);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                L[i][j] = lap[i][j];
            }
        }

        auto [eigvals, _] = torch::linalg_eigh(L);

        // Pad and threshold eigenvalues
        torch::Tensor padded = torch::zeros(max_size_eigenvalues, torch::kFloat32);
        padded.slice(0, 0, eigvals.size(0)) = eigvals;

        float threshold = 1e-5;
        padded = padded.where(padded.abs() >= threshold, torch::zeros_like(padded));

        // Store in death eigenvalues matrix
        for (int j = 0; j < max_size_eigenvalues; ++j) {
            eigvals_storage[storage_index][j] = padded[j].item<float_t>();
        }
        if( (int)draw_v_e.size() > 0){
            for(auto[draw_index , is_edge]: draw_v_e){
            // for(int i=0; i< (int)draw_v_e.size(); i++){
            //     auto[draw_index, is_edge]= draw_v_e[i];
                if(is_edge){
                    for (int j = 0; j < max_size_eigenvalues; ++j) {
                      birth_eigvals[draw_index][j] = padded[j].item<float_t>();
                    }
                }
                else{
                    for (int j = 0; j < max_size_eigenvalues; ++j) {
                      death_eigvals[draw_index][j] = padded[j].item<float_t>();
                    }
                }

            }
            draw_v_e.clear();
        }
    };




    for (auto i = 0; i < n_edges; i++) {
        auto cur_edge_index = sorting_space[edge_begin + i];
        //std::cout << "cur edge index " << cur_edge_index << std::endl;
        //std::cout << "i: " << i << std::endl;
        auto node1 = edge_index[cur_edge_index][0];
        auto node2 = edge_index[cur_edge_index][1];

        //std::cout << "Processing edge " << i << ": (" << node1 << ", " << node2 << ")" << std::endl;

        if (pers_indices[node1][1] == -1) {
            pers_indices[node1][1] = cur_edge_index;
        }

        if (pers_indices[node2][1] == -1) {
            pers_indices[node2][1] = cur_edge_index;
        }

        try {
            auto younger = UnionFind<int_t>::find(parents, node1);
            auto older = UnionFind<int_t>::find(parents, node2);

            //std::cout << "younger=" << younger << ", older=" << older << std::endl;

            if (younger == older) {

                pers1_indices[cur_edge_index][0] = cur_edge_index;

                adjacencies[older].push_back(cur_edge_index);

                if( (i== n_edges -1) || filtered_e[cur_edge_index] != filtered_e[sorting_space[edge_begin + i + 1]]){
                    //std::cout<< "OUUUUUUUU\n";
                    process_component(older, birth_eigvals, cur_edge_index);
                }
                else{
                    draw_v_e.push_back({cur_edge_index,true});
                }

                // birth_eigenvals
                continue;
            } else {
                if (filtered_v[younger] == filtered_v[older]) {
                    if (filtered_e[pers_indices[younger][1]] > filtered_e[pers_indices[older][1]]) {
                        std::swap(younger, older);
                        std::swap(node1, node2);
                    }
                } else if (filtered_v[younger] < filtered_v[older]) {
                    std::swap(younger, older);
                    std::swap(node1, node2);
                }
            }
            pers_indices[younger][0] = cur_edge_index;

            adjacencies[older].insert(adjacencies[older].begin(), adjacencies[younger].begin(), adjacencies[younger].end());

            adjacencies[younger]= std::vector<int>();

            UnionFind<int_t>::merge(parents, node1, node2);


            adjacencies[older].push_back(cur_edge_index);



            //edge_begin + i < edge_begin + n_edges;

            if( (i== n_edges -1) || filtered_e[cur_edge_index] != filtered_e[sorting_space[edge_begin + i + 1]]){
                //std::cout<< "OUUUUUUUU\n";
                process_component(older, death_eigvals, younger);
            }
            else{
                draw_v_e.push_back({younger, false});
            }

        } catch (const std::exception &e) {
            std::cerr << "Exception occurred while processing edge: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "Unknown exception occurred while processing edge." << std::endl;
        }
    }



    for (auto i = 0; i < n_vertices; i++) {
        auto vertex_index = vertex_begin + i;
        auto parent_value = parents[vertex_index];
        if (vertex_index == parent_value) {
            pers_indices[vertex_index][0] = -1;

            process_component(vertex_index, death_eigvals, vertex_index);
        }
    }
    //std::cout << "Finished compute_rephine_raw" << std::endl;
}

template <typename float_t, typename int_t>
void compute_spectre_ptrs(
    torch::TensorAccessor<float_t, 2> filtered_v,
    torch::TensorAccessor<float_t, 2> filtered_e,
    torch::TensorAccessor<float_t, 3> death_eigvals,
    torch::TensorAccessor<float_t, 3> birth_eigvals,
    torch::TensorAccessor<int_t, 2> edge_index,
    torch::TensorAccessor<int_t, 1> vertex_slices,
    torch::TensorAccessor<int_t, 1> edge_slices,
    torch::TensorAccessor<int_t, 2> parents,
    torch::TensorAccessor<int_t, 2> sorting_space,
    torch::TensorAccessor<int_t, 3> pers_ind,
    torch::TensorAccessor<int_t, 3> pers1_ind) {
  auto n_graphs = vertex_slices.size(0) - 1;
  auto n_filtrations = filtered_v.size(0);

  at::parallel_for(
      0, n_graphs * n_filtrations, 0, [&](int64_t begin, int64_t end) {
        for (auto i = begin; i < end; i++) {
          auto instance = i / n_filtrations;
          auto filtration = i % n_filtrations;
          compute_spectre_raw<float_t, int_t>(
              filtered_v[filtration], filtered_e[filtration], death_eigvals[filtration], birth_eigvals[filtration],
              edge_index,
              parents[filtration], sorting_space[filtration],
              pers_ind[filtration], pers1_ind[filtration],
              vertex_slices[instance], vertex_slices[instance + 1],
              edge_slices[instance], edge_slices[instance + 1]);
        }
      });
}

std::tuple<torch::Tensor, torch::Tensor>
compute_spectre_batched_mt(torch::Tensor filtered_v,
                                        torch::Tensor filtered_e,
                                        torch::Tensor edge_index,
                                        torch::Tensor vertex_slices,
                                        torch::Tensor edge_slices) {
  bool set_invalid_to_nan = true;

  std::vector<int> slices_diff;
  for(int i = 0; i < vertex_slices.size(0) - 1; i++) {
    slices_diff.push_back(vertex_slices[i + 1].item<int>() - vertex_slices[i].item<int>());
  }
  max_size_eigenvalues = std::max(*(std::max_element(slices_diff.begin(), slices_diff.end())), max_size_eigenvalues);

  auto n_nodes = filtered_v.size(1);
  auto n_edges = filtered_e.size(1);
  auto n_filtrations = filtered_v.size(0);
  auto integer_no_grad = torch::TensorOptions();
  integer_no_grad = integer_no_grad.requires_grad(false);
  integer_no_grad = integer_no_grad.device(edge_index.options().device());
  integer_no_grad = integer_no_grad.dtype(edge_index.options().dtype());

  // Output indicators
  auto pers_ind = torch::full({n_filtrations, n_nodes, 2}, -1, integer_no_grad);
  auto pers1_ind = torch::full({n_filtrations, n_edges, 3}, -1, integer_no_grad);

  auto death_eigvals = torch::full({n_filtrations, n_nodes, max_size_eigenvalues}, -1.0, filtered_v.options());

  auto birth_eigvals = torch::full({n_filtrations, n_edges, max_size_eigenvalues}, 0.0, filtered_e.options());
  // Datastructure for UnionFind and sorting operations
  auto parents = torch::arange(0, n_nodes, integer_no_grad)
                     .unsqueeze(0)
                     .repeat({n_filtrations, 1});
  auto sorting_space = torch::arange(0, n_edges, integer_no_grad)
                           .unsqueeze(0)
                           .repeat({n_filtrations, 1})
                           .contiguous();



  // Double dispatch over int and float types
  AT_DISPATCH_FLOATING_TYPES(
      filtered_v.scalar_type(), "compute_spectre_batched_mt1", ([&] {
        using float_t = scalar_t;
        AT_DISPATCH_INTEGRAL_TYPES(
            edge_index.scalar_type(),
            "compute_spectre_batched_"
            "mt2",
            ([&] {
              using int_t = scalar_t;
              compute_spectre_ptrs<float_t, int_t>(
                  filtered_v.accessor<float_t, 2>(),
                  filtered_e.accessor<float_t, 2>(),
                  death_eigvals.accessor<float_t, 3>(),
                  birth_eigvals.accessor<float_t, 3>(),
                  edge_index.accessor<int_t, 2>(),
                  vertex_slices.accessor<int_t, 1>(),
                  edge_slices.accessor<int_t, 1>(),
                  parents.accessor<int_t, 2>(),
                  sorting_space.accessor<int_t, 2>(),
                  pers_ind.accessor<int_t, 3>(),
                  pers1_ind.accessor<int_t, 3>());
            }));
      }));

  // Extract only the first two elements (edge indices) from pers_ind
  auto pers_ind_edge = pers_ind.slice(2, 0, 2);

  auto pers_e = filtered_e
          .index({torch::arange(0, n_filtrations, integer_no_grad).unsqueeze(1),
                  pers_ind_edge.view({n_filtrations, -1})})
          .view({n_filtrations, n_nodes, 2});


  float_t invalid_fill_value;
  if (set_invalid_to_nan)
    invalid_fill_value = std::numeric_limits<float_t>::quiet_NaN();
  else
    invalid_fill_value = 0;

  pers_e.index_put_({pers_ind_edge == -1}, invalid_fill_value);

  auto pers_v_ind = torch::full({n_filtrations, n_nodes, 1}, -1, integer_no_grad);
  pers_v_ind.index_put_({"...", 0}, torch::arange(0, n_nodes, integer_no_grad));

  auto pers_v = filtered_v
          .index({torch::arange(0, n_filtrations, integer_no_grad).unsqueeze(1),
                  pers_v_ind.view({n_filtrations, -1})})
          .view({n_filtrations, n_nodes, 1});

  //std::cout<< "Pers_v: " << pers_v << std::endl;

  // Extract eigenvalues from pers_ind
  //auto pers_eigvals = pers_ind.slice(2, 2, 2 + max_size_eigenvalues);

  // Concatenate pers_e, pers_v, and pers_eigvals
  auto pers = torch::cat({pers_e, pers_v, death_eigvals}, 2);

  // Gather filtration values according to the indices defined in pers1_ind.
  invalid_fill_value = 0;
   auto pers1 =
      torch::cat(
          {filtered_e, torch::full({n_filtrations, 1}, invalid_fill_value,
                                   filtered_v.options())},
         1)
          .index({torch::arange(n_filtrations, integer_no_grad).unsqueeze(1),
                  pers1_ind.view({n_filtrations, -1})})
          .view({n_filtrations, n_edges, 3});

    pers1= torch::cat({pers1, birth_eigvals}, 2);
    //std::cout<< "Pers: " << pers << std::endl;
    //std::cout<< "Pers1: " << pers1 << std::endl;
  return std::make_tuple(std::move(pers), std::move(pers1));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compute_spectre_batched_mt",
        &compute_spectre_batched_mt,
        py::call_guard<py::gil_scoped_release>(),
        "Persistence routine multi threading");
}