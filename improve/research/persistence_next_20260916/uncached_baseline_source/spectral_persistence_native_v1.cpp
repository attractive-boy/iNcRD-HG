// Independent implementation of the SpectRe event definition (arXiv:2506.06571v3).
// Spectra are computed after a complete equal-filtration-value block, separately
// for each resulting component. No global padding state or silent error recovery.
#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <numeric>
#include <tuple>
#include <vector>

using I = int64_t;

static I root_of(std::vector<I>& parent, I node) {
    I root = node;
    while (parent[root] != root) root = parent[root];
    while (parent[node] != node) {
        I next = parent[node];
        parent[node] = root;
        node = next;
    }
    return root;
}

std::vector<torch::Tensor> event_spectra(
    torch::Tensor vertices, torch::Tensor edge_values, torch::Tensor edges,
    torch::Tensor vertex_slices, torch::Tensor edge_slices) {
    TORCH_CHECK(vertices.device().is_cpu() && edge_values.device().is_cpu() && edges.device().is_cpu()
                && vertex_slices.device().is_cpu() && edge_slices.device().is_cpu(),
                "Native persistence takes CPU arrays");
    TORCH_CHECK(vertices.scalar_type() == torch::kFloat64 && edge_values.scalar_type() == torch::kFloat64,
                "Native filtration arrays must be FP64");
    TORCH_CHECK(edges.scalar_type() == torch::kInt64 && vertex_slices.scalar_type() == torch::kInt64
                && edge_slices.scalar_type() == torch::kInt64, "Native indices must be int64");
    TORCH_CHECK(vertices.is_contiguous() && edge_values.is_contiguous() && edges.is_contiguous()
                && vertex_slices.is_contiguous() && edge_slices.is_contiguous(), "Contiguous arrays required");
    TORCH_CHECK(vertices.dim() == 1 && edge_values.dim() == 1 && edges.dim() == 2 && edges.size(1) == 2
                && vertex_slices.dim() == 1 && edge_slices.dim() == 1,
                "Unexpected native input dimensions");
    const I groups = vertex_slices.numel() - 1;
    TORCH_CHECK(groups >= 1 && edge_slices.numel() == groups + 1, "Invalid graph slices");
    TORCH_CHECK(edge_values.numel() == edges.size(0), "Edge value count mismatch");
    const auto* vs = vertex_slices.data_ptr<I>();
    const auto* es = edge_slices.data_ptr<I>();
    const auto* v = vertices.data_ptr<double>();
    const auto* f = edge_values.data_ptr<double>();
    const auto* ee = edges.data_ptr<I>();
    TORCH_CHECK(vs[0] == 0 && es[0] == 0 && vs[groups] == vertices.numel()
                && es[groups] == edge_values.numel(), "Slices must cover the packed input");
    I width = 0;
    for (I g = 0; g < groups; ++g) {
        TORCH_CHECK(vs[g + 1] > vs[g] && es[g + 1] >= es[g], "Empty graph or invalid slices");
        width = std::max(width, vs[g + 1] - vs[g]);
    }
    auto indices = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    auto values = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
    auto death = torch::full({vertices.numel()}, -1, indices);
    auto first = torch::full({vertices.numel()}, -1, indices);
    auto cycle = torch::zeros({edges.size(0)}, indices);
    auto rho0 = torch::zeros({vertices.numel(), width}, values);
    auto rho1 = torch::zeros({edges.size(0), width}, values);
    auto final_rho = torch::zeros({groups, width}, values);
    auto counts = torch::zeros({groups, 3}, indices); // components, cycles, eigendecompositions
    auto* d = death.data_ptr<I>();
    auto* gamma = first.data_ptr<I>();
    auto* cyc = cycle.data_ptr<I>();
    auto* r0 = rho0.data_ptr<double>();
    auto* r1 = rho1.data_ptr<double>();
    auto* rf = final_rho.data_ptr<double>();
    auto* stats = counts.data_ptr<I>();
    at::parallel_for(0, groups, 1, [&](I begin, I end) {
        for (I g = begin; g < end; ++g) {
            const I vb = vs[g], ve = vs[g + 1], eb = es[g], eend = es[g + 1], n = ve - vb;
            std::vector<I> parent(n), order(eend - eb), processed;
            std::iota(parent.begin(), parent.end(), 0);
            std::iota(order.begin(), order.end(), eb);
            for (I i = vb; i < ve; ++i) TORCH_CHECK(std::isfinite(v[i]), "Nonfinite vertex filter");
            for (I e = eb; e < eend; ++e) {
                const I a = ee[2 * e], b = ee[2 * e + 1];
                TORCH_CHECK(a >= vb && a < ve && b >= vb && b < ve && a != b,
                            "Edge crosses a graph boundary or is a self-loop");
                TORCH_CHECK(std::isfinite(f[e]) && f[e] > 0. && f[e] < 1.,
                            "Finite edge filters in (0,1) required");
                for (I node : {a, b}) if (gamma[node] < 0 || f[e] < f[gamma[node]]) gamma[node] = e;
            }
            std::stable_sort(order.begin(), order.end(), [&](I a, I b) { return f[a] < f[b]; });
            auto spectrum = [&](I root, bool entire_graph) {
                std::vector<I> map(n, -1), members;
                for (I node = 0; node < n; ++node) {
                    if (entire_graph || root_of(parent, node) == root) {
                        map[node] = members.size();
                        members.push_back(node);
                    }
                }
                const I size = members.size();
                TORCH_CHECK(size >= 1, "Empty event component");
                std::vector<double> answer(width, 0.);
                if (size == 1) return answer;
                auto laplacian = torch::zeros({size, size}, values);
                auto* matrix = laplacian.data_ptr<double>();
                I component_edges = 0;
                for (I e : processed) {
                    const I a = map[ee[2 * e] - vb], b = map[ee[2 * e + 1] - vb];
                    if (a < 0 || b < 0) continue;
                    TORCH_CHECK(matrix[a * size + b] == 0., "Duplicate undirected edge");
                    matrix[a * size + b] = matrix[b * size + a] = -1.;
                    matrix[a * size + a] += 1.;
                    matrix[b * size + b] += 1.;
                    ++component_edges;
                }
                double trace_square = 2. * component_edges;
                for (I node = 0; node < size; ++node) trace_square += std::pow(matrix[node * size + node], 2);
                auto eigen = torch::linalg_eigvalsh(laplacian);
                const auto* ev = eigen.data_ptr<double>();
                double sum = 0., sum_square = 0.;
                I nonzero = 0;
                for (I j = 0; j < size; ++j) {
                    TORCH_CHECK(std::isfinite(ev[j]) && ev[j] >= -1e-8, "Invalid Laplacian eigenvalue");
                    answer[j] = std::abs(ev[j]) <= 1e-10 ? 0. : ev[j];
                    sum += answer[j];
                    sum_square += answer[j] * answer[j];
                    nonzero += answer[j] != 0.;
                }
                TORCH_CHECK(std::abs(sum - 2. * component_edges) <= 1e-8 * (1. + component_edges),
                            "Laplacian spectral trace check failed");
                TORCH_CHECK(std::abs(sum_square - trace_square) <= 1e-8 * (1. + trace_square),
                            "Laplacian second spectral moment check failed");
                TORCH_CHECK(entire_graph || nonzero == size - 1, "Event component spectrum has incorrect rank");
                ++stats[3 * g + 2];
                return answer;
            };
            for (I start = 0; start < (I)order.size();) {
                I stop = start + 1;
                while (stop < (I)order.size() && f[order[stop]] == f[order[start]]) ++stop;
                std::vector<I> dead_nodes, cycle_edges;
                for (I pos = start; pos < stop; ++pos) {
                    const I e = order[pos], a = ee[2 * e] - vb, b = ee[2 * e + 1] - vb;
                    I ra = root_of(parent, a), rb = root_of(parent, b);
                    processed.push_back(e);
                    if (ra == rb) {
                        cyc[e] = 1;
                        cycle_edges.push_back(e);
                        ++stats[3 * g + 1];
                        continue;
                    }
                    auto older_than = [&](I left, I right) {
                        if (v[vb + left] != v[vb + right]) return v[vb + left] < v[vb + right];
                        const double gl = gamma[vb + left] < 0 ? INFINITY : f[gamma[vb + left]];
                        const double gr = gamma[vb + right] < 0 ? INFINITY : f[gamma[vb + right]];
                        if (gl != gr) return gl > gr;
                        return left < right;
                    };
                    if (older_than(ra, rb)) std::swap(ra, rb);
                    d[vb + ra] = e;
                    dead_nodes.push_back(ra);
                    parent[ra] = rb;
                }
                // Every event at this level sees its own final component after
                // all simultaneous edges. Distinct components never share spectra.
                std::map<I, std::vector<double>> spectra;
                auto at_root = [&](I root) -> const std::vector<double>& {
                    auto found = spectra.find(root);
                    if (found == spectra.end()) found = spectra.emplace(root, spectrum(root, false)).first;
                    return found->second;
                };
                for (I node : dead_nodes) {
                    const auto& eigen = at_root(root_of(parent, node));
                    std::copy(eigen.begin(), eigen.end(), r0 + (vb + node) * width);
                }
                for (I e : cycle_edges) {
                    const auto& eigen = at_root(root_of(parent, ee[2 * e] - vb));
                    std::copy(eigen.begin(), eigen.end(), r1 + e * width);
                }
                start = stop;
            }
            for (I node = 0; node < n; ++node) {
                if (root_of(parent, node) == node) {
                    ++stats[3 * g];
                    const auto eigen = spectrum(node, false);
                    std::copy(eigen.begin(), eigen.end(), r0 + (vb + node) * width);
                }
            }
            const auto eigen = spectrum(0, true);
            std::copy(eigen.begin(), eigen.end(), rf + g * width);
            TORCH_CHECK(stats[3 * g + 1] == eend - eb - n + stats[3 * g], "Euler cycle rank mismatch");
        }
    });
    return {death, first, cycle, rho0, rho1, final_rho, counts};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("event_spectra", &event_spectra, py::call_guard<py::gil_scoped_release>());
}
