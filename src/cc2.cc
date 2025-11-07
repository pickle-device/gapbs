// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifdef _OPENMP
  #define _GLIBCXX_PARALLEL
#endif

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <iostream>
#include <unordered_map>
#include <vector>

#if ENABLE_PICKLEDEVICE==1
#pragma message("Compiling with Pickle device")
#include "pickle_graph.h"
#else
#pragma message("NOT compiling with Pickle device")
//#include "pickle_device_manager.h"
#include "graph.h"
#include "pvector.h"
#include "sliding_queue.h"
#endif

#include "benchmark.h"
#include "bitmap.h"
#include "builder.h"
#include "command_line.h"

#if ENABLE_GEM5==1
#pragma message("Compiling with gem5 instructions")
#include <gem5/m5ops.h>
#include "m5_mmap.h"
#endif // ENABLE_GEM5

#if ENABLE_PICKLEDEVICE==1
std::unique_ptr<PickleDeviceManager> pdev(new PickleDeviceManager());
#endif

/*
GAP Benchmark Suite
Kernel: Connected Components (CC)
Author: Scott Beamer

Will return comp array labelling each vertex with a connected component ID

This CC implementation makes use of the Shiloach-Vishkin [2] algorithm with
implementation optimizations from Bader et al. [1]. Michael Sutton contributed
a fix for directed graphs using the min-max swap from [3], and it also produces
more consistent performance for undirected graphs.

[1] David A Bader, Guojing Cong, and John Feo. "On the architectural
    requirements for efficient execution of graph algorithms." International
    Conference on Parallel Processing, Jul 2005.

[2] Yossi Shiloach and Uzi Vishkin. "An o(logn) parallel connectivity algorithm"
    Journal of Algorithms, 3(1):57–67, 1982.

[3] Kishore Kothapalli, Jyothish Soman, and P. J. Narayanan. "Fast GPU
    algorithms for graph connectivity." Workshop on Large Scale Parallel
    Processing, 2010.
*/


using namespace std;

uint64_t* UCPage = NULL;
uint64_t* PerfPage = NULL;

const uint64_t PERF_THREAD_START = 0;
const uint64_t PERF_THREAD_COMPLETE = 1;

// The maximum number of iterations of the Shiloach-Vishkin algorithm.
// This is to constraint simulation time.
const int MAX_SV_NUM_ITERS = 1;

// The hooking condition (comp_u < comp_v) may not coincide with the edge's
// direction, so we use a min-max swap such that lower component IDs propagate
// independent of the edge's direction.

void ShiloachVishkin(const Graph &g, const int max_iters, pvector<NodeID>& comp) {
  #pragma omp parallel for
  for (NodeID n=0; n < g.num_nodes(); n++) {
    comp[n] = n;
  }
  bool change = true;
  int num_iter = 0;
  while (change && (num_iter < max_iters)) {
    change = false;
    num_iter++;
    #pragma omp parallel
    {
#if ENABLE_PICKLEDEVICE==1
      const uint64_t thread_id = (uint64_t)omp_get_thread_num();
      *PerfPage = (thread_id << 1) | PERF_THREAD_START;
#endif
      #pragma omp for
      for (NodeID u=0; u < g.num_nodes(); u++) {
        for (NodeID v : g.out_neigh(u)) {
          NodeID comp_u = comp[u];
          NodeID comp_v = comp[v];
          if (comp_u == comp_v) continue;
          // Hooking condition so lower component ID wins independent of direction
          NodeID high_comp = comp_u > comp_v ? comp_u : comp_v;
          NodeID low_comp = comp_u + (comp_v - high_comp);
          if (high_comp == comp[high_comp]) {
            change = true;
            comp[high_comp] = low_comp;
          }
        }
      }
#if ENABLE_PICKLEDEVICE==1
      *PerfPage = (thread_id << 1) | PERF_THREAD_COMPLETE;
#endif
    }
    #pragma omp parallel for
    for (NodeID n=0; n < g.num_nodes(); n++) {
      while (comp[n] != comp[comp[n]]) {
        comp[n] = comp[comp[n]];
      }
    }
  }
  cout << "Shiloach-Vishkin took " << num_iter << " iterations" << endl;
}

void ShiloachVishkinWithPrefetch(const Graph &g, const int max_iters, pvector<NodeID>& comp) {
  #pragma omp parallel for
  for (NodeID n=0; n < g.num_nodes(); n++) {
    comp[n] = n;
  }
  bool change = true;
  int num_iter = 0;
  while (change && (num_iter < max_iters)) {
    change = false;
    num_iter++;
    #pragma omp parallel
    {
#if ENABLE_PICKLEDEVICE==1
      const uint64_t thread_id = (uint64_t)omp_get_thread_num();
      *PerfPage = (thread_id << 1) | PERF_THREAD_START;
#endif
      #pragma omp for
      for (NodeID u=0; u < g.num_nodes(); u++) {
#if ENABLE_PICKLEDEVICE==1
        *UCPage = (uint64_t)(u);
#endif
        for (NodeID v : g.out_neigh(u)) {
          NodeID comp_u = comp[u];
          NodeID comp_v = comp[v];
          if (comp_u == comp_v) continue;
          // Hooking condition so lower component ID wins independent of direction
          NodeID high_comp = comp_u > comp_v ? comp_u : comp_v;
          NodeID low_comp = comp_u + (comp_v - high_comp);
          if (high_comp == comp[high_comp]) {
            change = true;
            comp[high_comp] = low_comp;
          }
        }
      }
#if ENABLE_PICKLEDEVICE==1
    *PerfPage = (thread_id << 1) | PERF_THREAD_COMPLETE;
#endif
    }
    #pragma omp parallel for
    for (NodeID n=0; n < g.num_nodes(); n++) {
      while (comp[n] != comp[comp[n]]) {
        comp[n] = comp[comp[n]];
      }
    }
  }
  cout << "Shiloach-Vishkin took " << num_iter << " iterations" << endl;
}


void PrintCompStats(const Graph &g, const pvector<NodeID> &comp) {
  cout << endl;
  unordered_map<NodeID, NodeID> count;
  for (NodeID comp_i : comp)
    count[comp_i] += 1;
  int k = 5;
  vector<pair<NodeID, NodeID>> count_vector;
  count_vector.reserve(count.size());
  for (auto kvp : count)
    count_vector.push_back(kvp);
  vector<pair<NodeID, NodeID>> top_k = TopK(count_vector, k);
  k = min(k, static_cast<int>(top_k.size()));
  cout << k << " biggest clusters" << endl;
  for (auto kvp : top_k)
    cout << kvp.second << ":" << kvp.first << endl;
  cout << "There are " << count.size() << " components" << endl;
}


// Verifies CC result by performing a BFS from a vertex in each component
// - Asserts search does not reach a vertex with a different component label
// - If the graph is directed, it performs the search as if it was undirected
// - Asserts every vertex is visited (degree-0 vertex should have own label)
bool CCVerifier(const Graph &g, const pvector<NodeID> &comp) {
  unordered_map<NodeID, NodeID> label_to_source;
  for (NodeID n : g.vertices())
    label_to_source[comp[n]] = n;
  Bitmap visited(g.num_nodes());
  visited.reset();
  vector<NodeID> frontier;
  frontier.reserve(g.num_nodes());
  for (auto label_source_pair : label_to_source) {
    NodeID curr_label = label_source_pair.first;
    NodeID source = label_source_pair.second;
    frontier.clear();
    frontier.push_back(source);
    visited.set_bit(source);
    for (auto it = frontier.begin(); it != frontier.end(); it++) {
      NodeID u = *it;
      for (NodeID v : g.out_neigh(u)) {
        if (comp[v] != curr_label)
          return false;
        if (!visited.get_bit(v)) {
          visited.set_bit(v);
          frontier.push_back(v);
        }
      }
      if (g.directed()) {
        for (NodeID v : g.in_neigh(u)) {
          if (comp[v] != curr_label)
            return false;
          if (!visited.get_bit(v)) {
            visited.set_bit(v);
            frontier.push_back(v);
          }
        }
      }
    }
  }
  for (NodeID n=0; n < g.num_nodes(); n++)
    if (!visited.get_bit(n))
      return false;
  return true;
}

pvector<NodeID> DoCC(const Graph &g, int trial_num) {
  pvector<NodeID> result(g.num_nodes());
  if (trial_num == 0) { // ----- First trial: warm up phase -----
    std::cout << "ROI Start" << std::endl; // ----- ROI Start -----
#if ENABLE_GEM5==1
    m5_exit_addr(0); // exit 1
#endif // ENABLE_GEM5

#if ENABLE_PICKLEDEVICE==1
    // Set up PerfPage
    PerfPage = (uint64_t*) pdev->getPerfPagePtr();
    std::cout << "PerfPage: 0x" << std::hex << (uint64_t)PerfPage << std::dec << std::endl;
    assert(PerfPage != nullptr);
#endif
    ShiloachVishkin(g, MAX_SV_NUM_ITERS, result);

#if ENABLE_GEM5==1
    m5_exit_addr(0); // exit 2
#endif // ENABLE_GEM5
    std::cout << "ROI End" << std::endl; // ----- ROI End -----
  } else if (trial_num == 1) { // ----- Second trial: measured phase -----
    // Read device specs
    uint64_t use_pdev = 0;
    uint64_t prefetch_distance = 0;
#if ENABLE_PICKLEDEVICE==1
    PickleDevicePrefetcherSpecs specs = pdev->getDevicePrefetcherSpecs();
    use_pdev = specs.availability;
    prefetch_distance = specs.prefetch_distance;
#endif
    std::cout << "Use pdev: " << use_pdev << "; Prefetch distance: " << prefetch_distance << std::endl;

    // Set up pickle job
#if ENABLE_PICKLEDEVICE==1
    if (use_pdev == 1) {
        PickleJob job(/*kernel_name*/"cc_kernel");
        // We get the array descriptors from the graph. Note that the relation between the arrays here
        // is already set up by the graph's constructor.
        std::shared_ptr<PickleArrayDescriptor> out_index_array_descriptor = g.getOutIndexArrayDescriptor();
        out_index_array_descriptor->access_type = AccessType::Ranged;
        out_index_array_descriptor->addressing_mode = AddressingMode::Pointer;
        job.addArrayDescriptor(out_index_array_descriptor);
        std::shared_ptr<PickleArrayDescriptor> out_neighbors_array_descriptor = g.getOutNeighborsArrayDescriptor();
        out_neighbors_array_descriptor->access_type = AccessType::SingleElement;
        out_neighbors_array_descriptor->addressing_mode = AddressingMode::Index;
        job.addArrayDescriptor(out_neighbors_array_descriptor);
        // We add the result array descriptor
        auto result_array_descriptor = result.getArrayDescriptor();
        result_array_descriptor->access_type = AccessType::SingleElement;
        result_array_descriptor->addressing_mode = AddressingMode::Index;
        out_neighbors_array_descriptor->dst_indexing_array_id = result_array_descriptor->getArrayId();
        job.addArrayDescriptor(result_array_descriptor);
        // Done
        job.print();
        std::cout << "Sent job" << std::endl;
        pdev->sendJob(job);

        UCPage = (uint64_t*) pdev->getUCPagePtr(0);
        std::cout << "UCPage: 0x" << std::hex << (uint64_t)UCPage << std::dec << std::endl;
        assert(UCPage != nullptr);
    }
#endif

    std::cout << "ROI Start" << std::endl; // ----- ROI Start -----
#if ENABLE_GEM5==1
    m5_exit_addr(0); // exit 3
#endif // ENABLE_GEM5
    if (use_pdev == 1) {
        ShiloachVishkinWithPrefetch(g, MAX_SV_NUM_ITERS, result);
    } else {
        ShiloachVishkin(g, MAX_SV_NUM_ITERS, result);
    }
#if ENABLE_GEM5==1
    m5_exit_addr(0); // exit 4
#endif // ENABLE_GEM5
    std::cout << "ROI End" << std::endl; // ----- ROI End -----
  }
  return result;
}

int main(int argc, char* argv[]) {
  CLApp cli(argc, argv, "connected-components");
  if (!cli.ParseArgs())
    return -1;
  Builder b(cli);
  Graph g = b.MakeGraph();
#if ENABLE_GEM5==1
  map_m5_mem();
#endif // ENABLE_GEM5
  BenchmarkKernel(cli, g, DoCC, PrintCompStats, CCVerifier);
#if ENABLE_GEM5==1
  //unmap_m5_mem();
#endif // ENABLE_GEM5
  return 0;
}

