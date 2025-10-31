// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

// Encourage use of gcc's parallel algorithms (for sort for relabeling)
#ifdef _OPENMP
  #define _GLIBCXX_PARALLEL
#endif

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <iostream>
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
Kernel: Triangle Counting (TC)
Author: Scott Beamer

Will count the number of triangles (cliques of size 3)

Input graph requirements:
  - undirected
  - has no duplicate edges (or else will be counted as multiple triangles)
  - neighborhoods are sorted by vertex identifiers

Other than symmetrizing, the rest of the requirements are done by SquishCSR
during graph building.

This implementation reduces the search space by counting each triangle only
once. A naive implementation will count the same triangle six times because
each of the three vertices (u, v, w) will count it in both ways. To count
a triangle only once, this implementation only counts a triangle if u > v > w.
Once the remaining unexamined neighbors identifiers get too big, it can break
out of the loop, but this requires that the neighbors are sorted.

This implementation relabels the vertices by degree. This optimization is
beneficial if the average degree is sufficiently high and if the degree
distribution is sufficiently non-uniform. To decide whether to relabel the
graph, we use the heuristic in WorthRelabelling.
*/


using namespace std;

uint64_t* UCPage = NULL;
uint64_t* PerfPage = NULL;

const uint64_t PERF_THREAD_START = 0;
const uint64_t PERF_THREAD_COMPLETE = 1;

size_t OrderedCount(const Graph &g) {
  size_t total = 0;
  #pragma omp parallel
  {
    const uint64_t thread_id = (uint64_t)omp_get_thread_num();
    *PerfPage = (thread_id << 1) | PERF_THREAD_START;
    //#pragma omp parallel for reduction(+ : total) schedule(dynamic, 64)
    #pragma omp parallel for reduction(+ : total) schedule(dynamic, 16384)
    for (NodeID u=0; u < g.num_nodes(); u++) {
      for (NodeID v : g.out_neigh(u)) {
        if (v > u)
          break;
        auto it = g.out_neigh(v).begin();
        for (NodeID w : g.out_neigh(u)) {
          if (w > v)
            break;
          while (*it < w)
            it++;
          if (w == *it)
            total++;
        }
      }
    }
    PerfPage[thread_id] = (thread_id << 1) | PERF_THREAD_COMPLETE;
  }
  return total;
}

size_t OrderedCountWithPrefetch(const Graph &g) {
  size_t total = 0;
  #pragma omp parallel
  {
    const uint64_t thread_id = (uint64_t)omp_get_thread_num();
    *PerfPage = (thread_id << 1) | PERF_THREAD_START;
    //#pragma omp parallel for reduction(+ : total) schedule(dynamic, 64)
    #pragma omp parallel for reduction(+ : total) schedule(dynamic, 16384)
    for (NodeID u=0; u < g.num_nodes(); u++) {
      *UCPage = (uint64_t)(u);
      for (NodeID v : g.out_neigh(u)) {
        if (v > u)
          break;
        auto it = g.out_neigh(v).begin();
        for (NodeID w : g.out_neigh(u)) {
          if (w > v)
            break;
          while (*it < w)
            it++;
          if (w == *it)
            total++;
        }
      }
    }
    PerfPage[thread_id] = (thread_id << 1) | PERF_THREAD_COMPLETE;
  }
  return total;
}


// Heuristic to see if sufficiently dense power-law graph
bool WorthRelabelling(const Graph &g) {
  int64_t average_degree = g.num_edges() / g.num_nodes();
  if (average_degree < 10)
    return false;
  SourcePicker<Graph> sp(g);
  int64_t num_samples = min(int64_t(1000), g.num_nodes());
  int64_t sample_total = 0;
  pvector<int64_t> samples(num_samples);
  for (int64_t trial=0; trial < num_samples; trial++) {
    samples[trial] = g.out_degree(sp.PickNext());
    sample_total += samples[trial];
  }
  sort(samples.begin(), samples.end());
  double sample_average = static_cast<double>(sample_total) / num_samples;
  double sample_median = samples[num_samples/2];
  return sample_average / 1.3 > sample_median;
}


// Uses heuristic to see if worth relabeling
size_t DoTC(const Graph &g, int iter_num) {
  size_t result = 0;
  if (iter_num == 0) { // ----- First iteration: warm up phase -----
#if ENABLE_GEM5==1
    m5_exit_addr(0); // exit 1
#endif // ENABLE_GEM5
    std::cout << "ROI Start" << std::endl;

    // Set up PerfPage
    PerfPage = (uint64_t*) pdev->getPerfPagePtr();
    std::cout << "PerfPage: 0x" << std::hex << (uint64_t)PerfPage << std::dec << std::endl;
    assert(PerfPage != nullptr);

    // Main logic
    result = OrderedCount(g);

#if ENABLE_GEM5==1
    m5_exit_addr(0); // exit 2
#endif // ENABLE_GEM5
    std::cout << "ROI End" << std::endl;
  } else if (iter_num == 1) { // ----- Second iteration: measured phase -----
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
        PickleJob job(/*kernel_name*/"tc_kernel");
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
        job.print();
        std::cout << "Sent job" << std::endl;
        pdev->sendJob(job);

        UCPage = (uint64_t*) pdev->getUCPagePtr(0);
        std::cout << "UCPage: 0x" << std::hex << (uint64_t)UCPage << std::dec << std::endl;
        assert(UCPage != nullptr);
    }
#endif

    std::cout << "ROI Start" << std::endl;
#if ENABLE_GEM5==1
    m5_exit_addr(0); // exit 3
#endif // ENABLE_GEM5
    if (use_pdev == 1) {
      result = OrderedCountWithPrefetch(g);
    } else {
      result = OrderedCount(g);
    }
#if ENABLE_GEM5==1
    m5_exit_addr(0); // exit 4
#endif // ENABLE_GEM5
    std::cout << "ROI End" << std::endl;
  }
  return result;
}


void PrintTriangleStats(const Graph &g, size_t total_triangles) {
  cout << total_triangles << " triangles" << endl;
}


// Compares with simple serial implementation that uses std::set_intersection
bool TCVerifier(const Graph &g, size_t test_total) {
  size_t total = 0;
  vector<NodeID> intersection;
  intersection.reserve(g.num_nodes());
  for (NodeID u : g.vertices()) {
    for (NodeID v : g.out_neigh(u)) {
      auto new_end = set_intersection(g.out_neigh(u).begin(),
                                      g.out_neigh(u).end(),
                                      g.out_neigh(v).begin(),
                                      g.out_neigh(v).end(),
                                      intersection.begin());
      intersection.resize(new_end - intersection.begin());
      total += intersection.size();
    }
  }
  total = total / 6;  // each triangle was counted 6 times
  if (total != test_total)
    cout << total << " != " << test_total << endl;
  return total == test_total;
}


int main(int argc, char* argv[]) {
  CLApp cli(argc, argv, "triangle count");
  if (!cli.ParseArgs())
    return -1;
  Builder b(cli);
  Graph g = b.MakeGraph();
  if (g.directed()) {
    cout << "Input graph is directed but tc requires undirected" << endl;
    return -2;
  }
  if (WorthRelabelling(g)) {
    g = Builder::RelabelByDegree(g);
  }
#if ENABLE_GEM5==1
  map_m5_mem();
#endif // ENABLE_GEM5
  BenchmarkKernel(cli, g, DoTC, PrintTriangleStats, TCVerifier);
#if ENABLE_GEM5==1
  //unmap_m5_mem();
#endif // ENABLE_GEM5
  return 0;
}
