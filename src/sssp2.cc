// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <cassert>
#include <cinttypes>
#include <limits>
#include <iostream>
#include <queue>
#include <vector>

#include <omp.h>

#if ENABLE_PICKLEDEVICE==1
#pragma message("Compiling with Pickle device")
#include "pickle_graph.h"
#else
#pragma message("NOT compiling with Pickle device")
#include "graph.h"
#include "pvector.h"
#endif

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "platform_atomics.h"
#include "timer.h"

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
Kernel: Single-source Shortest Paths (SSSP)
Author: Scott Beamer, Yunming Zhang

Returns array of distances for all vertices from given source vertex

This SSSP implementation makes use of the ∆-stepping algorithm [1]. The type
used for weights and distances (WeightT) is typedefined in benchmark.h. The
delta parameter (-d) should be set for each input graph. This implementation
incorporates a new bucket fusion optimization [2] that significantly reduces
the number of iterations (& barriers) needed.

The bins of width delta are actually all thread-local and of type std::vector,
so they can grow but are otherwise capacity-proportional. Each iteration is
done in two phases separated by barriers. In the first phase, the current
shared bin is processed by all threads. As they find vertices whose distance
they are able to improve, they add them to their thread-local bins. During this
phase, each thread also votes on what the next bin should be (smallest
non-empty bin). In the next phase, each thread copies its selected
thread-local bin into the shared bin.

Once a vertex is added to a bin, it is not removed, even if its distance is
later updated and, it now appears in a lower bin. We find ignoring vertices if
their distance is less than the min distance for the current bin removes
enough redundant work to be faster than removing the vertex from older bins.

The bucket fusion optimization [2] executes the next thread-local bin in
the same iteration if the vertices in the next thread-local bin have the
same priority as those in the current shared bin. This optimization greatly
reduces the number of iterations needed without violating the priority-based
execution order, leading to significant speedup on large diameter road networks.

[1] Ulrich Meyer and Peter Sanders. "δ-stepping: a parallelizable shortest path
    algorithm." Journal of Algorithms, 49(1):114–152, 2003.

[2] Yunming Zhang, Ajay Brahmakshatriya, Xinyi Chen, Laxman Dhulipala,
    Shoaib Kamil, Saman Amarasinghe, and Julian Shun. "Optimizing ordered graph
    algorithms with GraphIt." The 18th International Symposium on Code Generation
    and Optimization (CGO), pages 158-170, 2020.
*/


using namespace std;

uint64_t* UCPage = NULL;
uint64_t* UCPage_Kernel1 = NULL; // Prefetch RelaxEdges with checking threshold
uint64_t* UCPage_Kernel2 = NULL; // Prefetch RelaxEdges without checking threshold
uint64_t* UCPage_Kernel3 = NULL; // Tell the prefetcher about threshold
uint64_t* PerfPage = NULL;

const uint64_t PERF_THREAD_START = 0;
const uint64_t PERF_THREAD_COMPLETE = 1;

const WeightT kDistInf = numeric_limits<WeightT>::max()/2;
const size_t kMaxBin = numeric_limits<size_t>::max()/2;
const size_t kBinSizeThreshold = 1000;

inline
void RelaxEdges(const WGraph &g, NodeID u, WeightT delta,
                pvector<WeightT> &dist, vector <vector<NodeID>> &local_bins) {
  for (WNode wn : g.out_neigh(u)) {
    WeightT old_dist = dist[wn.v];
    WeightT new_dist = dist[u] + wn.w;
    while (new_dist < old_dist) {
      if (compare_and_swap(dist[wn.v], old_dist, new_dist)) {
        size_t dest_bin = new_dist/delta;
        if (dest_bin >= local_bins.size())
          local_bins.resize(dest_bin+1);
        local_bins[dest_bin].push_back(wn.v);
        break;
      }
      old_dist = dist[wn.v];      // swap failed, recheck dist update & retry
    }
  }
}

void DeltaStepWithPrefetch(
  const WGraph &g, NodeID source, WeightT delta,
  pvector<WeightT>& dist, pvector<NodeID>& frontier,
  const uint64_t prefetch_distance,
  bool logging_enabled = false
) {
  dist[source] = 0;
  // two element arrays for double buffering curr=iter&1, next=(iter+1)&1
  size_t shared_indexes[2] = {0, kMaxBin};
  size_t frontier_tails[2] = {1, 0};
  frontier[0] = source;
#if ENABLE_GEM5==1
  m5_exit_addr(0); // exit 3 (for trial 1)
#endif // ENABLE_GEM5
  #pragma omp parallel
  {
#if ENABLE_PICKLEDEVICE==1
    const uint64_t thread_id = (uint64_t)omp_get_thread_num();
    *PerfPage = (thread_id << 1) | PERF_THREAD_START;
#endif
    vector<vector<NodeID> > local_bins(0);
    size_t iter = 0;
    while (shared_indexes[iter&1] != kMaxBin) {
      size_t &curr_bin_index = shared_indexes[iter&1];
      size_t &next_bin_index = shared_indexes[(iter+1)&1];
      size_t &curr_frontier_tail = frontier_tails[iter&1];
      size_t &next_frontier_tail = frontier_tails[(iter+1)&1];
      // New threshold for the new iter
      const WeightT threshold = delta * static_cast<WeightT>(curr_bin_index);
#if ENABLE_PICKLEDEVICE==1
      *UCPage_Kernel3 = static_cast<uint64_t>(threshold);
#endif
      //#pragma omp for nowait schedule(dynamic, 64)
      // Let's use static scheduling for now.
      // Benchmarking shows this make no difference performance-wise.
      #pragma omp for schedule(static) nowait
      for (size_t i=0; i < curr_frontier_tail; i++) {
        NodeID u = frontier[i];
#if ENABLE_PICKLEDEVICE==1
        if (i + prefetch_distance < curr_frontier_tail) {
            *UCPage_Kernel1 = (uint64_t)(&(frontier[i+prefetch_distance]));
        }
#endif
        if (dist[u] >= threshold)
          RelaxEdges(g, u, delta, dist, local_bins);
      }
      while (curr_bin_index < local_bins.size() &&
             !local_bins[curr_bin_index].empty() &&
             local_bins[curr_bin_index].size() < kBinSizeThreshold) {
        vector<NodeID> curr_bin_copy = local_bins[curr_bin_index];
        local_bins[curr_bin_index].resize(0);
        //for (NodeID u : curr_bin_copy)
        //  RelaxEdges(g, u, delta, dist, local_bins);
        for (auto u_iter = curr_bin_copy.begin(); u_iter < curr_bin_copy.end(); u_iter++) {
#if ENABLE_PICKLEDEVICE==1
          auto prefetch_iter = u_iter + prefetch_distance;
          if (prefetch_iter < curr_bin_copy.end()) {
            *UCPage_Kernel2 = (uint64_t)(&(*prefetch_iter));
          }
#endif
          RelaxEdges(g, *u_iter, delta, dist, local_bins);
        }
      }
      for (size_t i=curr_bin_index; i < local_bins.size(); i++) {
        if (!local_bins[i].empty()) {
          #pragma omp critical
          next_bin_index = min(next_bin_index, i);
          break;
        }
      }
      #pragma omp barrier
      #pragma omp single nowait
      {
        curr_bin_index = kMaxBin;
        curr_frontier_tail = 0;
      }
      if (next_bin_index < local_bins.size()) {
        size_t copy_start = fetch_and_add(next_frontier_tail,
                                          local_bins[next_bin_index].size());
        copy(local_bins[next_bin_index].begin(),
             local_bins[next_bin_index].end(), frontier.data() + copy_start);
        local_bins[next_bin_index].resize(0);
      }
      iter++;
      #pragma omp barrier
    }
    #pragma omp single
    {
      cout << "took " << iter << " iterations" << endl;
    }
#if ENABLE_PICKLEDEVICE==1
    *PerfPage = (thread_id << 1) | PERF_THREAD_START;
#endif
  }
#if ENABLE_GEM5==1
  m5_exit_addr(0); // exit 4 (for trial 1)
#endif // ENABLE_GEM5
}

void DeltaStep(
  const WGraph &g, NodeID source, WeightT delta,
  pvector<WeightT>& dist, pvector<NodeID>& frontier,
  bool logging_enabled = false
) {
  dist[source] = 0;
  // two element arrays for double buffering curr=iter&1, next=(iter+1)&1
  size_t shared_indexes[2] = {0, kMaxBin};
  size_t frontier_tails[2] = {1, 0};
  frontier[0] = source;

#if ENABLE_GEM5==1
  m5_exit_addr(0); // exit 1 (for trial 0) or exit 3 (for trial 1)
#endif // ENABLE_GEM5

#if ENABLE_PICKLEDEVICE==1
  // Turn on the performance watcher
  PerfPage = (uint64_t*) pdev->getPerfPagePtr();
  std::cout << "PerfPage: 0x" << std::hex << (uint64_t)PerfPage << std::dec << std::endl;
  assert(PerfPage != nullptr);
#endif
  #pragma omp parallel
  {
#if ENABLE_PICKLEDEVICE==1
    const uint64_t thread_id = (uint64_t)omp_get_thread_num();
    *PerfPage = (thread_id << 1) | PERF_THREAD_START;
#endif
    vector<vector<NodeID> > local_bins(0);
    size_t iter = 0;
    while (shared_indexes[iter&1] != kMaxBin) {
      size_t &curr_bin_index = shared_indexes[iter&1];
      size_t &next_bin_index = shared_indexes[(iter+1)&1];
      size_t &curr_frontier_tail = frontier_tails[iter&1];
      size_t &next_frontier_tail = frontier_tails[(iter+1)&1];
      const WeightT threshold = delta * static_cast<WeightT>(curr_bin_index);
      //#pragma omp for nowait schedule(dynamic, 64)
      // Let's use static scheduling for now.
      // Benchmarking shows this make no difference performance-wise.
      #pragma omp for schedule(static) nowait
      for (size_t i=0; i < curr_frontier_tail; i++) {
        NodeID u = frontier[i];
        if (dist[u] >= threshold)
          RelaxEdges(g, u, delta, dist, local_bins);
      }
      while (curr_bin_index < local_bins.size() &&
             !local_bins[curr_bin_index].empty() &&
             local_bins[curr_bin_index].size() < kBinSizeThreshold) {
        vector<NodeID> curr_bin_copy = local_bins[curr_bin_index];
        local_bins[curr_bin_index].resize(0);
        for (NodeID u : curr_bin_copy)
          RelaxEdges(g, u, delta, dist, local_bins);
      }
      for (size_t i=curr_bin_index; i < local_bins.size(); i++) {
        if (!local_bins[i].empty()) {
          #pragma omp critical
          next_bin_index = min(next_bin_index, i);
          break;
        }
      }
      #pragma omp barrier
      #pragma omp single nowait
      {
        curr_bin_index = kMaxBin;
        curr_frontier_tail = 0;
      }
      if (next_bin_index < local_bins.size()) {
        size_t copy_start = fetch_and_add(next_frontier_tail,
                                          local_bins[next_bin_index].size());
        copy(local_bins[next_bin_index].begin(),
             local_bins[next_bin_index].end(), frontier.data() + copy_start);
        local_bins[next_bin_index].resize(0);
      }
      iter++;
      #pragma omp barrier
    }
    #pragma omp single
    {
      cout << "took " << iter << " iterations" << endl;
    }
#if ENABLE_PICKLEDEVICE==1
    *PerfPage = (thread_id << 1) | PERF_THREAD_START;
#endif
  }
#if ENABLE_GEM5==1
  m5_exit_addr(0); // exit 2 (for trial 0) or exit 4 (for trial 1)
#endif // ENABLE_GEM5
}


pvector<WeightT> DoSSSP(
  const WGraph &g, NodeID source, WeightT delta, int trial_num,
  bool logging_enabled = false
) {
  pvector<WeightT> dist(g.num_nodes(), kDistInf);
  pvector<NodeID> frontier(g.num_edges_directed());

  if (trial_num == 0) {
    dist[source] = 0;
    DeltaStep(g, source, delta, dist, frontier);
  } else if (trial_num == 1) {
    dist.fill(kDistInf);
    dist[source] = 0;

    uint64_t use_pdev = 0;
#if ENABLE_PICKLEDEVICE==1
    uint64_t prefetch_distance = 0;
    PrefetchMode prefetch_mode = PrefetchMode::UNKNOWN;
    uint64_t bulk_mode_chunk_size = 0;
#endif

#if ENABLE_PICKLEDEVICE==1
    // Only use pdev in the second trial
    PickleDevicePrefetcherSpecs specs = pdev->getDevicePrefetcherSpecs();
    use_pdev = specs.availability;
    prefetch_distance = specs.prefetch_distance;
    prefetch_mode = specs.prefetch_mode;
    bulk_mode_chunk_size = specs.bulk_mode_chunk_size;
    std::cout << "Device specs: " << std::endl;
    std::cout << "  . Use pdev: " << use_pdev << std::endl;
    std::cout << "  . Prefetch distance: " << prefetch_distance << std::endl;
    std::cout << "  . Prefetch mode (0: unknown, 1: single, 2: bulk): " << prefetch_mode << std::endl;
    std::cout << "  . Chunk size (should be non-zero in bulk mode): " << bulk_mode_chunk_size << std::endl;
#endif

    if (use_pdev == 1) {
#if ENABLE_PICKLEDEVICE==1
      // Send prefech job descriptions
      // Job 1 (Kernel 1: prefetch if dist[u] meets threshold)
      {
        PickleJob job(/*kernel_name*/"sssp_kernel_1");
        std::shared_ptr<PickleArrayDescriptor> frontier_array_descriptor = frontier.getArrayDescriptor();
        frontier_array_descriptor->name = "frontier";
        frontier_array_descriptor->access_type = AccessType::SingleElement;
        frontier_array_descriptor->addressing_mode = AddressingMode::Index;
        job.addArrayDescriptor(frontier_array_descriptor);
        // We get the array descriptors from the graph. Note that the relation between the arrays here
        // is already set up by the graph's constructor.
        std::shared_ptr<PickleArrayDescriptor> out_index_array_descriptor = g.getOutIndexArrayDescriptor();
        out_index_array_descriptor->name = "out_index";
        out_index_array_descriptor->access_type = AccessType::Ranged;
        out_index_array_descriptor->addressing_mode = AddressingMode::Pointer;
        frontier_array_descriptor->dst_indexing_array_id = out_index_array_descriptor->getArrayId();
        job.addArrayDescriptor(out_index_array_descriptor);
        std::shared_ptr<PickleArrayDescriptor> out_neighbors_array_descriptor = g.getOutNeighborsArrayDescriptor();
        out_neighbors_array_descriptor->name = "out_neighbors";
        out_neighbors_array_descriptor->access_type = AccessType::SingleElement;
        out_neighbors_array_descriptor->addressing_mode = AddressingMode::Index;
        job.addArrayDescriptor(out_neighbors_array_descriptor);
        // Add the `dist` array descriptor
        auto dist_array_descriptor = dist.getArrayDescriptor();
        dist_array_descriptor->name = "dist";
        dist_array_descriptor->access_type = AccessType::SingleElement;
        dist_array_descriptor->addressing_mode = AddressingMode::Index;
        out_neighbors_array_descriptor->dst_indexing_array_id = dist_array_descriptor->getArrayId();
        job.addArrayDescriptor(dist_array_descriptor);
        // Send the job
        job.print();
        pdev->sendJob(job);
        std::cout << "Sent kernel_1" << std::endl;
      }
      // Job 2 (Kernel 2: Prefetch dist[v] for all v in the neighbors of u)
      {
        PickleJob job(/*kernel_name*/"sssp_kernel_2");
        // The curr_bin_copy array is a local array that is created and deleted multiple times over the course
        // the algorithm runtime.
        // However, we don't really need to know the base address, the array size, or the element size of the
        // array as we don't need to index to this array. So we'll just place a dummy value for all fields.
        std::shared_ptr<PickleArrayDescriptor> curr_bin_copy_array_descriptor(new PickleArrayDescriptor());
        curr_bin_copy_array_descriptor->name = "curr_bin_copy";
        curr_bin_copy_array_descriptor->vaddr_start = 0;
        curr_bin_copy_array_descriptor->vaddr_end = 0;
        curr_bin_copy_array_descriptor->element_size = 0;
        curr_bin_copy_array_descriptor->access_type = AccessType::SingleElement;
        curr_bin_copy_array_descriptor->addressing_mode = AddressingMode::Index;
        job.addArrayDescriptor(curr_bin_copy_array_descriptor);
        // We get the array descriptors from the graph. Note that the relation between the arrays here
        // is already set up by the graph's constructor.
        std::shared_ptr<PickleArrayDescriptor> out_index_array_descriptor = g.getOutIndexArrayDescriptor();
        out_index_array_descriptor->name = "out_index";
        out_index_array_descriptor->access_type = AccessType::Ranged;
        out_index_array_descriptor->addressing_mode = AddressingMode::Pointer;
        curr_bin_copy_array_descriptor->dst_indexing_array_id = out_index_array_descriptor->getArrayId();
        job.addArrayDescriptor(out_index_array_descriptor);
        std::shared_ptr<PickleArrayDescriptor> out_neighbors_array_descriptor = g.getOutNeighborsArrayDescriptor();
        out_neighbors_array_descriptor->name = "out_neighbors";
        out_neighbors_array_descriptor->access_type = AccessType::SingleElement;
        out_neighbors_array_descriptor->addressing_mode = AddressingMode::Index;
        job.addArrayDescriptor(out_neighbors_array_descriptor);
         // Add the `dist` array descriptor
        auto dist_array_descriptor = dist.getArrayDescriptor();
        dist_array_descriptor->name = "dist";
        dist_array_descriptor->access_type = AccessType::SingleElement;
        dist_array_descriptor->addressing_mode = AddressingMode::Index;
        out_neighbors_array_descriptor->dst_indexing_array_id = dist_array_descriptor->getArrayId();
        job.addArrayDescriptor(dist_array_descriptor);
        // Send the job
        job.print();
        pdev->sendJob(job);
        std::cout << "Sent kernel_2" << std::endl;
      }
      // Job 3 (Kernel 3: A dummy kernel)
      {
        PickleJob job(/*kernel_name*/"sssp_kernel_3");
        // This is a kernel that keeps the value of the `threshold` variable sent by the program at runtime.
        job.print();
        pdev->sendJob(job);
        std::cout << "Sent kernel_3" << std::endl;
      }
      // Create communication page
      UCPage = (uint64_t*) pdev->getUCPagePtr(0);
      UCPage_Kernel1 = UCPage;
      UCPage_Kernel2 = UCPage + 1;
      UCPage_Kernel3 = UCPage + 2;
      std::cout << "UCPage_Kernel1: 0x" << std::hex << (uint64_t)UCPage_Kernel1 << std::dec << std::endl;
      std::cout << "UCPage_Kernel2: 0x" << std::hex << (uint64_t)UCPage_Kernel2 << std::dec << std::endl;
      std::cout << "UCPage_Kernel3: 0x" << std::hex << (uint64_t)UCPage_Kernel3 << std::dec << std::endl;
      assert(UCPage != nullptr);
#endif
      DeltaStepWithPrefetch(
        g, source, delta, dist, frontier, prefetch_distance
      );
    } else {
      DeltaStep(g, source, delta, dist, frontier);
    }
  }
  return dist;
}


void PrintSSSPStats(const WGraph &g, const pvector<WeightT> &dist) {
  auto NotInf = [](WeightT d) { return d != kDistInf; };
  int64_t num_reached = count_if(dist.begin(), dist.end(), NotInf);
  cout << "SSSP Tree reaches " << num_reached << " nodes" << endl;
}


// Compares against simple serial implementation
bool SSSPVerifier(const WGraph &g, NodeID source,
                  const pvector<WeightT> &dist_to_test) {
  // Serial Dijkstra implementation to get oracle distances
  pvector<WeightT> oracle_dist(g.num_nodes(), kDistInf);
  oracle_dist[source] = 0;
  typedef pair<WeightT, NodeID> WN;
  priority_queue<WN, vector<WN>, greater<WN>> mq;
  mq.push(make_pair(0, source));
  while (!mq.empty()) {
    WeightT td = mq.top().first;
    NodeID u = mq.top().second;
    mq.pop();
    if (td == oracle_dist[u]) {
      for (WNode wn : g.out_neigh(u)) {
        if (td + wn.w < oracle_dist[wn.v]) {
          oracle_dist[wn.v] = td + wn.w;
          mq.push(make_pair(td + wn.w, wn.v));
        }
      }
    }
  }
  // Report any mismatches
  bool all_ok = true;
  for (NodeID n : g.vertices()) {
    if (dist_to_test[n] != oracle_dist[n]) {
      cout << n << ": " << dist_to_test[n] << " != " << oracle_dist[n] << endl;
      all_ok = false;
    }
  }
  return all_ok;
}


int main(int argc, char* argv[]) {
  CLDelta<WeightT> cli(argc, argv, "single-source shortest-path");
  if (!cli.ParseArgs())
    return -1;
  WeightedBuilder b(cli);
  WGraph g = b.MakeGraph();
  SourcePicker<WGraph> sp(g, cli.start_vertex());
  auto SSSPBound = [&sp, &cli] (const WGraph &g, int trial_num) {
    return DoSSSP(g, sp.PickNext(), cli.delta(), trial_num);
  };
  SourcePicker<WGraph> vsp(g, cli.start_vertex());
  auto VerifierBound = [&vsp] (const WGraph &g, const pvector<WeightT> &dist) {
    return SSSPVerifier(g, vsp.PickNext(), dist);
  };
#if ENABLE_GEM5==1
  map_m5_mem();
#endif // ENABLE_GEM5
  BenchmarkKernel(cli, g, SSSPBound, PrintSSSPStats, VerifierBound);
#if ENABLE_GEM5==1
  //m5_work_end_addr(0, 0);
  //unmap_m5_mem();
#endif // ENABLE_GEM5
  return 0;
}
