// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <atomic>
#include <cassert>
#include <cstdint>
#include <functional>
#include <iostream>
#include <vector>

#include <omp.h>

#if ENABLE_PICKLEDEVICE==1
#pragma message("Compiling with Pickle device")
#include "pickle_graph.h"
#else
#pragma message("NOT compiling with Pickle device")
#include "graph.h"
#include "pvector.h"
#include "sliding_queue.h"
#endif

#include "benchmark.h"
#include "bitmap.h"
#include "builder.h"
#include "command_line.h"
#include "platform_atomics.h"
#include "timer.h"
#include "util.h"

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
Kernel: Betweenness Centrality (BC)
Author: Scott Beamer

Will return array of approx betweenness centrality scores for each vertex

This BC implementation makes use of the Brandes [1] algorithm with
implementation optimizations from Madduri et al. [2]. It is only approximate
because it does not compute the paths from every start vertex, but only a small
subset of them. Additionally, the scores are normalized to the range [0,1].

As an optimization to save memory, this implementation uses a Bitmap to hold
succ (list of successors) found during the BFS phase that are used in the back-
propagation phase.

[1] Ulrik Brandes. "A faster algorithm for betweenness centrality." Journal of
    Mathematical Sociology, 25(2):163–177, 2001.

[2] Kamesh Madduri, David Ediger, Karl Jiang, David A Bader, and Daniel
    Chavarria-Miranda. "A faster parallel algorithm and efficient multithreaded
    implementations for evaluating betweenness centrality on massive datasets."
    International Symposium on Parallel & Distributed Processing (IPDPS), 2009.
*/


using namespace std;
typedef float ScoreT;
typedef double CountT;

uint64_t* UCPage = NULL;
uint64_t* UCPage_Kernel2 = NULL;
uint64_t* UCPage_Kernel3 = NULL;
uint64_t* PerfPage = NULL;

const uint64_t PERF_THREAD_START = 0;
const uint64_t PERF_THREAD_COMPLETE = 1;

//atomic<int> num_nodes_visited = 0;
// The maximum number of iterations of the Brandes algorithm.
// This is to constraint simulation time.
const int MAX_BRANDES_NUM_ITERS = 1;

void PBFS(const Graph &g, NodeID source, pvector<NodeID>& depths,
    pvector<CountT> &path_counts,
    Bitmap &succ, vector<SlidingQueue<NodeID>::iterator> &depth_index,
    SlidingQueue<NodeID> &queue) {
  depths[source] = 0;
  path_counts[source] = 1;
  queue.push_back(source);
  depth_index.push_back(queue.begin());
  queue.slide_window();
  const NodeID* g_out_start = g.out_neigh(0).begin();
  #pragma omp parallel
  {
    NodeID depth = 0;
    QueueBuffer<NodeID> lqueue(queue);
    while (!queue.empty()) {
      depth++;
      //#pragma omp for schedule(dynamic, 64) nowait
      // Let's do static scheduling for now
      #pragma omp for schedule(static) nowait
      for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
        NodeID u = *q_iter;
        for (NodeID &v : g.out_neigh(u)) {
          if ((depths[v] == -1) &&
              (compare_and_swap(depths[v], static_cast<NodeID>(-1), depth))) {
            //num_nodes_visited++;
            lqueue.push_back(v);
          }
          if (depths[v] == depth) {
            succ.set_bit_atomic(&v - g_out_start);
            #pragma omp atomic
            path_counts[v] += path_counts[u];
          }
        }
      }
      lqueue.flush();
      #pragma omp barrier
      #pragma omp single
      {
        depth_index.push_back(queue.begin());
        queue.slide_window();
      }
    }
  }
  depth_index.push_back(queue.begin());
}

void PBFSWithPrefetch(const Graph &g, NodeID source, pvector<NodeID>& depths,
    pvector<CountT> &path_counts,
    Bitmap &succ, vector<SlidingQueue<NodeID>::iterator> &depth_index,
    SlidingQueue<NodeID> &queue, const uint64_t prefetch_distance) {
  depths[source] = 0;
  path_counts[source] = 1;
  queue.push_back(source);
  depth_index.push_back(queue.begin());
  queue.slide_window();
  const NodeID* g_out_start = g.out_neigh(0).begin();
  #pragma omp parallel
  {
#if ENABLE_PICKLEDEVICE==1
    const uint64_t thread_id = (uint64_t)omp_get_thread_num();
    *PerfPage = (thread_id << 1) | PERF_THREAD_START;
#endif
    NodeID depth = 0;
    QueueBuffer<NodeID> lqueue(queue);
    while (!queue.empty()) {
      depth++;
#if ENABLE_PICKLEDEVICE==1
      *UCPage_Kernel3 = (uint64_t)(depth);
#endif
      //#pragma omp for schedule(dynamic, 64) nowait
      // Let's do static scheduling for now
      #pragma omp for schedule(static) nowait
      for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
        NodeID u = *q_iter;
#if ENABLE_PICKLEDEVICE==1
        auto prefetch_iter = q_iter + prefetch_distance;
        if (prefetch_iter < queue.end()) {
          *UCPage = (uint64_t)(&(*q_iter)); // We send the current position of the work_queue
        }
#endif
        for (NodeID &v : g.out_neigh(u)) {
          if ((depths[v] == -1) &&
              (compare_and_swap(depths[v], static_cast<NodeID>(-1), depth))) {
            //num_nodes_visited++;
            lqueue.push_back(v);
          }
          if (depths[v] == depth) {
            succ.set_bit_atomic(&v - g_out_start);
            #pragma omp atomic
            path_counts[v] += path_counts[u];
          }
        }
      }
      lqueue.flush();
      #pragma omp barrier
      #pragma omp single
      {
        depth_index.push_back(queue.begin());
        queue.slide_window();
      }
    }
#if ENABLE_PICKLEDEVICE==1
    *PerfPage = (thread_id << 1) | PERF_THREAD_COMPLETE;
#endif
  }
  depth_index.push_back(queue.begin());
}

pvector<ScoreT> Brandes(const Graph &g, SourcePicker<Graph> &sp,
                        NodeID num_iters, uint64_t trial_num,
                        bool logging_enabled = false) {

  pvector<ScoreT> scores(g.num_nodes(), -1);
  pvector<CountT> path_counts(g.num_nodes());
  Bitmap succ(g.num_edges_directed());
  vector<SlidingQueue<NodeID>::iterator> depth_index;
  SlidingQueue<NodeID> queue(g.num_nodes());
  pvector<NodeID> depths(g.num_nodes(), -1);
  pvector<ScoreT> deltas(g.num_nodes(), 0);

  /* Determine if we use Pickle prefetcher */
#if ENABLE_PICKLEDEVICE==1
  uint64_t use_pdev = 0;
  uint64_t prefetch_distance = 0;
  PrefetchMode prefetch_mode = PrefetchMode::UNKNOWN;
  uint64_t bulk_mode_chunk_size = 0;
#endif

#if ENABLE_PICKLEDEVICE==1
  // Only use pdev in the second trial
  if (trial_num == 1) {
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
    if (use_pdev == 1) {
        // kernel_1: queue -> out_neigh_ptr -> out_neigh -> depths & path_counts
        // kernel_2: depth_index[d] -> out_neigh_ptr -> out_neigh -> path_counts & deltas
        //           depth_index[d] -> scores
        {
          PickleJob job(/*kernel_name*/"bc_kernel_1");
          std::shared_ptr<PickleArrayDescriptor> queue_array_descriptor = queue.getArrayDescriptor();
          queue_array_descriptor->name = "queue";
          queue_array_descriptor->access_type = AccessType::SingleElement;
          queue_array_descriptor->addressing_mode = AddressingMode::Index;
          job.addArrayDescriptor(queue_array_descriptor);
          // We get the array descriptors from the graph. Note that the relation between the arrays here
          // is already set up by the graph's constructor.
          std::shared_ptr<PickleArrayDescriptor> out_index_array_descriptor = g.getOutIndexArrayDescriptor();
          out_index_array_descriptor->name = "out_index";
          out_index_array_descriptor->access_type = AccessType::Ranged;
          out_index_array_descriptor->addressing_mode = AddressingMode::Pointer;
          queue_array_descriptor->dst_indexing_array_id = out_index_array_descriptor->getArrayId();
          job.addArrayDescriptor(out_index_array_descriptor);
          std::shared_ptr<PickleArrayDescriptor> out_neighbors_array_descriptor = g.getOutNeighborsArrayDescriptor();
          out_neighbors_array_descriptor->name = "out_neighbors";
          out_neighbors_array_descriptor->access_type = AccessType::SingleElement;
          out_neighbors_array_descriptor->addressing_mode = AddressingMode::Index;
          job.addArrayDescriptor(out_neighbors_array_descriptor);
          // Add the `depths` array descriptor
          auto depths_array_descriptor = depths.getArrayDescriptor();
          depths_array_descriptor->name = "depths";
          depths_array_descriptor->access_type = AccessType::SingleElement;
          depths_array_descriptor->addressing_mode = AddressingMode::Index;
          out_neighbors_array_descriptor->dst_indexing_array_id = depths_array_descriptor->getArrayId();
          job.addArrayDescriptor(depths_array_descriptor);
          // Add the `path_counts` array descriptor
          auto path_counts_array_descriptor = path_counts.getArrayDescriptor();
          path_counts_array_descriptor->name = "path_counts";
          path_counts_array_descriptor->access_type = AccessType::SingleElement;
          path_counts_array_descriptor->addressing_mode = AddressingMode::Index;
          out_neighbors_array_descriptor->dst_indexing_array_id = path_counts_array_descriptor->getArrayId(); // TODO: add support for multiple destination
          job.addArrayDescriptor(path_counts_array_descriptor);
          job.print();
          pdev->sendJob(job);
          std::cout << "Sent kernel_1" << std::endl;
        }
        {
          PickleJob job(/*kernel_name*/"bc_kernel_2");
          // For the depth_index, it's a vector of iterators where the iterators are added during the run time.
          // For prefetching, this is the first layer of indirection, and we don't actually need to know the base
          // address each iterator as we send the address of the current_node and we guarantee that the address of
          // the current_node + prefetch_distance * node_size is available. So, we'll just place a dummy value for
          // the base address.
          std::shared_ptr<PickleArrayDescriptor> depth_index_array_descriptor(new PickleArrayDescriptor());
          depth_index_array_descriptor->name = "depth_index_iterator";
          depth_index_array_descriptor->vaddr_start = 0;
          depth_index_array_descriptor->vaddr_end = 0;
          depth_index_array_descriptor->element_size = sizeof(NodeID);
          depth_index_array_descriptor->access_type = AccessType::SingleElement;
          depth_index_array_descriptor->addressing_mode = AddressingMode::Index;
          job.addArrayDescriptor(depth_index_array_descriptor);
          // We get the array descriptors from the graph. Note that the relation between the arrays here
          // is already set up by the graph's constructor.
          std::shared_ptr<PickleArrayDescriptor> out_index_array_descriptor = g.getOutIndexArrayDescriptor();
          out_index_array_descriptor->name = "out_index";
          out_index_array_descriptor->access_type = AccessType::Ranged;
          out_index_array_descriptor->addressing_mode = AddressingMode::Pointer;
          depth_index_array_descriptor->dst_indexing_array_id = out_index_array_descriptor->getArrayId();
          job.addArrayDescriptor(out_index_array_descriptor);
          std::shared_ptr<PickleArrayDescriptor> out_neighbors_array_descriptor = g.getOutNeighborsArrayDescriptor();
          out_neighbors_array_descriptor->name = "out_neighbors";
          out_neighbors_array_descriptor->access_type = AccessType::SingleElement;
          out_neighbors_array_descriptor->addressing_mode = AddressingMode::Index;
          job.addArrayDescriptor(out_neighbors_array_descriptor);
          // Add the `path_counts` array descriptor
          auto path_counts_array_descriptor = path_counts.getArrayDescriptor();
          path_counts_array_descriptor->name = "path_counts";
          path_counts_array_descriptor->access_type = AccessType::SingleElement;
          path_counts_array_descriptor->addressing_mode = AddressingMode::Index;
          out_neighbors_array_descriptor->dst_indexing_array_id = path_counts_array_descriptor->getArrayId();
          job.addArrayDescriptor(path_counts_array_descriptor);
          // Add the `deltas` array descriptor
          auto deltas_array_descriptor = deltas.getArrayDescriptor();
          deltas_array_descriptor->name = "deltas";
          deltas_array_descriptor->access_type = AccessType::SingleElement;
          deltas_array_descriptor->addressing_mode = AddressingMode::Index;
          out_neighbors_array_descriptor->dst_indexing_array_id = deltas_array_descriptor->getArrayId(); // TODO: add support for multiple destination
          job.addArrayDescriptor(deltas_array_descriptor);
          // Add the `scores` array descriptor
          auto scores_array_descriptor = scores.getArrayDescriptor();
          scores_array_descriptor->name = "scores";
          scores_array_descriptor->access_type = AccessType::SingleElement;
          scores_array_descriptor->addressing_mode = AddressingMode::Index;
          //out_neighbors_array_descriptor->dst_indexing_array_id = scores_array_descriptor->getArrayId(); // TODO: add support for multiple destination
          job.addArrayDescriptor(scores_array_descriptor);
          job.print();
          std::cout << "Sent kernel_2" << std::endl;
          pdev->sendJob(job);
        }
        // Job 3 (Kernel 3: A dummy kernel)
        {
          PickleJob job(/*kernel_name*/"bc_kernel_3");
          // This is a kernel that keeps the value of the `depth` variable sent by the program at runtime.
          job.print();
          pdev->sendJob(job);
          std::cout << "Sent kernel_3" << std::endl;
        }
        // Create communication page
        UCPage = (uint64_t*) pdev->getUCPagePtr(0);
        UCPage_Kernel2 = UCPage + 1;
        UCPage_Kernel3 = UCPage + 2;
        std::cout << "UCPage: 0x" << std::hex << (uint64_t)UCPage << std::dec << std::endl;
        std::cout << "UCPage_Kernel2: 0x" << std::hex << (uint64_t)UCPage_Kernel2 << std::dec << std::endl;
        std::cout << "UCPage_Kernel3: 0x" << std::hex << (uint64_t)UCPage_Kernel3 << std::dec << std::endl;
        assert(UCPage != nullptr);
    }
  }
#endif

  // The main algorithm
  const NodeID* g_out_start = g.out_neigh(0).begin();
  for (NodeID iter=0; iter < num_iters; iter++) {
    NodeID source = sp.PickNext();
    if (logging_enabled)
      PrintStep("Source", static_cast<int64_t>(source));
    path_counts.fill(0);
    depth_index.resize(0);
    queue.reset();
    succ.reset();
    depths.fill(-1);
#if ENABLE_GEM5==1
    m5_exit_addr(0); // exit 1 (for trial 0) or exit 3 (for trial 1)
#endif // ENABLE_GEM5

#if ENABLE_PICKLEDEVICE==1
    // Turn on the performance watcher
    PerfPage = (uint64_t*) pdev->getPerfPagePtr();
    std::cout << "PerfPage: 0x" << std::hex << (uint64_t)PerfPage << std::dec << std::endl;
    assert(PerfPage != nullptr);
#endif

    if ((trial_num == 1 && use_pdev == 1)) {
      if (prefetch_mode == PrefetchMode::BULK_PREFETCH) {
        assert(false && "The bulk prefetch mode has not been supported yet");
      }
      PBFSWithPrefetch(g, source, depths, path_counts, succ, depth_index, queue, prefetch_distance);
      for (int d=depth_index.size()-2; d >= 0; d--) {
        //#pragma omp parallel for schedule(dynamic, 64)
        // Let's do static scheduling for now
        #pragma omp parallel for schedule(static)
        for (auto it = depth_index[d]; it < depth_index[d+1]; it++) {
          NodeID u = *it;
          ScoreT delta_u = 0;
#if ENABLE_PICKLEDEVICE==1
          auto prefetch_it = it + prefetch_distance;
          if (prefetch_it < depth_index[d+1]) {
            *UCPage_Kernel2 = (uint64_t)(&(*it));
          }
#endif
          for (NodeID &v : g.out_neigh(u)) {
            if (succ.get_bit(&v - g_out_start)) {
              delta_u += (path_counts[u] / path_counts[v]) * (1 + deltas[v]);
            }
          }
          deltas[u] = delta_u;
          scores[u] += delta_u;
        }
      }
    } else {
      PBFS(g, source, depths, path_counts, succ, depth_index, queue);
      for (int d=depth_index.size()-2; d >= 0; d--) {
        //#pragma omp parallel for schedule(dynamic, 64)
        // Let's do static scheduling for now
        #pragma omp parallel for schedule(static)
        for (auto it = depth_index[d]; it < depth_index[d+1]; it++) {
          NodeID u = *it;
          ScoreT delta_u = 0;
          for (NodeID &v : g.out_neigh(u)) {
            if (succ.get_bit(&v - g_out_start)) {
              delta_u += (path_counts[u] / path_counts[v]) * (1 + deltas[v]);
            }
          }
          deltas[u] = delta_u;
          scores[u] += delta_u;
        }
      }
    }
#if ENABLE_GEM5==1
    m5_exit_addr(0); // exit 2 (for trial 0) or exit 4 (for trial 1)
#endif // ENABLE_GEM5
  }
  // Let's not normalize scores as it's not part of the benchmarking
  /*
  ScoreT biggest_score = 0;
  #pragma omp parallel for reduction(max : biggest_score)
  for (NodeID n=0; n < g.num_nodes(); n++)
    biggest_score = max(biggest_score, scores[n]);
  #pragma omp parallel for
  for (NodeID n=0; n < g.num_nodes(); n++)
    scores[n] = scores[n] / biggest_score;
  */
  return scores;
}


void PrintTopScores(const Graph &g, const pvector<ScoreT> &scores) {
  vector<pair<NodeID, ScoreT>> score_pairs(g.num_nodes());
  for (NodeID n : g.vertices())
    score_pairs[n] = make_pair(n, scores[n]);
  int k = 5;
  vector<pair<ScoreT, NodeID>> top_k = TopK(score_pairs, k);
  for (auto kvp : top_k)
    cout << kvp.second << ":" << kvp.first << endl;
}


// Still uses Brandes algorithm, but has the following differences:
// - serial (no need for atomics or dynamic scheduling)
// - uses vector for BFS queue
// - regenerates farthest to closest traversal order from depths
// - regenerates successors from depths
bool BCVerifier(const Graph &g, SourcePicker<Graph> &sp, NodeID num_iters,
                const pvector<ScoreT> &scores_to_test) {
  pvector<ScoreT> scores(g.num_nodes(), 0);
  for (int iter=0; iter < num_iters; iter++) {
    NodeID source = sp.PickNext();
    // BFS phase, only records depth & path_counts
    pvector<int> depths(g.num_nodes(), -1);
    depths[source] = 0;
    vector<CountT> path_counts(g.num_nodes(), 0);
    path_counts[source] = 1;
    vector<NodeID> to_visit;
    to_visit.reserve(g.num_nodes());
    to_visit.push_back(source);
    for (auto it = to_visit.begin(); it != to_visit.end(); it++) {
      NodeID u = *it;
      for (NodeID v : g.out_neigh(u)) {
        if (depths[v] == -1) {
          depths[v] = depths[u] + 1;
          to_visit.push_back(v);
        }
        if (depths[v] == depths[u] + 1)
          path_counts[v] += path_counts[u];
      }
    }
    // Get lists of vertices at each depth
    vector<vector<NodeID>> verts_at_depth;
    for (NodeID n : g.vertices()) {
      if (depths[n] != -1) {
        if (depths[n] >= static_cast<int>(verts_at_depth.size()))
          verts_at_depth.resize(depths[n] + 1);
        verts_at_depth[depths[n]].push_back(n);
      }
    }
    // Going from farthest to closest, compute "dependencies" (deltas)
    pvector<ScoreT> deltas(g.num_nodes(), 0);
    for (int depth=verts_at_depth.size()-1; depth >= 0; depth--) {
      for (NodeID u : verts_at_depth[depth]) {
        for (NodeID v : g.out_neigh(u)) {
          if (depths[v] == depths[u] + 1) {
            deltas[u] += (path_counts[u] / path_counts[v]) * (1 + deltas[v]);
          }
        }
        scores[u] += deltas[u];
      }
    }
  }
  // Normalize scores
  ScoreT biggest_score = *max_element(scores.begin(), scores.end());
  for (NodeID n : g.vertices())
    scores[n] = scores[n] / biggest_score;
  // Compare scores
  bool all_ok = true;
  for (NodeID n : g.vertices()) {
    ScoreT delta = abs(scores_to_test[n] - scores[n]);
    if (delta > std::numeric_limits<ScoreT>::epsilon()) {
      cout << n << ": " << scores[n] << " != " << scores_to_test[n];
      cout << "(" << delta << ")" << endl;
      all_ok = false;
    }
  }
  return all_ok;
}


int main(int argc, char* argv[]) {
  CLIterApp cli(argc, argv, "betweenness-centrality", 1);
  if (!cli.ParseArgs())
    return -1;
  if (cli.num_iters() > 1 && cli.start_vertex() != -1)
    cout << "Warning: iterating from same source (-r & -i)" << endl;
  Builder b(cli);
  Graph g = b.MakeGraph();
  SourcePicker<Graph> sp(g, cli.start_vertex());
  auto BCBound = [&sp, &cli] (const Graph &g, int trial_num) {
    return Brandes(g, sp, MAX_BRANDES_NUM_ITERS, trial_num);
  };
  SourcePicker<Graph> vsp(g, cli.start_vertex());
  auto VerifierBound = [&vsp, &cli] (const Graph &g,
                                     const pvector<ScoreT> &scores) {
    return BCVerifier(g, vsp, MAX_BRANDES_NUM_ITERS, scores);
  };
#if ENABLE_GEM5==1
  map_m5_mem();
#endif // ENABLE_GEM5
  BenchmarkKernel(cli, g, BCBound, PrintTopScores, VerifierBound);
#if ENABLE_GEM5==1
  //m5_work_end_addr(0, 0);
  //unmap_m5_mem();
#endif // ENABLE_GEM5
  //printf("Num node visited: %d\n", num_nodes_visited.load());
  return 0;
}
