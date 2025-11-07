// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

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
#include "graph.h"
#include "pvector.h"
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
Kernel: PageRank (PR)
Author: Scott Beamer

Will return pagerank scores for all vertices once total change < epsilon

This PR implementation uses the traditional iterative approach. It performs
updates in the pull direction to remove the need for atomics, and it allows
new values to be immediately visible (like Gauss-Seidel method). The prior PR
implementation is still available in src/pr_spmv.cc.
*/


using namespace std;

uint64_t* UCPage = NULL;
uint64_t* PerfPage = NULL;

const uint64_t PERF_THREAD_START = 0;
const uint64_t PERF_THREAD_COMPLETE = 1;

typedef float ScoreT;
const float kDamp = 0.85;


// Do 1 iteration of PR using GS method
void PageRankPullGS(
  const Graph &g, pvector<ScoreT>& scores, pvector<ScoreT>& outgoing_contrib
) {
  const ScoreT init_score = 1.0f / g.num_nodes();
  const ScoreT base_score = (1.0f - kDamp) / g.num_nodes();
  #pragma omp parallel for
  for (NodeID n=0; n < g.num_nodes(); n++)
    outgoing_contrib[n] = init_score / g.out_degree(n);
  //for (int iter=0; iter < max_iters; iter++) {
    double error = 0;
    #pragma omp parallel
    {
#if ENABLE_PICKLEDEVICE==1
      const uint64_t thread_id = (uint64_t)omp_get_thread_num();
      *PerfPage = (thread_id << 1) | PERF_THREAD_START;
#endif
      #pragma omp for reduction(+ : error) schedule(dynamic, 16384)
      for (NodeID u=0; u < g.num_nodes(); u++) {
        ScoreT incoming_total = 0;
        for (NodeID v : g.in_neigh(u))
          incoming_total += outgoing_contrib[v];
        ScoreT old_score = scores[u];
        scores[u] = base_score + kDamp * incoming_total;
        error += fabs(scores[u] - old_score);
        outgoing_contrib[u] = scores[u] / g.out_degree(u);
      }
#if ENABLE_PICKLEDEVICE==1
      *PerfPage = (thread_id << 1) | PERF_THREAD_COMPLETE;
#endif
    }
  //  if (logging_enabled)
  //    PrintStep(iter, error);
  //  if (error < epsilon)
  //    break;
  //}
}

// Do 1 iteration of PR using GS method with prefetcher
void PageRankPullGSWithPrefetch(
  const Graph &g, pvector<ScoreT>& scores, pvector<ScoreT>& outgoing_contrib
) {
  const ScoreT init_score = 1.0f / g.num_nodes();
  const ScoreT base_score = (1.0f - kDamp) / g.num_nodes();
  #pragma omp parallel for
  for (NodeID n=0; n < g.num_nodes(); n++)
    outgoing_contrib[n] = init_score / g.out_degree(n);
  //for (int iter=0; iter < max_iters; iter++) {
    double error = 0;
    #pragma omp parallel
    {
#if ENABLE_PICKLEDEVICE==1
      const uint64_t thread_id = (uint64_t)omp_get_thread_num();
      *PerfPage = (thread_id << 1) | PERF_THREAD_START;
#endif
      #pragma omp for reduction(+ : error) schedule(dynamic, 16384)
      for (NodeID u=0; u < g.num_nodes(); u++) {
#if ENABLE_PICKLEDEVICE==1
        *UCPage = (uint64_t)(u);
#endif
        ScoreT incoming_total = 0;
        for (NodeID v : g.in_neigh(u))
          incoming_total += outgoing_contrib[v];
        ScoreT old_score = scores[u];
        scores[u] = base_score + kDamp * incoming_total;
        error += fabs(scores[u] - old_score);
        outgoing_contrib[u] = scores[u] / g.out_degree(u);
      }
#if ENABLE_PICKLEDEVICE==1
      *PerfPage = (thread_id << 1) | PERF_THREAD_COMPLETE;
#endif
    }
  //  if (logging_enabled)
  //    PrintStep(iter, error);
  //  if (error < epsilon)
  //    break;
  //}
}


void PrintTopScores(const Graph &g, const pvector<ScoreT> &scores) {
  vector<pair<NodeID, ScoreT>> score_pairs(g.num_nodes());
  for (NodeID n=0; n < g.num_nodes(); n++) {
    score_pairs[n] = make_pair(n, scores[n]);
  }
  int k = 5;
  vector<pair<ScoreT, NodeID>> top_k = TopK(score_pairs, k);
  for (auto kvp : top_k)
    cout << kvp.second << ":" << kvp.first << endl;
}


// Verifies by asserting a single serial iteration in push direction has
//   error < target_error
bool PRVerifier(const Graph &g, const pvector<ScoreT> &scores,
                        double target_error) {
  const ScoreT base_score = (1.0f - kDamp) / g.num_nodes();
  pvector<ScoreT> incoming_sums(g.num_nodes(), 0);
  double error = 0;
  for (NodeID u : g.vertices()) {
    ScoreT outgoing_contrib = scores[u] / g.out_degree(u);
    for (NodeID v : g.out_neigh(u))
      incoming_sums[v] += outgoing_contrib;
  }
  for (NodeID n : g.vertices()) {
    error += fabs(base_score + kDamp * incoming_sums[n] - scores[n]);
    incoming_sums[n] = 0;
  }
  PrintTime("Total Error", error);
  return error < target_error;
}

pvector<ScoreT> DoPR(const Graph& g, int trial_num) {
  const ScoreT init_score = 1.0f / g.num_nodes();
  pvector<ScoreT> scores(g.num_nodes(), init_score);
  pvector<ScoreT> outgoing_contrib(g.num_nodes());

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
    PageRankPullGS(g, scores, outgoing_contrib);
#if ENABLE_GEM5==1
    m5_exit_addr(0); // exit 2
#endif // ENABLE_GEM5
    std::cout << "ROI End" << std::endl; // ----- ROI End -----
  } else if (trial_num == 1) { // ----- Second iteration: measured phase -----
    // reset the score
    scores.fill(init_score);
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
        PickleJob job(/*kernel_name*/"pr_kernel");
        // We get the array descriptors from the graph. Note that the relation between the arrays here
        // is already set up by the graph's constructor.
        std::shared_ptr<PickleArrayDescriptor> in_index_array_descriptor = g.getInIndexArrayDescriptor();
        in_index_array_descriptor->access_type = AccessType::Ranged;
        in_index_array_descriptor->addressing_mode = AddressingMode::Pointer;
        job.addArrayDescriptor(in_index_array_descriptor);
        std::shared_ptr<PickleArrayDescriptor> in_neighbors_array_descriptor = g.getInNeighborsArrayDescriptor();
        in_neighbors_array_descriptor->access_type = AccessType::SingleElement;
        in_neighbors_array_descriptor->addressing_mode = AddressingMode::Index;
        job.addArrayDescriptor(in_neighbors_array_descriptor);
        // We need to add the scores and outgoing_contrib arrays as well
        std::shared_ptr<PickleArrayDescriptor> scores_array_descriptor = scores.getArrayDescriptor();
        scores_array_descriptor->access_type = AccessType::SingleElement;
        scores_array_descriptor->addressing_mode = AddressingMode::Index;
        in_neighbors_array_descriptor->dst_indexing_array_id = scores_array_descriptor->getArrayId();
        job.addArrayDescriptor(scores_array_descriptor);
        std::shared_ptr<PickleArrayDescriptor> outgoing_contrib_array_descriptor = outgoing_contrib.getArrayDescriptor();
        outgoing_contrib_array_descriptor->access_type = AccessType::SingleElement;
        outgoing_contrib_array_descriptor->addressing_mode = AddressingMode::Index;
        // we currently do not have a way to add 2 destinations per array so we need to fix this
        job.addArrayDescriptor(outgoing_contrib_array_descriptor);
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
        PageRankPullGSWithPrefetch(g, scores, outgoing_contrib);;
    } else {
        PageRankPullGS(g, scores, outgoing_contrib);;
    }
#if ENABLE_GEM5==1
    m5_exit_addr(0); // exit 4
#endif // ENABLE_GEM5
    std::cout << "ROI End" << std::endl; // ----- ROI End -----
  }
  return scores;
}


int main(int argc, char* argv[]) {
  CLPageRank cli(argc, argv, "pagerank", 1e-4, 20);
  if (!cli.ParseArgs())
    return -1;
  Builder b(cli);
  Graph g = b.MakeGraph();
  auto VerifierBound = [&cli] (const Graph &g, const pvector<ScoreT> &scores) {
    return PRVerifier(g, scores, cli.tolerance());
  };
#if ENABLE_GEM5==1
  map_m5_mem();
#endif // ENABLE_GEM5
  BenchmarkKernel(cli, g, DoPR, PrintTopScores, VerifierBound);
#if ENABLE_GEM5==1
  //unmap_m5_mem();
#endif // ENABLE_GEM5
  return 0;
}

