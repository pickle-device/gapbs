// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <sstream>
#include <vector>

#include <omp.h>
#include <stdlib.h>

#if CHECK_NUM_EDGES==1
#pragma message("Checking number of edges enabled")
#else
#pragma message("Not enable checking number of edges")
#endif

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
#include "platform_atomics.h"
#include "timer.h"

#if ENABLE_GEM5==1
#pragma message("Compiling with gem5 instructions")
#include <gem5/m5ops.h>
#include "m5_mmap.h"
#endif // ENABLE_GEM5

#if ENABLE_PICKLEDEVICE
std::unique_ptr<PickleDeviceManager> pdev(new PickleDeviceManager());
#endif

/*
GAP Benchmark Suite
Kernel: Breadth-First Search (BFS)
Author: Scott Beamer

Will return parent array for a BFS traversal from a source vertex

This BFS implementation makes use of the Direction-Optimizing approach [1].
It uses the alpha and beta parameters to determine whether to switch search
directions. For representing the frontier, it uses a SlidingQueue for the
top-down approach and a Bitmap for the bottom-up approach. To reduce
false-sharing for the top-down approach, thread-local QueueBuffer's are used.

To save time computing the number of edges exiting the frontier, this
implementation precomputes the degrees in bulk at the beginning by storing
them in parent array as negative numbers. Thus the encoding of parent is:
  parent[x] < 0 implies x is unvisited and parent[x] = -out_degree(x)
  parent[x] >= 0 implies x been visited

[1] Scott Beamer, Krste Asanović, and David Patterson. "Direction-Optimizing
    Breadth-First Search." International Conference on High Performance
    Computing, Networking, Storage and Analysis (SC), Salt Lake City, Utah,
    November 2012.
*/


using namespace std;
uint64_t* UCPage = NULL;

int64_t TDStepWithPrefetch(const Graph &g, pvector<NodeID> &parent,
                           SlidingQueue<NodeID> &queue, const uint64_t prefetch_distance) {
  int64_t scout_count = 0;
  #pragma omp parallel
  {
    QueueBuffer<NodeID> lqueue(queue);
    #pragma omp for reduction(+ : scout_count) nowait
    for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
      NodeID u = *q_iter;
#if ENABLE_PICKLEDEVICE==1
      auto prefetch_iter = q_iter + prefetch_distance;
      if (prefetch_iter < queue.end())
      {
          *UCPage = (uint64_t)(&(*prefetch_iter));
          //std::cout << "Prefetch hint at 0x" << std::hex << (uint64_t)(&(*prefetch_iter)) << std::dec << std::endl;
          //std::cout << "Hinting prefetch node id 0x" << std::hex << (*check_iter) << ", prefetch hint: 0x" << (uint64_t)(&(*prefetch_iter)) << std::dec << std::endl;
      }
#endif
      for (NodeID v : g.out_neigh(u)) {
        NodeID curr_val = parent[v];
        if (curr_val < 0) {
          if (compare_and_swap(parent[v], curr_val, u)) {
            lqueue.push_back(v);
            scout_count += -curr_val;
          }
        }
      }
    }
    lqueue.flush();
  }
  return scout_count;
}

int64_t TDStep(const Graph &g, pvector<NodeID> &parent,
               SlidingQueue<NodeID> &queue, uint64_t& num_visited_edges, uint64_t& num_visited_nodes) {
  int64_t scout_count = 0;
  #pragma omp parallel
  {
    QueueBuffer<NodeID> lqueue(queue);
    #pragma omp for reduction(+ : scout_count) nowait
    for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
      NodeID u = *q_iter;
#if CHECK_NUM_EDGES==1
      num_visited_nodes += 1;
#endif
        //uint64_t thread_id = omp_get_thread_num();
        //uint64_t addr = (uint64_t)(&(*q_iter));
        //std::ofstream out_stream;
        //std::stringstream ss;
        //ss << "/home/ubuntu/traces/" << thread_id << ".txt";
        //std::string filename = ss.str();
        //out_stream.open(filename, std::ios::app);
        //out_stream << "0x" << std::hex << addr << std::endl;
        //out_stream.close();
      for (NodeID v : g.out_neigh(u)) {
        NodeID curr_val = parent[v];
#if CHECK_NUM_EDGES==1
        num_visited_edges += 1;
#endif
        if (curr_val < 0) {
          if (compare_and_swap(parent[v], curr_val, u)) {
            lqueue.push_back(v);
            scout_count += -curr_val;
          }
        }
      }
    }
    lqueue.flush();
  }
  return scout_count;
}


void QueueToBitmap(const SlidingQueue<NodeID> &queue, Bitmap &bm) {
  #pragma omp parallel for
  for (auto q_iter = queue.begin(); q_iter < queue.end(); q_iter++) {
    NodeID u = *q_iter;
    bm.set_bit_atomic(u);
  }
}

void BitmapToQueue(const Graph &g, const Bitmap &bm,
                   SlidingQueue<NodeID> &queue) {
  #pragma omp parallel
  {
    QueueBuffer<NodeID> lqueue(queue);
    #pragma omp for nowait
    for (NodeID n=0; n < g.num_nodes(); n++)
      if (bm.get_bit(n))
        lqueue.push_back(n);
    lqueue.flush();
  }
  queue.slide_window();
}

pvector<NodeID> InitParent(const Graph &g) {
  pvector<NodeID> parent(g.num_nodes());
  #pragma omp parallel for
  for (NodeID n=0; n < g.num_nodes(); n++)
    parent[n] = g.out_degree(n) != 0 ? -g.out_degree(n) : -1;
  return parent;
}

pvector<NodeID> DOBFS(const Graph &g, NodeID source, int iter_num,
                      int alpha = 15, int beta = 18) {
  PrintStep("Source", static_cast<int64_t>(source));

  pvector<NodeID> parent = InitParent(g);

  parent[source] = source;
  SlidingQueue<NodeID> queue(g.num_edges());
  queue.push_back(source);
  queue.slide_window();
  Bitmap curr(g.num_nodes());
  curr.reset();
  Bitmap front(g.num_nodes());
  front.reset();
  int64_t edges_to_check = g.num_edges_directed();
  int64_t scout_count = g.out_degree(source);

  std::cout << "Trial " << iter_num+1 << std::endl;

  uint64_t num_visited_edges = 0;
  uint64_t num_visited_nodes = 0;

  // Iter 0
  if (iter_num == 0) {
    #if ENABLE_GEM5==1
      m5_exit_addr(0);
    #endif // ENABLE_GEM5
    while (!queue.empty()) {
      edges_to_check -= scout_count;
      scout_count = TDStep(g, parent, queue, num_visited_edges, num_visited_nodes);
      queue.slide_window();
    }
    #if CHECK_NUM_EDGES==1
    std::cout << "Number of visited nodes: " << num_visited_nodes << std::endl;
    std::cout << "Number of visited edges: " << num_visited_edges << std::endl;
    #endif
    #if ENABLE_GEM5==1
    m5_exit_addr(0);
    #endif // ENABLE_GEM5
    #pragma omp parallel for
    for (NodeID n = 0; n < g.num_nodes(); n++)
      if (parent[n] < -1)
        parent[n] = -1;
    //std::cout << "Trial " << iter_num+1 << " finished" << std::endl;
    std::cout << "Trial " << iter_num+1 << " finished; scout_count = " << scout_count << std::endl;
    return parent;
  }
  // Iter 1
  uint64_t use_pdev = 0;
  uint64_t prefetch_distance = 0;
#if ENABLE_PICKLEDEVICE==1
  PickleDevicePrefetcherSpecs specs = pdev->getDevicePrefetcherSpecs();
  use_pdev = specs.availability;
  prefetch_distance = specs.prefetch_distance;
#endif
  std::cout << "Use pdev: " << use_pdev << "; Prefetch distance: " << prefetch_distance << std::endl;

  if (use_pdev == 0) {
    #if ENABLE_GEM5==1
    m5_exit_addr(0);
    #endif // ENABLE_GEM5
    std::cout << "ROI Start" << std::endl;
    while (!queue.empty()) {
      edges_to_check -= scout_count;
      scout_count = TDStep(g, parent, queue, num_visited_edges, num_visited_nodes);
      queue.slide_window();
    }
    std::cout << "ROI End" << std::endl;
    #if ENABLE_GEM5==1
    m5_exit_addr(0);
    #endif // ENABLE_GEM5
    #pragma omp parallel for
    for (NodeID n = 0; n < g.num_nodes(); n++)
      if (parent[n] < -1)
        parent[n] = -1;
    //std::cout << "Trial " << iter_num+1 << " finished; scout_count = " << scout_count << std::endl;
    return parent;
  } else {
    // Write dynamic info per run, ie., memory address of structures
    // doint before ROI so that the info will be printed out every time
#if ENABLE_PICKLEDEVICE==1
    auto job = createGraphJobUsingOutgoingEdges(&g, "bfs_kernel", &queue, &parent);

    // ---
    // We have to manually set the AccessType and AddressingMode as we currently
    // do not have any mechanism to detect these values.
    // queue I S
    job.changeAccessTypeByArrayId(queue.getArrayDescriptor()->getArrayId(), AccessType::SingleElement);
    job.changeAddressingModeByArrayId(queue.getArrayDescriptor()->getArrayId(), AddressingMode::Index);
    // outIndex P R
    job.changeAccessTypeByArrayId(g.getOutIndexArrayDescriptor()->getArrayId(), AccessType::Ranged);
    job.changeAddressingModeByArrayId(g.getOutIndexArrayDescriptor()->getArrayId(), AddressingMode::Pointer);
    // outNeighbors I S
    job.changeAccessTypeByArrayId(g.getOutNeighborsArrayDescriptor()->getArrayId(), AccessType::SingleElement);
    job.changeAddressingModeByArrayId(g.getOutNeighborsArrayDescriptor()->getArrayId(), AddressingMode::Index);
    // nprop I S
    job.changeAccessTypeByArrayId(parent.getArrayDescriptor()->getArrayId(), AccessType::SingleElement);
    job.changeAddressingModeByArrayId(parent.getArrayDescriptor()->getArrayId(), AddressingMode::Index);
    // ---
    job.print();
    std::cout << "Sent job" << std::endl;
    pdev->sendJob(createGraphJobUsingOutgoingEdges(&g, "bfs_kernel", &queue, &parent));

    UCPage = (uint64_t*) pdev->getUCPagePtr(0);
    std::cout << "UCPage: 0x" << std::hex<< (uint64_t)UCPage <<std::dec<< std::endl;
    assert(UCPage != nullptr);
#endif
    #if ENABLE_GEM5==1
    m5_exit_addr(0);
    #endif // ENABLE_GEM5
    std::cout << "ROI Start" << std::endl;

    num_visited_edges = 0;
    while (!queue.empty()) {
        edges_to_check -= scout_count;
        scout_count = TDStepWithPrefetch(g, parent, queue, prefetch_distance);
        queue.slide_window();
    }
    #if CHECK_NUM_EDGES==1
    std::cout << "Number of visited edges: " << num_visited_edges << std::endl;
    #endif
    std::cout << "Trial " << iter_num+1 << " finished; scout_count = " << scout_count << std::endl;
    std::cout << "ROI End" << std::endl;
    #if ENABLE_GEM5==1
    m5_exit_addr(0);
    #endif // ENABLE_GEM5

    #pragma omp parallel for
    for (NodeID n = 0; n < g.num_nodes(); n++)
      if (parent[n] < -1)
        parent[n] = -1;
    return parent;
  }
  return parent;
}


void PrintBFSStats(const Graph &g, const pvector<NodeID> &bfs_tree) {
  int64_t tree_size = 0;
  int64_t n_edges = 0;
  for (NodeID n : g.vertices()) {
    if (bfs_tree[n] >= 0) {
      n_edges += g.out_degree(n);
      tree_size++;
    }
  }
  cout << "BFS Tree has " << tree_size << " nodes and ";
  cout << n_edges << " edges" << endl;
}


// BFS verifier does a serial BFS from same source and asserts:
// - parent[source] = source
// - parent[v] = u  =>  depth[v] = depth[u] + 1 (except for source)
// - parent[v] = u  => there is edge from u to v
// - all vertices reachable from source have a parent
bool BFSVerifier(const Graph &g, NodeID source,
                 const pvector<NodeID> &parent) {
  pvector<int> depth(g.num_nodes(), -1);
  depth[source] = 0;
  vector<NodeID> to_visit;
  to_visit.reserve(g.num_nodes());
  to_visit.push_back(source);
  for (auto it = to_visit.begin(); it != to_visit.end(); it++) {
    NodeID u = *it;
    for (NodeID v : g.out_neigh(u)) {
      if (depth[v] == -1) {
        depth[v] = depth[u] + 1;
        to_visit.push_back(v);
      }
    }
  }
  for (NodeID u : g.vertices()) {
    if ((depth[u] != -1) && (parent[u] != -1)) {
      if (u == source) {
        if (!((parent[u] == u) && (depth[u] == 0))) {
          cout << "Source wrong" << endl;
          return false;
        }
        continue;
      }
      bool parent_found = false;
      for (NodeID v : g.in_neigh(u)) {
        if (v == parent[u]) {
          if (depth[v] != depth[u] - 1) {
            cout << "Wrong depths for " << u << " & " << v << endl;
            return false;
          }
          parent_found = true;
          break;
        }
      }
      if (!parent_found) {
        cout << "Couldn't find edge from " << parent[u] << " to " << u << endl;
        return false;
      }
    } else if (depth[u] != parent[u]) {
      cout << "Reachability mismatch" << endl;
      return false;
    }
  }
  return true;
}


int main(int argc, char* argv[]) {
  CLApp cli(argc, argv, "breadth-first search");
  uint64_t num_threads = 0;
  #pragma omp parallel reduction(+:num_threads)
  {
    num_threads += 1;
  }
  printf("Number of threads: %d\n", num_threads);
  if (!cli.ParseArgs())
    return -1;
  Builder b(cli);
  Graph g = b.MakeGraph();
  //g.printGraph();
  SourcePicker<Graph> sp(g, cli.start_vertex());
  auto BFSBound = [&sp] (const Graph &g, const int& iter_num) { return DOBFS(g, sp.PickNext(), iter_num); };
  SourcePicker<Graph> vsp(g, cli.start_vertex());
  auto VerifierBound = [&vsp] (const Graph &g, const pvector<NodeID> &parent) {
    return BFSVerifier(g, vsp.PickNext(), parent);
  };

#if ENABLE_GEM5==1
  map_m5_mem();
#endif // ENABLE_GEM5
  BenchmarkKernel(cli, g, BFSBound, PrintBFSStats, VerifierBound);
#if ENABLE_GEM5==1
  //m5_work_end_addr(0, 0);
  //unmap_m5_mem();
#endif // ENABLE_GEM5

  return 0;
}
