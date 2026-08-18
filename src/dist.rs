//! All-pairs shortest paths over the directed reconfiguration graph.
//!
//! Edge `i -> j` exists iff `topo.time(i, j) >= 0` (negative entries in the
//! raw matrix mean "no edge"); self-loops are never used. Edge cost is the
//! *mean* reconfiguration time, which makes these the distances `d(i,j)` of
//! the system model. Every policy routes on them, so a mean-preserving
//! change of reconfiguration-time family alters realized durations without
//! moving a single routing decision.
//!
//! Distances to unreachable targets are `f64::INFINITY` and the matching
//! next-hop entry is `None`. Configurations are validated to be mutually
//! reachable before a run starts, so the infinite case only arises in unit
//! tests of partial graphs.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::topology::Topology;

pub type DistMatrix = Vec<Vec<f64>>;
pub type NextHopMatrix = Vec<Vec<Option<usize>>>;

/// Result of all-pairs Dijkstra over the reconfiguration graph.
///
/// `next_hop[i][j]` is the configuration immediately following `i` on the
/// shortest path from `i` to `j`. Transit-aware policies use it to take
/// one-edge moves toward a committed target. `None` when `i == j` or when
/// `j` is unreachable from `i`.
#[derive(Debug, Clone)]
pub struct PathInfo {
    pub dist: DistMatrix,
    pub next_hop: NextHopMatrix,
}

/// All-pairs shortest paths on the mean reconfiguration times.
pub fn all_pairs_shortest_paths(topo: &Topology) -> PathInfo {
    let n = topo.n;
    let mut dist = vec![vec![f64::INFINITY; n]; n];
    let mut next_hop = vec![vec![None; n]; n];
    for src in 0..n {
        let (d, nh) = dijkstra(topo, src);
        dist[src] = d;
        next_hop[src] = nh;
    }
    PathInfo { dist, next_hop }
}

fn dijkstra(topo: &Topology, src: usize) -> (Vec<f64>, Vec<Option<usize>>) {
    let n = topo.n;
    let mut dist = vec![f64::INFINITY; n];
    let mut pred: Vec<Option<usize>> = vec![None; n];
    dist[src] = 0.0;

    let mut heap: BinaryHeap<MinNode> = BinaryHeap::new();
    heap.push(MinNode { d: 0.0, v: src });
    while let Some(MinNode { d, v }) = heap.pop() {
        if d > dist[v] {
            continue;
        }
        for &w in topo.neighbors(v) {
            let nd = d + topo.time(v, w);
            if nd < dist[w] {
                dist[w] = nd;
                pred[w] = Some(v);
                heap.push(MinNode { d: nd, v: w });
            }
        }
    }

    // For each reachable destination j, walk back via predecessors until
    // reaching the node whose predecessor is `src`: that is the first hop
    // along the shortest path from `src`.
    let mut next_hop = vec![None; n];
    for j in 0..n {
        if j == src || !dist[j].is_finite() {
            continue;
        }
        let mut cur = j;
        loop {
            match pred[cur] {
                Some(p) if p == src => {
                    next_hop[j] = Some(cur);
                    break;
                }
                Some(p) => cur = p,
                None => break, // unreachable; cannot happen when dist is finite
            }
        }
    }
    (dist, next_hop)
}

#[derive(Copy, Clone)]
struct MinNode {
    d: f64,
    v: usize,
}
impl PartialEq for MinNode {
    fn eq(&self, o: &Self) -> bool {
        self.d == o.d && self.v == o.v
    }
}
impl Eq for MinNode {}
impl PartialOrd for MinNode {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for MinNode {
    fn cmp(&self, o: &Self) -> Ordering {
        // BinaryHeap is a max-heap; flip the comparison to get a min-heap.
        o.d.partial_cmp(&self.d).unwrap_or(Ordering::Equal)
    }
}
