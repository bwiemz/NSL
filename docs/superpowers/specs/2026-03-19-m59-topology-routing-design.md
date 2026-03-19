# M59: Topology-Aware Network Routing — Design Spec

**Date:** 2026-03-19
**Status:** Planned
**Milestone:** M59
**Prerequisites:** M30 (Tensor Parallelism & NCCL — collective communication), M34 (Ring Attention)
**Dependencies:** M58 benefits from topology-aware communicator rebuild; M61 uses topology info for cross-rank trace analysis

---

## Overview

Feed the physical datacenter network topology to the NSL compiler so it can optimize collective communication (all-reduce, all-gather, send/recv) to minimize cross-rack traffic and maximize bandwidth utilization. The compiler reads a topology specification file (`cluster.json`) describing the physical switch hierarchy, NIC types, bandwidths, and latencies, then produces a compile-time routing table that maps each communication primitive to the optimal physical path.

**Key design:** Topology awareness is a compile-time optimization, not a runtime discovery. The compiler ingests the topology spec, computes optimal rank placement (TP within a switch, PP across racks, DP across the cluster), emits hierarchical collective implementations (NVLink -> InfiniBand -> optical tiers), and embeds the routing table directly in the binary. At runtime, ranks read their routing table entry instead of relying on NCCL's runtime topology detection, which is suboptimal for multi-tier networks.

**Why this is different from NCCL's auto-detection:** NCCL discovers topology at runtime via PCI-e bus queries and NVLink probing. This works well for single-node multi-GPU but fails at cluster scale because: (1) NCCL cannot see beyond the node boundary — it doesn't know which nodes share a ToR switch; (2) NCCL's ring algorithm treats all inter-node links as equal bandwidth, ignoring the 10x difference between intra-rack InfiniBand and inter-rack optical; (3) NCCL cannot pre-compute ring orderings at compile time, paying discovery cost at every communicator creation.

**Why this requires a compiler:** The topology information is static for a given cluster. Embedding it at compile time eliminates runtime discovery overhead, enables the compiler to prove bandwidth bounds (feeding into M53 WCET proofs), and allows the linker to select hardware-specific transport libraries (NVLink vs. InfiniBand vs. RoCE).

---

## Section 1: Language Surface

### CLI Integration

```
# Build with topology awareness
nsl build --topology cluster.json model.nsl

# Validate topology file without building
nsl check --topology cluster.json

# Generate a template topology file from runtime detection
nsl topology detect --output cluster.json

# Visualize the topology and rank placement
nsl topology show cluster.json --ranks 256 --tp 8 --pp 4
```

### Topology Reference in train/serve Blocks

```
@fault_tolerant
train DistributedPretraining:
    model: LLaMA70B
    data: WebCorpus
    optimizer: AdamW(lr=3e-4)

    distribute: "dp=32, tp=8, pp=4"
    topology: "cluster.json"             # NEW: topology spec file

    # Optional: override automatic placement
    placement_strategy: "tp_intra_switch"   # default: TP within switch
```

```
serve DistributedLLM:
    model: LLaMA70B
    distribute: "tp=8"
    topology: "cluster.json"

    @endpoint
    fn generate(prompt: str) -> str:
        return autoregressive_decode(model, tokenizer.encode(prompt), 512)
```

### Topology Specification Format

```json
{
    "version": 1,
    "name": "prod-cluster-a",
    "description": "8-rack GPU cluster, 256 A100 GPUs",

    "nodes": [
        {
            "id": "node-000",
            "rack": "rack-0",
            "gpus": [
                {
                    "id": "gpu-0",
                    "type": "A100-80GB-SXM4",
                    "pcie_bus": "0000:06:00.0",
                    "nvlink_peers": ["gpu-1", "gpu-2", "gpu-3", "gpu-4", "gpu-5", "gpu-6", "gpu-7"],
                    "nvlink_bw_gbps": 600
                },
                {
                    "id": "gpu-1",
                    "type": "A100-80GB-SXM4",
                    "pcie_bus": "0000:07:00.0",
                    "nvlink_peers": ["gpu-0", "gpu-2", "gpu-3", "gpu-4", "gpu-5", "gpu-6", "gpu-7"],
                    "nvlink_bw_gbps": 600
                }
            ],
            "nics": [
                {
                    "id": "nic-0",
                    "type": "ConnectX-7",
                    "speed_gbps": 400,
                    "connected_to": "switch-tor-rack-0-port-1"
                }
            ]
        }
    ],

    "switches": [
        {
            "id": "switch-tor-rack-0",
            "type": "tor",
            "rack": "rack-0",
            "ports": 64,
            "port_speed_gbps": 400,
            "latency_us": 0.5
        },
        {
            "id": "switch-spine-0",
            "type": "spine",
            "ports": 128,
            "port_speed_gbps": 400,
            "latency_us": 1.0
        },
        {
            "id": "switch-core-0",
            "type": "core",
            "ports": 256,
            "port_speed_gbps": 800,
            "latency_us": 2.0
        }
    ],

    "links": [
        {
            "from": "switch-tor-rack-0",
            "to": "switch-spine-0",
            "bandwidth_gbps": 400,
            "latency_us": 1.5
        },
        {
            "from": "switch-spine-0",
            "to": "switch-core-0",
            "bandwidth_gbps": 800,
            "latency_us": 3.0
        }
    ],

    "tiers": [
        {
            "name": "nvlink",
            "scope": "intra_node",
            "bandwidth_gbps": 600,
            "latency_us": 0.01
        },
        {
            "name": "infiniband",
            "scope": "intra_rack",
            "bandwidth_gbps": 400,
            "latency_us": 1.0
        },
        {
            "name": "optical",
            "scope": "inter_rack",
            "bandwidth_gbps": 800,
            "latency_us": 5.0
        }
    ]
}
```

### Topology File Validation

`nsl check --topology cluster.json` validates:
- All node GPU counts are consistent
- NVLink peer references are bidirectional (if A lists B, B must list A)
- All NIC `connected_to` references resolve to real switch ports
- All links reference existing switches
- No orphan nodes (every node reachable from every other)
- Bandwidth/latency values are positive
- Switch port counts match actual connections

---

## Section 2: Architecture

### Compilation Pipeline

```
cluster.json ──> TopologyParser ──> TopologyGraph
                                        │
                                        ▼
                                   RankPlacer
                                   (assigns ranks to GPUs)
                                        │
                                        ▼
                                   RouteComputer
                                   (computes optimal paths for each collective)
                                        │
                                        ▼
                                   RoutingTable
                                   (serialized into binary)
                                        │
                                        ▼
                              CollectiveEmitter
                              (emits hierarchical all-reduce/all-gather/send-recv)
                                        │
                                        ▼
                                   Cranelift IR
                                   (routing table as const data section)
```

### Topology Graph

```rust
/// Parsed and validated topology representation.
pub struct TopologyGraph {
    pub nodes: Vec<TopologyNode>,
    pub switches: Vec<TopologySwitch>,
    pub links: Vec<TopologyLink>,
    pub tiers: Vec<NetworkTier>,

    // Derived: adjacency + shortest paths
    pub node_to_rack: HashMap<NodeId, RackId>,
    pub rack_to_nodes: HashMap<RackId, Vec<NodeId>>,
    pub shortest_paths: ShortestPathMatrix,
    pub bandwidth_matrix: BandwidthMatrix,
}

pub struct TopologyNode {
    pub id: NodeId,
    pub rack: RackId,
    pub gpus: Vec<GpuInfo>,
    pub nics: Vec<NicInfo>,
}

pub struct GpuInfo {
    pub id: GpuId,
    pub gpu_type: String,
    pub pcie_bus: String,
    pub nvlink_peers: Vec<GpuId>,
    pub nvlink_bw_gbps: f64,
}

pub struct NicInfo {
    pub id: NicId,
    pub nic_type: String,
    pub speed_gbps: f64,
    pub connected_to: PortRef,
}

pub struct TopologySwitch {
    pub id: SwitchId,
    pub switch_type: SwitchTier,   // ToR, Spine, Core
    pub rack: Option<RackId>,      // only for ToR switches
    pub ports: u32,
    pub port_speed_gbps: f64,
    pub latency_us: f64,
}

pub enum SwitchTier {
    ToR,      // Top-of-Rack
    Spine,    // Spine (aggregation)
    Core,     // Core (backbone)
}

pub struct TopologyLink {
    pub from: SwitchId,
    pub to: SwitchId,
    pub bandwidth_gbps: f64,
    pub latency_us: f64,
}

pub struct NetworkTier {
    pub name: String,
    pub scope: TierScope,
    pub bandwidth_gbps: f64,
    pub latency_us: f64,
}

pub enum TierScope {
    IntraNode,     // NVLink
    IntraRack,     // InfiniBand (same ToR switch)
    InterRack,     // Optical (cross spine/core)
}

// Compact IDs for fast lookup
pub type NodeId = u32;
pub type GpuId = u32;
pub type NicId = u32;
pub type SwitchId = u32;
pub type RackId = u32;
```

### Shortest Path and Bandwidth Matrices

```rust
/// Pre-computed shortest path and bandwidth between all node pairs.
pub struct ShortestPathMatrix {
    /// paths[i][j] = ordered list of switches from node i to node j
    paths: Vec<Vec<Vec<SwitchId>>>,
}

pub struct BandwidthMatrix {
    /// bw[i][j] = bottleneck bandwidth (Gbps) on the path from node i to node j
    bw: Vec<Vec<f64>>,
    /// latency[i][j] = total latency (us) on the path from node i to node j
    latency: Vec<Vec<f64>>,
}

impl BandwidthMatrix {
    /// Compute using Dijkstra with bandwidth as edge weight (maximize minimum bandwidth).
    pub fn from_topology(topo: &TopologyGraph) -> Self {
        let n = topo.nodes.len();
        let mut bw = vec![vec![0.0; n]; n];
        let mut latency = vec![vec![f64::MAX; n]; n];

        for i in 0..n {
            // Modified Dijkstra: maximize minimum bandwidth on path
            // (bottleneck shortest path problem)
            let (node_bw, node_lat) = Self::dijkstra_max_bandwidth(topo, i);
            bw[i] = node_bw;
            latency[i] = node_lat;
        }

        // Intra-node: override with NVLink bandwidth
        for node in &topo.nodes {
            for (gi, gpu_a) in node.gpus.iter().enumerate() {
                for (gj, gpu_b) in node.gpus.iter().enumerate() {
                    if gi != gj {
                        let a_idx = gpu_a.id as usize;
                        let b_idx = gpu_b.id as usize;
                        if gpu_a.nvlink_peers.contains(&gpu_b.id) {
                            bw[a_idx][b_idx] = gpu_a.nvlink_bw_gbps;
                            latency[a_idx][b_idx] = 0.01; // NVLink ~10ns
                        }
                    }
                }
            }
        }

        Self { bw, latency }
    }
}
```

---

## Section 3: Rank Placement

### Placement Strategy

The compiler assigns ranks to physical GPUs to minimize communication cost for the given parallelism configuration. The default strategy is:

1. **TP group within a single node** (NVLink): TP requires all-reduce after every layer — highest communication frequency, needs highest bandwidth.
2. **PP stages across nodes within a rack** (InfiniBand): PP requires point-to-point send/recv between stages — moderate frequency, benefits from low latency.
3. **DP replicas across racks** (optical): DP requires all-reduce once per step — lowest frequency, tolerates higher latency.

```rust
pub struct RankPlacer {
    topology: TopologyGraph,
    tp_degree: u32,
    pp_degree: u32,
    dp_degree: u32,
}

pub struct RankPlacement {
    /// rank_to_gpu[rank] = (node_id, gpu_index_within_node)
    pub rank_to_gpu: Vec<(NodeId, u32)>,
    /// gpu_to_rank[(node_id, gpu_index)] = rank
    pub gpu_to_rank: HashMap<(NodeId, u32), u16>,
    /// TP groups: tp_groups[tp_group_id] = [rank0, rank1, ...]
    pub tp_groups: Vec<Vec<u16>>,
    /// PP groups: pp_groups[pp_group_id] = [stage0_rank, stage1_rank, ...]
    pub pp_groups: Vec<Vec<u16>>,
    /// DP groups: dp_groups[dp_group_id] = [replica0_rank, replica1_rank, ...]
    pub dp_groups: Vec<Vec<u16>>,
}

impl RankPlacer {
    pub fn place(&self) -> Result<RankPlacement, PlacementError> {
        let total_ranks = self.tp_degree * self.pp_degree * self.dp_degree;
        let total_gpus: u32 = self.topology.nodes.iter()
            .map(|n| n.gpus.len() as u32)
            .sum();

        if total_ranks > total_gpus {
            return Err(PlacementError::InsufficientGpus {
                needed: total_ranks,
                available: total_gpus,
            });
        }

        let gpus_per_node = self.topology.nodes[0].gpus.len() as u32;

        // Validate TP fits within a single node
        if self.tp_degree > gpus_per_node {
            return Err(PlacementError::TpExceedsNodeSize {
                tp_degree: self.tp_degree,
                gpus_per_node,
            });
        }

        // Phase 1: Assign TP groups to nodes
        // Each TP group uses tp_degree GPUs on a single node
        let tp_groups_per_node = gpus_per_node / self.tp_degree;

        // Phase 2: Assign PP stages to TP groups within the same rack
        // pp_degree consecutive TP groups form a PP pipeline
        let mut rank_to_gpu = vec![(0u32, 0u32); total_ranks as usize];
        let mut rank = 0u16;

        // Sort racks by size (descending) for balanced placement
        let mut racks: Vec<(RackId, Vec<&TopologyNode>)> = self.topology
            .rack_to_nodes.iter()
            .map(|(rack, node_ids)| {
                let nodes: Vec<&TopologyNode> = node_ids.iter()
                    .map(|id| &self.topology.nodes[*id as usize])
                    .collect();
                (*rack, nodes)
            })
            .collect();
        racks.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

        // Assign: iterate DP dimension across racks,
        // PP dimension across nodes within rack,
        // TP dimension within each node
        for dp_idx in 0..self.dp_degree {
            let rack_idx = (dp_idx as usize) % racks.len();
            let (_, ref rack_nodes) = racks[rack_idx];

            for pp_idx in 0..self.pp_degree {
                let node_in_rack = pp_idx as usize % rack_nodes.len();
                let node = rack_nodes[node_in_rack];

                for tp_idx in 0..self.tp_degree {
                    let gpu_idx = tp_idx;
                    rank_to_gpu[rank as usize] = (node.id, gpu_idx);
                    rank += 1;
                }
            }
        }

        Ok(self.build_groups(rank_to_gpu))
    }
}
```

### Placement Strategies

```rust
pub enum PlacementStrategy {
    /// Default: TP within node, PP within rack, DP across racks
    TpIntraSwitch,

    /// Expert-parallel: MoE experts spread across nodes for load balance
    MoeBalanced,

    /// Ring-optimized: physical ring follows NVLink/IB topology
    RingOptimal,

    /// Custom: user provides explicit rank-to-GPU mapping
    Custom(Vec<(NodeId, u32)>),
}
```

---

## Section 4: Hierarchical Collectives

### Three-Tier All-Reduce

Standard flat all-reduce sends data across the slowest link (inter-rack) for every byte. Hierarchical all-reduce exploits the bandwidth tiers:

```
Tier 1 (NVLink, 600 GB/s): Intra-node reduce within each node
Tier 2 (InfiniBand, 400 Gb/s): Inter-node reduce within each rack
Tier 3 (Optical, 800 Gb/s): Cross-rack reduce across racks
Tier 3 → Tier 2 → Tier 1: Broadcast results back down the hierarchy
```

```rust
pub struct HierarchicalAllReduce {
    /// Ranks within the same node (NVLink tier)
    pub intra_node_groups: Vec<Vec<u16>>,
    /// Ranks within the same rack (InfiniBand tier)
    pub intra_rack_groups: Vec<Vec<u16>>,
    /// Cross-rack leaders (optical tier)
    pub cross_rack_leaders: Vec<u16>,
}

impl HierarchicalAllReduce {
    pub fn from_placement(
        placement: &RankPlacement,
        topology: &TopologyGraph,
    ) -> Self {
        // 1. Group ranks by node → intra_node_groups
        let mut intra_node: HashMap<NodeId, Vec<u16>> = HashMap::new();
        for (rank, (node, _gpu)) in placement.rank_to_gpu.iter().enumerate() {
            intra_node.entry(*node).or_default().push(rank as u16);
        }

        // 2. Group node-leaders by rack → intra_rack_groups
        let mut intra_rack: HashMap<RackId, Vec<u16>> = HashMap::new();
        for (node_id, ranks) in &intra_node {
            let rack = topology.node_to_rack[node_id];
            let leader = ranks[0]; // first rank in each node is the leader
            intra_rack.entry(rack).or_default().push(leader);
        }

        // 3. Rack-leaders form cross-rack group
        let cross_rack: Vec<u16> = intra_rack.values()
            .map(|ranks| ranks[0])
            .collect();

        Self {
            intra_node_groups: intra_node.into_values().collect(),
            intra_rack_groups: intra_rack.into_values().collect(),
            cross_rack_leaders: cross_rack,
        }
    }
}
```

### Emitted All-Reduce Code

For a DP gradient all-reduce, the compiler emits:

```
# Phase 1: Intra-node reduce (NVLink)
for each intra_node_group:
    ncclReduceScatter(group_comm, grad, reduced_grad, count/group_size, ncclSum)

# Phase 2: Intra-rack reduce (InfiniBand)
if I am a node leader:
    ncclAllReduce(rack_comm, reduced_grad, reduced_grad, count/group_size, ncclSum)

# Phase 3: Cross-rack reduce (Optical)
if I am a rack leader:
    ncclAllReduce(cross_rack_comm, reduced_grad, reduced_grad, count/group_size, ncclSum)

# Phase 4: Broadcast back down the hierarchy
if I am a rack leader:
    ncclBroadcast(rack_comm, reduced_grad, reduced_grad, count/group_size, 0)
if I am a node leader:
    ncclAllGather(group_comm, reduced_grad, grad, count/group_size)
```

### Ring Attention Path Optimization

M34 Ring Attention sends KV-cache chunks around a ring. The ring ordering should follow physical network topology to minimize cross-rack hops:

```rust
/// Compute optimal ring ordering for ring attention based on physical topology.
pub fn compute_ring_order(
    ring_ranks: &[u16],
    bandwidth_matrix: &BandwidthMatrix,
    placement: &RankPlacement,
) -> Vec<u16> {
    // Greedy nearest-neighbor TSP heuristic on the bandwidth graph.
    // Start from rank 0, always choose the neighbor with highest bandwidth
    // among unvisited ranks.

    let n = ring_ranks.len();
    let mut visited = vec![false; n];
    let mut order = Vec::with_capacity(n);

    let mut current = 0;
    visited[current] = true;
    order.push(ring_ranks[current]);

    for _ in 1..n {
        let current_gpu = placement.rank_to_gpu[ring_ranks[current] as usize];
        let mut best_next = None;
        let mut best_bw = 0.0;

        for j in 0..n {
            if !visited[j] {
                let j_gpu = placement.rank_to_gpu[ring_ranks[j] as usize];
                let bw = bandwidth_matrix.bw[current_gpu.0 as usize][j_gpu.0 as usize];
                if bw > best_bw {
                    best_bw = bw;
                    best_next = Some(j);
                }
            }
        }

        if let Some(next) = best_next {
            visited[next] = true;
            order.push(ring_ranks[next]);
            current = next;
        }
    }

    order
}
```

### Bandwidth-Aware Pipeline Scheduling

For pipeline parallelism, the inter-stage communication bandwidth affects the optimal number of micro-batches. Slower links need more micro-batches to hide communication latency:

```rust
/// Compute optimal micro-batch count based on inter-stage bandwidth.
pub fn compute_microbatch_count(
    pp_group: &[u16],
    placement: &RankPlacement,
    bandwidth_matrix: &BandwidthMatrix,
    activation_size_bytes: u64,
) -> u32 {
    // Find the slowest inter-stage link
    let mut min_bw_gbps = f64::MAX;
    for i in 0..pp_group.len() - 1 {
        let src_gpu = placement.rank_to_gpu[pp_group[i] as usize];
        let dst_gpu = placement.rank_to_gpu[pp_group[i + 1] as usize];
        let bw = bandwidth_matrix.bw[src_gpu.0 as usize][dst_gpu.0 as usize];
        if bw < min_bw_gbps {
            min_bw_gbps = bw;
        }
    }

    // Communication time for one activation transfer
    let comm_time_us = (activation_size_bytes as f64 * 8.0) / (min_bw_gbps * 1e3);

    // Compute time per micro-batch (from cost model, M37)
    let compute_time_us = estimate_compute_time(activation_size_bytes);

    // Optimal micro-batches: enough to hide communication
    // Standard 1F1B: need at least pp_stages micro-batches
    let min_microbatches = pp_group.len() as u32;
    let hiding_microbatches = (comm_time_us / compute_time_us).ceil() as u32 + 1;

    std::cmp::max(min_microbatches, hiding_microbatches)
}
```

---

## Section 5: Compile-Time Routing Table

### Routing Table Structure

The routing table is a const data section embedded in the compiled binary. Each rank reads its entry at startup to configure NCCL communicators and transport selection.

```rust
/// Per-rank routing entry, embedded as const data in the binary.
#[repr(C)]
pub struct RoutingEntry {
    pub rank: u16,
    pub node_id: u16,
    pub gpu_index: u8,
    pub rack_id: u8,
    pub _pad: u16,

    // TP communication
    pub tp_group_id: u16,
    pub tp_group_size: u16,
    pub tp_transport: TransportKind,    // NVLink
    pub tp_peers: [u16; 8],            // max 8 TP peers (0xFFFF = unused)

    // PP communication
    pub pp_group_id: u16,
    pub pp_stage: u16,
    pub pp_transport: TransportKind,    // InfiniBand or NVLink
    pub pp_prev_rank: u16,             // 0xFFFF if first stage
    pub pp_next_rank: u16,             // 0xFFFF if last stage

    // DP communication
    pub dp_group_id: u16,
    pub dp_group_size: u16,
    pub dp_transport: TransportKind,    // Hierarchical
    pub dp_intra_node_leader: bool,
    pub dp_intra_rack_leader: bool,
    pub dp_cross_rack_leader: bool,

    // Ring attention
    pub ring_prev_rank: u16,
    pub ring_next_rank: u16,
    pub ring_transport: TransportKind,

    // Hierarchical all-reduce group IDs
    pub hier_intra_node_group: u16,
    pub hier_intra_rack_group: u16,
    pub hier_cross_rack_group: u16,
}

#[repr(u8)]
pub enum TransportKind {
    NVLink = 0,
    InfiniBand = 1,
    RoCE = 2,
    TCP = 3,
    SharedMem = 4,
}
```

### Embedding in Binary

```rust
/// Emit the routing table as a const data section in Cranelift.
fn emit_routing_table(
    &mut self,
    entries: &[RoutingEntry],
) {
    // Serialize entries to bytes
    let entry_size = std::mem::size_of::<RoutingEntry>();
    let total_size = entries.len() * entry_size;
    let mut data = vec![0u8; total_size];

    for (i, entry) in entries.iter().enumerate() {
        let offset = i * entry_size;
        unsafe {
            std::ptr::copy_nonoverlapping(
                entry as *const RoutingEntry as *const u8,
                data[offset..].as_mut_ptr(),
                entry_size,
            );
        }
    }

    // Emit as Cranelift data section
    let data_id = self.module.declare_data(
        "nsl_routing_table",
        cranelift_module::Linkage::Export,
        false, // not writable
        false, // not TLS
    ).unwrap();

    let mut data_ctx = cranelift_module::DataContext::new();
    data_ctx.define(data.into_boxed_slice());
    self.module.define_data(data_id, &data_ctx).unwrap();
}
```

### Runtime Routing Table Access

```rust
// crates/nsl-runtime/src/routing.rs

extern "C" {
    /// Linker-resolved reference to the embedded routing table.
    static nsl_routing_table: [u8; 0]; // flexible array, actual size known at link time
}

/// Read this rank's routing entry from the embedded table.
pub fn get_my_routing_entry(my_rank: u16, world_size: u32) -> &'static RoutingEntry {
    let entry_size = std::mem::size_of::<RoutingEntry>();
    let offset = my_rank as usize * entry_size;
    unsafe {
        let ptr = nsl_routing_table.as_ptr().add(offset) as *const RoutingEntry;
        &*ptr
    }
}

/// Configure NCCL communicators based on routing table.
pub fn configure_from_routing(entry: &RoutingEntry) -> NcclConfig {
    NcclConfig {
        tp_comm: create_sub_communicator(entry.tp_group_id, entry.tp_group_size, entry.tp_transport),
        pp_prev: if entry.pp_prev_rank != 0xFFFF { Some(entry.pp_prev_rank) } else { None },
        pp_next: if entry.pp_next_rank != 0xFFFF { Some(entry.pp_next_rank) } else { None },
        dp_hier: HierarchicalConfig {
            intra_node_group: entry.hier_intra_node_group,
            intra_rack_group: entry.hier_intra_rack_group,
            cross_rack_group: entry.hier_cross_rack_group,
            is_node_leader: entry.dp_intra_node_leader,
            is_rack_leader: entry.dp_intra_rack_leader,
            is_cross_rack_leader: entry.dp_cross_rack_leader,
        },
    }
}
```

---

## Section 6: Topology Auto-Detection

### `nsl topology detect`

For clusters without a manually-written topology file, the `detect` subcommand queries hardware and generates a topology file:

```rust
// crates/nsl-cli/src/topology.rs

pub fn detect_topology(output_path: &Path) -> Result<(), DetectError> {
    // Phase 1: Detect local node GPUs via NVML
    let gpus = detect_local_gpus()?;

    // Phase 2: Detect NVLink topology via nvmlDeviceGetNvLinkRemotePciInfo
    let nvlink_topo = detect_nvlink_topology(&gpus)?;

    // Phase 3: Detect NICs via ibv_get_device_list (InfiniBand) or ethtool (Ethernet)
    let nics = detect_network_interfaces()?;

    // Phase 4: If multi-node, use MPI/NCCL to exchange topology from all nodes
    let all_nodes = exchange_topology_all_nodes(&gpus, &nvlink_topo, &nics)?;

    // Phase 5: Infer switch topology from subnet information
    let switches = infer_switch_hierarchy(&all_nodes)?;

    // Phase 6: Serialize to JSON
    let topo = TopologySpec::from_detected(all_nodes, switches);
    let json = serde_json::to_string_pretty(&topo)?;
    std::fs::write(output_path, json)?;

    Ok(())
}
```

**Limitations:** Auto-detection cannot determine physical rack placement or inter-rack link bandwidth. These fields are populated with defaults and flagged with `"auto_detected": true` for the user to review and correct.

---

## Section 7: Codegen Changes

### Module: `crates/nsl-codegen/src/topology.rs` (NEW)

```rust
pub struct TopologyRouter {
    topology: TopologyGraph,
    placement: RankPlacement,
    routing_table: Vec<RoutingEntry>,
    hierarchical_allreduce: HierarchicalAllReduce,
}

impl TopologyRouter {
    /// Load topology from JSON, compute placement and routing.
    pub fn new(
        topology_path: &Path,
        tp_degree: u32,
        pp_degree: u32,
        dp_degree: u32,
        strategy: PlacementStrategy,
    ) -> Result<Self, TopologyError> {
        let topology = TopologyParser::parse(topology_path)?;
        let placer = RankPlacer { topology: topology.clone(), tp_degree, pp_degree, dp_degree };
        let placement = placer.place()?;
        let routing_table = Self::compute_routing_table(&topology, &placement, tp_degree, pp_degree, dp_degree);
        let hierarchical_allreduce = HierarchicalAllReduce::from_placement(&placement, &topology);

        Ok(Self { topology, placement, routing_table, hierarchical_allreduce })
    }

    /// Emit the routing table as a const data section.
    pub fn emit_routing_data(&self, compiler: &mut Compiler) {
        compiler.emit_routing_table(&self.routing_table);
    }

    /// Replace flat all-reduce calls with hierarchical all-reduce.
    pub fn rewrite_allreduce(&self, compiler: &mut Compiler, allreduce_sites: &[AllReduceSite]) {
        for site in allreduce_sites {
            compiler.replace_flat_allreduce_with_hierarchical(
                site,
                &self.hierarchical_allreduce,
            );
        }
    }
}
```

### Integration with Existing Codegen

**Module:** `crates/nsl-codegen/src/stmt.rs`

```rust
fn compile_train_block(&mut self, train: &TrainBlock) {
    // ... existing distribute/TP/PP setup ...

    // NEW: If topology is specified, use TopologyRouter
    if let Some(topo_path) = &train.topology {
        let router = TopologyRouter::new(
            topo_path,
            train.tp_degree,
            train.pp_degree,
            train.dp_degree,
            train.placement_strategy.unwrap_or(PlacementStrategy::TpIntraSwitch),
        )?;

        // Emit routing table into binary
        router.emit_routing_data(self);

        // Replace flat all-reduce with hierarchical
        router.rewrite_allreduce(self, &self.allreduce_sites);

        // Optimize ring attention ordering
        if self.uses_ring_attention {
            let ring_order = router.compute_optimal_ring_order();
            self.set_ring_order(ring_order);
        }

        // Adjust pipeline micro-batch count for bandwidth
        if train.pp_degree > 1 {
            let optimal_microbatches = router.compute_optimal_microbatches();
            self.set_microbatch_count(optimal_microbatches);
        }
    }
}
```

### CompileOptions Extension

```rust
pub struct CompileOptions {
    pub no_autotune: bool,
    pub autotune_fresh: bool,
    pub world_size: usize,
    pub fusion_report: bool,
    pub trace: bool,
    pub deterministic: bool,
    pub fault_tolerant: bool,
    pub topology_path: Option<PathBuf>,     // NEW: --topology flag
}
```

### AST Extension

**Module:** `crates/nsl-ast/src/lib.rs`

```rust
pub struct TrainBlock {
    // ... existing fields ...
    pub topology: Option<String>,                  // NEW
    pub placement_strategy: Option<PlacementStrategy>, // NEW
}
```

---

## Section 8: Runtime Changes

### Module: `crates/nsl-runtime/src/routing.rs` (NEW)

```rust
/// Initialize NCCL communicators from the embedded routing table.
#[no_mangle]
pub extern "C" fn nsl_topology_init(
    my_rank: u16,
    world_size: u32,
) -> i32 {
    let entry = get_my_routing_entry(my_rank, world_size);
    let config = configure_from_routing(entry);

    // Create sub-communicators for each communication tier
    // TP comm: NVLink peers
    // PP comm: prev/next stage
    // DP comm: hierarchical all-reduce groups

    TOPOLOGY_STATE.lock().unwrap().replace(config);
    0 // success
}

/// Hierarchical all-reduce using topology-aware sub-communicators.
#[no_mangle]
pub extern "C" fn nsl_topology_allreduce(
    sendbuf: *const c_void,
    recvbuf: *mut c_void,
    count: usize,
    dtype: u8,
) -> i32 {
    let state = TOPOLOGY_STATE.lock().unwrap();
    let config = state.as_ref().unwrap();

    // Phase 1: Intra-node reduce (NVLink)
    nccl_reduce_scatter(config.tp_comm, sendbuf, recvbuf, count / config.tp_size);

    // Phase 2: Intra-rack all-reduce (InfiniBand) — node leaders only
    if config.dp_hier.is_node_leader {
        nccl_allreduce(config.rack_comm, recvbuf, recvbuf, count / config.tp_size);
    }

    // Phase 3: Cross-rack all-reduce (Optical) — rack leaders only
    if config.dp_hier.is_rack_leader {
        nccl_allreduce(config.cross_rack_comm, recvbuf, recvbuf, count / config.tp_size);
    }

    // Phase 4: Broadcast back down
    if config.dp_hier.is_rack_leader {
        nccl_broadcast(config.rack_comm, recvbuf, recvbuf, count / config.tp_size);
    }
    nccl_allgather(config.tp_comm, recvbuf, recvbuf, count / config.tp_size);

    0 // success
}

/// Get the optimal ring order for ring attention.
#[no_mangle]
pub extern "C" fn nsl_topology_get_ring_next(my_rank: u16) -> u16 {
    let entry = get_my_routing_entry(my_rank, WORLD_SIZE.load(Ordering::SeqCst));
    entry.ring_next_rank
}

/// Get the optimal ring predecessor for ring attention.
#[no_mangle]
pub extern "C" fn nsl_topology_get_ring_prev(my_rank: u16) -> u16 {
    let entry = get_my_routing_entry(my_rank, WORLD_SIZE.load(Ordering::SeqCst));
    entry.ring_prev_rank
}
```

---

## Section 9: File Changes

### New Files

| File | Description |
|------|-------------|
| `crates/nsl-codegen/src/topology.rs` | TopologyRouter, RankPlacer, RouteComputer |
| `crates/nsl-runtime/src/routing.rs` | Runtime routing table access, hierarchical all-reduce |
| `crates/nsl-runtime/src/topology_parser.rs` | JSON topology file parser and validator |
| `crates/nsl-cli/src/topology.rs` | `nsl topology detect`, `nsl topology show` subcommands |
| `crates/nsl-semantic/src/topology.rs` | Semantic validation of topology config in train/serve blocks |

### Modified Files

| File | Change |
|------|--------|
| `crates/nsl-parser/src/lib.rs` | Parse `topology:` and `placement_strategy:` in train/serve blocks |
| `crates/nsl-ast/src/lib.rs` | Add `topology` and `placement_strategy` fields to TrainBlock/ServeBlock |
| `crates/nsl-codegen/src/stmt.rs` | Integrate TopologyRouter into train/serve block compilation |
| `crates/nsl-codegen/src/context.rs` | Add `topology_path` to `CompileOptions` |
| `crates/nsl-codegen/src/context_parallel.rs` | Use topology ring ordering for ring attention |
| `crates/nsl-codegen/src/pipeline.rs` | Use topology-aware micro-batch count |
| `crates/nsl-codegen/src/tensor_parallel.rs` | Use topology-aware TP communicator config |
| `crates/nsl-cli/src/main.rs` | Add `--topology` CLI flag, `topology` subcommand |
| `crates/nsl-runtime/src/lib.rs` | Export new FFI functions |
| `crates/nsl-codegen/src/linker.rs` | Link NVML for auto-detection |

---

## Section 10: Testing Strategy

### Unit Tests

**`crates/nsl-runtime/src/topology_parser.rs`:**
- Parse valid cluster.json with 4 racks, 32 nodes, 256 GPUs
- Reject cluster.json with asymmetric NVLink references (A->B but not B->A)
- Reject cluster.json with orphan nodes (unreachable from root switch)
- Reject cluster.json with negative bandwidth values
- Parse topology with optional fields missing (use defaults)

**`crates/nsl-codegen/src/topology.rs`:**
- RankPlacer with tp=8, pp=4, dp=8 on 256 GPUs: TP groups are intra-node
- RankPlacer with tp > gpus_per_node: returns error
- RankPlacer with total_ranks > total_gpus: returns error
- HierarchicalAllReduce groups: node leaders are correct, rack leaders are correct
- Ring ordering follows physical bandwidth (high-BW neighbors are adjacent)
- Bandwidth-aware micro-batch count increases for slower inter-rack PP links

**`crates/nsl-runtime/src/routing.rs`:**
- RoutingEntry serialization/deserialization round-trip
- `get_my_routing_entry` returns correct entry for each rank
- `nsl_topology_allreduce` produces correct results on 4-rank test (simulated)

**`crates/nsl-semantic/src/topology.rs`:**
- `topology: "nonexistent.json"` produces compile error
- `topology:` without `distribute:` produces compile error
- `topology:` with valid file and matching distribute passes

### E2E Tests

- **`examples/m59_topology_basic.nsl`** — train block with topology, verify routing table embedded in binary
- **`examples/m59_hierarchical_allreduce.nsl`** — 4-node cluster, verify hierarchical all-reduce produces same result as flat all-reduce
- **`examples/m59_ring_ordering.nsl`** — ring attention with topology, verify ring follows physical topology
- **`examples/m59_topology_detect.nsl`** — `nsl topology detect` produces valid JSON on local machine

---

## Section 11: Deliverables

- `nsl build --topology cluster.json` reads physical network topology at compile time
- Topology specification format: JSON with nodes, switches, links, bandwidth tiers
- `nsl check --topology` validates topology files without building
- `nsl topology detect` auto-generates topology from hardware queries
- Topology-aware rank placement: TP within node (NVLink), PP within rack (IB), DP across racks
- Hierarchical all-reduce: three-tier (NVLink -> InfiniBand -> optical)
- Ring attention ordering follows physical topology for maximum bandwidth
- Bandwidth-aware pipeline scheduling adjusts micro-batch count for slow links
- Compile-time routing table embedded as const data in binary (zero runtime discovery cost)
- Multiple placement strategies: default, MoE-balanced, ring-optimal, custom

## Not in Scope

- Dynamic topology changes at runtime (cluster reconfiguration without recompile)
- Congestion-aware routing (reacting to real-time network congestion)
- Multi-cluster topology (spanning multiple datacenters)
- QoS-aware scheduling (traffic prioritization between training jobs)
- Topology-aware data placement (co-locating data shards with compute)
- Optical circuit switching (runtime reconfiguration of optical fabric)
- InfiniBand adaptive routing (handled by the switch firmware, not the compiler)
