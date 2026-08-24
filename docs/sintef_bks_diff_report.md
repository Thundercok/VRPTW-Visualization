# SINTEF Benchmark Ground-Truth Audit & Provenance Report

### Provenance & Web Fetch Metadata
- **Source Portal**: SINTEF Top Project Web VRPTW Benchmarks
- **Timestamp (UTC)**: `2026-08-14T12:26:47Z`
- **Fetch Protocol**: `read_url_content agent tool (direct HTML parse from live SINTEF server)`
- **Objective Hierarchy**: `Hierarchical (1. Minimize NV, 2. Minimize TD with double precision L2 norm)`
- **Official Benchmark Instance Coverage**: 176 instances (Solomon: 56, H200: 60, H400: 60)
- **Local `src/vrptw/config.py::BKS` Entries Present**: 176
- **Exact Matches**: 176 / 176 (100.0%)
- **Mismatches (|ΔTD| > 0.01% or NV mismatch)**: 0
- **Missing in Local BKS**: 0

---

## 1. Ground-Truth Verification for Specific Flagged Instances

| Instance | SINTEF NV | SINTEF TD | Ref Code | Primary Citation | Official Comment | Source URL |
|---|---|---|---|---|---|---|
| **RC101** | 14 | 1696.95 | `TBGGP` | E. Taillard, P. Badeau, M. Gendreau, F. Geurtin, and J.Y. Potvin, "A Tabu Search Heuristic for VRPTW", Transportation Science 31, 170-186, 1997. | 1696.94 reported by TBGGP is believed to result from a rounding error | [SINTEF Solomon 100](https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/100-customers/) |
| **RC102** | 12 | 1554.75 | `TBGGP` | E. Taillard, P. Badeau, M. Gendreau, F. Geurtin, and J.Y. Potvin, "A Tabu Search Heuristic for VRPTW", Transportation Science 31, 170-186, 1997. | Detailed solution by Shobb | [SINTEF Solomon 100](https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/100-customers/) |
| **RC105** | 13 | 1629.44 | `BBB` | J. Berger, M. Barkaoui and O. Bräysy, "A Parallel Hybrid Genetic Algorithm for VRPTW", DREV Canada, 2001. | Detailed solution from BVH | [SINTEF Solomon 100](https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/100-customers/) |
| **RC106** | 11 | 1424.73 | `BBB` | J. Berger, M. Barkaoui and O. Bräysy, "A Parallel Hybrid Genetic Algorithm for VRPTW", DREV Canada, 2001. | Detailed solution from BVH | [SINTEF Solomon 100](https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/100-customers/) |
| **RC201** | 4 | 1406.94 | `MBD` | D. Mester, O. Bräysy and W. Dullaert, "A Multi-parametric Evolution Strategies Algorithm for VRP", University of Haifa, 2005. | See note below | [SINTEF Solomon 100](https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/100-customers/) |
| **RC202** | 3 | 1365.65 | `GCC` | Agnieszka Debudaj-Grabysz, Zbigniew J. Czech and Piotr Czarnas, Silesia University of Technology & University of Wroclaw, 2004. | Detailed solution by Victor Allis | [SINTEF Solomon 100](https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/100-customers/) |
| **RC205** | 4 | 1297.65 | `MBD` | D. Mester, O. Bräysy and W. Dullaert, "A Multi-parametric Evolution Strategies Algorithm for VRP", University of Haifa, 2005. | See note below | [SINTEF Solomon 100](https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/100-customers/) |
| **R211**  | 2 | 885.71 | `WL` | M. Woch, P. Lebkowski, "Sequential Simulated Annealing for VRPTW", Decision Making in Manufacturing and Services 3, 87-100, 2009. | Detailed solution by Victor Allis | [SINTEF Solomon 100](https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/100-customers/) |

> [!IMPORTANT]
> **Hierarchical Objective Clarification for RC202**:
> On the live SINTEF portal (`https://www.sintef.no/projectweb/top/vrptw/solomon-benchmark/100-customers/`),
> RC202 is officially cataloged as **NV=3, TD=1365.65** (Reference: GCC — Debudaj-Grabysz, Czech & Czarnas 2004, detailed solution by Victor Allis).
> While non-hierarchical/relaxed heuristics allowing NV=4 can achieve TD=1153.84, our benchmark strictly adheres to SINTEF's primary vehicle minimization hierarchy ($NV=3$).

---

## 2. Complete Audit Summary across All 176 Benchmark Instances

✅ **100% Exact Parity**: All 176 instances in `src/vrptw/config.py::BKS` have zero mismatches with official published SINTEF ground truth.