// velocity_data -- nanobind module: load + re-block the ICON R02B05 velocity_tendencies
// artifact data for the dace-fortran velocity samples.
//
//   load(data_dir, timestep=1)  -> dict of {flat_name: numpy array (Fortran order), scalars, "lbounds"}
//   reblock(arrays, nproma=32)  -> same dict re-blocked to the new nproma
//   selftest()                  -> parse + reblock on a synthetic nproma=8 fixture (throws on mismatch)
//
// The parser is the vendored struct-aware serde reader (third_party/serde_velocity.hpp,
// from VelocityTendenciesPipeline); the per-struct read pattern below follows
// VelocityTendenciesPipeline/main.cpp (open_ifstream / read<T> / t0_t1_pair, t0 half only).
// Flat names are the harness c_f_pointer locals of tests/icon/ocean/_ocean_e2e.py
// (scheme <dummy>_<component>[_<sub>], see velocity_tendencies_binding.json).
//
// Re-blocking math (nproma_old -> nproma_new), applied to dim0/dim2 of every
// nproma-carrying array: global 0-based id g = (blk-1)*np_old + (idx-1);
// new_idx = g % np_new + 1, new_blk = g / np_new + 1; nblks_new = ceil(valid/np_new).
// Trailing pad of the last block: 0 for data arrays, a replicated valid element for
// idx/blk connectivity tables (gathers stay in-bounds).  Refin-ctrl start/end
// index/block arrays are value-recomputed with the same g mapping (extent unchanged).

// nanobind (and through it Python.h) must precede every system header, else glibc's
// feature macros are set first and pyconfig.h redefines _POSIX_C_SOURCE/_XOPEN_SOURCE.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "third_party/serde_velocity.hpp"

namespace nb = nanobind;
using namespace nb::literals;
namespace fs = std::filesystem;

namespace {

// ---------------------------------------------------------------------------
// ndarray helpers
// ---------------------------------------------------------------------------

// Wrap a serde-allocated (new[]) buffer as a Fortran-ordered numpy array, zero-copy;
// the capsule owns the buffer.  nanobind strides are in ELEMENTS.
template <typename T>
nb::object make_f_array(T* buf, const std::vector<std::size_t>& shape) {
  nb::capsule owner(buf, [](void* p) noexcept { delete[] static_cast<T*>(p); });
  std::vector<std::int64_t> strides(shape.size());
  std::int64_t s = 1;
  for (std::size_t i = 0; i < shape.size(); ++i) {
    strides[i] = s;
    s *= static_cast<std::int64_t>(shape[i]);
  }
  nb::ndarray<nb::numpy, T> arr(buf, shape.size(), shape.data(), owner, strides.data());
  return nb::cast(std::move(arr));
}

[[nodiscard]] bool is_f_contiguous(const nb::ndarray<>& a) {
  std::int64_t s = 1;
  for (std::size_t i = 0; i < a.ndim(); ++i) {
    if (a.shape(i) > 1 && a.stride(i) != s) return false;
    s *= static_cast<std::int64_t>(a.shape(i));
  }
  return true;
}

nb::ndarray<> get_array(const nb::dict& d, const char* name) {
  auto arr = nb::cast<nb::ndarray<>>(d[name]);
  if (!is_f_contiguous(arr)) throw std::runtime_error(std::string(name) + ": expected a Fortran-contiguous array");
  return arr;
}

// ---------------------------------------------------------------------------
// load: artifact .data files -> flat-name dict
// ---------------------------------------------------------------------------

// From VelocityTendenciesPipeline/main.cpp::open_ifstream, exit() -> throw.
std::ifstream open_data(const fs::path& root, const std::string& name, int timestep) {
  const fs::path p = root / (name + "." + std::to_string(timestep) + ".data");
  if (!fs::exists(p)) throw std::runtime_error("missing data file: " + p.string());
  return std::ifstream{p};
}

template <typename T>
T read_scalar_file(const fs::path& root, const std::string& name, int timestep) {
  auto s = open_data(root, name, timestep);
  T value{};
  serde::read_scalar(value, s);
  return value;
}

const serde::array_meta& meta_of(const void* p, const char* flat) {
  auto it = serde::ARRAY_META_DICT()->find(const_cast<void*>(p));
  if (it == serde::ARRAY_META_DICT()->end())
    throw std::runtime_error(std::string(flat) + ": no serde metadata recorded");
  return it->second;
}

template <typename T>
void put_array(nb::dict& d, nb::dict& lbounds, const char* flat, T* p) {
  if (p == nullptr)
    throw std::runtime_error(std::string(flat) +
                             ": member absent in the artifact data "
                             "(did download_data.sh apply the is_associated patch?)");
  const serde::array_meta& m = meta_of(p, flat);
  std::vector<std::size_t> shape(m.size.begin(), m.size.end());
  nb::list lb;
  for (int b : m.lbound) lb.append(b);
  lbounds[flat] = nb::tuple(lb);
  d[flat] = make_f_array(p, shape);
}

nb::dict load(const std::string& data_dir, int timestep) {
  const fs::path root(data_dir);
  if (!fs::is_directory(root)) throw std::runtime_error("not a directory: " + data_dir);

  t_patch patch{};
  t_int_state p_int{};
  t_nh_diag p_diag{};
  t_nh_metrics p_metrics{};
  t_nh_prog p_prog{};
  global_data_type gdata{};
  std::pair<serde::array_meta, double*> z_kin{}, z_vt{}, z_wc{};
  int istep = 0, ldeepatmo = 0, lvn_only = 0, ntnd = 0;
  double dtime = 0.0, dt_linintp_ubc = 0.0;

  {
    // Pure C++ file parsing; no Python objects touched.  Sequential on purpose:
    // main.cpp parallelizes per-file but races on the shared ARRAY_META_DICT map.
    nb::gil_scoped_release rel;
    {
      auto s = open_data(root, "p_patch", timestep);
      serde::deserialize(&patch, s);
    }
    {
      auto s = open_data(root, "p_int", timestep);
      serde::deserialize(&p_int, s);
    }
    {
      auto s = open_data(root, "p_diag.t0", timestep);
      serde::deserialize(&p_diag, s);
    }
    {
      auto s = open_data(root, "p_metrics.t0", timestep);
      serde::deserialize(&p_metrics, s);
    }
    {
      auto s = open_data(root, "p_prog.t0", timestep);
      serde::deserialize(&p_prog, s);
    }
    {
      auto s = open_data(root, "global_data.t0", timestep);
      serde::deserialize_global_data(&gdata, s);
    }
    {
      auto s = open_data(root, "z_kin_hor_e.t0", timestep);
      z_kin = serde::read_array<double>(s);
    }
    {
      auto s = open_data(root, "z_vt_ie.t0", timestep);
      z_vt = serde::read_array<double>(s);
    }
    {
      auto s = open_data(root, "z_w_concorr_me.t0", timestep);
      z_wc = serde::read_array<double>(s);
    }
    istep = read_scalar_file<int>(root, "istep", timestep);
    ldeepatmo = read_scalar_file<int>(root, "ldeepatmo", timestep);
    lvn_only = read_scalar_file<int>(root, "lvn_only", timestep);
    ntnd = read_scalar_file<int>(root, "ntnd", timestep);
    dtime = read_scalar_file<double>(root, "dtime", timestep);
    dt_linintp_ubc = read_scalar_file<double>(root, "dt_linintp_ubc", timestep);
  }

  nb::dict d;
  nb::dict lb;

  // p_patch % cells (+ nested decomp_info)
  put_array(d, lb, "p_patch_cells_area", patch.cells->area);
  put_array(d, lb, "p_patch_cells_decomp_info_owner_mask", patch.cells->decomp_info->owner_mask);
  put_array(d, lb, "p_patch_cells_edge_blk", patch.cells->edge_blk);
  put_array(d, lb, "p_patch_cells_edge_idx", patch.cells->edge_idx);
  put_array(d, lb, "p_patch_cells_end_block", patch.cells->end_block);
  put_array(d, lb, "p_patch_cells_end_index", patch.cells->end_index);
  put_array(d, lb, "p_patch_cells_neighbor_blk", patch.cells->neighbor_blk);
  put_array(d, lb, "p_patch_cells_neighbor_idx", patch.cells->neighbor_idx);
  put_array(d, lb, "p_patch_cells_start_block", patch.cells->start_block);
  put_array(d, lb, "p_patch_cells_start_index", patch.cells->start_index);
  // p_patch % edges
  put_array(d, lb, "p_patch_edges_area_edge", patch.edges->area_edge);
  put_array(d, lb, "p_patch_edges_cell_blk", patch.edges->cell_blk);
  put_array(d, lb, "p_patch_edges_cell_idx", patch.edges->cell_idx);
  put_array(d, lb, "p_patch_edges_end_block", patch.edges->end_block);
  put_array(d, lb, "p_patch_edges_end_index", patch.edges->end_index);
  put_array(d, lb, "p_patch_edges_f_e", patch.edges->f_e);
  put_array(d, lb, "p_patch_edges_fn_e", patch.edges->fn_e);
  put_array(d, lb, "p_patch_edges_ft_e", patch.edges->ft_e);
  put_array(d, lb, "p_patch_edges_inv_dual_edge_length", patch.edges->inv_dual_edge_length);
  put_array(d, lb, "p_patch_edges_inv_primal_edge_length", patch.edges->inv_primal_edge_length);
  put_array(d, lb, "p_patch_edges_quad_blk", patch.edges->quad_blk);
  put_array(d, lb, "p_patch_edges_quad_idx", patch.edges->quad_idx);
  put_array(d, lb, "p_patch_edges_start_block", patch.edges->start_block);
  put_array(d, lb, "p_patch_edges_start_index", patch.edges->start_index);
  put_array(d, lb, "p_patch_edges_tangent_orientation", patch.edges->tangent_orientation);
  put_array(d, lb, "p_patch_edges_vertex_blk", patch.edges->vertex_blk);
  put_array(d, lb, "p_patch_edges_vertex_idx", patch.edges->vertex_idx);
  // p_patch % verts
  put_array(d, lb, "p_patch_verts_cell_blk", patch.verts->cell_blk);
  put_array(d, lb, "p_patch_verts_cell_idx", patch.verts->cell_idx);
  put_array(d, lb, "p_patch_verts_edge_blk", patch.verts->edge_blk);
  put_array(d, lb, "p_patch_verts_edge_idx", patch.verts->edge_idx);
  put_array(d, lb, "p_patch_verts_end_block", patch.verts->end_block);
  put_array(d, lb, "p_patch_verts_end_index", patch.verts->end_index);
  put_array(d, lb, "p_patch_verts_start_block", patch.verts->start_block);
  put_array(d, lb, "p_patch_verts_start_index", patch.verts->start_index);
  // p_int
  put_array(d, lb, "p_int_c_lin_e", p_int.c_lin_e);
  put_array(d, lb, "p_int_cells_aw_verts", p_int.cells_aw_verts);
  put_array(d, lb, "p_int_e_bln_c_s", p_int.e_bln_c_s);
  put_array(d, lb, "p_int_geofac_grdiv", p_int.geofac_grdiv);
  put_array(d, lb, "p_int_geofac_n2s", p_int.geofac_n2s);
  put_array(d, lb, "p_int_geofac_rot", p_int.geofac_rot);
  put_array(d, lb, "p_int_rbf_vec_coeff_e", p_int.rbf_vec_coeff_e);
  // p_metrics
  put_array(d, lb, "p_metrics_coeff1_dwdz", p_metrics.coeff1_dwdz);
  put_array(d, lb, "p_metrics_coeff2_dwdz", p_metrics.coeff2_dwdz);
  put_array(d, lb, "p_metrics_coeff_gradekin", p_metrics.coeff_gradekin);
  put_array(d, lb, "p_metrics_ddqz_z_full_e", p_metrics.ddqz_z_full_e);
  put_array(d, lb, "p_metrics_ddqz_z_half", p_metrics.ddqz_z_half);
  put_array(d, lb, "p_metrics_ddxn_z_full", p_metrics.ddxn_z_full);
  put_array(d, lb, "p_metrics_ddxt_z_full", p_metrics.ddxt_z_full);
  put_array(d, lb, "p_metrics_deepatmo_gradh_ifc", p_metrics.deepatmo_gradh_ifc);
  put_array(d, lb, "p_metrics_deepatmo_gradh_mc", p_metrics.deepatmo_gradh_mc);
  put_array(d, lb, "p_metrics_deepatmo_invr_ifc", p_metrics.deepatmo_invr_ifc);
  put_array(d, lb, "p_metrics_deepatmo_invr_mc", p_metrics.deepatmo_invr_mc);
  put_array(d, lb, "p_metrics_wgtfac_c", p_metrics.wgtfac_c);
  put_array(d, lb, "p_metrics_wgtfac_e", p_metrics.wgtfac_e);
  put_array(d, lb, "p_metrics_wgtfacq_e", p_metrics.wgtfacq_e);
  // p_diag (t0 state; ddt_vn_cor_pc / vn_ie_ubc are patched OUT of the artifact set --
  // convert_data.py synthesizes them, see there)
  put_array(d, lb, "p_diag_ddt_vn_apc_pc", p_diag.ddt_vn_apc_pc);
  put_array(d, lb, "p_diag_ddt_w_adv_pc", p_diag.ddt_w_adv_pc);
  put_array(d, lb, "p_diag_vn_ie", p_diag.vn_ie);
  put_array(d, lb, "p_diag_vt", p_diag.vt);
  put_array(d, lb, "p_diag_w_concorr_c", p_diag.w_concorr_c);
  // p_prog + top-level z arrays
  put_array(d, lb, "p_prog_vn", p_prog.vn);
  put_array(d, lb, "p_prog_w", p_prog.w);
  put_array(d, lb, "z_kin_hor_e", z_kin.second);
  put_array(d, lb, "z_vt_ie", z_vt.second);
  put_array(d, lb, "z_w_concorr_me", z_wc.second);
  // global_data module arrays (per-domain namelist config)
  put_array(d, lb, "nflatlev", gdata.nflatlev);
  put_array(d, lb, "nrdmax", gdata.nrdmax);

  // scalars
  d["nproma"] = gdata.nproma;
  d["nblks_c"] = patch.nblks_c;
  d["nblks_e"] = patch.nblks_e;
  d["nblks_v"] = patch.nblks_v;
  d["lextra_diffu"] = gdata.lextra_diffu;
  d["istep"] = istep;
  d["ldeepatmo"] = ldeepatmo;
  d["lvn_only"] = lvn_only;
  d["ntnd"] = ntnd;
  d["dtime"] = dtime;
  d["dt_linintp_ubc"] = dt_linintp_ubc;
  d["p_diag_max_vcfl_dyn"] = p_diag.max_vcfl_dyn;
  d["lbounds"] = lb;

  // Struct shells: the array buffers are owned by capsules now, the shells leak
  // a few hundred bytes -- free them.
  delete patch.cells->decomp_info;
  delete patch.cells;
  delete patch.edges;
  delete patch.verts;

  return d;
}

// ---------------------------------------------------------------------------
// reblock: nproma_old -> nproma_new
// ---------------------------------------------------------------------------

enum Space : std::uint8_t { SP_C = 0, SP_E = 1, SP_V = 2 };

struct DataSpec {
  const char* name;
  int space;
  int proma_dim;
  int block_dim;
};

struct PairSpec {
  const char* idx;
  const char* blk;
  int owner;
  int target;
};

struct RefinSpec {
  const char* index;
  const char* block;
  int space;
};

// Every nproma-carrying data array in velocity_tendencies_binding.json.
// (proma_dim, block_dim) are fixed per array; rbf_vec_coeff_e is the one (4, nproma,
// nblks_e) outlier.  ddt_vn_cor_pc / vn_ie_ubc are synthesized post-reblock but listed
// so a dict that does carry them re-blocks correctly.
constexpr DataSpec DATA_SPECS[] = {
    {"p_patch_cells_area", SP_C, 0, 1},
    {"p_patch_cells_decomp_info_owner_mask", SP_C, 0, 1},
    {"p_int_e_bln_c_s", SP_C, 0, 2},
    {"p_int_geofac_n2s", SP_C, 0, 2},
    {"p_metrics_coeff1_dwdz", SP_C, 0, 2},
    {"p_metrics_coeff2_dwdz", SP_C, 0, 2},
    {"p_metrics_ddqz_z_half", SP_C, 0, 2},
    {"p_metrics_wgtfac_c", SP_C, 0, 2},
    {"p_diag_ddt_w_adv_pc", SP_C, 0, 2},
    {"p_diag_w_concorr_c", SP_C, 0, 2},
    {"p_prog_w", SP_C, 0, 2},
    {"p_diag_ddt_vn_apc_pc", SP_E, 0, 2},
    {"p_diag_ddt_vn_cor_pc", SP_E, 0, 2},
    {"p_diag_vn_ie", SP_E, 0, 2},
    {"p_diag_vn_ie_ubc", SP_E, 0, 2},
    {"p_diag_vt", SP_E, 0, 2},
    {"p_int_c_lin_e", SP_E, 0, 2},
    {"p_int_geofac_grdiv", SP_E, 0, 2},
    {"p_int_rbf_vec_coeff_e", SP_E, 1, 2},
    {"p_metrics_coeff_gradekin", SP_E, 0, 2},
    {"p_metrics_ddqz_z_full_e", SP_E, 0, 2},
    {"p_metrics_ddxn_z_full", SP_E, 0, 2},
    {"p_metrics_ddxt_z_full", SP_E, 0, 2},
    {"p_metrics_wgtfac_e", SP_E, 0, 2},
    {"p_metrics_wgtfacq_e", SP_E, 0, 2},
    {"p_patch_edges_area_edge", SP_E, 0, 1},
    {"p_patch_edges_f_e", SP_E, 0, 1},
    {"p_patch_edges_fn_e", SP_E, 0, 1},
    {"p_patch_edges_ft_e", SP_E, 0, 1},
    {"p_patch_edges_inv_dual_edge_length", SP_E, 0, 1},
    {"p_patch_edges_inv_primal_edge_length", SP_E, 0, 1},
    {"p_patch_edges_tangent_orientation", SP_E, 0, 1},
    {"p_prog_vn", SP_E, 0, 2},
    {"z_kin_hor_e", SP_E, 0, 2},
    {"z_vt_ie", SP_E, 0, 2},
    {"z_w_concorr_me", SP_E, 0, 2},
    {"p_int_cells_aw_verts", SP_V, 0, 2},
    {"p_int_geofac_rot", SP_V, 0, 2},
};

// Split (idx, blk) connectivity tables: (owner space, degree, TARGET space), shape
// (nproma, nblks_owner, degree), both 1-based; gather is A[idx-1, jk, blk-1].
constexpr PairSpec PAIR_SPECS[] = {
    {"p_patch_cells_neighbor_idx", "p_patch_cells_neighbor_blk", SP_C, SP_C},
    {"p_patch_cells_edge_idx", "p_patch_cells_edge_blk", SP_C, SP_E},
    {"p_patch_edges_cell_idx", "p_patch_edges_cell_blk", SP_E, SP_C},
    {"p_patch_edges_vertex_idx", "p_patch_edges_vertex_blk", SP_E, SP_V},
    {"p_patch_edges_quad_idx", "p_patch_edges_quad_blk", SP_E, SP_E},
    {"p_patch_verts_cell_idx", "p_patch_verts_cell_blk", SP_V, SP_C},
    {"p_patch_verts_edge_idx", "p_patch_verts_edge_blk", SP_V, SP_E},
};

constexpr RefinSpec REFIN_SPECS[] = {
    {"p_patch_cells_start_index", "p_patch_cells_start_block", SP_C},
    {"p_patch_cells_end_index", "p_patch_cells_end_block", SP_C},
    {"p_patch_edges_start_index", "p_patch_edges_start_block", SP_E},
    {"p_patch_edges_end_index", "p_patch_edges_end_block", SP_E},
    {"p_patch_verts_start_index", "p_patch_verts_start_block", SP_V},
    {"p_patch_verts_end_index", "p_patch_verts_end_block", SP_V},
};

constexpr const char* SPACE_NAMES[] = {"cells", "edges", "verts"};
constexpr const char* NBLKS_KEYS[] = {"nblks_c", "nblks_e", "nblks_v"};
constexpr const char* VALID_KEYS[] = {"valid_cells", "valid_edges", "valid_verts"};

// Valid element count of one space: the interior-most refin-ctrl zone ends at the
// last valid element, so max over levels of (end_block-1)*nproma + end_index (levels
// with end_block/end_index < 1 are empty).
std::int64_t derive_valid(const nb::ndarray<>& end_index, const nb::ndarray<>& end_block, std::int64_t np_old) {
  if (end_index.ndim() != 1 || end_block.ndim() != 1 || end_index.shape(0) != end_block.shape(0))
    throw std::runtime_error("refin-ctrl end_index/end_block extent mismatch");
  const int* ei = static_cast<const int*>(end_index.data());
  const int* eb = static_cast<const int*>(end_block.data());
  std::int64_t best = 0;
  for (std::size_t k = 0; k < end_index.shape(0); ++k) {
    if (eb[k] < 1 || ei[k] < 1) continue;
    const std::int64_t total = static_cast<std::int64_t>(eb[k] - 1) * np_old + ei[k];
    if (total > best) best = total;
  }
  if (best <= 0) throw std::runtime_error("refin-ctrl arrays yield no valid elements");
  return best;
}

template <typename T>
struct Raw {
  std::unique_ptr<T[]> buf;  // released into the ndarray capsule once fully built
  std::vector<std::size_t> shape;
  std::int64_t volume;
};

// Position remap of the (proma_dim, block_dim) plane by the g mapping; the remaining
// dims are copied per element.  Pad past `valid`: replicate element valid-1
// (replicate_pad) or zero-fill.
template <typename T>
Raw<T> remap_raw(const nb::ndarray<>& src, int proma_dim, int block_dim, std::int64_t np_old, std::int64_t np_new,
                 std::int64_t nb_new, std::int64_t valid, bool replicate_pad, const char* name) {
  const std::size_t rank = src.ndim();
  std::vector<std::size_t> shp(rank);
  std::vector<std::int64_t> istr(rank);
  for (std::size_t i = 0; i < rank; ++i) {
    shp[i] = src.shape(i);
    istr[i] = src.stride(i);
  }
  const auto pd = static_cast<std::size_t>(proma_dim);
  const auto bd = static_cast<std::size_t>(block_dim);
  if (pd >= rank || bd >= rank || static_cast<std::int64_t>(shp[pd]) != np_old)
    throw std::runtime_error(std::string(name) + ": shape does not match nproma_old on the expected dim");
  if (valid > np_old * static_cast<std::int64_t>(shp[bd]))
    throw std::runtime_error(std::string(name) + ": derived valid count exceeds the array extent");

  std::vector<std::size_t> out_shape = shp;
  out_shape[pd] = static_cast<std::size_t>(np_new);
  out_shape[bd] = static_cast<std::size_t>(nb_new);
  std::vector<std::int64_t> ostr(rank);
  std::int64_t vol = 1;
  for (std::size_t i = 0; i < rank; ++i) {
    ostr[i] = vol;
    vol *= static_cast<std::int64_t>(out_shape[i]);
  }
  // dims other than (proma, block): iterated as one flattened rest loop
  std::vector<std::pair<std::int64_t, std::pair<std::int64_t, std::int64_t>>> rest;  // extent, (istride, ostride)
  std::int64_t rest_vol = 1;
  for (std::size_t i = 0; i < rank; ++i) {
    if (i == pd || i == bd) continue;
    rest.push_back({static_cast<std::int64_t>(shp[i]), {istr[i], ostr[i]}});
    rest_vol *= static_cast<std::int64_t>(shp[i]);
  }

  std::unique_ptr<T[]> out(new T[static_cast<std::size_t>(vol)]);
  const T* in = static_cast<const T*>(src.data());
  const std::int64_t slots = np_new * nb_new;
  for (std::int64_t slot = 0; slot < slots; ++slot) {
    const std::int64_t sg = slot < valid ? slot : (replicate_pad ? valid - 1 : -1);
    const std::int64_t d_base = (slot % np_new) * ostr[pd] + (slot / np_new) * ostr[bd];
    const std::int64_t s_base = sg >= 0 ? (sg % np_old) * istr[pd] + (sg / np_old) * istr[bd] : 0;
    for (std::int64_t r = 0; r < rest_vol; ++r) {
      std::int64_t rr = r, so = s_base, dofs = d_base;
      for (const auto& [extent, strides] : rest) {
        const std::int64_t k = rr % extent;
        rr /= extent;
        so += k * strides.first;
        dofs += k * strides.second;
      }
      out[dofs] = sg >= 0 ? in[so] : T(0);
    }
  }
  return {std::move(out), std::move(out_shape), vol};
}

// Rewrite (idx, blk) VALUES into the target space's new blocking.  Slots with
// idx/blk < 1 (unused pentagon-point slots) are preserved as-is.
void rewrite_pair_values(int* idx, int* blk, std::int64_t n, std::int64_t np_old, std::int64_t np_new,
                         std::int64_t valid_target, const char* name) {
  for (std::int64_t i = 0; i < n; ++i) {
    const int iv = idx[i];
    const int bv = blk[i];
    if (iv < 1 || bv < 1) continue;
    const std::int64_t g = static_cast<std::int64_t>(bv - 1) * np_old + iv - 1;
    if (g >= valid_target)
      throw std::runtime_error(std::string(name) + ": connectivity entry past the valid target range");
    idx[i] = static_cast<int>(g % np_new + 1);
    blk[i] = static_cast<int>(g / np_new + 1);
  }
}

// Refin-ctrl (index, block) value recompute.  The linear element order is invariant
// under re-blocking, so each zone boundary maps by the same g formula; g = -1 ("one
// before the first element", ICON's empty-zone end encoding) maps to (idx 0, blk 1).
void rewrite_refin_values(int* idx, int* blk, std::int64_t n, std::int64_t np_old, std::int64_t np_new) {
  for (std::int64_t i = 0; i < n; ++i) {
    std::int64_t g = static_cast<std::int64_t>(blk[i] - 1) * np_old + idx[i] - 1;
    if (g < -1) g = -1;
    const std::int64_t nb_ = g >= 0 ? g / np_new + 1 : 1;
    blk[i] = static_cast<int>(nb_);
    idx[i] = static_cast<int>(g - (nb_ - 1) * np_new + 1);
  }
}

template <typename T>
nb::object remap_data_typed(const nb::ndarray<>& src, const DataSpec& spec, std::int64_t np_old, std::int64_t np_new,
                            std::int64_t nb_new, std::int64_t valid) {
  Raw<T> r = remap_raw<T>(src, spec.proma_dim, spec.block_dim, np_old, np_new, nb_new, valid, false, spec.name);
  return make_f_array(r.buf.release(), r.shape);
}

nb::dict reblock(const nb::dict& arrays, std::int64_t nproma) {
  const std::int64_t np_new = nproma;
  if (np_new < 1) throw std::runtime_error("nproma must be >= 1");
  if (!arrays.contains("nproma"))
    throw std::runtime_error("arrays dict carries no 'nproma' scalar (load() provides it)");
  const auto np_old = nb::cast<std::int64_t>(arrays["nproma"]);

  // Valid counts per space from the refin-ctrl end arrays; a space whose refin
  // arrays are absent has no count and none of its arrays may be present.
  bool have[3] = {false, false, false};
  std::int64_t valid[3] = {0, 0, 0};
  std::int64_t nb_new[3] = {0, 0, 0};
  for (const RefinSpec& rs : REFIN_SPECS) {
    if (std::string_view(rs.index).find("end_index") == std::string_view::npos || !arrays.contains(rs.index)) continue;
    valid[rs.space] = derive_valid(get_array(arrays, rs.index), get_array(arrays, rs.block), np_old);
    nb_new[rs.space] = (valid[rs.space] + np_new - 1) / np_new;
    have[rs.space] = true;
  }
  // Cross-check against the true icosahedral R02B05 element counts whenever the
  // input is the 80k artifact set (nproma 81920); other sets (nproma32 debug,
  // selftest fixtures) have other counts and skip this gate.
  if (np_old == 81920) {
    constexpr std::int64_t R02B05[3] = {81920, 122880, 40962};
    for (int sp = 0; sp < 3; ++sp)
      if (have[sp] && valid[sp] != R02B05[sp])
        throw std::runtime_error(std::string("R02B05 cross-check failed for ") + SPACE_NAMES[sp] + ": derived " +
                                 std::to_string(valid[sp]) + ", expected " + std::to_string(R02B05[sp]));
  }

  nb::dict out;
  for (auto [k, v] : arrays)  // default: passthrough (scalars, lbounds, no-nproma arrays)
    out[k] = v;

  for (const DataSpec& spec : DATA_SPECS) {
    if (!arrays.contains(spec.name)) continue;
    if (!have[spec.space])
      throw std::runtime_error(std::string(spec.name) + ": no refin-ctrl arrays for its space in the dict");
    nb::ndarray<> src = get_array(arrays, spec.name);
    const std::int64_t vd = valid[spec.space], nbn = nb_new[spec.space];
    if (src.dtype() == nb::dtype<double>())
      out[spec.name] = remap_data_typed<double>(src, spec, np_old, np_new, nbn, vd);
    else if (src.dtype() == nb::dtype<int>())
      out[spec.name] = remap_data_typed<int>(src, spec, np_old, np_new, nbn, vd);
    else if (src.dtype() == nb::dtype<std::int8_t>())
      out[spec.name] = remap_data_typed<std::int8_t>(src, spec, np_old, np_new, nbn, vd);
    else
      throw std::runtime_error(std::string(spec.name) + ": unsupported dtype");
  }

  for (const PairSpec& spec : PAIR_SPECS) {
    if (!arrays.contains(spec.idx)) continue;
    if (!have[spec.owner] || !have[spec.target])
      throw std::runtime_error(std::string(spec.idx) + ": missing refin-ctrl arrays for owner/target space");
    nb::ndarray<> src_i = get_array(arrays, spec.idx);
    nb::ndarray<> src_b = get_array(arrays, spec.blk);
    if (src_i.dtype() != nb::dtype<int>() || src_b.dtype() != nb::dtype<int>())
      throw std::runtime_error(std::string(spec.idx) + ": connectivity tables must be int32");
    Raw<int> ri = remap_raw<int>(src_i, 0, 1, np_old, np_new, nb_new[spec.owner], valid[spec.owner], true, spec.idx);
    Raw<int> rb = remap_raw<int>(src_b, 0, 1, np_old, np_new, nb_new[spec.owner], valid[spec.owner], true, spec.blk);
    rewrite_pair_values(ri.buf.get(), rb.buf.get(), ri.volume, np_old, np_new, valid[spec.target], spec.idx);
    out[spec.idx] = make_f_array(ri.buf.release(), ri.shape);
    out[spec.blk] = make_f_array(rb.buf.release(), rb.shape);
  }

  for (const RefinSpec& spec : REFIN_SPECS) {
    if (!arrays.contains(spec.index)) continue;
    nb::ndarray<> src_i = get_array(arrays, spec.index);
    nb::ndarray<> src_b = get_array(arrays, spec.block);
    const auto n = static_cast<std::int64_t>(src_i.shape(0));
    std::unique_ptr<int[]> ni(new int[static_cast<std::size_t>(n)]);
    std::unique_ptr<int[]> nbk(new int[static_cast<std::size_t>(n)]);
    std::copy_n(static_cast<const int*>(src_i.data()), n, ni.get());
    std::copy_n(static_cast<const int*>(src_b.data()), n, nbk.get());
    rewrite_refin_values(ni.get(), nbk.get(), n, np_old, np_new);
    const std::vector<std::size_t> shape1{static_cast<std::size_t>(n)};
    out[spec.index] = make_f_array(ni.release(), shape1);
    out[spec.block] = make_f_array(nbk.release(), shape1);
  }

  out["nproma"] = np_new;
  for (int sp = 0; sp < 3; ++sp) {
    if (!have[sp]) continue;
    out[NBLKS_KEYS[sp]] = nb_new[sp];
    out[VALID_KEYS[sp]] = valid[sp];
  }
  return out;
}

// ---------------------------------------------------------------------------
// selftest: synthetic serde parse + reblock 8 -> 4, exact expected values
// ---------------------------------------------------------------------------

void check(bool ok, const std::string& what) {
  if (!ok) throw std::runtime_error("selftest FAILED: " + what);
}

std::string selftest() {
  constexpr std::int64_t NP_OLD = 8, NP_NEW = 4, VALID = 13, NLEVP1 = 2, NB_OLD = 2;

  // --- 1. serde parse on a synthetic t_nh_prog record (# assoc / # missing / meta /
  // Fortran-ordered entries), exercising the vendored reader end to end.
  std::ostringstream text;
  auto emit_array3 = [&text](const char* name, std::int64_t d0, std::int64_t d1, std::int64_t d2, double base) {
    text << "# " << name << "\n# assoc\n1\n# missing\n1\n# rank\n3\n# size\n"
         << d0 << "\n"
         << d1 << "\n"
         << d2 << "\n# lbound\n1\n1\n1\n# entries\n";
    for (std::int64_t b = 0; b < d2; ++b)
      for (std::int64_t k = 0; k < d1; ++k)
        for (std::int64_t i = 0; i < d0; ++i)
          text << (static_cast<double>(b * d0 + i) * 10.0 + static_cast<double>(k) + base) << "\n";
  };
  emit_array3("w", NP_OLD, NLEVP1, NB_OLD, 0.0);
  emit_array3("vn", NP_OLD, NLEVP1, NB_OLD, 0.5);
  std::istringstream in(text.str());
  t_nh_prog prog{};
  serde::deserialize(&prog, in);
  check(prog.w != nullptr && prog.vn != nullptr, "t_nh_prog members parsed");
  const serde::array_meta& wm = meta_of(prog.w, "w");
  check(wm.rank == 3 && wm.size[0] == NP_OLD && wm.size[1] == NLEVP1 && wm.size[2] == NB_OLD, "w meta");
  check(prog.w[1 + NP_OLD * (1 + NLEVP1 * 1)] == (1 * NP_OLD + 1) * 10.0 + 1.0, "w Fortran-order value");
  delete[] prog.vn;  // only w goes into the reblock fixture

  // --- 2. reblock fixture: cells space, valid=13 of 16 slots, nproma 8 -> 4.
  nb::dict d;
  d["nproma"] = static_cast<int>(NP_OLD);
  d["nblks_c"] = static_cast<int>(NB_OLD);
  std::vector<std::size_t> shape_w{NP_OLD, NLEVP1, NB_OLD};
  d["p_prog_w"] = make_f_array(prog.w, shape_w);

  // refin-ctrl, extent 3: zone 0 covers [0,7], zone 1 covers [8,12], zone 2 empty.
  {
    const std::vector<std::size_t> s1{3};
    d["p_patch_cells_start_index"] = make_f_array(new int[3]{1, 1, 1}, s1);
    d["p_patch_cells_start_block"] = make_f_array(new int[3]{1, 2, 1}, s1);
    d["p_patch_cells_end_index"] = make_f_array(new int[3]{8, 5, 0}, s1);
    d["p_patch_cells_end_block"] = make_f_array(new int[3]{1, 2, 1}, s1);
  }

  // connectivity (c -> c), degree 3: slot g targets t = (7g + 3 + j) mod 13.
  constexpr std::int64_t DEG = 3;
  {
    std::unique_ptr<int[]> ci(new int[NP_OLD * NB_OLD * DEG]());
    std::unique_ptr<int[]> cb(new int[NP_OLD * NB_OLD * DEG]());
    for (std::int64_t j = 0; j < DEG; ++j)
      for (std::int64_t g = 0; g < VALID; ++g) {
        const std::int64_t t = (7 * g + 3 + j) % VALID;
        const std::int64_t at = (g % NP_OLD) + NP_OLD * ((g / NP_OLD) + NB_OLD * j);
        ci[at] = static_cast<int>(t % NP_OLD + 1);
        cb[at] = static_cast<int>(t / NP_OLD + 1);
      }
    const std::vector<std::size_t> s3{NP_OLD, NB_OLD, DEG};
    d["p_patch_cells_neighbor_idx"] = make_f_array(ci.release(), s3);
    d["p_patch_cells_neighbor_blk"] = make_f_array(cb.release(), s3);
  }

  nb::dict r = reblock(d, NP_NEW);

  check(nb::cast<std::int64_t>(r["nproma"]) == NP_NEW, "nproma updated");
  check(nb::cast<std::int64_t>(r["valid_cells"]) == VALID, "valid_cells == 13");
  check(nb::cast<std::int64_t>(r["nblks_c"]) == 4, "nblks_c == ceil(13/4)");

  {  // data array: value at new slot g is g*10 + k, pad 0
    auto w = nb::cast<nb::ndarray<>>(r["p_prog_w"]);
    check(w.ndim() == 3 && w.shape(0) == 4 && w.shape(1) == NLEVP1 && w.shape(2) == 4, "w reblocked shape");
    const double* p = static_cast<const double*>(w.data());
    for (std::int64_t b = 0; b < 4; ++b)
      for (std::int64_t k = 0; k < NLEVP1; ++k)
        for (std::int64_t i = 0; i < 4; ++i) {
          const std::int64_t g = b * NP_NEW + i;
          const double want = g < VALID ? static_cast<double>(g) * 10.0 + static_cast<double>(k) : 0.0;
          check(p[i + 4 * (k + NLEVP1 * b)] == want, "w value at g=" + std::to_string(g));
        }
  }
  {  // connectivity: recomputed target coords; pads replicate slot 12
    auto ci = nb::cast<nb::ndarray<>>(r["p_patch_cells_neighbor_idx"]);
    auto cb = nb::cast<nb::ndarray<>>(r["p_patch_cells_neighbor_blk"]);
    const int* pi = static_cast<const int*>(ci.data());
    const int* pb = static_cast<const int*>(cb.data());
    check(ci.shape(0) == 4 && ci.shape(1) == 4 && ci.shape(2) == DEG, "neighbor_idx reblocked shape");
    for (std::int64_t j = 0; j < DEG; ++j)
      for (std::int64_t slot = 0; slot < 16; ++slot) {
        const std::int64_t g = slot < VALID ? slot : VALID - 1;
        const std::int64_t t = (7 * g + 3 + j) % VALID;
        const std::int64_t at = (slot % NP_NEW) + NP_NEW * ((slot / NP_NEW) + 4 * j);
        check(pi[at] == t % NP_NEW + 1 && pb[at] == t / NP_NEW + 1, "neighbor entry at slot=" + std::to_string(slot));
      }
  }
  {  // refin-ctrl: zone boundaries re-expressed; empty zone stays empty (idx 0)
    auto si = nb::cast<nb::ndarray<>>(r["p_patch_cells_start_index"]);
    auto sb = nb::cast<nb::ndarray<>>(r["p_patch_cells_start_block"]);
    auto ei = nb::cast<nb::ndarray<>>(r["p_patch_cells_end_index"]);
    auto eb = nb::cast<nb::ndarray<>>(r["p_patch_cells_end_block"]);
    const int* psi = static_cast<const int*>(si.data());
    const int* psb = static_cast<const int*>(sb.data());
    const int* pei = static_cast<const int*>(ei.data());
    const int* peb = static_cast<const int*>(eb.data());
    // zone 0: [0,7] -> start (1,1), end (4,2); zone 1: [8,12] -> start (1,3), end (1,4)
    check(psi[0] == 1 && psb[0] == 1 && pei[0] == 4 && peb[0] == 2, "zone 0 boundaries");
    check(psi[1] == 1 && psb[1] == 3 && pei[1] == 1 && peb[1] == 4, "zone 1 boundaries");
    // zone 2 was empty (end idx 0 = "one before start"): start (1,1), end (0,1)
    check(psi[2] == 1 && psb[2] == 1 && pei[2] == 0 && peb[2] == 1, "empty zone stays empty");
  }

  return "velocity_data selftest OK: parse + reblock 8->4 (valid 13, 4 blocks), "
         "connectivity + refin-ctrl recompute verified";
}

}  // namespace

NB_MODULE(velocity_data, m) {
  m.doc() = "ICON velocity_tendencies artifact data: serde load + nproma re-blocking";
  m.def("load", &load, "data_dir"_a, "timestep"_a = 1,
        "Parse one timestep of the artifact .data set into {flat_name: F-ordered numpy array}");
  m.def("reblock", &reblock, "arrays"_a, "nproma"_a = 32, "Re-block a load() dict to a new nproma");
  m.def("selftest", &selftest, "Synthetic parse + reblock check; throws on any mismatch");
}
