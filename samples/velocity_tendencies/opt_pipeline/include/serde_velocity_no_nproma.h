#ifndef __DACE_SERDE__
#define __DACE_SERDE__

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <istream>
#include <map>
#include <optional>
#include <ranges>
#include <set>
#include <sstream>
#include <string_view>
#include <vector>

#include "velocity_tendencies_no_nproma.h"

namespace serde {
std::vector<std::string_view> split(std::string_view s, char delim) {
  std::vector<std::string_view> parts;
  for (uint start_pos = 0, next_pos; start_pos < s.length(); start_pos = next_pos + 1) {
    next_pos = s.find(delim, start_pos);
    if (next_pos == s.npos) {
      parts.push_back({s.begin() + start_pos, s.length() - start_pos});
      break;
    }
    parts.push_back({s.begin() + start_pos, static_cast<size_t>(next_pos - start_pos)});
  }
  return parts;
}

std::string scroll_space(std::istream &s) {
  std::string out;
  while (!s.eof() && (!s.peek() || isspace(s.peek()))) {
    out += s.get();
    assert(s.good());
  }
  return out;
}

std::string read_line(std::istream &s, const std::optional<std::string> &should_contain = {}) {
  if (s.eof())
    return "<eof>";
  scroll_space(s);
  char bin[101];
  s.getline(bin, 100);
  assert(s.good());
  if (should_contain) {
    bool ok = (std::string(bin).find(*should_contain) != std::string::npos);
    if (!ok) {
      std::cerr << "Expected: '" << *should_contain << "'; got: '" << bin << "'" << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  return {bin};
}

struct array_meta;
std::map<void *, array_meta> *ARRAY_META_DICT();

struct array_meta {
  int rank = 0;
  std::vector<int> size, lbound;

  int volume() const { return std::reduce(size.begin(), size.end(), 1, std::multiplies<int>()); }

  template <typename T> T *read(std::istream &s) const;
};
std::map<void *, array_meta> *ARRAY_META_DICT() {
  static auto *M = new std::map<void *, array_meta>();
  return M;
}
template <typename T> const array_meta &ARRAY_META_DICT_AT(T *a) {
  if constexpr (std::is_pointer_v<T>) {
    return ARRAY_META_DICT_AT(*a);
  } else {
    return ARRAY_META_DICT()->at(a);
  }
}

template <typename T> void read_scalar(T &x, std::istream &s) {
  if (s.eof())
    return;
  scroll_space(s);
  s >> x;
}

void read_scalar(float &x, std::istream &s) {
  if (s.eof())
    return;
  scroll_space(s);
  long double y;
  s >> y;
  x = y;
}

void read_scalar(double &x, std::istream &s) {
  if (s.eof())
    return;
  scroll_space(s);
  long double y;
  s >> y;
  x = y;
}

void read_scalar(bool &x, std::istream &s) {
  char c;
  read_scalar(c, s);
  assert(c == '1' or c == '0');
  x = (c == '1');
}

array_meta read_array_meta(std::istream &s) {
  array_meta m;
  read_line(s, {"# rank"}); // Should contain '# rank'
  read_scalar(m.rank, s);
  m.size.resize(m.rank);
  m.lbound.resize(m.rank);
  read_line(s, {"# size"}); // Should contain '# size'
  for (int i = 0; i < m.rank; ++i) {
    read_scalar(m.size[i], s);
  }
  read_line(s, {"# lbound"}); // Should contain '# lbound'
  for (int i = 0; i < m.rank; ++i) {
    read_scalar(m.lbound[i], s);
  }
  return m;
}

template <typename T> std::pair<array_meta, T *> read_array(std::istream &s) {
  auto m = serde::read_array_meta(s);
  auto *y = m.read<T>(s);
  return {m, y};
}

template <typename T> std::pair<array_meta, T *> read_pointer(std::istream &s) {
  read_line(s, {"# missing"}); // Should contain '# missing'
  int missing;
  read_scalar(missing, s);
  assert(missing == 1);
  return read_array<T>(s);
}

template <typename T> std::string serialize_array(T *arr);

void deserialize(float *x, std::istream &s) { read_scalar(*x, s); }
void deserialize(double *x, std::istream &s) { read_scalar(*x, s); }
void deserialize(long double *x, std::istream &s) { read_scalar(*x, s); }
void deserialize(int *x, std::istream &s) { read_scalar(*x, s); }
void deserialize(long *x, std::istream &s) { read_scalar(*x, s); }
void deserialize(long long *x, std::istream &s) { read_scalar(*x, s); }
void deserialize(bool *x, std::istream &s) { read_scalar(*x, s); }
void deserialize(float &x, std::istream &s) { read_scalar(x, s); }
void deserialize(double &x, std::istream &s) { read_scalar(x, s); }
void deserialize(long double &x, std::istream &s) { read_scalar(x, s); }
void deserialize(int &x, std::istream &s) { read_scalar(x, s); }
void deserialize(long &x, std::istream &s) { read_scalar(x, s); }
void deserialize(long long &x, std::istream &s) { read_scalar(x, s); }
void deserialize(bool &x, std::istream &s) { read_scalar(x, s); }

void deserialize(t_grid_domain_decomp_info *x, std::istream &s) {
  bool yep;
  array_meta m;
  read_line(s, {"# owner_mask"}); // Should contain '# owner_mask'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_owner_mask_d_0_s_2 = m.size[0];
    x->__f2dace_SA_owner_mask_d_1_s_3 = m.size[1];
    x->__f2dace_SOA_owner_mask_d_0_s_2 = m.lbound[0];
    x->__f2dace_SOA_owner_mask_d_1_s_3 = m.lbound[1];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->owner_mask = m.read<std::remove_pointer<decltype(x->owner_mask)>::type>(s);

  } // CONCLUDING IF
}

void deserialize(t_int_state *x, std::istream &s) {
  bool yep;
  array_meta m;
  read_line(s, {"# c_lin_e"}); // Should contain '# c_lin_e'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_c_lin_e_d_0_s_25 = m.size[0];
    x->__f2dace_SA_c_lin_e_d_1_s_26 = m.size[1];
    x->__f2dace_SA_c_lin_e_d_2_s_27 = m.size[2];
    x->__f2dace_SOA_c_lin_e_d_0_s_25 = m.lbound[0];
    x->__f2dace_SOA_c_lin_e_d_1_s_26 = m.lbound[1];
    x->__f2dace_SOA_c_lin_e_d_2_s_27 = m.lbound[2];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->c_lin_e = m.read<std::remove_pointer<decltype(x->c_lin_e)>::type>(s);

  } // CONCLUDING IF
  read_line(s, {"# e_bln_c_s"}); // Should contain '# e_bln_c_s'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_e_bln_c_s_d_0_s_28 = m.size[0];
    x->__f2dace_SA_e_bln_c_s_d_1_s_29 = m.size[1];
    x->__f2dace_SA_e_bln_c_s_d_2_s_30 = m.size[2];
    x->__f2dace_SOA_e_bln_c_s_d_0_s_28 = m.lbound[0];
    x->__f2dace_SOA_e_bln_c_s_d_1_s_29 = m.lbound[1];
    x->__f2dace_SOA_e_bln_c_s_d_2_s_30 = m.lbound[2];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->e_bln_c_s = m.read<std::remove_pointer<decltype(x->e_bln_c_s)>::type>(s);

  } // CONCLUDING IF
  read_line(s, {"# cells_aw_verts"}); // Should contain '# cells_aw_verts'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_cells_aw_verts_d_0_s_31 = m.size[0];
    x->__f2dace_SA_cells_aw_verts_d_1_s_32 = m.size[1];
    x->__f2dace_SA_cells_aw_verts_d_2_s_33 = m.size[2];
    x->__f2dace_SOA_cells_aw_verts_d_0_s_31 = m.lbound[0];
    x->__f2dace_SOA_cells_aw_verts_d_1_s_32 = m.lbound[1];
    x->__f2dace_SOA_cells_aw_verts_d_2_s_33 = m.lbound[2];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->cells_aw_verts = m.read<std::remove_pointer<decltype(x->cells_aw_verts)>::type>(s);

  } // CONCLUDING IF
  read_line(s, {"# rbf_vec_coeff_e"}); // Should contain '# rbf_vec_coeff_e'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_rbf_vec_coeff_e_d_0_s_34 = m.size[0];
    x->__f2dace_SA_rbf_vec_coeff_e_d_1_s_35 = m.size[1];
    x->__f2dace_SA_rbf_vec_coeff_e_d_2_s_36 = m.size[2];
    x->__f2dace_SOA_rbf_vec_coeff_e_d_0_s_34 = m.lbound[0];
    x->__f2dace_SOA_rbf_vec_coeff_e_d_1_s_35 = m.lbound[1];
    x->__f2dace_SOA_rbf_vec_coeff_e_d_2_s_36 = m.lbound[2];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->rbf_vec_coeff_e = m.read<std::remove_pointer<decltype(x->rbf_vec_coeff_e)>::type>(s);

  } // CONCLUDING IF
  read_line(s, {"# geofac_grdiv"}); // Should contain '# geofac_grdiv'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_geofac_grdiv_d_0_s_37 = m.size[0];
    x->__f2dace_SA_geofac_grdiv_d_1_s_38 = m.size[1];
    x->__f2dace_SA_geofac_grdiv_d_2_s_39 = m.size[2];
    x->__f2dace_SOA_geofac_grdiv_d_0_s_37 = m.lbound[0];
    x->__f2dace_SOA_geofac_grdiv_d_1_s_38 = m.lbound[1];
    x->__f2dace_SOA_geofac_grdiv_d_2_s_39 = m.lbound[2];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->geofac_grdiv = m.read<std::remove_pointer<decltype(x->geofac_grdiv)>::type>(s);

  } // CONCLUDING IF
  read_line(s, {"# geofac_rot"}); // Should contain '# geofac_rot'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_geofac_rot_d_0_s_40 = m.size[0];
    x->__f2dace_SA_geofac_rot_d_1_s_41 = m.size[1];
    x->__f2dace_SA_geofac_rot_d_2_s_42 = m.size[2];
    x->__f2dace_SOA_geofac_rot_d_0_s_40 = m.lbound[0];
    x->__f2dace_SOA_geofac_rot_d_1_s_41 = m.lbound[1];
    x->__f2dace_SOA_geofac_rot_d_2_s_42 = m.lbound[2];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->geofac_rot = m.read<std::remove_pointer<decltype(x->geofac_rot)>::type>(s);

  } // CONCLUDING IF
  read_line(s, {"# geofac_n2s"}); // Should contain '# geofac_n2s'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_geofac_n2s_d_0_s_43 = m.size[0];
    x->__f2dace_SA_geofac_n2s_d_1_s_44 = m.size[1];
    x->__f2dace_SA_geofac_n2s_d_2_s_45 = m.size[2];
    x->__f2dace_SOA_geofac_n2s_d_0_s_43 = m.lbound[0];
    x->__f2dace_SOA_geofac_n2s_d_1_s_44 = m.lbound[1];
    x->__f2dace_SOA_geofac_n2s_d_2_s_45 = m.lbound[2];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->geofac_n2s = m.read<std::remove_pointer<decltype(x->geofac_n2s)>::type>(s);

  } // CONCLUDING IF
}

void deserialize(t_grid_cells *x, std::istream &s) {
  bool yep;
  array_meta m;
  read_line(s, {"# neighbor_idx"}); // Should contain '# neighbor_idx'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_neighbor_idx_d_0_s_146 = m.size[0];
    x->__f2dace_SA_neighbor_idx_d_1_s_147 = m.size[1];
    x->__f2dace_SA_neighbor_idx_d_2_s_148 = m.size[2];
    x->__f2dace_SOA_neighbor_idx_d_0_s_146 = m.lbound[0];
    x->__f2dace_SOA_neighbor_idx_d_1_s_147 = m.lbound[1];
    x->__f2dace_SOA_neighbor_idx_d_2_s_148 = m.lbound[2];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->neighbor_idx = m.read<std::remove_pointer<decltype(x->neighbor_idx)>::type>(s);

  } // CONCLUDING IF
  read_line(s, {"# neighbor_blk"}); // Should contain '# neighbor_blk'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_neighbor_blk_d_0_s_149 = m.size[0];
    x->__f2dace_SA_neighbor_blk_d_1_s_150 = m.size[1];
    x->__f2dace_SA_neighbor_blk_d_2_s_151 = m.size[2];
    x->__f2dace_SOA_neighbor_blk_d_0_s_149 = m.lbound[0];
    x->__f2dace_SOA_neighbor_blk_d_1_s_150 = m.lbound[1];
    x->__f2dace_SOA_neighbor_blk_d_2_s_151 = m.lbound[2];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->neighbor_blk = m.read<std::remove_pointer<decltype(x->neighbor_blk)>::type>(s);

  } // CONCLUDING IF
  read_line(s, {"# edge_idx"}); // Should contain '# edge_idx'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_edge_idx_d_0_s_152 = m.size[0];
    x->__f2dace_SA_edge_idx_d_1_s_153 = m.size[1];
    x->__f2dace_SA_edge_idx_d_2_s_154 = m.size[2];
    x->__f2dace_SOA_edge_idx_d_0_s_152 = m.lbound[0];
    x->__f2dace_SOA_edge_idx_d_1_s_153 = m.lbound[1];
    x->__f2dace_SOA_edge_idx_d_2_s_154 = m.lbound[2];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->edge_idx = m.read<std::remove_pointer<decltype(x->edge_idx)>::type>(s);

  } // CONCLUDING IF
  read_line(s, {"# edge_blk"}); // Should contain '# edge_blk'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_edge_blk_d_0_s_155 = m.size[0];
    x->__f2dace_SA_edge_blk_d_1_s_156 = m.size[1];
    x->__f2dace_SA_edge_blk_d_2_s_157 = m.size[2];
    x->__f2dace_SOA_edge_blk_d_0_s_155 = m.lbound[0];
    x->__f2dace_SOA_edge_blk_d_1_s_156 = m.lbound[1];
    x->__f2dace_SOA_edge_blk_d_2_s_157 = m.lbound[2];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->edge_blk = m.read<std::remove_pointer<decltype(x->edge_blk)>::type>(s);

  } // CONCLUDING IF
  read_line(s, {"# area"}); // Should contain '# area'

  read_line(s, {"# assoc"}); // Should contain '# assoc'
  deserialize(&yep, s);

  if (yep) {
    auto [m, arr] = read_pointer<std::remove_pointer<decltype(x->area)>::type>(s);
    x->__f2dace_SA_area_d_0_s_158 = m.size.at(0);
    x->__f2dace_SA_area_d_1_s_159 = m.size.at(1);
    x->__f2dace_SOA_area_d_0_s_158 = m.lbound.at(0);
    x->__f2dace_SOA_area_d_1_s_159 = m.lbound.at(1);
    x->area = arr;
  }

  read_line(s, {"# start_index"}); // Should contain '# start_index'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_start_index_d_0_s_160 = m.size[0];
    x->__f2dace_SOA_start_index_d_0_s_160 = m.lbound[0];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->start_index = m.read<std::remove_pointer<decltype(x->start_index)>::type>(s);

  } // CONCLUDING IF
  read_line(s, {"# end_index"}); // Should contain '# end_index'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_end_index_d_0_s_161 = m.size[0];
    x->__f2dace_SOA_end_index_d_0_s_161 = m.lbound[0];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->end_index = m.read<std::remove_pointer<decltype(x->end_index)>::type>(s);

  } // CONCLUDING IF
  read_line(s, {"# start_block"}); // Should contain '# start_block'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_start_block_d_0_s_162 = m.size[0];
    x->__f2dace_SOA_start_block_d_0_s_162 = m.lbound[0];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->start_block = m.read<std::remove_pointer<decltype(x->start_block)>::type>(s);

  } // CONCLUDING IF
  read_line(s, {"# end_block"}); // Should contain '# end_block'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_end_block_d_0_s_163 = m.size[0];
    x->__f2dace_SOA_end_block_d_0_s_163 = m.lbound[0];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->end_block = m.read<std::remove_pointer<decltype(x->end_block)>::type>(s);

  } // CONCLUDING IF
  read_line(s, {"# decomp_info"}); // Should contain '# decomp_info'

  x->decomp_info = new std::remove_pointer<decltype(x->decomp_info)>::type;
  deserialize(x->decomp_info, s);
}

void deserialize(t_grid_edges *x, std::istream &s) {
  bool yep;
  array_meta m;
  read_line(s, {"# cell_idx"}); // Should contain '# cell_idx'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_cell_idx_d_0_s_164 = m.size[0];
    x->__f2dace_SA_cell_idx_d_1_s_165 = m.size[1];
    x->__f2dace_SA_cell_idx_d_2_s_166 = m.size[2];
    x->__f2dace_SOA_cell_idx_d_0_s_164 = m.lbound[0];
    x->__f2dace_SOA_cell_idx_d_1_s_165 = m.lbound[1];
    x->__f2dace_SOA_cell_idx_d_2_s_166 = m.lbound[2];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->cell_idx = m.read<std::remove_pointer<decltype(x->cell_idx)>::type>(s);

  } // CONCLUDING IF
  read_line(s, {"# cell_blk"}); // Should contain '# cell_blk'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_cell_blk_d_0_s_167 = m.size[0];
    x->__f2dace_SA_cell_blk_d_1_s_168 = m.size[1];
    x->__f2dace_SA_cell_blk_d_2_s_169 = m.size[2];
    x->__f2dace_SOA_cell_blk_d_0_s_167 = m.lbound[0];
    x->__f2dace_SOA_cell_blk_d_1_s_168 = m.lbound[1];
    x->__f2dace_SOA_cell_blk_d_2_s_169 = m.lbound[2];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->cell_blk = m.read<std::remove_pointer<decltype(x->cell_blk)>::type>(s);

  } // CONCLUDING IF
  read_line(s, {"# vertex_idx"}); // Should contain '# vertex_idx'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_vertex_idx_d_0_s_170 = m.size[0];
    x->__f2dace_SA_vertex_idx_d_1_s_171 = m.size[1];
    x->__f2dace_SA_vertex_idx_d_2_s_172 = m.size[2];
    x->__f2dace_SOA_vertex_idx_d_0_s_170 = m.lbound[0];
    x->__f2dace_SOA_vertex_idx_d_1_s_171 = m.lbound[1];
    x->__f2dace_SOA_vertex_idx_d_2_s_172 = m.lbound[2];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->vertex_idx = m.read<std::remove_pointer<decltype(x->vertex_idx)>::type>(s);

  } // CONCLUDING IF
  read_line(s, {"# vertex_blk"}); // Should contain '# vertex_blk'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_vertex_blk_d_0_s_173 = m.size[0];
    x->__f2dace_SA_vertex_blk_d_1_s_174 = m.size[1];
    x->__f2dace_SA_vertex_blk_d_2_s_175 = m.size[2];
    x->__f2dace_SOA_vertex_blk_d_0_s_173 = m.lbound[0];
    x->__f2dace_SOA_vertex_blk_d_1_s_174 = m.lbound[1];
    x->__f2dace_SOA_vertex_blk_d_2_s_175 = m.lbound[2];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->vertex_blk = m.read<std::remove_pointer<decltype(x->vertex_blk)>::type>(s);

  } // CONCLUDING IF
  read_line(s, {"# tangent_orientation"}); // Should contain '# tangent_orientation'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_tangent_orientation_d_0_s_176 = m.size[0];
    x->__f2dace_SA_tangent_orientation_d_1_s_177 = m.size[1];
    x->__f2dace_SOA_tangent_orientation_d_0_s_176 = m.lbound[0];
    x->__f2dace_SOA_tangent_orientation_d_1_s_177 = m.lbound[1];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->tangent_orientation = m.read<std::remove_pointer<decltype(x->tangent_orientation)>::type>(s);

  } // CONCLUDING IF
  read_line(s, {"# quad_idx"}); // Should contain '# quad_idx'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_quad_idx_d_0_s_178 = m.size[0];
    x->__f2dace_SA_quad_idx_d_1_s_179 = m.size[1];
    x->__f2dace_SA_quad_idx_d_2_s_180 = m.size[2];
    x->__f2dace_SOA_quad_idx_d_0_s_178 = m.lbound[0];
    x->__f2dace_SOA_quad_idx_d_1_s_179 = m.lbound[1];
    x->__f2dace_SOA_quad_idx_d_2_s_180 = m.lbound[2];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->quad_idx = m.read<std::remove_pointer<decltype(x->quad_idx)>::type>(s);

  } // CONCLUDING IF
  read_line(s, {"# quad_blk"}); // Should contain '# quad_blk'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_quad_blk_d_0_s_181 = m.size[0];
    x->__f2dace_SA_quad_blk_d_1_s_182 = m.size[1];
    x->__f2dace_SA_quad_blk_d_2_s_183 = m.size[2];
    x->__f2dace_SOA_quad_blk_d_0_s_181 = m.lbound[0];
    x->__f2dace_SOA_quad_blk_d_1_s_182 = m.lbound[1];
    x->__f2dace_SOA_quad_blk_d_2_s_183 = m.lbound[2];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->quad_blk = m.read<std::remove_pointer<decltype(x->quad_blk)>::type>(s);

  } // CONCLUDING IF
  read_line(s, {"# inv_primal_edge_length"}); // Should contain '#
                                              // inv_primal_edge_length'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_inv_primal_edge_length_d_0_s_184 = m.size[0];
    x->__f2dace_SA_inv_primal_edge_length_d_1_s_185 = m.size[1];
    x->__f2dace_SOA_inv_primal_edge_length_d_0_s_184 = m.lbound[0];
    x->__f2dace_SOA_inv_primal_edge_length_d_1_s_185 = m.lbound[1];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->inv_primal_edge_length = m.read<std::remove_pointer<decltype(x->inv_primal_edge_length)>::type>(s);

  } // CONCLUDING IF
  read_line(s, {"# inv_dual_edge_length"}); // Should contain '# inv_dual_edge_length'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_inv_dual_edge_length_d_0_s_186 = m.size[0];
    x->__f2dace_SA_inv_dual_edge_length_d_1_s_187 = m.size[1];
    x->__f2dace_SOA_inv_dual_edge_length_d_0_s_186 = m.lbound[0];
    x->__f2dace_SOA_inv_dual_edge_length_d_1_s_187 = m.lbound[1];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->inv_dual_edge_length = m.read<std::remove_pointer<decltype(x->inv_dual_edge_length)>::type>(s);

  } // CONCLUDING IF
  read_line(s, {"# area_edge"}); // Should contain '# area_edge'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_area_edge_d_0_s_188 = m.size[0];
    x->__f2dace_SA_area_edge_d_1_s_189 = m.size[1];
    x->__f2dace_SOA_area_edge_d_0_s_188 = m.lbound[0];
    x->__f2dace_SOA_area_edge_d_1_s_189 = m.lbound[1];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->area_edge = m.read<std::remove_pointer<decltype(x->area_edge)>::type>(s);

  } // CONCLUDING IF
  read_line(s, {"# f_e"}); // Should contain '# f_e'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_f_e_d_0_s_190 = m.size[0];
    x->__f2dace_SA_f_e_d_1_s_191 = m.size[1];
    x->__f2dace_SOA_f_e_d_0_s_190 = m.lbound[0];
    x->__f2dace_SOA_f_e_d_1_s_191 = m.lbound[1];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->f_e = m.read<std::remove_pointer<decltype(x->f_e)>::type>(s);

  } // CONCLUDING IF
  read_line(s, {"# fn_e"}); // Should contain '# fn_e'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_fn_e_d_0_s_192 = m.size[0];
    x->__f2dace_SA_fn_e_d_1_s_193 = m.size[1];
    x->__f2dace_SOA_fn_e_d_0_s_192 = m.lbound[0];
    x->__f2dace_SOA_fn_e_d_1_s_193 = m.lbound[1];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->fn_e = m.read<std::remove_pointer<decltype(x->fn_e)>::type>(s);

  } // CONCLUDING IF
  read_line(s, {"# ft_e"}); // Should contain '# ft_e'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_ft_e_d_0_s_194 = m.size[0];
    x->__f2dace_SA_ft_e_d_1_s_195 = m.size[1];
    x->__f2dace_SOA_ft_e_d_0_s_194 = m.lbound[0];
    x->__f2dace_SOA_ft_e_d_1_s_195 = m.lbound[1];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->ft_e = m.read<std::remove_pointer<decltype(x->ft_e)>::type>(s);

  } // CONCLUDING IF
  read_line(s, {"# start_index"}); // Should contain '# start_index'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_start_index_d_0_s_196 = m.size[0];
    x->__f2dace_SOA_start_index_d_0_s_196 = m.lbound[0];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->start_index = m.read<std::remove_pointer<decltype(x->start_index)>::type>(s);

  } // CONCLUDING IF
  read_line(s, {"# end_index"}); // Should contain '# end_index'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_end_index_d_0_s_197 = m.size[0];
    x->__f2dace_SOA_end_index_d_0_s_197 = m.lbound[0];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->end_index = m.read<std::remove_pointer<decltype(x->end_index)>::type>(s);

  } // CONCLUDING IF
  read_line(s, {"# start_block"}); // Should contain '# start_block'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_start_block_d_0_s_198 = m.size[0];
    x->__f2dace_SOA_start_block_d_0_s_198 = m.lbound[0];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->start_block = m.read<std::remove_pointer<decltype(x->start_block)>::type>(s);

  } // CONCLUDING IF
  read_line(s, {"# end_block"}); // Should contain '# end_block'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_end_block_d_0_s_199 = m.size[0];
    x->__f2dace_SOA_end_block_d_0_s_199 = m.lbound[0];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->end_block = m.read<std::remove_pointer<decltype(x->end_block)>::type>(s);

  } // CONCLUDING IF
}

void deserialize(t_grid_vertices *x, std::istream &s) {
  bool yep;
  array_meta m;
  read_line(s, {"# cell_idx"}); // Should contain '# cell_idx'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_cell_idx_d_0_s_200 = m.size[0];
    x->__f2dace_SA_cell_idx_d_1_s_201 = m.size[1];
    x->__f2dace_SA_cell_idx_d_2_s_202 = m.size[2];
    x->__f2dace_SOA_cell_idx_d_0_s_200 = m.lbound[0];
    x->__f2dace_SOA_cell_idx_d_1_s_201 = m.lbound[1];
    x->__f2dace_SOA_cell_idx_d_2_s_202 = m.lbound[2];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->cell_idx = m.read<std::remove_pointer<decltype(x->cell_idx)>::type>(s);

  } // CONCLUDING IF
  read_line(s, {"# cell_blk"}); // Should contain '# cell_blk'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_cell_blk_d_0_s_203 = m.size[0];
    x->__f2dace_SA_cell_blk_d_1_s_204 = m.size[1];
    x->__f2dace_SA_cell_blk_d_2_s_205 = m.size[2];
    x->__f2dace_SOA_cell_blk_d_0_s_203 = m.lbound[0];
    x->__f2dace_SOA_cell_blk_d_1_s_204 = m.lbound[1];
    x->__f2dace_SOA_cell_blk_d_2_s_205 = m.lbound[2];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->cell_blk = m.read<std::remove_pointer<decltype(x->cell_blk)>::type>(s);

  } // CONCLUDING IF
  read_line(s, {"# edge_idx"}); // Should contain '# edge_idx'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_edge_idx_d_0_s_206 = m.size[0];
    x->__f2dace_SA_edge_idx_d_1_s_207 = m.size[1];
    x->__f2dace_SA_edge_idx_d_2_s_208 = m.size[2];
    x->__f2dace_SOA_edge_idx_d_0_s_206 = m.lbound[0];
    x->__f2dace_SOA_edge_idx_d_1_s_207 = m.lbound[1];
    x->__f2dace_SOA_edge_idx_d_2_s_208 = m.lbound[2];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->edge_idx = m.read<std::remove_pointer<decltype(x->edge_idx)>::type>(s);

  } // CONCLUDING IF
  read_line(s, {"# edge_blk"}); // Should contain '# edge_blk'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_edge_blk_d_0_s_209 = m.size[0];
    x->__f2dace_SA_edge_blk_d_1_s_210 = m.size[1];
    x->__f2dace_SA_edge_blk_d_2_s_211 = m.size[2];
    x->__f2dace_SOA_edge_blk_d_0_s_209 = m.lbound[0];
    x->__f2dace_SOA_edge_blk_d_1_s_210 = m.lbound[1];
    x->__f2dace_SOA_edge_blk_d_2_s_211 = m.lbound[2];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->edge_blk = m.read<std::remove_pointer<decltype(x->edge_blk)>::type>(s);

  } // CONCLUDING IF
  read_line(s, {"# start_index"}); // Should contain '# start_index'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_start_index_d_0_s_212 = m.size[0];
    x->__f2dace_SOA_start_index_d_0_s_212 = m.lbound[0];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->start_index = m.read<std::remove_pointer<decltype(x->start_index)>::type>(s);

  } // CONCLUDING IF
  read_line(s, {"# end_index"}); // Should contain '# end_index'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_end_index_d_0_s_213 = m.size[0];
    x->__f2dace_SOA_end_index_d_0_s_213 = m.lbound[0];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->end_index = m.read<std::remove_pointer<decltype(x->end_index)>::type>(s);

  } // CONCLUDING IF
  read_line(s, {"# start_block"}); // Should contain '# start_block'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_start_block_d_0_s_214 = m.size[0];
    x->__f2dace_SOA_start_block_d_0_s_214 = m.lbound[0];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->start_block = m.read<std::remove_pointer<decltype(x->start_block)>::type>(s);

  } // CONCLUDING IF
  read_line(s, {"# end_block"}); // Should contain '# end_block'

  read_line(s, {"# alloc"}); // Should contain '# alloc'
  deserialize(&yep, s);
  if (yep) { // BEGINING IF

    m = read_array_meta(s);
    x->__f2dace_SA_end_block_d_0_s_215 = m.size[0];
    x->__f2dace_SOA_end_block_d_0_s_215 = m.lbound[0];
    // We only need to allocate a volume of contiguous memory, and let DaCe
    // interpret (assuming it follows the same protocol as us).
    x->end_block = m.read<std::remove_pointer<decltype(x->end_block)>::type>(s);

  } // CONCLUDING IF
}

void deserialize(t_patch *x, std::istream &s) {
  bool yep;
  array_meta m;
  read_line(s, {"# nblks_c"}); // Should contain '# nblks_c'

  deserialize(&(x->nblks_c), s);

  read_line(s, {"# nblks_e"}); // Should contain '# nblks_e'

  deserialize(&(x->nblks_e), s);

  read_line(s, {"# nblks_v"}); // Should contain '# nblks_v'

  deserialize(&(x->nblks_v), s);

  read_line(s, {"# cells"}); // Should contain '# cells'

  x->cells = new std::remove_pointer<decltype(x->cells)>::type;
  deserialize(x->cells, s);

  read_line(s, {"# edges"}); // Should contain '# edges'

  x->edges = new std::remove_pointer<decltype(x->edges)>::type;
  deserialize(x->edges, s);

  read_line(s, {"# verts"}); // Should contain '# verts'

  x->verts = new std::remove_pointer<decltype(x->verts)>::type;
  deserialize(x->verts, s);
}

void deserialize(t_nh_prog *x, std::istream &s) {
  bool yep;
  array_meta m;
  read_line(s, {"# w"}); // Should contain '# w'

  read_line(s, {"# assoc"}); // Should contain '# assoc'
  deserialize(&yep, s);

  if (yep) {
    auto [m, arr] = read_pointer<std::remove_pointer<decltype(x->w)>::type>(s);
    x->__f2dace_SA_w_d_0_s_285 = m.size.at(0);
    x->__f2dace_SA_w_d_1_s_286 = m.size.at(1);
    x->__f2dace_SA_w_d_2_s_287 = m.size.at(2);
    x->__f2dace_SOA_w_d_0_s_285 = m.lbound.at(0);
    x->__f2dace_SOA_w_d_1_s_286 = m.lbound.at(1);
    x->__f2dace_SOA_w_d_2_s_287 = m.lbound.at(2);
    x->w = arr;
  }

  read_line(s, {"# vn"}); // Should contain '# vn'

  read_line(s, {"# assoc"}); // Should contain '# assoc'
  deserialize(&yep, s);

  if (yep) {
    auto [m, arr] = read_pointer<std::remove_pointer<decltype(x->vn)>::type>(s);
    x->__f2dace_SA_vn_d_0_s_288 = m.size.at(0);
    x->__f2dace_SA_vn_d_1_s_289 = m.size.at(1);
    x->__f2dace_SA_vn_d_2_s_290 = m.size.at(2);
    x->__f2dace_SOA_vn_d_0_s_288 = m.lbound.at(0);
    x->__f2dace_SOA_vn_d_1_s_289 = m.lbound.at(1);
    x->__f2dace_SOA_vn_d_2_s_290 = m.lbound.at(2);
    x->vn = arr;
  }
}

void deserialize(t_nh_diag *x, std::istream &s) {
  bool yep;
  array_meta m;
  read_line(s, {"# vt"}); // Should contain '# vt'

  read_line(s, {"# assoc"}); // Should contain '# assoc'
  deserialize(&yep, s);

  if (yep) {
    auto [m, arr] = read_pointer<std::remove_pointer<decltype(x->vt)>::type>(s);
    x->__f2dace_SA_vt_d_0_s_291 = m.size.at(0);
    x->__f2dace_SA_vt_d_1_s_292 = m.size.at(1);
    x->__f2dace_SA_vt_d_2_s_293 = m.size.at(2);
    x->__f2dace_SOA_vt_d_0_s_291 = m.lbound.at(0);
    x->__f2dace_SOA_vt_d_1_s_292 = m.lbound.at(1);
    x->__f2dace_SOA_vt_d_2_s_293 = m.lbound.at(2);
    x->vt = arr;
  }

  read_line(s, {"# vn_ie"}); // Should contain '# vn_ie'

  read_line(s, {"# assoc"}); // Should contain '# assoc'
  deserialize(&yep, s);

  if (yep) {
    auto [m, arr] = read_pointer<std::remove_pointer<decltype(x->vn_ie)>::type>(s);
    x->__f2dace_SA_vn_ie_d_0_s_294 = m.size.at(0);
    x->__f2dace_SA_vn_ie_d_1_s_295 = m.size.at(1);
    x->__f2dace_SA_vn_ie_d_2_s_296 = m.size.at(2);
    x->__f2dace_SOA_vn_ie_d_0_s_294 = m.lbound.at(0);
    x->__f2dace_SOA_vn_ie_d_1_s_295 = m.lbound.at(1);
    x->__f2dace_SOA_vn_ie_d_2_s_296 = m.lbound.at(2);
    x->vn_ie = arr;
  }

  read_line(s, {"# w_concorr_c"}); // Should contain '# w_concorr_c'

  read_line(s, {"# assoc"}); // Should contain '# assoc'
  deserialize(&yep, s);

  if (yep) {
    auto [m, arr] = read_pointer<std::remove_pointer<decltype(x->w_concorr_c)>::type>(s);
    x->__f2dace_SA_w_concorr_c_d_0_s_297 = m.size.at(0);
    x->__f2dace_SA_w_concorr_c_d_1_s_298 = m.size.at(1);
    x->__f2dace_SA_w_concorr_c_d_2_s_299 = m.size.at(2);
    x->__f2dace_SOA_w_concorr_c_d_0_s_297 = m.lbound.at(0);
    x->__f2dace_SOA_w_concorr_c_d_1_s_298 = m.lbound.at(1);
    x->__f2dace_SOA_w_concorr_c_d_2_s_299 = m.lbound.at(2);
    x->w_concorr_c = arr;
  }

  read_line(s, {"# ddt_vn_apc_pc"}); // Should contain '# ddt_vn_apc_pc'

  read_line(s, {"# assoc"}); // Should contain '# assoc'
  deserialize(&yep, s);

  if (yep) {
    auto [m, arr] = read_pointer<std::remove_pointer<decltype(x->ddt_vn_apc_pc)>::type>(s);
    x->__f2dace_SA_ddt_vn_apc_pc_d_0_s_300 = m.size.at(0);
    x->__f2dace_SA_ddt_vn_apc_pc_d_1_s_301 = m.size.at(1);
    x->__f2dace_SA_ddt_vn_apc_pc_d_2_s_302 = m.size.at(2);
    x->__f2dace_SA_ddt_vn_apc_pc_d_3_s_303 = m.size.at(3);
    x->__f2dace_SOA_ddt_vn_apc_pc_d_0_s_300 = m.lbound.at(0);
    x->__f2dace_SOA_ddt_vn_apc_pc_d_1_s_301 = m.lbound.at(1);
    x->__f2dace_SOA_ddt_vn_apc_pc_d_2_s_302 = m.lbound.at(2);
    x->__f2dace_SOA_ddt_vn_apc_pc_d_3_s_303 = m.lbound.at(3);
    x->ddt_vn_apc_pc = arr;
  }

  read_line(s, {"# ddt_w_adv_pc"}); // Should contain '# ddt_w_adv_pc'

  read_line(s, {"# assoc"}); // Should contain '# assoc'
  deserialize(&yep, s);

  if (yep) {
    auto [m, arr] = read_pointer<std::remove_pointer<decltype(x->ddt_w_adv_pc)>::type>(s);
    x->__f2dace_SA_ddt_w_adv_pc_d_0_s_304 = m.size.at(0);
    x->__f2dace_SA_ddt_w_adv_pc_d_1_s_305 = m.size.at(1);
    x->__f2dace_SA_ddt_w_adv_pc_d_2_s_306 = m.size.at(2);
    x->__f2dace_SA_ddt_w_adv_pc_d_3_s_307 = m.size.at(3);
    x->__f2dace_SOA_ddt_w_adv_pc_d_0_s_304 = m.lbound.at(0);
    x->__f2dace_SOA_ddt_w_adv_pc_d_1_s_305 = m.lbound.at(1);
    x->__f2dace_SOA_ddt_w_adv_pc_d_2_s_306 = m.lbound.at(2);
    x->__f2dace_SOA_ddt_w_adv_pc_d_3_s_307 = m.lbound.at(3);
    x->ddt_w_adv_pc = arr;
  }

  read_line(s, {"# max_vcfl_dyn"}); // Should contain '# max_vcfl_dyn'

  deserialize(&(x->max_vcfl_dyn), s);
}

void deserialize(t_nh_metrics *x, std::istream &s) {
  bool yep;
  array_meta m;
  read_line(s, {"# ddxn_z_full"}); // Should contain '# ddxn_z_full'

  read_line(s, {"# assoc"}); // Should contain '# assoc'
  deserialize(&yep, s);

  if (yep) {
    auto [m, arr] = read_pointer<std::remove_pointer<decltype(x->ddxn_z_full)>::type>(s);
    x->__f2dace_SA_ddxn_z_full_d_0_s_308 = m.size.at(0);
    x->__f2dace_SA_ddxn_z_full_d_1_s_309 = m.size.at(1);
    x->__f2dace_SA_ddxn_z_full_d_2_s_310 = m.size.at(2);
    x->__f2dace_SOA_ddxn_z_full_d_0_s_308 = m.lbound.at(0);
    x->__f2dace_SOA_ddxn_z_full_d_1_s_309 = m.lbound.at(1);
    x->__f2dace_SOA_ddxn_z_full_d_2_s_310 = m.lbound.at(2);
    x->ddxn_z_full = arr;
  }

  read_line(s, {"# ddxt_z_full"}); // Should contain '# ddxt_z_full'

  read_line(s, {"# assoc"}); // Should contain '# assoc'
  deserialize(&yep, s);

  if (yep) {
    auto [m, arr] = read_pointer<std::remove_pointer<decltype(x->ddxt_z_full)>::type>(s);
    x->__f2dace_SA_ddxt_z_full_d_0_s_311 = m.size.at(0);
    x->__f2dace_SA_ddxt_z_full_d_1_s_312 = m.size.at(1);
    x->__f2dace_SA_ddxt_z_full_d_2_s_313 = m.size.at(2);
    x->__f2dace_SOA_ddxt_z_full_d_0_s_311 = m.lbound.at(0);
    x->__f2dace_SOA_ddxt_z_full_d_1_s_312 = m.lbound.at(1);
    x->__f2dace_SOA_ddxt_z_full_d_2_s_313 = m.lbound.at(2);
    x->ddxt_z_full = arr;
  }

  read_line(s, {"# ddqz_z_full_e"}); // Should contain '# ddqz_z_full_e'

  read_line(s, {"# assoc"}); // Should contain '# assoc'
  deserialize(&yep, s);

  if (yep) {
    auto [m, arr] = read_pointer<std::remove_pointer<decltype(x->ddqz_z_full_e)>::type>(s);
    x->__f2dace_SA_ddqz_z_full_e_d_0_s_314 = m.size.at(0);
    x->__f2dace_SA_ddqz_z_full_e_d_1_s_315 = m.size.at(1);
    x->__f2dace_SA_ddqz_z_full_e_d_2_s_316 = m.size.at(2);
    x->__f2dace_SOA_ddqz_z_full_e_d_0_s_314 = m.lbound.at(0);
    x->__f2dace_SOA_ddqz_z_full_e_d_1_s_315 = m.lbound.at(1);
    x->__f2dace_SOA_ddqz_z_full_e_d_2_s_316 = m.lbound.at(2);
    x->ddqz_z_full_e = arr;
  }

  read_line(s, {"# ddqz_z_half"}); // Should contain '# ddqz_z_half'

  read_line(s, {"# assoc"}); // Should contain '# assoc'
  deserialize(&yep, s);

  if (yep) {
    auto [m, arr] = read_pointer<std::remove_pointer<decltype(x->ddqz_z_half)>::type>(s);
    x->__f2dace_SA_ddqz_z_half_d_0_s_317 = m.size.at(0);
    x->__f2dace_SA_ddqz_z_half_d_1_s_318 = m.size.at(1);
    x->__f2dace_SA_ddqz_z_half_d_2_s_319 = m.size.at(2);
    x->__f2dace_SOA_ddqz_z_half_d_0_s_317 = m.lbound.at(0);
    x->__f2dace_SOA_ddqz_z_half_d_1_s_318 = m.lbound.at(1);
    x->__f2dace_SOA_ddqz_z_half_d_2_s_319 = m.lbound.at(2);
    x->ddqz_z_half = arr;
  }

  read_line(s, {"# wgtfac_c"}); // Should contain '# wgtfac_c'

  read_line(s, {"# assoc"}); // Should contain '# assoc'
  deserialize(&yep, s);

  if (yep) {
    auto [m, arr] = read_pointer<std::remove_pointer<decltype(x->wgtfac_c)>::type>(s);
    x->__f2dace_SA_wgtfac_c_d_0_s_320 = m.size.at(0);
    x->__f2dace_SA_wgtfac_c_d_1_s_321 = m.size.at(1);
    x->__f2dace_SA_wgtfac_c_d_2_s_322 = m.size.at(2);
    x->__f2dace_SOA_wgtfac_c_d_0_s_320 = m.lbound.at(0);
    x->__f2dace_SOA_wgtfac_c_d_1_s_321 = m.lbound.at(1);
    x->__f2dace_SOA_wgtfac_c_d_2_s_322 = m.lbound.at(2);
    x->wgtfac_c = arr;
  }

  read_line(s, {"# wgtfac_e"}); // Should contain '# wgtfac_e'

  read_line(s, {"# assoc"}); // Should contain '# assoc'
  deserialize(&yep, s);

  if (yep) {
    auto [m, arr] = read_pointer<std::remove_pointer<decltype(x->wgtfac_e)>::type>(s);
    x->__f2dace_SA_wgtfac_e_d_0_s_323 = m.size.at(0);
    x->__f2dace_SA_wgtfac_e_d_1_s_324 = m.size.at(1);
    x->__f2dace_SA_wgtfac_e_d_2_s_325 = m.size.at(2);
    x->__f2dace_SOA_wgtfac_e_d_0_s_323 = m.lbound.at(0);
    x->__f2dace_SOA_wgtfac_e_d_1_s_324 = m.lbound.at(1);
    x->__f2dace_SOA_wgtfac_e_d_2_s_325 = m.lbound.at(2);
    x->wgtfac_e = arr;
  }

  read_line(s, {"# wgtfacq_e"}); // Should contain '# wgtfacq_e'

  read_line(s, {"# assoc"}); // Should contain '# assoc'
  deserialize(&yep, s);

  if (yep) {
    auto [m, arr] = read_pointer<std::remove_pointer<decltype(x->wgtfacq_e)>::type>(s);
    x->__f2dace_SA_wgtfacq_e_d_0_s_326 = m.size.at(0);
    x->__f2dace_SA_wgtfacq_e_d_1_s_327 = m.size.at(1);
    x->__f2dace_SA_wgtfacq_e_d_2_s_328 = m.size.at(2);
    x->__f2dace_SOA_wgtfacq_e_d_0_s_326 = m.lbound.at(0);
    x->__f2dace_SOA_wgtfacq_e_d_1_s_327 = m.lbound.at(1);
    x->__f2dace_SOA_wgtfacq_e_d_2_s_328 = m.lbound.at(2);
    x->wgtfacq_e = arr;
  }

  read_line(s, {"# coeff_gradekin"}); // Should contain '# coeff_gradekin'

  read_line(s, {"# assoc"}); // Should contain '# assoc'
  deserialize(&yep, s);

  if (yep) {
    auto [m, arr] = read_pointer<std::remove_pointer<decltype(x->coeff_gradekin)>::type>(s);
    x->__f2dace_SA_coeff_gradekin_d_0_s_329 = m.size.at(0);
    x->__f2dace_SA_coeff_gradekin_d_1_s_330 = m.size.at(1);
    x->__f2dace_SA_coeff_gradekin_d_2_s_331 = m.size.at(2);
    x->__f2dace_SOA_coeff_gradekin_d_0_s_329 = m.lbound.at(0);
    x->__f2dace_SOA_coeff_gradekin_d_1_s_330 = m.lbound.at(1);
    x->__f2dace_SOA_coeff_gradekin_d_2_s_331 = m.lbound.at(2);
    x->coeff_gradekin = arr;
  }

  read_line(s, {"# coeff1_dwdz"}); // Should contain '# coeff1_dwdz'

  read_line(s, {"# assoc"}); // Should contain '# assoc'
  deserialize(&yep, s);

  if (yep) {
    auto [m, arr] = read_pointer<std::remove_pointer<decltype(x->coeff1_dwdz)>::type>(s);
    x->__f2dace_SA_coeff1_dwdz_d_0_s_332 = m.size.at(0);
    x->__f2dace_SA_coeff1_dwdz_d_1_s_333 = m.size.at(1);
    x->__f2dace_SA_coeff1_dwdz_d_2_s_334 = m.size.at(2);
    x->__f2dace_SOA_coeff1_dwdz_d_0_s_332 = m.lbound.at(0);
    x->__f2dace_SOA_coeff1_dwdz_d_1_s_333 = m.lbound.at(1);
    x->__f2dace_SOA_coeff1_dwdz_d_2_s_334 = m.lbound.at(2);
    x->coeff1_dwdz = arr;
  }

  read_line(s, {"# coeff2_dwdz"}); // Should contain '# coeff2_dwdz'

  read_line(s, {"# assoc"}); // Should contain '# assoc'
  deserialize(&yep, s);

  if (yep) {
    auto [m, arr] = read_pointer<std::remove_pointer<decltype(x->coeff2_dwdz)>::type>(s);
    x->__f2dace_SA_coeff2_dwdz_d_0_s_335 = m.size.at(0);
    x->__f2dace_SA_coeff2_dwdz_d_1_s_336 = m.size.at(1);
    x->__f2dace_SA_coeff2_dwdz_d_2_s_337 = m.size.at(2);
    x->__f2dace_SOA_coeff2_dwdz_d_0_s_335 = m.lbound.at(0);
    x->__f2dace_SOA_coeff2_dwdz_d_1_s_336 = m.lbound.at(1);
    x->__f2dace_SOA_coeff2_dwdz_d_2_s_337 = m.lbound.at(2);
    x->coeff2_dwdz = arr;
  }

  read_line(s, {"# deepatmo_gradh_mc"}); // Should contain '# deepatmo_gradh_mc'

  read_line(s, {"# assoc"}); // Should contain '# assoc'
  deserialize(&yep, s);

  if (yep) {
    auto [m, arr] = read_pointer<std::remove_pointer<decltype(x->deepatmo_gradh_mc)>::type>(s);
    x->__f2dace_SA_deepatmo_gradh_mc_d_0_s_338 = m.size.at(0);
    x->__f2dace_SOA_deepatmo_gradh_mc_d_0_s_338 = m.lbound.at(0);
    x->deepatmo_gradh_mc = arr;
  }

  read_line(s, {"# deepatmo_invr_mc"}); // Should contain '# deepatmo_invr_mc'

  read_line(s, {"# assoc"}); // Should contain '# assoc'
  deserialize(&yep, s);

  if (yep) {
    auto [m, arr] = read_pointer<std::remove_pointer<decltype(x->deepatmo_invr_mc)>::type>(s);
    x->__f2dace_SA_deepatmo_invr_mc_d_0_s_339 = m.size.at(0);
    x->__f2dace_SOA_deepatmo_invr_mc_d_0_s_339 = m.lbound.at(0);
    x->deepatmo_invr_mc = arr;
  }

  read_line(s, {"# deepatmo_gradh_ifc"}); // Should contain '# deepatmo_gradh_ifc'

  read_line(s, {"# assoc"}); // Should contain '# assoc'
  deserialize(&yep, s);

  if (yep) {
    auto [m, arr] = read_pointer<std::remove_pointer<decltype(x->deepatmo_gradh_ifc)>::type>(s);
    x->__f2dace_SA_deepatmo_gradh_ifc_d_0_s_340 = m.size.at(0);
    x->__f2dace_SOA_deepatmo_gradh_ifc_d_0_s_340 = m.lbound.at(0);
    x->deepatmo_gradh_ifc = arr;
  }

  read_line(s, {"# deepatmo_invr_ifc"}); // Should contain '# deepatmo_invr_ifc'

  read_line(s, {"# assoc"}); // Should contain '# assoc'
  deserialize(&yep, s);

  if (yep) {
    auto [m, arr] = read_pointer<std::remove_pointer<decltype(x->deepatmo_invr_ifc)>::type>(s);
    x->__f2dace_SA_deepatmo_invr_ifc_d_0_s_341 = m.size.at(0);
    x->__f2dace_SOA_deepatmo_invr_ifc_d_0_s_341 = m.lbound.at(0);
    x->deepatmo_invr_ifc = arr;
  }
}

template <typename T> void add_line(const T &x, std::ostream &s, bool trailing_newline = true) {
  s << x;
  if (trailing_newline)
    s << std::endl;
}
void add_line(long long x, std::ostream &s, bool trailing_newline = true) {
  s << x;
  if (trailing_newline)
    s << std::endl;
}
void add_line(long double x, std::ostream &s, bool trailing_newline = true) {
  s << std::setprecision(20) << x;
  if (trailing_newline)
    s << std::endl;
}
void add_line(bool x, std::ostream &s, bool trailing_newline = true) { add_line(int(x), s, trailing_newline); }
template <typename T> std::string serialize(const T *x) {
  if constexpr (std::is_pointer_v<T>) {
    return serialize(*x);
  } else {
    std::stringstream s;
    add_line(*x, s, false);
    return s.str();
  }
}
std::string serialize(int x) {
  std::stringstream s;
  s << x;
  return s.str();
}
std::string serialize(long x) {
  std::stringstream s;
  s << x;
  return s.str();
}
std::string serialize(long long x) {
  std::stringstream s;
  s << x;
  return s.str();
}
std::string serialize(float x) {
  std::stringstream s;
  s << std::setprecision(20) << x;
  return s.str();
}
std::string serialize(double x) {
  std::stringstream s;
  s << std::setprecision(20) << x;
  return s.str();
}
std::string serialize(long double x) {
  std::stringstream s;
  s << std::setprecision(20) << x;
  return s.str();
}
std::string serialize(bool x) { return serialize(int(x)); }

std::string serialize(const t_grid_domain_decomp_info *x) {
  std::stringstream s;
  add_line("# owner_mask", s);

  add_line("# alloc", s);
  add_line(serialize(x->owner_mask != nullptr), s);
  if (x->owner_mask) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->owner_mask);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->owner_mask[i]), s);
      }
    }

  } // CONCLUDING IF
  std::string out = s.str();
  if (out.length() > 0)
    out.pop_back();
  return out;
}

std::string serialize(const t_int_state *x) {
  std::stringstream s;
  add_line("# c_lin_e", s);

  add_line("# alloc", s);
  add_line(serialize(x->c_lin_e != nullptr), s);
  if (x->c_lin_e) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->c_lin_e);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->c_lin_e[i]), s);
      }
    }

  } // CONCLUDING IF
  add_line("# e_bln_c_s", s);

  add_line("# alloc", s);
  add_line(serialize(x->e_bln_c_s != nullptr), s);
  if (x->e_bln_c_s) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->e_bln_c_s);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->e_bln_c_s[i]), s);
      }
    }

  } // CONCLUDING IF
  add_line("# cells_aw_verts", s);

  add_line("# alloc", s);
  add_line(serialize(x->cells_aw_verts != nullptr), s);
  if (x->cells_aw_verts) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->cells_aw_verts);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->cells_aw_verts[i]), s);
      }
    }

  } // CONCLUDING IF
  add_line("# rbf_vec_coeff_e", s);

  add_line("# alloc", s);
  add_line(serialize(x->rbf_vec_coeff_e != nullptr), s);
  if (x->rbf_vec_coeff_e) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->rbf_vec_coeff_e);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->rbf_vec_coeff_e[i]), s);
      }
    }

  } // CONCLUDING IF
  add_line("# geofac_grdiv", s);

  add_line("# alloc", s);
  add_line(serialize(x->geofac_grdiv != nullptr), s);
  if (x->geofac_grdiv) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->geofac_grdiv);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->geofac_grdiv[i]), s);
      }
    }

  } // CONCLUDING IF
  add_line("# geofac_rot", s);

  add_line("# alloc", s);
  add_line(serialize(x->geofac_rot != nullptr), s);
  if (x->geofac_rot) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->geofac_rot);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->geofac_rot[i]), s);
      }
    }

  } // CONCLUDING IF
  add_line("# geofac_n2s", s);

  add_line("# alloc", s);
  add_line(serialize(x->geofac_n2s != nullptr), s);
  if (x->geofac_n2s) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->geofac_n2s);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->geofac_n2s[i]), s);
      }
    }

  } // CONCLUDING IF
  std::string out = s.str();
  if (out.length() > 0)
    out.pop_back();
  return out;
}

std::string serialize(const t_grid_cells *x) {
  std::stringstream s;
  add_line("# neighbor_idx", s);

  add_line("# alloc", s);
  add_line(serialize(x->neighbor_idx != nullptr), s);
  if (x->neighbor_idx) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->neighbor_idx);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->neighbor_idx[i]), s);
      }
    }

  } // CONCLUDING IF
  add_line("# neighbor_blk", s);

  add_line("# alloc", s);
  add_line(serialize(x->neighbor_blk != nullptr), s);
  if (x->neighbor_blk) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->neighbor_blk);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->neighbor_blk[i]), s);
      }
    }

  } // CONCLUDING IF
  add_line("# edge_idx", s);

  add_line("# alloc", s);
  add_line(serialize(x->edge_idx != nullptr), s);
  if (x->edge_idx) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->edge_idx);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->edge_idx[i]), s);
      }
    }

  } // CONCLUDING IF
  add_line("# edge_blk", s);

  add_line("# alloc", s);
  add_line(serialize(x->edge_blk != nullptr), s);
  if (x->edge_blk) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->edge_blk);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->edge_blk[i]), s);
      }
    }

  } // CONCLUDING IF
  add_line("# area", s);

  add_line("# assoc", s);
  add_line(serialize(x->area != nullptr), s);

  if (x->area)
    add_line(serialize_array(x->area), s);

  add_line("# start_index", s);

  add_line("# alloc", s);
  add_line(serialize(x->start_index != nullptr), s);
  if (x->start_index) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->start_index);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->start_index[i]), s);
      }
    }

  } // CONCLUDING IF
  add_line("# end_index", s);

  add_line("# alloc", s);
  add_line(serialize(x->end_index != nullptr), s);
  if (x->end_index) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->end_index);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->end_index[i]), s);
      }
    }

  } // CONCLUDING IF
  add_line("# start_block", s);

  add_line("# alloc", s);
  add_line(serialize(x->start_block != nullptr), s);
  if (x->start_block) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->start_block);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->start_block[i]), s);
      }
    }

  } // CONCLUDING IF
  add_line("# end_block", s);

  add_line("# alloc", s);
  add_line(serialize(x->end_block != nullptr), s);
  if (x->end_block) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->end_block);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->end_block[i]), s);
      }
    }

  } // CONCLUDING IF
  add_line("# decomp_info", s);
  add_line(serialize(x->decomp_info), s);
  std::string out = s.str();
  if (out.length() > 0)
    out.pop_back();
  return out;
}

std::string serialize(const t_grid_edges *x) {
  std::stringstream s;
  add_line("# cell_idx", s);

  add_line("# alloc", s);
  add_line(serialize(x->cell_idx != nullptr), s);
  if (x->cell_idx) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->cell_idx);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->cell_idx[i]), s);
      }
    }

  } // CONCLUDING IF
  add_line("# cell_blk", s);

  add_line("# alloc", s);
  add_line(serialize(x->cell_blk != nullptr), s);
  if (x->cell_blk) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->cell_blk);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->cell_blk[i]), s);
      }
    }

  } // CONCLUDING IF
  add_line("# vertex_idx", s);

  add_line("# alloc", s);
  add_line(serialize(x->vertex_idx != nullptr), s);
  if (x->vertex_idx) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->vertex_idx);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->vertex_idx[i]), s);
      }
    }

  } // CONCLUDING IF
  add_line("# vertex_blk", s);

  add_line("# alloc", s);
  add_line(serialize(x->vertex_blk != nullptr), s);
  if (x->vertex_blk) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->vertex_blk);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->vertex_blk[i]), s);
      }
    }

  } // CONCLUDING IF
  add_line("# tangent_orientation", s);

  add_line("# alloc", s);
  add_line(serialize(x->tangent_orientation != nullptr), s);
  if (x->tangent_orientation) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->tangent_orientation);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->tangent_orientation[i]), s);
      }
    }

  } // CONCLUDING IF
  add_line("# quad_idx", s);

  add_line("# alloc", s);
  add_line(serialize(x->quad_idx != nullptr), s);
  if (x->quad_idx) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->quad_idx);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->quad_idx[i]), s);
      }
    }

  } // CONCLUDING IF
  add_line("# quad_blk", s);

  add_line("# alloc", s);
  add_line(serialize(x->quad_blk != nullptr), s);
  if (x->quad_blk) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->quad_blk);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->quad_blk[i]), s);
      }
    }

  } // CONCLUDING IF
  add_line("# inv_primal_edge_length", s);

  add_line("# alloc", s);
  add_line(serialize(x->inv_primal_edge_length != nullptr), s);
  if (x->inv_primal_edge_length) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->inv_primal_edge_length);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->inv_primal_edge_length[i]), s);
      }
    }

  } // CONCLUDING IF
  add_line("# inv_dual_edge_length", s);

  add_line("# alloc", s);
  add_line(serialize(x->inv_dual_edge_length != nullptr), s);
  if (x->inv_dual_edge_length) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->inv_dual_edge_length);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->inv_dual_edge_length[i]), s);
      }
    }

  } // CONCLUDING IF
  add_line("# area_edge", s);

  add_line("# alloc", s);
  add_line(serialize(x->area_edge != nullptr), s);
  if (x->area_edge) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->area_edge);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->area_edge[i]), s);
      }
    }

  } // CONCLUDING IF
  add_line("# f_e", s);

  add_line("# alloc", s);
  add_line(serialize(x->f_e != nullptr), s);
  if (x->f_e) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->f_e);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->f_e[i]), s);
      }
    }

  } // CONCLUDING IF
  add_line("# fn_e", s);

  add_line("# alloc", s);
  add_line(serialize(x->fn_e != nullptr), s);
  if (x->fn_e) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->fn_e);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->fn_e[i]), s);
      }
    }

  } // CONCLUDING IF
  add_line("# ft_e", s);

  add_line("# alloc", s);
  add_line(serialize(x->ft_e != nullptr), s);
  if (x->ft_e) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->ft_e);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->ft_e[i]), s);
      }
    }

  } // CONCLUDING IF
  add_line("# start_index", s);

  add_line("# alloc", s);
  add_line(serialize(x->start_index != nullptr), s);
  if (x->start_index) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->start_index);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->start_index[i]), s);
      }
    }

  } // CONCLUDING IF
  add_line("# end_index", s);

  add_line("# alloc", s);
  add_line(serialize(x->end_index != nullptr), s);
  if (x->end_index) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->end_index);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->end_index[i]), s);
      }
    }

  } // CONCLUDING IF
  add_line("# start_block", s);

  add_line("# alloc", s);
  add_line(serialize(x->start_block != nullptr), s);
  if (x->start_block) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->start_block);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->start_block[i]), s);
      }
    }

  } // CONCLUDING IF
  add_line("# end_block", s);

  add_line("# alloc", s);
  add_line(serialize(x->end_block != nullptr), s);
  if (x->end_block) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->end_block);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->end_block[i]), s);
      }
    }

  } // CONCLUDING IF
  std::string out = s.str();
  if (out.length() > 0)
    out.pop_back();
  return out;
}

std::string serialize(const t_grid_vertices *x) {
  std::stringstream s;
  add_line("# cell_idx", s);

  add_line("# alloc", s);
  add_line(serialize(x->cell_idx != nullptr), s);
  if (x->cell_idx) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->cell_idx);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->cell_idx[i]), s);
      }
    }

  } // CONCLUDING IF
  add_line("# cell_blk", s);

  add_line("# alloc", s);
  add_line(serialize(x->cell_blk != nullptr), s);
  if (x->cell_blk) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->cell_blk);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->cell_blk[i]), s);
      }
    }

  } // CONCLUDING IF
  add_line("# edge_idx", s);

  add_line("# alloc", s);
  add_line(serialize(x->edge_idx != nullptr), s);
  if (x->edge_idx) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->edge_idx);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->edge_idx[i]), s);
      }
    }

  } // CONCLUDING IF
  add_line("# edge_blk", s);

  add_line("# alloc", s);
  add_line(serialize(x->edge_blk != nullptr), s);
  if (x->edge_blk) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->edge_blk);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->edge_blk[i]), s);
      }
    }

  } // CONCLUDING IF
  add_line("# start_index", s);

  add_line("# alloc", s);
  add_line(serialize(x->start_index != nullptr), s);
  if (x->start_index) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->start_index);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->start_index[i]), s);
      }
    }

  } // CONCLUDING IF
  add_line("# end_index", s);

  add_line("# alloc", s);
  add_line(serialize(x->end_index != nullptr), s);
  if (x->end_index) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->end_index);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->end_index[i]), s);
      }
    }

  } // CONCLUDING IF
  add_line("# start_block", s);

  add_line("# alloc", s);
  add_line(serialize(x->start_block != nullptr), s);
  if (x->start_block) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->start_block);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->start_block[i]), s);
      }
    }

  } // CONCLUDING IF
  add_line("# end_block", s);

  add_line("# alloc", s);
  add_line(serialize(x->end_block != nullptr), s);
  if (x->end_block) { // BEGINING IF

    {
      const array_meta &m = ARRAY_META_DICT_AT(x->end_block);
      add_line("# rank", s);
      add_line(m.rank, s);
      add_line("# size", s);
      for (auto i : m.size)
        add_line(i, s);
      add_line("# lbound", s);
      for (auto i : m.lbound)
        add_line(i, s);
      add_line("# entries", s);
      for (int i = 0; i < m.volume(); ++i) {
        add_line(serialize(x->end_block[i]), s);
      }
    }

  } // CONCLUDING IF
  std::string out = s.str();
  if (out.length() > 0)
    out.pop_back();
  return out;
}

std::string serialize(const t_patch *x) {
  std::stringstream s;
  add_line("# nblks_c", s);
  add_line(serialize(x->nblks_c), s);
  add_line("# nblks_e", s);
  add_line(serialize(x->nblks_e), s);
  add_line("# nblks_v", s);
  add_line(serialize(x->nblks_v), s);
  add_line("# cells", s);
  add_line(serialize(x->cells), s);
  add_line("# edges", s);
  add_line(serialize(x->edges), s);
  add_line("# verts", s);
  add_line(serialize(x->verts), s);
  std::string out = s.str();
  if (out.length() > 0)
    out.pop_back();
  return out;
}

std::string serialize(const t_nh_prog *x) {
  std::stringstream s;
  add_line("# w", s);

  add_line("# assoc", s);
  add_line(serialize(x->w != nullptr), s);

  if (x->w)
    add_line(serialize_array(x->w), s);

  add_line("# vn", s);

  add_line("# assoc", s);
  add_line(serialize(x->vn != nullptr), s);

  if (x->vn)
    add_line(serialize_array(x->vn), s);

  std::string out = s.str();
  if (out.length() > 0)
    out.pop_back();
  return out;
}

std::string serialize(const t_nh_diag *x) {
  std::stringstream s;
  add_line("# vt", s);

  add_line("# assoc", s);
  add_line(serialize(x->vt != nullptr), s);

  if (x->vt)
    add_line(serialize_array(x->vt), s);

  add_line("# vn_ie", s);

  add_line("# assoc", s);
  add_line(serialize(x->vn_ie != nullptr), s);

  if (x->vn_ie)
    add_line(serialize_array(x->vn_ie), s);

  add_line("# w_concorr_c", s);

  add_line("# assoc", s);
  add_line(serialize(x->w_concorr_c != nullptr), s);

  if (x->w_concorr_c)
    add_line(serialize_array(x->w_concorr_c), s);

  add_line("# ddt_vn_apc_pc", s);

  add_line("# assoc", s);
  add_line(serialize(x->ddt_vn_apc_pc != nullptr), s);

  if (x->ddt_vn_apc_pc)
    add_line(serialize_array(x->ddt_vn_apc_pc), s);

  add_line("# ddt_w_adv_pc", s);

  add_line("# assoc", s);
  add_line(serialize(x->ddt_w_adv_pc != nullptr), s);

  if (x->ddt_w_adv_pc)
    add_line(serialize_array(x->ddt_w_adv_pc), s);

  add_line("# max_vcfl_dyn", s);
  add_line(serialize(x->max_vcfl_dyn), s);
  std::string out = s.str();
  if (out.length() > 0)
    out.pop_back();
  return out;
}

std::string serialize(const t_nh_metrics *x) {
  std::stringstream s;
  add_line("# ddxn_z_full", s);

  add_line("# assoc", s);
  add_line(serialize(x->ddxn_z_full != nullptr), s);

  if (x->ddxn_z_full)
    add_line(serialize_array(x->ddxn_z_full), s);

  add_line("# ddxt_z_full", s);

  add_line("# assoc", s);
  add_line(serialize(x->ddxt_z_full != nullptr), s);

  if (x->ddxt_z_full)
    add_line(serialize_array(x->ddxt_z_full), s);

  add_line("# ddqz_z_full_e", s);

  add_line("# assoc", s);
  add_line(serialize(x->ddqz_z_full_e != nullptr), s);

  if (x->ddqz_z_full_e)
    add_line(serialize_array(x->ddqz_z_full_e), s);

  add_line("# ddqz_z_half", s);

  add_line("# assoc", s);
  add_line(serialize(x->ddqz_z_half != nullptr), s);

  if (x->ddqz_z_half)
    add_line(serialize_array(x->ddqz_z_half), s);

  add_line("# wgtfac_c", s);

  add_line("# assoc", s);
  add_line(serialize(x->wgtfac_c != nullptr), s);

  if (x->wgtfac_c)
    add_line(serialize_array(x->wgtfac_c), s);

  add_line("# wgtfac_e", s);

  add_line("# assoc", s);
  add_line(serialize(x->wgtfac_e != nullptr), s);

  if (x->wgtfac_e)
    add_line(serialize_array(x->wgtfac_e), s);

  add_line("# wgtfacq_e", s);

  add_line("# assoc", s);
  add_line(serialize(x->wgtfacq_e != nullptr), s);

  if (x->wgtfacq_e)
    add_line(serialize_array(x->wgtfacq_e), s);

  add_line("# coeff_gradekin", s);

  add_line("# assoc", s);
  add_line(serialize(x->coeff_gradekin != nullptr), s);

  if (x->coeff_gradekin)
    add_line(serialize_array(x->coeff_gradekin), s);

  add_line("# coeff1_dwdz", s);

  add_line("# assoc", s);
  add_line(serialize(x->coeff1_dwdz != nullptr), s);

  if (x->coeff1_dwdz)
    add_line(serialize_array(x->coeff1_dwdz), s);

  add_line("# coeff2_dwdz", s);

  add_line("# assoc", s);
  add_line(serialize(x->coeff2_dwdz != nullptr), s);

  if (x->coeff2_dwdz)
    add_line(serialize_array(x->coeff2_dwdz), s);

  add_line("# deepatmo_gradh_mc", s);

  add_line("# assoc", s);
  add_line(serialize(x->deepatmo_gradh_mc != nullptr), s);

  if (x->deepatmo_gradh_mc)
    add_line(serialize_array(x->deepatmo_gradh_mc), s);

  add_line("# deepatmo_invr_mc", s);

  add_line("# assoc", s);
  add_line(serialize(x->deepatmo_invr_mc != nullptr), s);

  if (x->deepatmo_invr_mc)
    add_line(serialize_array(x->deepatmo_invr_mc), s);

  add_line("# deepatmo_gradh_ifc", s);

  add_line("# assoc", s);
  add_line(serialize(x->deepatmo_gradh_ifc != nullptr), s);

  if (x->deepatmo_gradh_ifc)
    add_line(serialize_array(x->deepatmo_gradh_ifc), s);

  add_line("# deepatmo_invr_ifc", s);

  add_line("# assoc", s);
  add_line(serialize(x->deepatmo_invr_ifc != nullptr), s);

  if (x->deepatmo_invr_ifc)
    add_line(serialize_array(x->deepatmo_invr_ifc), s);

  std::string out = s.str();
  if (out.length() > 0)
    out.pop_back();
  return out;
}

template <typename T> T *array_meta::read(std::istream &s) const {
  auto *buf = new T[volume()];
  if constexpr (std::is_pointer_v<T>) {
    auto *bufc = read<std::remove_pointer_t<T>>(s);
    for (int i = 0; i < volume(); ++i) {
      buf[i] = &bufc[i];
    }
  } else {
    read_line(s, {"# entries"});
    for (int i = 0; i < volume(); ++i) {
      deserialize(&buf[i], s);
    }
    (*ARRAY_META_DICT())[buf] = *this;
  }
  return buf;
}

template <typename T> std::string serialize_array(T *arr) {
  const auto m = ARRAY_META_DICT_AT(static_cast<void *>(arr));
  std::stringstream s;
  add_line("# rank", s);
  add_line(m.rank, s);
  add_line("# size", s);
  for (auto i : m.size)
    add_line(i, s);
  add_line("# lbound", s);
  for (auto i : m.lbound)
    add_line(i, s);
  add_line("# entries", s);
  for (int i = 0; i < m.volume(); ++i)
    add_line(serialize(arr[i]), s);
  return s.str();
}

void deserialize_global_data(global_data_type *g, std::istream &s) {
  {
    read_line(s, "# nflatlev");
    auto [m, arr] = read_array<int>(s);
    g->nflatlev = arr;
  }

  read_line(s, "# i_am_accel_node");
  deserialize(g->i_am_accel_node, s);

  read_line(s, "# lextra_diffu");
  deserialize(g->lextra_diffu, s);

  read_line(s, "# nproma");
  deserialize(g->nproma, s);

  read_line(s, "# timers_level");
  deserialize(g->timers_level, s);

  read_line(s, "# timer_solve_nh_veltend");
  deserialize(g->timer_solve_nh_veltend, s);

  read_line(s, "# timer_intp");
  deserialize(g->timer_intp, s);

  {
    read_line(s, "# nrdmax");
    auto [m, arr] = read_array<int>(s);
    g->nrdmax = arr;
  }
}

std::string serialize_global_data(const global_data_type *g) {
  std::stringstream s;

  add_line(serialize_array(g->nflatlev), s);

  add_line(serialize(g->i_am_accel_node), s);

  add_line(serialize(g->lextra_diffu), s);

  add_line(serialize(g->nproma), s);

  add_line(serialize(g->timers_level), s);

  add_line(serialize(g->timer_solve_nh_veltend), s);

  add_line(serialize(g->timer_intp), s);

  add_line(serialize_array(g->nrdmax), s);

  return s.str();
}

enum class SerializationType { INVALID, PLAIN, CONST_INJECTION, F90_MODULE };

std::string serialize_consistent_global_data(std::vector<const global_data_type *> &gs,
                                             SerializationType serialization_type = SerializationType::INVALID) {
  assert(serialization_type != SerializationType::INVALID);
  if (gs.empty())
    return "";

  std::map<std::string, std::set<std::string>> consistent;
  for (const auto *g : gs) {
    consistent["mo_mpi.i_am_accel_node"].insert((g->i_am_accel_node ? ".true." : ".false."));

    consistent["mo_nonhydrostatic_config.lextra_diffu"].insert((g->lextra_diffu ? ".true." : ".false."));

    consistent["mo_parallel_config.nproma"].insert(serialize(g->nproma));

    consistent["mo_run_config.timers_level"].insert(serialize(g->timers_level));

    consistent["mo_timer.timer_solve_nh_veltend"].insert(serialize(g->timer_solve_nh_veltend));

    consistent["mo_timer.timer_intp"].insert(serialize(g->timer_intp));
  }

  std::stringstream s;
  if (serialization_type == SerializationType::F90_MODULE) {
    s << R"(
module global_data_assertion
contains
subroutine assert_global_data()
)";
    for (const auto &[k, vs] : consistent) {
      std::vector<std::string_view> parts = split(k, '.');
      assert(parts.size() == 2);
      const auto mname = std::string_view(parts[0]);
      s << "use " << mname << std::endl;
    }
    s << R"(
implicit none
)";
  }
  for (const auto &[k, vs] : consistent) {
    if (vs.size() != 1)
      continue;
    const auto &v = *vs.begin();
    if (serialization_type == SerializationType::PLAIN) {
      s << k << " = " << v << std::endl;
    } else if (serialization_type == SerializationType::CONST_INJECTION) {
      const std::string vval = (v == ".true." ? "1" : (v == ".false." ? "0" : v));
      s << R"({ "type": "ConstInstanceInjection", "scope": null, )";
      s << R"("root": ")" << k << R"(", "component": null, "value": ")" << vval << R"(" })" << std::endl;
    } else if (serialization_type == SerializationType::F90_MODULE) {
      std::vector<std::string_view> parts = split(k, '.');
      assert(parts.size() == 2);
      const auto vname = std::string_view(parts[1]);
      const auto neqop = (v == ".true." || v == ".false.") ? " .neqv. " : " .ne. ";
      s << "if (" << vname << neqop << v << ") then" << std::endl;
      s << R"(	print *, "mismatched )" << vname << "; want " << v << R"(, got: ", )" << vname << std::endl;
      s << "	call abort" << std::endl << "endif" << std::endl;
    }
  }
  if (serialization_type == SerializationType::F90_MODULE) {
    s << R"(
end subroutine assert_global_data
end module global_data_assertion
)";
  }
  return s.str();
}

} // namespace serde

#endif // __DACE_SERDE__
