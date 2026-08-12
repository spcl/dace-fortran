// Walk every *.data file under <data_dir> and emit a CSV of three record
// kinds used by the extent-dealiasing passes:
//
//   kind,field,dim,values
//   size,<field>,<dim>,<v0;v1;...>        per-dim array size observations
//   lbound,<field>,<dim>,<v0;v1;...>      per-dim array lbound observations
//   scalar,<name>,0,<v0;v1;...>           whole-file scalars like nproma/nlev
//
// ``values`` is a semicolon-separated list of the distinct values
// observed across every ``.data`` file scanned.
//
// The .data file format (from serde_velocity_no_nproma.h) uses:
//   # <field_name>
//   [# alloc / # missing  for allocatables / pointers]
//   # rank
//   <N>
//   # size
//   <s0> ... <sN-1>
//   # lbound
//   <l0> ... <lN-1>
//   ...array contents...
//
// Nested structs concatenate headers; the leaf field for a given array
// is the nearest non-structural `# <name>` before its `# rank` block.
// Top-level scalars appear as `# <name>\n<number>\n` with no enclosing
// `# rank` block.
//
// Build: g++ -O2 -std=c++17 tools/analyze_lbounds.cpp -o tools/analyze_lbounds
// Run:   tools/analyze_lbounds data_r02b05 baseline/lbounds.csv

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static const std::set<std::string> STRUCTURAL = {
    "rank", "size", "lbound", "alloc", "missing", "entries"
};

// Known top-level scalar names we care about for SA-symbol dealiasing.
// (nproma, nlev, nblks_c, nblks_e, nblks_v; nlev+1 is derived.)
static const std::set<std::string> KNOWN_SCALARS = {
    "nproma", "nlev", "nblks_c", "nblks_e", "nblks_v"
};

static std::string trim(const std::string& s) {
    auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

static bool is_int_line(const std::string& s) {
    std::string t = trim(s);
    if (t.empty()) return false;
    std::size_t i = 0;
    if (t[0] == '-' || t[0] == '+') ++i;
    if (i == t.size()) return false;
    for (; i < t.size(); ++i)
        if (!std::isdigit(static_cast<unsigned char>(t[i]))) return false;
    return true;
}

struct ArrayRec {
    std::string field;
    int dim;
    int size;
    int lbound;
};

struct FileResult {
    std::vector<ArrayRec> arrays;
    std::vector<std::pair<std::string, int>> scalars;
};

static FileResult scan_file(const fs::path& path) {
    FileResult out;
    std::ifstream f(path);
    if (!f) return out;

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(f, line)) lines.push_back(line);

    // Pass 1: scalar headers (``# <known>\n<int>\n``).
    for (std::size_t i = 0; i + 1 < lines.size(); ++i) {
        std::string t = trim(lines[i]);
        if (t.size() < 2 || t[0] != '#') continue;
        std::string name = trim(t.substr(1));
        if (!KNOWN_SCALARS.count(name)) continue;
        if (!is_int_line(lines[i + 1])) continue;
        out.scalars.emplace_back(name, std::stoi(trim(lines[i + 1])));
    }

    // Pass 2: per-array (size, lbound). Anchor on "# rank"; look back for
    // the nearest non-structural "# <field>" header, then forward through
    // "# size" and "# lbound" blocks.
    for (std::size_t i = 0; i < lines.size(); ++i) {
        if (trim(lines[i]) != "# rank") continue;
        if (i + 1 >= lines.size() || !is_int_line(lines[i + 1])) continue;
        int rank = std::stoi(trim(lines[i + 1]));
        if (rank <= 0) continue;

        // Walk back for field name.
        std::string field;
        for (long j = (long)i - 1; j >= 0; --j) {
            std::string t = trim(lines[(std::size_t)j]);
            if (t.size() < 2 || t[0] != '#') continue;
            std::string name = trim(t.substr(1));
            if (!STRUCTURAL.count(name)) { field = name; break; }
        }
        if (field.empty()) continue;

        std::vector<int> sizes, lbounds;
        std::size_t cursor = i + 2;  // past "# rank" + rank value
        auto read_block = [&](const std::string& header,
                              std::vector<int>& dst) -> bool {
            while (cursor < lines.size() && trim(lines[cursor]) != header) ++cursor;
            if (cursor >= lines.size()) return false;
            ++cursor;  // past the header line
            for (int k = 0; k < rank && cursor < lines.size(); ++k, ++cursor) {
                if (!is_int_line(lines[cursor])) return false;
                dst.push_back(std::stoi(trim(lines[cursor])));
            }
            return (int)dst.size() == rank;
        };
        if (!read_block("# size", sizes)) continue;
        if (!read_block("# lbound", lbounds)) continue;

        for (int d = 0; d < rank; ++d)
            out.arrays.push_back({field, d, sizes[d], lbounds[d]});
    }

    return out;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <data_dir> [out.csv]\n";
        return 1;
    }
    fs::path data_dir = argv[1];
    if (!fs::is_directory(data_dir)) {
        std::cerr << "not a directory: " << data_dir << "\n";
        return 1;
    }

    std::ostream* os = &std::cout;
    std::ofstream of;
    if (argc >= 3) {
        fs::create_directories(fs::path(argv[2]).parent_path());
        of.open(argv[2]);
        if (!of) { std::cerr << "cannot open " << argv[2] << "\n"; return 1; }
        os = &of;
    }

    // Dedup by filename prefix: ``<prefix>.<timestep>.data`` files share
    // identical metadata across timesteps, so one per prefix is enough.
    auto prefix_of = [](const fs::path& p) {
        std::string stem = p.filename().string();
        if (stem.size() > 5 && stem.substr(stem.size() - 5) == ".data")
            stem = stem.substr(0, stem.size() - 5);
        auto dot = stem.find_last_of('.');
        if (dot != std::string::npos) {
            std::string tail = stem.substr(dot + 1);
            bool all_digits = !tail.empty()
                && std::all_of(tail.begin(), tail.end(),
                               [](char c){ return std::isdigit((unsigned char)c); });
            if (all_digits) stem = stem.substr(0, dot);
        }
        return stem;
    };

    std::map<std::pair<std::string, int>, std::set<int>> sizes;
    std::map<std::pair<std::string, int>, std::set<int>> lbounds;
    std::map<std::string, std::set<int>> scalars;

    std::set<std::string> seen_prefixes;
    std::size_t n_files = 0, n_scanned = 0;
    for (auto& ent : fs::directory_iterator(data_dir)) {
        if (!ent.is_regular_file()) continue;
        if (ent.path().extension() != ".data") continue;
        ++n_files;
        if (!seen_prefixes.insert(prefix_of(ent.path())).second) continue;
        ++n_scanned;
        auto r = scan_file(ent.path());
        for (auto& a : r.arrays) {
            sizes[{a.field, a.dim}].insert(a.size);
            lbounds[{a.field, a.dim}].insert(a.lbound);
        }
        for (auto& [name, value] : r.scalars) scalars[name].insert(value);
    }

    auto emit_values = [&](const std::set<int>& vs) {
        bool first = true;
        for (int v : vs) { if (!first) *os << ";"; *os << v; first = false; }
    };

    *os << "kind,field,dim,values\n";
    for (auto& [key, vs] : sizes) {
        *os << "size," << key.first << "," << key.second << ",";
        emit_values(vs);
        *os << "\n";
    }
    for (auto& [key, vs] : lbounds) {
        *os << "lbound," << key.first << "," << key.second << ",";
        emit_values(vs);
        *os << "\n";
    }
    for (auto& [name, vs] : scalars) {
        *os << "scalar," << name << ",0,";
        emit_values(vs);
        *os << "\n";
    }

    std::cerr << "scanned " << n_scanned << " of " << n_files
              << " .data files; " << sizes.size() << " array fields, "
              << scalars.size() << " scalars\n";
    return 0;
}
