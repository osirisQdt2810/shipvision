#include "shipvision/tracking/association.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace shipvision::tracking {

    namespace {

        struct BoxPair {
                float overlap;
                float union_area;
        };

        float area_of(const float* box) {
            const float width = std::max(box[2] - box[0], 0.f);
            const float height = std::max(box[3] - box[1], 0.f);
            return width * height;
        }

        BoxPair intersect(const float* a, const float* b) {
            const float left = std::max(a[0], b[0]);
            const float top = std::max(a[1], b[1]);
            const float right = std::min(a[2], b[2]);
            const float bottom = std::min(a[3], b[3]);
            const float overlap = std::max(right - left, 0.f) * std::max(bottom - top, 0.f);
            return BoxPair{overlap, area_of(a) + area_of(b) - overlap};
        }

    }  // namespace

    std::vector<float> gather_boxes(const float* boxes, const std::vector<int>& indices) {
        std::vector<float> gathered(indices.size() * 4);
        for (size_t index = 0; index < indices.size(); ++index) {
            for (int i = 0; i < 4; ++i)
                gathered[index * 4 + i] = boxes[static_cast<size_t>(indices[index]) * 4 + i];
        }
        return gathered;
    }

    std::vector<float> gather_submatrix(const float* matrix, int m, const std::vector<int>& rows,
                                        const std::vector<int>& columns) {
        std::vector<float> gathered(rows.size() * columns.size());
        for (size_t r = 0; r < rows.size(); ++r) {
            for (size_t c = 0; c < columns.size(); ++c) {
                gathered[r * columns.size() + c] =
                    matrix[static_cast<size_t>(rows[r]) * m + columns[c]];
            }
        }
        return gathered;
    }

    std::vector<float> iou_matrix(const float* boxes_a, int n, const float* boxes_b, int m) {
        std::vector<float> result(static_cast<size_t>(n) * m, 0.f);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                const BoxPair pair = intersect(boxes_a + i * 4, boxes_b + j * 4);
                // The 1e-9 floor is the numpy version's `np.maximum(union, 1e-9)`: two
                // zero-area boxes are legal input and 0/0 would put a NaN in the cost matrix,
                // which the solver would then treat as comparing false against every
                // threshold — so the pair is never matched and never reported.
                result[static_cast<size_t>(i) * m + j] =
                    pair.overlap / std::max(pair.union_area, 1e-9f);
            }
        }
        return result;
    }

    std::vector<float> giou_matrix(const float* boxes_a, int n, const float* boxes_b, int m) {
        std::vector<float> result(static_cast<size_t>(n) * m, 0.f);
        for (int i = 0; i < n; ++i) {
            const float* a = boxes_a + i * 4;
            for (int j = 0; j < m; ++j) {
                const float* b = boxes_b + j * 4;
                const BoxPair pair = intersect(a, b);
                const float iou = pair.overlap / std::max(pair.union_area, 1e-9f);
                const float hull_width = std::max(std::max(a[2], b[2]) - std::min(a[0], b[0]), 0.f);
                const float hull_height =
                    std::max(std::max(a[3], b[3]) - std::min(a[1], b[1]), 0.f);
                const float hull = hull_width * hull_height;
                result[static_cast<size_t>(i) * m + j] =
                    iou - (hull - pair.union_area) / std::max(hull, 1e-9f);
            }
        }
        return result;
    }

    std::vector<float> iou_cost(const float* boxes_a, int n, const float* boxes_b, int m) {
        std::vector<float> cost = iou_matrix(boxes_a, n, boxes_b, m);
        for (float& value : cost)
            value = 1.f - value;
        return cost;
    }

    std::vector<float> giou_cost(const float* boxes_a, int n, const float* boxes_b, int m) {
        std::vector<float> cost = giou_matrix(boxes_a, n, boxes_b, m);
        for (float& value : cost)
            value = 1.f - value;
        return cost;
    }

    std::vector<float> direction_cost(const float* directions, const float* origins, int n,
                                      const float* detection_boxes, int m) {
        std::vector<float> cost(static_cast<size_t>(n) * std::max(m, 0), 0.f);
        constexpr float kPi = 3.14159265358979323846f;
        for (int i = 0; i < n; ++i) {
            const float* heading = directions + i * 2;
            const float heading_norm = std::sqrt(heading[0] * heading[0] + heading[1] * heading[1]);
            // A track the ring cannot yet measure a heading for carries no information, so
            // every candidate scores zero and the term stays additive-neutral.
            if (heading_norm < 1e-6f)
                continue;
            const float* origin = origins + i * 4;
            const float origin_cx = (origin[0] + origin[2]) * 0.5f;
            const float origin_cy = (origin[1] + origin[3]) * 0.5f;
            for (int j = 0; j < m; ++j) {
                const float* box = detection_boxes + j * 4;
                const float offset_x = (box[0] + box[2]) * 0.5f - origin_cx;
                const float offset_y = (box[1] + box[3]) * 0.5f - origin_cy;
                const float offset_norm = std::sqrt(offset_x * offset_x + offset_y * offset_y);
                // A candidate sitting exactly on the origin has no direction either, and the
                // numpy version scores it zero for the same reason.
                if (offset_norm < 1e-6f)
                    continue;
                const float scale = offset_norm > 1e-6f ? offset_norm : 1e-6f;
                float cosine = (offset_x / scale) * heading[0] + (offset_y / scale) * heading[1];
                cosine = std::min(std::max(cosine, -1.f), 1.f);
                cost[static_cast<size_t>(i) * m + j] = std::acos(cosine) / kPi;
            }
        }
        return cost;
    }

    void min_fuse(std::vector<float>& motion, const float* appearance, int n, int m,
                  float motion_gate, float appearance_gate, float appearance_weight) {
        for (size_t index = 0; index < static_cast<size_t>(n) * m; ++index) {
            const float geometry = motion[index];
            const bool motion_admitted = geometry <= motion_gate;
            const float admitted_motion = motion_admitted ? geometry : 1.f;
            // Appearance is admitted only where the motion gate ALSO passed. A pair the
            // geometry has already vetoed must not be rescued by a cosine distance alone,
            // which is what makes the two gates a conjunction rather than a choice.
            const float admitted_appearance =
                (appearance[index] <= appearance_gate && motion_admitted)
                    ? appearance_weight * appearance[index]
                    : 1.f;
            motion[index] = std::min(admitted_motion, admitted_appearance);
        }
    }

    void fuse_score(std::vector<float>& cost, int n, int m, const float* detection_scores) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                float& value = cost[static_cast<size_t>(i) * m + j];
                value = 1.f - (1.f - value) * detection_scores[j];
            }
        }
    }

    void gate_cost(std::vector<float>& cost, int n, int m, const float* gating_distances,
                   float threshold) {
        for (size_t index = 0; index < static_cast<size_t>(n) * m; ++index) {
            if (gating_distances[index] > threshold)
                cost[index] = kInfeasible;
        }
    }

    std::vector<int> solve_assignment(const std::vector<float>& cost, int n, int m) {
        if (cost.size() != static_cast<size_t>(n) * static_cast<size_t>(m)) {
            throw std::invalid_argument("the cost matrix does not match the shape it was given");
        }
        std::vector<int> assignment(static_cast<size_t>(std::max(n, 0)), -1);
        if (n == 0 || m == 0)
            return assignment;

        // The augmenting-path formulation needs at least as many columns as rows. Transposing
        // rather than padding with a large constant: padding changes the objective whenever a
        // real cost is comparable to the pad, which for a fully gated matrix (every entry
        // kInfeasible) is exactly the case that matters.
        const bool transposed = n > m;
        const int rows = transposed ? m : n;
        const int columns = transposed ? n : m;
        auto at = [&](int i, int j) -> double {
            return transposed ? cost[static_cast<size_t>(j) * m + i]
                              : cost[static_cast<size_t>(i) * m + j];
        };

        constexpr double kInfinity = std::numeric_limits<double>::infinity();
        // One-based, with index 0 as the sentinel "no row/column", which is what keeps the
        // augmenting loop free of special cases. `potential_*` are the dual variables; `mate`
        // maps a column to the row holding it.
        std::vector<double> potential_row(rows + 1, 0.0);
        std::vector<double> potential_column(columns + 1, 0.0);
        std::vector<int> mate(columns + 1, 0);
        std::vector<int> path(columns + 1, 0);
        // Hoisted out of the row loop: this runs once per camera per frame, and two
        // allocations per track is the kind of cost a native tracker exists to remove.
        std::vector<double> shortest(columns + 1, kInfinity);
        std::vector<char> visited(columns + 1, 0);

        for (int row = 1; row <= rows; ++row) {
            mate[0] = row;
            int column = 0;
            std::fill(shortest.begin(), shortest.end(), kInfinity);
            std::fill(visited.begin(), visited.end(), 0);
            do {
                visited[column] = 1;
                const int current_row = mate[column];
                double delta = kInfinity;
                int next_column = 0;
                for (int candidate = 1; candidate <= columns; ++candidate) {
                    if (visited[candidate])
                        continue;
                    const double reduced = at(current_row - 1, candidate - 1) -
                                           potential_row[current_row] - potential_column[candidate];
                    if (reduced < shortest[candidate]) {
                        shortest[candidate] = reduced;
                        path[candidate] = column;
                    }
                    if (shortest[candidate] < delta) {
                        delta = shortest[candidate];
                        next_column = candidate;
                    }
                }
                for (int candidate = 0; candidate <= columns; ++candidate) {
                    if (visited[candidate]) {
                        potential_row[mate[candidate]] += delta;
                        potential_column[candidate] -= delta;
                    } else {
                        shortest[candidate] -= delta;
                    }
                }
                column = next_column;
            } while (mate[column] != 0);
            // Walk the alternating path back, flipping every edge on it.
            do {
                const int previous = path[column];
                mate[column] = mate[previous];
                column = previous;
            } while (column != 0);
        }

        for (int candidate = 1; candidate <= columns; ++candidate) {
            const int row = mate[candidate];
            if (row == 0)
                continue;
            if (transposed)
                assignment[candidate - 1] = row - 1;
            else
                assignment[row - 1] = candidate - 1;
        }
        return assignment;
    }

    Association associate(const std::vector<float>& cost, int n, int m, float max_cost) {
        Association result;
        if (n == 0 || m == 0) {
            for (int i = 0; i < n; ++i)
                result.unmatched_rows.push_back(i);
            for (int j = 0; j < m; ++j)
                result.unmatched_columns.push_back(j);
            return result;
        }

        const std::vector<int> assignment = solve_assignment(cost, n, m);
        std::vector<char> matched_columns(static_cast<size_t>(m), 0);
        for (int row = 0; row < n; ++row) {
            const int column = assignment[static_cast<size_t>(row)];
            if (column < 0 || cost[static_cast<size_t>(row) * m + column] > max_cost) {
                result.unmatched_rows.push_back(row);
                continue;
            }
            result.matches.emplace_back(row, column);
            matched_columns[static_cast<size_t>(column)] = 1;
        }
        for (int column = 0; column < m; ++column) {
            if (!matched_columns[static_cast<size_t>(column)])
                result.unmatched_columns.push_back(column);
        }
        return result;
    }

    Association cascade_associate(const CostBuilder& build_cost, float max_cost,
                                  const std::vector<int>& rows, const std::vector<int>& columns,
                                  const std::vector<int>& ages, int stride, int max_depth) {
        if (stride < 1)
            throw std::invalid_argument("a cascade stride must be >= 1");

        Association result;
        result.unmatched_columns = columns;
        if (rows.empty() || columns.empty()) {
            result.unmatched_rows = rows;
            return result;
        }

        std::vector<char> matched(rows.size(), 0);
        size_t matched_count = 0;
        for (int start = 0; start < std::max(max_depth, 1); start += stride) {
            if (result.unmatched_columns.empty() || matched_count == rows.size())
                break;
            std::vector<int> band;
            for (int row : rows) {
                const int age = ages[static_cast<size_t>(row)];
                if (age >= start && age < start + stride)
                    band.push_back(row);
            }
            if (band.empty())
                continue;

            const Association solved = associate_subset(build_cost(band, result.unmatched_columns),
                                                        band, result.unmatched_columns, max_cost);
            for (const auto& match : solved.matches) {
                result.matches.push_back(match);
                for (size_t index = 0; index < rows.size(); ++index) {
                    if (rows[index] == match.first && !matched[index]) {
                        matched[index] = 1;
                        ++matched_count;
                        break;
                    }
                }
            }
            result.unmatched_columns = solved.unmatched_columns;
        }

        for (size_t index = 0; index < rows.size(); ++index) {
            if (!matched[index])
                result.unmatched_rows.push_back(rows[index]);
        }
        return result;
    }

    Association associate_subset(const std::vector<float>& submatrix, const std::vector<int>& rows,
                                 const std::vector<int>& columns, float max_cost) {
        Association translated;
        if (rows.empty() || columns.empty()) {
            translated.unmatched_rows = rows;
            translated.unmatched_columns = columns;
            return translated;
        }
        const Association local = associate(submatrix, static_cast<int>(rows.size()),
                                            static_cast<int>(columns.size()), max_cost);
        for (const auto& [row, column] : local.matches)
            translated.matches.emplace_back(rows[static_cast<size_t>(row)],
                                            columns[static_cast<size_t>(column)]);
        for (int row : local.unmatched_rows)
            translated.unmatched_rows.push_back(rows[static_cast<size_t>(row)]);
        for (int column : local.unmatched_columns)
            translated.unmatched_columns.push_back(columns[static_cast<size_t>(column)]);
        return translated;
    }

}  // namespace shipvision::tracking
