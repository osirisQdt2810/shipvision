// Which detection is which track: the cost matrices, and the solver that consumes them.
//
// The split mirrors `shipvision/mot/association/` exactly, and for the same reason:
// building a cost is domain knowledge that differs between algorithms, while solving the
// assignment is a solved problem that does not. Every tracker here differs in *which of these
// it combines* and none of them differ in how the assignment works.
//
// ABOUT THE SOLVER, because the house rule says not to write one. `shipvision`'s ponytail
// principle is explicit — "if you are writing a Hungarian solver, stop" — and the Python side
// obeys it by calling `scipy.optimize.linear_sum_assignment`. There is no scipy in a C++
// translation unit, and the alternatives are worse than the ~70 lines below: crossing back
// into Python once per association stage costs the GIL and a marshalling pass per frame per
// camera, which is the entire reason a native tracker exists, and adding a linear-algebra
// dependency to a library whose only other need is fixed-size 8x8 arithmetic would make the
// parent build a solver package to get a tracker. What keeps this honest is that scipy stays
// the oracle: the parity tests run both trackers over one sequence and compare identities.
//
// Everything here is row-major `(n_tracks, n_detections)` float, matching the numpy shapes so
// the two can be diffed.

#pragma once

#include <functional>
#include <utility>
#include <vector>

namespace shipvision::mot {

    /// The cost given to a pair a gate has forbidden.
    ///
    /// Large enough that the solver never prefers it to a real alternative, FINITE so the
    /// solve still terminates on a matrix that is entirely gated — the same reason the Python
    /// constant is 1e5 rather than `np.inf`, which makes `linear_sum_assignment` raise.
    constexpr float kInfeasible = 1e5f;

    /// `(indices.size(), 4)` xyxy gathered from a row-major `(n, 4)` set of boxes.
    ///
    /// Here rather than in each tracker because every staged association needs it and the
    /// mistake it removes is the same one `associate_subset` removes: a stage that indexes the
    /// full box array with a submatrix position silently scores the wrong objects against each
    /// other, and every tracker still runs.
    std::vector<float> gather_boxes(const float* boxes, const std::vector<int>& indices);

    /// `(rows.size(), columns.size())` slice of a row-major `(n, m)` matrix.
    ///
    /// The frame's appearance distances are computed once, over every track and every
    /// detection, and each stage reads the corner of them it is allowed to see. Slicing here
    /// keeps the row/column translation in one place for the same reason `associate_subset`
    /// does.
    std::vector<float> gather_submatrix(const float* matrix, int m, const std::vector<int>& rows,
                                        const std::vector<int>& columns);

    /// `(n, m)` pairwise IoU between two sets of xyxy boxes.
    std::vector<float> iou_matrix(const float* boxes_a, int n, const float* boxes_b, int m);

    /// `(n, m)` generalised IoU, in `[-1, 1]`.
    ///
    /// IoU is *flat at zero* for every non-overlapping pair, so an assignment on IoU alone
    /// cannot tell "just missed" from "the other side of the frame". GIoU keeps decreasing as
    /// the boxes separate, which is what lets a cascade rank two equally-unmatched candidates.
    std::vector<float> giou_matrix(const float* boxes_a, int n, const float* boxes_b, int m);

    /// `1 - IoU`. The workhorse cost: cheap, and right whenever motion between frames is
    /// small.
    std::vector<float> iou_cost(const float* boxes_a, int n, const float* boxes_b, int m);

    /// `1 - GIoU`, so the range is `[0, 2]` rather than `[0, 1]`.
    std::vector<float> giou_cost(const float* boxes_a, int n, const float* boxes_b, int m);

    /// `(n, m)` direction-consistency cost in `[0, 1]`. OC-SORT's observation-centric
    /// momentum.
    ///
    /// The angle between the way a track has been travelling and the way from where it was
    /// last *seen* to a candidate, divided by pi. Zero is straight ahead, one is directly
    /// backwards. This is the piece that resolves a crossing without appearance: two objects
    /// passing each other are geometrically interchangeable at the moment they overlap and are
    /// not interchangeable in heading — and the heading is measured between two real
    /// observations rather than read off a filter that has been extrapolating.
    ///
    /// A track with no measured heading, and a candidate sitting exactly on the origin, both
    /// score zero rather than the mean. Zero keeps the term additive-neutral; 0.5 would
    /// quietly penalise every newly-born track for having no history yet.
    ///
    /// @param directions      `(n, 2)` unit headings, `(0, 0)` where there is no history
    /// @param origins         `(n, 4)` xyxy observations the headings were measured *to*
    /// @param detection_boxes `(m, 4)` xyxy candidates
    std::vector<float> direction_cost(const float* directions, const float* origins, int n,
                                      const float* detection_boxes, int m);

    /// BoT-SORT's fusion: element-wise **minimum** of two independently gated costs, in place
    /// over `motion`.
    ///
    /// The alternative — a weighted sum, which is what DeepSORT does — has a specific failure:
    /// a pair that is unambiguous on one signal gets dragged over the threshold by the other.
    /// Two people in identical uniforms have a near-zero appearance distance, so a sum lets
    /// appearance veto a geometrically obvious match; a person turning a corner has poor
    /// overlap, so a sum lets geometry veto an appearance-obvious match. The minimum means
    /// *either* signal on its own suffices, and the two gates are what stop that from becoming
    /// "anything matches anything".
    ///
    /// A pair failing both gates comes out at 1.0, which every caller's threshold rejects.
    ///
    /// @param appearance        `(n, m)` cosine distances
    /// @param appearance_weight the paper halves the cosine distance before the minimum,
    ///                          because `1 - IoU` and a cosine distance are not on the same
    ///                          scale and the raw cosine term would almost never win
    void min_fuse(std::vector<float>& motion, const float* appearance, int n, int m,
                  float motion_gate, float appearance_gate, float appearance_weight);

    /// Fold detector confidence into a cost, in place. ByteTrack's formulation.
    ///
    /// A high-IoU match with a 0.3-confidence detection is weaker evidence than the same IoU
    /// with a 0.9 one, and an assignment that ignores that will happily attach an identity to
    /// a barely-there box.
    void fuse_score(std::vector<float>& cost, int n, int m, const float* detection_scores);

    /// Forbid the pairs the motion model says are impossible, in place.
    ///
    /// Gating BEFORE the assignment rather than weighting inside it: a detection the filter
    /// says cannot belong to this track must not be selectable at any price, because an ID
    /// switch is not recoverable the way a missed frame is.
    void gate_cost(std::vector<float>& cost, int n, int m, const float* gating_distances,
                   float threshold);

    /// Minimum-cost one-to-one assignment. Returns one column per row, or -1 for a row left
    /// unassigned (which happens only when there are fewer columns than rows).
    ///
    /// O(n^3) Jonker-Volgenant with potentials — the same objective `linear_sum_assignment`
    /// optimises, so the total cost is identical. TIES ARE NOT: two assignments of equal total
    /// cost are both optimal and the two implementations may pick different ones. That is why
    /// the parity tests use sequences where the right answer is unambiguous, and why a tracker
    /// must never depend on which of two equal-cost matches it gets.
    std::vector<int> solve_assignment(const std::vector<float>& cost, int n, int m);

    /// The result of one association stage, in the caller's own indices.
    struct Association {
            std::vector<std::pair<int, int>> matches;  ///< (row, column)
            std::vector<int> unmatched_rows;
            std::vector<int> unmatched_columns;
    };

    /// Solve, then threshold. The order matters and is the same as the Python's.
    ///
    /// The solver optimises the *total*, so it will accept an expensive pair to enable two
    /// cheap ones — correct globally and wrong for that pair. Dropping the over-threshold
    /// matches afterwards keeps the global optimum where it helps and refuses the individual
    /// assignments that are not actually evidence.
    Association associate(const std::vector<float>& cost, int n, int m, float max_cost);

    /// `associate` on a sub-problem, translating positions back to the caller's indices.
    ///
    /// Every multi-stage tracker needs this and every one of them gets the translation wrong
    /// at least once: the solver returns positions within the submatrix, and using those as
    /// track indices silently associates the wrong objects. Done once, here, so it cannot
    /// drift between one tracker's stages and another's.
    Association associate_subset(const std::vector<float>& submatrix, const std::vector<int>& rows,
                                 const std::vector<int>& columns, float max_cost);

    /// Builds one sub-problem's cost matrix, row-major over `(rows, columns)`.
    ///
    /// A callback rather than a prebuilt matrix because a cascade solves a different subset on
    /// every band, and materialising the whole matrix up front would compute costs for pairs
    /// an earlier band has already taken off the table.
    using CostBuilder = std::function<std::vector<float>(const std::vector<int>& rows,
                                                         const std::vector<int>& columns)>;

    /// DeepSORT's matching cascade: recently-seen tracks choose first.
    ///
    /// A single global assignment treats a track last seen one frame ago and one last seen
    /// twenty frames ago as equally credible bidders for the same detection. They are not: the
    /// older track's covariance has been inflated once per frame, so its gate is wide open and
    /// it will happily claim a detection that belongs to its neighbour. Solving in bands of
    /// increasing `time_since_update` gives the well-supported tracks first refusal.
    ///
    /// `unmatched_columns` carries the detections no band took, so the caller can hand them
    /// straight to the next stage.
    ///
    /// @param ages       `time_since_update` per track, indexed by the values in `rows`
    /// @param stride     band width in frames; one is DeepSORT's original formulation, and the
    ///                   internal reference uses five — a deliberate trade of a little
    ///                   precedence for a fifth of the solves
    /// @param max_depth  bands stop once this age is reached
    Association cascade_associate(const CostBuilder& build_cost, float max_cost,
                                  const std::vector<int>& rows, const std::vector<int>& columns,
                                  const std::vector<int>& ages, int stride, int max_depth);

}  // namespace shipvision::mot
