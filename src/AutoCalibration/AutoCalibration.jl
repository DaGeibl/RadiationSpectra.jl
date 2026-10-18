function calculate_ratios(pos::Array{T, 1})::Array{T, 2} where {T <: AbstractFloat}
    n::Int = length(pos)
    ratios::Array{T, 2} = zeros(Float64, n, n)
    @fastmath @inbounds for i in 1:n
        for j in i+1:n
            ratios[i,j] = pos[i] / pos[j]
        end
    end
    return ratios
end

# Default number of reference lines that must be matched: all for up to three lines, all but one for four or five, four beyond.
_default_min_n_matches(n_lines::Int) = min(4, n_lines, max(3, n_lines - 1))

# No span requirement for two-match searches, half of the reference range otherwise.
_default_min_match_span(min_n_matches::Int) = min_n_matches <= 2 ? 0.0 : 0.5

function determine_calibration_constant_through_peak_ratios(fPositionX::Array{<:Real,1}, photon_lines::Array{<:Real,1};
        α::Real=0.01, rtol::Real=5e-3, min_n_matches::Int=_default_min_n_matches(length(photon_lines)), min_match_span::Real=_default_min_match_span(min_n_matches))
    solutions = calibration_solutions(fPositionX, photon_lines; α, rtol, min_n_matches, min_match_span)
    return select_calibration_solution(solutions; rtol, min_n_matches).c
end

# Match sorted peaks to sorted references, using each peak and reference at most once.
function match_calibration_lines(positions, photon_lines, c; rtol, min_match_span=0.0)
    n, m = length(positions), length(photon_lines)
    peak_indices, line_indices = Int[], Int[]
    best_count, best_cost = 0, Inf
    required_span = m == 0 ? 0.0 : min_match_span * (last(photon_lines) - first(photon_lines))
    counts = Matrix{Int}(undef, n + 1, m + 1)
    costs = Matrix{Float64}(undef, n + 1, m + 1)
    directions = Matrix{UInt8}(undef, n + 1, m + 1)
    # One pass per first matched reference line, so the span is known when the last one is chosen.
    # Each prefix keeps the most matches, ties broken by the sum of squared relative residuals.
    for first_line in 1:m
        photon_lines[end] - photon_lines[first_line] >= required_span || break
        fill!(counts, -1)
        fill!(costs, Inf)
        fill!(directions, 0x00)
        counts[:, first_line] .= 0
        costs[:, first_line] .= 0
        last_peak, last_line = 0, 0
        for i in 1:n, j in first_line:m
            count, cost, direction = counts[i, j+1], costs[i, j+1], 0x01
            if j > first_line && (counts[i+1, j] > count || (counts[i+1, j] == count && costs[i+1, j] < cost))
                count, cost, direction = counts[i+1, j], costs[i+1, j], 0x02
            end
            residual = (c * positions[i] - photon_lines[j]) / photon_lines[j]
            if counts[i, j] >= 0 && abs(residual) <= rtol + 8eps(Float64)
                matched_count, matched_cost = counts[i, j] + 1, costs[i, j] + residual^2
                if matched_count > count || (matched_count == count && matched_cost < cost)
                    count, cost, direction = matched_count, matched_cost, 0x03
                end
                if photon_lines[j] - photon_lines[first_line] >= required_span &&
                        (matched_count > best_count || (matched_count == best_count && matched_cost < best_cost))
                    best_count, best_cost = matched_count, matched_cost
                    last_peak, last_line = i, j
                end
            end
            counts[i+1, j+1], costs[i+1, j+1], directions[i+1, j+1] = count, cost, direction
        end
        last_peak == 0 && continue
        # The final pair is forced; its prefix follows the ordinary optimal-assignment choices.
        empty!(peak_indices)
        empty!(line_indices)
        push!(peak_indices, last_peak)
        push!(line_indices, last_line)
        i, j = last_peak - 1, last_line - 1
        while i > 0 && j >= first_line
            direction = directions[i+1, j+1]
            if direction == 0x03
                push!(peak_indices, i)
                push!(line_indices, j)
                i -= 1
                j -= 1
            elseif direction == 0x01
                i -= 1
            else
                j -= 1
            end
        end
        reverse!(peak_indices)
        reverse!(line_indices)
    end
    score = best_count == 0 ? Inf : best_cost / best_count
    return (; peak_indices, line_indices, score)
end

function _check_matching_arguments(n_lines; rtol, min_n_matches, min_match_span)
    0 <= rtol < 1 || throw(ArgumentError("rtol must satisfy 0 ≤ rtol < 1"))
    2 <= min_n_matches <= n_lines || throw(ArgumentError("min_n_matches must be between 2 and the number of reference lines"))
    0 <= min_match_span <= 1 || throw(ArgumentError("min_match_span must be between 0 and 1"))
    return nothing
end

# Generate proportional gains from peak/reference pairs, then refine their distinct-line assignments.
function calibration_solutions(fPositionX, photon_lines; α, rtol, min_n_matches, min_match_span)
    _check_matching_arguments(length(photon_lines); rtol, min_n_matches, min_match_span)
    photon_lines = sort(Float64.(photon_lines))
    fPositionX = sort(unique(Float64.(fPositionX)))
    theo_ratios = calculate_ratios(photon_lines)
    exp_ratios = calculate_ratios(fPositionX)
    pcfs = Float64[]
    for i in eachindex(fPositionX), j in i+1:length(fPositionX), it in eachindex(photon_lines), jt in it+1:length(photon_lines)
        if 1 - α <= theo_ratios[it, jt] / exp_ratios[i, j] <= 1 + α
            # Probe the overlap of both allowed gain intervals, including its boundaries.
            gains = (photon_lines[it] / fPositionX[i], photon_lines[jt] / fPositionX[j])
            c_low = max(0, (1 - rtol) * maximum(gains))
            c_high = (1 + rtol) * minimum(gains)
            c_low <= c_high && append!(pcfs, (c_low, (c_low + c_high) / 2, c_high))
        end
    end
    solutions = @NamedTuple{c::Float64, n_matches::Int, score::Float64, matched_lines::Vector{Float64}}[]
    for c in unique(pcfs)
        matches = match_calibration_lines(fPositionX, photon_lines, c; rtol, min_match_span)
        converged = false
        for iteration in eachindex(photon_lines)
            length(matches.line_indices) >= min_n_matches || break
            peaks, lines = fPositionX[matches.peak_indices], photon_lines[matches.line_indices]
            # Minimize squared relative residuals while keeping all current matches within tolerance.
            # Reassign at the refined gain and repeat until the matched pairs stop changing.
            ratios = peaks ./ lines
            c_low = max(0, maximum((1 - rtol) ./ ratios))
            c_high = minimum((1 + rtol) ./ ratios)
            c = clamp(sum(ratios) / sum(abs2, ratios), c_low, c_high)
            refined = match_calibration_lines(fPositionX, photon_lines, c; rtol, min_match_span)
            if refined.peak_indices == matches.peak_indices && refined.line_indices == matches.line_indices
                matches = refined
                converged = true
                break
            end
            matches = refined
        end
        converged || continue
        push!(solutions, (; c, n_matches=length(matches.line_indices), score=matches.score, matched_lines=photon_lines[matches.line_indices]))
    end
    return solutions
end

# Prefer more matched references, then higher thresholds and smaller residuals; check competing gains.
function select_calibration_solution(solutions; rtol, min_n_matches)
    isempty(solutions) && error("No peaks identified. Try different values for keyword parameters: `α`, `rtol`")
    sort!(solutions; by=s -> (-s.n_matches, -get(s, :threshold, 0.0), s.score))
    unique!(s -> s.c, solutions)
    best = first(solutions)
    # Two-match searches allow competing gains because different partial reference sequences can fit equally well.
    for other in Iterators.drop(solutions, 1)
        if min_n_matches > 2 && other.n_matches == best.n_matches && get(other, :threshold, 0.0) == get(best, :threshold, 0.0) && !isapprox(other.c, best.c; rtol=rtol) && other.score <= 2 * best.score + eps(Float64)
            error("Ambiguous calibration: competing scales have comparable reference-line support")
        end
    end
    @debug "Calibration identified $(best.n_matches) reference lines with relative RMS residual $(sqrt(best.score))"
    return best
end


function _filter_peaks_from_peakfinder(h::Histogram{<:Real, 1}, peakPositions::Vector{T}, σ::Real) where {T <: Real}
    pfits = Vector{@NamedTuple{μ::Real, σ::Real}}(undef, length(peakPositions))
    success = zeros(Bool, length(peakPositions))
    Threads.@threads for ipf in eachindex(peakPositions)
        p = peakPositions[ipf]
        try
            searchrange = (p - 10σ*step(h.edges[1]), p + 10σ*step(h.edges[1]))
            h_search = subhist(h, searchrange)
            weights = h_search.weights
            imax = argmax(weights)
            centers = StatsBase.midpoints(h_search.edges[1])
            background = min(first(weights), last(weights))
            half_height = background + (weights[imax] - background) / 2
            weights[imax] > half_height || continue
            left = findlast(<=(half_height), view(weights, 1:imax-1))
            right_offset = findfirst(<=(half_height), view(weights, imax+1:length(weights)))
            (isnothing(left) || isnothing(right_offset)) && continue
            right = imax + right_offset
            # Nearest adjacent crossings must bracket half-height; never extrapolate.
            cut_left = _interpolate_linear((centers[left], weights[left]), (centers[left+1], weights[left+1]); threshold=half_height)
            cut_right = _interpolate_linear((centers[right-1], weights[right-1]), (centers[right], weights[right]); threshold=half_height)
            peak_σ = (cut_right - cut_left) / (2 * √(2log(2)))
            success[ipf] = true
            pfits[ipf] = (μ=centers[imax], σ=peak_σ)
        catch e
            @debug "Error filter peak at $p: $e"
        end
    end
    pfits = pfits[success]
    peakPositions = peakPositions[success]
    isempty(peakPositions) && return T[]
    d = [abs(pfits[i].μ - peakPositions[i]) for i in eachindex(peakPositions)]
    σs = [f.σ for f in pfits]
    hd, hσ = harmmean(d), harmmean(σs)
    # A peak on a bin center makes the harmonic-mean distance zero; retain a distance tolerance of at least one bin.
    distance_limit = max(4hd, step(h.edges[1]))
    return peakPositions[(d .<= distance_limit) .& (σs .< 2hσ)]
end

"""
    determine_calibration_constant_through_peak_ratios(h::Histogram, photon_lines; kwargs...)

Return `(c, h_deconv, peak_positions, threshold)` for the proportional calibration `energy = c * position`.
`peak_positions` contains all raw peak-finder positions at the selected threshold, in strength order.
See [`calibrate_spectrum`](@ref) for matching and discovery keywords. Here `threshold` defaults to 10
and `max_n_peaks` to `5 * length(photon_lines)`.
"""
function determine_calibration_constant_through_peak_ratios(h::Histogram{<:Real, 1}, photon_lines::Array{<:Real, 1};
            min_n_peaks::Int = _default_min_n_matches(length(photon_lines)), max_n_peaks::Int = 5 * length(photon_lines),
            threshold::Real = 10., α::Real = 0.01, σ::Real = 3.0, rtol::Real = 5e-3,
            min_n_matches::Int = min(min_n_peaks, _default_min_n_matches(length(photon_lines))), min_match_span::Real = _default_min_match_span(min_n_matches))
    best = discover_calibration(h, photon_lines; min_n_peaks, max_n_peaks, threshold, α, σ, rtol, min_n_matches, min_match_span)
    return best.c, best.h_deconv, best.positions, best.threshold
end

# Peak-finder thresholds from `threshold` down to 0.5, each three quarters of the previous one.
function _threshold_sequence(threshold::Real)
    0 < threshold < 100 || throw(ArgumentError("threshold must be between 0 and 100"))
    thresholds = Float64[threshold]
    while last(thresholds) > 0.5
        push!(thresholds, max(0.5, 0.75 * last(thresholds)))
    end
    return thresholds
end

# Shared by the ratio method and `calibrate_spectrum`, which also needs the matched lines.
function discover_calibration(h, photon_lines; min_n_peaks, max_n_peaks, threshold, α, σ, rtol, min_n_matches, min_match_span)
    min_n_peaks >= 2 || throw(ArgumentError("min_n_peaks must be at least 2"))
    max_n_peaks >= min_n_matches || throw(ArgumentError("max_n_peaks must be at least min_n_matches"))
    _check_matching_arguments(length(photon_lines); rtol, min_n_matches, min_match_span)
    two_match = min_n_matches <= 2
    solutions = @NamedTuple{c::Float64, n_matches::Int, score::Float64, matched_lines::Vector{Float64}, threshold::Float64, positions::Vector{Float64}, h_deconv::Histogram}[]
    # Two-match searches try the shape-filtered peaks first and stop at the first threshold with a solution.
    # Searches for more matches use all peaks and collect the solutions of all thresholds.
    for filter_peaks in (two_match ? (true, false) : (false,)), threshold_reduced in _threshold_sequence(threshold)
        h_deconv, peakPositions = peakfinder(h; threshold=threshold_reduced, σ)
        length(peakPositions) >= min_n_peaks || continue
        candidates = first(filter_peaks ? _filter_peaks_from_peakfinder(h, peakPositions, σ) : peakPositions, max_n_peaks)
        found = calibration_solutions(candidates, photon_lines; α, rtol, min_n_matches, min_match_span)
        @debug "Calibration discovery" threshold=threshold_reduced n_raw=length(peakPositions) n_candidates=length(candidates) n_solutions=length(found)
        for candidate in found
            push!(solutions, (; candidate..., threshold=Float64(threshold_reduced), positions=peakPositions, h_deconv))
        end
        two_match && !isempty(found) && break
    end
    best = select_calibration_solution(solutions; rtol, min_n_matches)
    @debug "Calibration selected" threshold=best.threshold n_raw=length(best.positions) n_matches=best.n_matches relative_rms=sqrt(best.score)
    return best
end

function determine_calibration_constant_through_peak_fitting(h::Histogram{<:Real, 1, E}, photon_lines::Vector{<:Real}, c_pre::Real=1.0) where {E}
    c_pre = Float64(c_pre)
    peak_fits = []
    photon_lines = Float64.(photon_lines)
    processed_photon_lines = Float64[]
    for pl in photon_lines
        line::Float64 = pl
        fitrange = (line - 20, line + 20)
        fitrange = fitrange ./ c_pre

        first_bin, last_bin = StatsBase.binindex(h, fitrange[1]), StatsBase.binindex(h, fitrange[2])
        if last_bin > length(h.weights) || first_bin < 1
            @warn("$pl keV line not sufficiently contained in Histogram, skipping...");
            continue
        end
        d = fit(NormalPeakUvD, subhist(h, fitrange))[1]
        push!(peak_fits, d)
        push!(processed_photon_lines, pl)
    end
    photon_lines = intersect(photon_lines, processed_photon_lines)
    fitted_peak_positions = [ fr.UvNormal.μ for fr in peak_fits ]

    @fastmath function _linear_function_fixed_offset_at_zero(p::Vector{T}) where {T}
        return sqrt( sum(((fitted_peak_positions * p[1]) .- photon_lines).^2) )
    end
    
    opt_result = Optim.optimize( 
        _linear_function_fixed_offset_at_zero, 
        Float64[0], Float64[Inf], Float64[c_pre], 
        Fminbox(BFGS()); 
        autodiff=AutoForwardDiff() )
    
    c = opt_result.minimizer[1]

    return c, peak_fits, opt_result
end


"""
    calibrate_spectrum(h_uncal::Histogram, photon_lines::Vector{<:Real}; kwargs...)

Calibrate with `energy = c * position`: discover peaks, identify reference matches, then fit only
those lines with Gaussian peaks and a linear background to refine the gain (offset fixed to zero).
Return `(h_cal, h_deconv, peak_positions, threshold, c, c_precal)`. Positions are all raw,
uncalibrated peak-finder positions at the selected threshold, in strength order.

# Keywords
- `σ::Real = 3.0`: Expected peak sigma, in bins.
- `threshold::Real = 50.0`: Initial peak-finder threshold, a percentage between 0 and 100; reduced down to 0.5 to discover weaker peaks.
- `min_n_peaks::Int`: Required raw discovery count; defaults to 2 for two references, 3 for three or four, and 4 for larger lists. Also caps the default `min_n_matches` for existing callers requesting two-peak searches.
- `max_n_peaks::Int = 4 * length(photon_lines)`: Match only this many strongest candidates; does not truncate returned positions.
- `min_n_matches::Int`: Required distinct reference matches; defaults as above, capped by `min_n_peaks`. Set explicitly to require more or fewer matches (at least two).
- `min_match_span::Real = min_n_matches <= 2 ? 0.0 : 0.5`: Minimum matched energy span as a fraction of the full reference span, between 0 and 1.
- `α::Real = 0.01`: Relative tolerance when comparing a reference-pair ratio with a discovered peak-pair ratio.
- `rtol::Real = 5e-3`: Maximum relative line residual, `abs(c * position - line) / line`; must satisfy `0 ≤ rtol < 1`. Small values are needed to discriminate nearby lines.

Solutions prefer more matched references, then higher discovery thresholds, then smaller squared
relative residuals. For more than two required matches, comparably supported competing gains at
the selected threshold raise an ambiguity error. Two-match searches stop at the first successful threshold.
Missing-line subsets must still satisfy `min_match_span`; lower it when using a narrow reference subset.
"""
function calibrate_spectrum(h_uncal::Histogram{<:Real, 1}, photon_lines::Vector{<:Real};
            σ::Real = 3.0, threshold::Real = 50.0, min_n_peaks::Int = _default_min_n_matches(length(photon_lines)),
            max_n_peaks::Int = 4 * length(photon_lines), α::Real = 0.01, rtol::Real = 5e-3,
            min_n_matches::Int = min(min_n_peaks, _default_min_n_matches(length(photon_lines))), min_match_span::Real = _default_min_match_span(min_n_matches))
    best = discover_calibration(h_uncal, photon_lines; min_n_peaks, max_n_peaks, threshold, α, σ, rtol, min_n_matches, min_match_span)
    c_precal = best.c
    c = determine_calibration_constant_through_peak_fitting(h_uncal, best.matched_lines, c_precal)[1]
    h_cal = Histogram(h_uncal.edges[1] .* c, :left)
    h_cal.weights = h_uncal.weights
    return h_cal, best.h_deconv, best.positions, best.threshold, c, c_precal
end
