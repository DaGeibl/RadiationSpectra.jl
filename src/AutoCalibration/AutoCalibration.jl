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
function calculate_abs_diffs(pos::Array{T, 1})::Array{T, 2} where {T <: AbstractFloat}
    n::Int = length(pos)
    ratios::Array{T, 2} = zeros(Float64, n, n)
    @fastmath @inbounds for i in 1:n
        for j in i+1:n
            ratios[i, j] = pos[j] - pos[i]
        end
    end
    return ratios
end


# Sorted, one-to-one matching: maximize support, then minimize relative residuals.
function match_calibration_lines(positions, photon_lines, c; rtol)
    n, m = length(positions), length(photon_lines)
    counts = zeros(Int, n + 1, m + 1)
    costs = zeros(Float64, n + 1, m + 1)
    directions = zeros(UInt8, n + 1, m + 1)
    for i in 1:n, j in 1:m
        count, cost, direction = counts[i, j+1], costs[i, j+1], 0x01
        if counts[i+1, j] > count || (counts[i+1, j] == count && costs[i+1, j] < cost)
            count, cost, direction = counts[i+1, j], costs[i+1, j], 0x02
        end
        residual = (c * positions[i] - photon_lines[j]) / photon_lines[j]
        if abs(residual) <= rtol
            matched_count, matched_cost = counts[i, j] + 1, costs[i, j] + residual^2
            if matched_count > count || (matched_count == count && matched_cost < cost)
                count, cost, direction = matched_count, matched_cost, 0x03
            end
        end
        counts[i+1, j+1], costs[i+1, j+1], directions[i+1, j+1] = count, cost, direction
    end
    peak_indices, line_indices = Int[], Int[]
    i, j = n, m
    while i > 0 && j > 0
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
    score = isempty(line_indices) ? Inf : costs[n+1, m+1] / length(line_indices)
    return (; peak_indices, line_indices, score)
end

"""
    determine_calibration_constant_through_peak_ratios(positions, photon_lines; α=0.01, rtol=0.005, min_n_matches=min(4, length(photon_lines)), min_match_span=min_n_matches <= 2 ? 0.0 : 0.5)

Find a proportional scale using distinct reference-line matches. Require
`min_n_matches` over at least `min_match_span` of the reference energy range.
Rank by matched-line count and relative residual, refine the scale from all
matched positions, and reject competing scales with comparable support and error.
Two-match discovery permits ambiguous partial reference series (e.g. SiPM peaks).
"""
function determine_calibration_constant_through_peak_ratios(fPositionX::Array{<:Real,1}, photon_lines::Array{<:Real,1};
        α::Real=0.01, rtol::Real=5e-3, min_n_matches::Int=min(4, length(photon_lines)), min_match_span::Real=min_n_matches <= 2 ? 0.0 : 0.5)
    solutions = calibration_solutions(fPositionX, photon_lines; α, rtol, min_n_matches, min_match_span)
    return select_calibration_solution(solutions; rtol, min_n_matches).c
end

# Shared by array matching and the bounded histogram discovery search.
function calibration_solutions(fPositionX, photon_lines; α, rtol, min_n_matches, min_match_span)
    photon_lines = sort(Float64.(photon_lines))
    fPositionX = sort(unique(Float64.(fPositionX)))
    2 <= min_n_matches <= length(photon_lines) || throw(ArgumentError("min_n_matches must lie between 2 and the number of reference lines"))
    0 <= min_match_span <= 1 || throw(ArgumentError("min_match_span must lie between 0 and 1"))
    all(>(0), photon_lines) && all(>(0), diff(photon_lines)) || throw(ArgumentError("Reference lines must be positive and distinct"))
    theo_ratios = calculate_ratios(photon_lines)
    exp_ratios = calculate_ratios(fPositionX)
    pcfs = Float64[]
    for i in eachindex(fPositionX), j in i+1:length(fPositionX), it in eachindex(photon_lines), jt in it+1:length(photon_lines)
        if 1 - α <= theo_ratios[it, jt] / exp_ratios[i, j] <= 1 + α
            c = (photon_lines[it] - photon_lines[jt]) / (fPositionX[i] - fPositionX[j])
            if abs(c * fPositionX[i] - photon_lines[it]) < rtol * photon_lines[it] && abs(c * fPositionX[j] - photon_lines[jt]) < rtol * photon_lines[jt]
                push!(pcfs, c)
            end
        end
    end
    solutions = @NamedTuple{c::Float64, n_matches::Int, score::Float64}[]
    for c in unique(pcfs)
        matches = match_calibration_lines(fPositionX, photon_lines, c; rtol)
        converged = false
        for iteration in eachindex(photon_lines)
            length(matches.line_indices) >= min_n_matches || break
            peaks, lines = fPositionX[matches.peak_indices], photon_lines[matches.line_indices]
            (last(lines) - first(lines)) >= min_match_span * (last(photon_lines) - first(photon_lines)) || break
            # Refinement may recover another line; stop only when the assignment stabilizes.
            c = sum(peaks .* lines) / sum(abs2, peaks)
            refined = match_calibration_lines(fPositionX, photon_lines, c; rtol)
            if refined.peak_indices == matches.peak_indices && refined.line_indices == matches.line_indices
                matches = refined
                converged = true
                break
            end
            matches = refined
        end
        converged || continue
        push!(solutions, (; c, n_matches=length(matches.line_indices), score=matches.score))
    end
    return solutions
end

function select_calibration_solution(solutions; rtol, min_n_matches)
    isempty(solutions) && error("Insufficient distinct reference-line support for calibration; adjust peak discovery or matching requirements")
    sort!(solutions; by=s -> (-s.n_matches, s.score))
    unique!(s -> s.c, solutions)
    best = first(solutions)
    for other in Iterators.drop(solutions, 1)
        if min_n_matches > 2 && other.n_matches == best.n_matches && !isapprox(other.c, best.c; rtol=rtol) && other.score <= 2 * best.score + eps(Float64)
            error("Ambiguous calibration: competing scales have comparable reference-line support")
        end
    end
    @debug "Calibration identified $(best.n_matches) reference lines with relative RMS residual $(sqrt(best.score))"
    return best
end


function _filter_peaks_from_peakfinder(h::Histogram{<:Real, 1}, peakPositions::Vector{T}, σ::Real) where {T <: Real}
    pfits = Vector{@NamedTuple{μ::Real, σ::Real}}(undef, length(peakPositions))
    # A byte per flag avoids concurrent writes to the same BitArray word.
    success = fill(false, length(peakPositions))
    Threads.@threads for ipf in eachindex(peakPositions)
        p = peakPositions[ipf]
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
    end
    pfits = pfits[success]
    peakPositions = peakPositions[success]
    isempty(peakPositions) && return T[]
    d = [abs(pfits[i].μ - peakPositions[i]) for i in eachindex(peakPositions)]
    σs = [f.σ for f in pfits]
    hd, hσ = harmmean(d), harmmean(σs)
    # Histogram maxima are quantized; a near-zero distance must not veto
    # other candidates displaced within one bin.
    distance_limit = max(4hd, step(h.edges[1]))
    return peakPositions[(d .<= distance_limit) .& (σs .< 2hσ)]
end

"""
    determine_calibration_constant_through_peak_ratios(h, photon_lines; ...)

Search raw deconvolution candidates down to threshold 0.5, using at most the
`max_n_peaks` strongest candidates per trial. Select the best support and reject
competing scales. Candidate count alone does not establish calibration support;
`min_n_peaks` sets the default minimum number of reference matches (at most four).
Two-match searches retain filtered, first-success discovery for SiPM callers.
"""
function determine_calibration_constant_through_peak_ratios(h::Histogram{<:Real, 1, E}, photon_lines::Array{<:Real, 1};
            min_n_peaks::Int = length(photon_lines), max_n_peaks::Int = 5 * length(photon_lines), threshold::Real = 10., α::Real = 0.01, σ::Real = 3.0, rtol::Real = 5e-3,
            min_n_matches::Int = min(4, min_n_peaks, length(photon_lines)), min_match_span::Real = min_n_matches <= 2 ? 0.0 : 0.5) where {E}
    max_n_peaks >= min_n_matches || throw(ArgumentError("max_n_peaks must allow at least min_n_matches candidates"))
    threshold_reduced = threshold
    solutions = @NamedTuple{c::Float64, n_matches::Int, score::Float64, threshold::Float64, positions::Vector{Float64}}[]
    while true
        h_deconv, peakPositions = peakfinder(h; threshold=threshold_reduced, σ)
        n_raw = length(peakPositions)
        if min_n_matches <= 2
            peakPositions = _filter_peaks_from_peakfinder(h, peakPositions, σ)
        end
        # peakfinder orders candidates by strength; cap work without skipping a
        # valid solution when a threshold step discovers several extra peaks.
        resize!(peakPositions, min(length(peakPositions), max_n_peaks))
        candidates = calibration_solutions(peakPositions, photon_lines; α, rtol, min_n_matches, min_match_span)
        @debug "Calibration discovery" threshold=threshold_reduced n_raw n_candidates=length(peakPositions) n_solutions=length(candidates)
        for candidate in candidates
            push!(solutions, (; candidate..., threshold=Float64(threshold_reduced), positions=peakPositions))
        end
        # SiPM callers use the first two returned positions directly; preserve
        # their first-success search rather than introducing weaker low peaks.
        min_n_matches <= 2 && length(peakPositions) >= min_n_peaks && !isempty(candidates) && break
        threshold_reduced <= 0.5 && break
        threshold_reduced = max(0.5, threshold_reduced * 0.75)
    end
    best = select_calibration_solution(solutions; rtol, min_n_matches)
    # Return the discovery products from the selected trial, not the final trial.
    h_deconv, _ = peakfinder(h; threshold=best.threshold, σ)
    positions = sort(best.positions)
    lines = sort(photon_lines)
    matches = match_calibration_lines(positions, lines, best.c; rtol)
    @debug "Calibration selected" threshold=best.threshold n_candidates=length(positions) matched_lines=lines[matches.line_indices] relative_rms=sqrt(best.score)
    return best.c, h_deconv, best.positions, best.threshold
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
    calibrate_spectrum(h_uncal::Histogram, photon_lines::Array{Real, 1}; <keyword arguments>)

Returns the calibrated histogram, the deconvoluted spectrum, the found (uncalibrated) peak positions and the final `threshold` value.

# Keywords
- `σ::Real = 2.0`: The expected sigma of a peak in the spectrum. In units of bins.
- `threshold::Real = 10.0`: Threshold for being identified as a peak in the deconvoluted spectrum. A single bin is identified as a peak when its weight exceeds the `threshold` and the previous bin was not identified as an peak.
- `min_n_peaks::Int = 0`: If the number of found peaks is smaller than `min_n_peaks` the functions lowers the parameter `threshold` until enough peaks are found.
- `max_n_peaks::Int = 50`: Use only the first (strongest) `max_n_peaks` peaks for peak identification.
- `α::Real = 0.005`:  = 0.5%. Acceptance level in the comparison of the peak position ratios in the peak indentification step. When the difference between the ratio of two found peak positions and the ratio of two photon lines (`photon_lines`) is smaller than `α`, the found peaks are identified as the two photon lines.
- `rtol::Real = 5e-3`:  = 5e-3. Acceptance level for tolerance of the absolute difference between true and found line position.

Calibrate the spectrum `h_uncal`. This is done by:
1) finding peaks through devoncolution
2) identifying them through comparison of the ratios of their positions with the ratios of the known `lines`
3) fitting all identified peaks (with a gaussian plus first order polynomial) to get their position more precisely
4) performe a linear fit (offset forced to 0) of these positions vs the true positions (`lines`) to get the calibration constant
"""
function calibrate_spectrum(h_uncal::Histogram{<:Real, 1}, photon_lines::Vector{<:Real}; 
            σ::Real = 3.0, threshold::Real = 50.0, min_n_peaks::Int = length(photon_lines), 
            max_n_peaks::Int = 4 * length(photon_lines), α::Real = 0.01, rtol::Real = 5e-3)
    c_precal, h_deconv, peakPositions, threshold = determine_calibration_constant_through_peak_ratios(h_uncal, photon_lines, 
                                min_n_peaks = min_n_peaks, max_n_peaks = max_n_peaks, threshold=threshold, α=α, σ=σ, rtol=rtol)
    c = determine_calibration_constant_through_peak_fitting(h_uncal, photon_lines, c_precal)[1]
    h_cal = Histogram( h_uncal.edges[1] .* c, :left )
    h_cal.weights = h_uncal.weights
    return h_cal, h_deconv, peakPositions, threshold, c, c_precal
end
