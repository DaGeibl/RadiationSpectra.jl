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
    return best.c
end


function _filter_peaks_from_peakfinder(h::Histogram{<:Real, 1}, peakPositions::Vector{T}, σ::Real) where {T <: Real}
    pfits = Vector{@NamedTuple{μ::Real, σ::Real}}(undef, length(peakPositions))
    success =  Bool.(zeros(length(peakPositions)))
    Threads.@threads for ipf in eachindex(peakPositions)
        p = peakPositions[ipf]
        try
            searchrange = (p - 10σ*step(h.edges[1]), p + 10σ*step(h.edges[1]))
            h_search = subhist(h, searchrange)
            # find peak
            cts_argmax = mapslices(argmax, h_search.weights, dims=1)[1]
            cts_max    = h_search.weights[cts_argmax]
            μ = getindex(h_search.edges[1], cts_argmax)

            # find FWHM of peak by interpolation of the two points at which the peak drops below half of its maximum
            threshold = 0.5
            left_peak_weights = h_search.weights[1:cts_argmax]
            cut_left_arg  = findfirst(idx -> left_peak_weights[idx] >= threshold*cts_max && all(left_peak_weights[idx+1:end] .> left_peak_weights[idx]), eachindex(left_peak_weights))
            left_point1 = (getindex(h_search.edges[1], ifelse(cut_left_arg == 1, cut_left_arg + 1, cut_left_arg -1 )), h_search.weights[ifelse(cut_left_arg == 1, cut_left_arg + 1, cut_left_arg -1 )]) 
            left_point_2 = (getindex(h_search.edges[1], cut_left_arg), h_search.weights[cut_left_arg])
            cut_left = _interpolate_linear(left_point1, left_point_2; threshold=threshold*cts_max)
            right_peak_weights = h_search.weights[cts_argmax:end]
            cut_right_arg = findfirst(idx -> right_peak_weights[idx] <= threshold*cts_max && all(right_peak_weights[idx+1:end] .< right_peak_weights[idx]), eachindex(right_peak_weights)) + cts_argmax - 1
            right_point1 = (getindex(h_search.edges[1], cut_right_arg-1), h_search.weights[cut_right_arg-1])
            right_point2 = (getindex(h_search.edges[1], cut_right_arg), h_search.weights[cut_right_arg])
            cut_right = _interpolate_linear(right_point1, right_point2; threshold=threshold*cts_max)
            peak_σ = (cut_right -  cut_left) * inv(2*√(2log(2)))
            success[ipf] = true
            pfits[ipf] = (μ = μ, σ = peak_σ)
        catch e
            @debug "Error filter peak at $p: $e"
        end
    end
    pfits = pfits[success]
    peakPositions = peakPositions[success]
    d = map(i -> abs(pfits[i].μ - peakPositions[i]), eachindex(peakPositions));
    σs = map(f -> f.σ, pfits);   
    hd = harmmean(d);
    hσ = harmmean(σs);
    accepted_peakPositions = T[]
    for (ipf, pf) in enumerate(pfits)
        if d[ipf] < 4 * hd && abs(pf.σ) < 2hσ
            push!(accepted_peakPositions, peakPositions[ipf])
        end
    end
    accepted_peakPositions
end


function determine_calibration_constant_through_peak_ratios(h::Histogram{<:Real, 1, E}, photon_lines::Array{<:Real, 1};
            min_n_peaks::Int = length(photon_lines), max_n_peaks::Int = 5 * length(photon_lines), threshold::Real = 10., α::Real = 0.01, σ::Real = 3.0, rtol::Real = 5e-3,
            min_n_matches::Int = min(4, min_n_peaks, length(photon_lines)), min_match_span::Real = min_n_matches <= 2 ? 0.0 : 0.5) where {E}
    h_deconv, peakPositions = peakfinder(h, threshold=threshold, σ = σ)
    fitted_peak_Positions = _filter_peaks_from_peakfinder(h, peakPositions, σ)
    threshold_reduced = threshold
    while length(fitted_peak_Positions) > max_n_peaks
        threshold_reduced *= 1.1
        if threshold_reduced > 70.0
            throw(ErrorException("Found too many peaks in the spectrum."))
        end
        h_deconv, peakPositions = peakfinder(h, threshold=threshold_reduced, σ = σ)
        fitted_peak_Positions = _filter_peaks_from_peakfinder(h, peakPositions, σ)
    end
    while length(fitted_peak_Positions) < min_n_peaks
        next_threshold = threshold_reduced * 0.75
        if next_threshold < 0.5
            @warn "Could not find enough peaks in the spectrum."
            break
        end
        threshold_reduced = next_threshold
        h_deconv, peakPositions = peakfinder(h, threshold=threshold_reduced, σ = σ)
        fitted_peak_Positions = _filter_peaks_from_peakfinder(h, peakPositions, σ)
    end
    if length(fitted_peak_Positions) < min_n_peaks
        threshold_reduced = threshold
        h_deconv, fitted_peak_Positions = peakfinder(h, threshold=threshold_reduced, σ = σ)
        while length(fitted_peak_Positions) < min_n_peaks
            next_threshold = threshold_reduced * 0.75
            if next_threshold < 0.5
                @warn "Could not find enough peaks in the spectrum."
                break
            end
            threshold_reduced = next_threshold
            h_deconv, fitted_peak_Positions = peakfinder(h, threshold=threshold_reduced, σ = σ)
        end
    end
    if length(fitted_peak_Positions) < min_n_peaks
        throw(ErrorException("Could not find enough peaks in the spectrum."))
    end
    @debug "Found peak positions at: $fitted_peak_Positions"
    c = determine_calibration_constant_through_peak_ratios(fitted_peak_Positions, photon_lines; α, rtol, min_n_matches, min_match_span)
    return c, h_deconv, fitted_peak_Positions, threshold_reduced
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
