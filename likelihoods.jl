module Likelihoods

function lnprior(m::Float64, b::Float64)

	# Non informative priors over a generous range
	if (-5.0 < m < 5.0) & (-20 < b < 20)
		return 	0.0
	end

	return -Inf
end


function lnlike(x::Array{Float64, 1}, y::Array{Float64, 1}, 
	u::Array{Float64, 1}, m::Float64, b::Float64)
	
	# linear model
	model = m * x + b
	inv_sigma2 = 1.0 ./ u.^2

	log_like = -0.5 * (
		sum((y .- model).^2 .* inv_sigma2 - log(inv_sigma2)))

	return log_like

end


function lnprob(x::Array{Float64, 1}, y::Array{Float64, 1}, 
	u::Array{Float64, 1}, m::Float64, b::Float64)

	lp = lnprior(m, b)
	if !isfinite(lp)
		return -Inf
	end

	return lp + lnlike(x, y, u, m, b)

end

end
