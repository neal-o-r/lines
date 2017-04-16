using PyPlot
using Distributions

include("./likelihoods.jl")
using Likelihoods
include("./sampler.jl")
using MH



srand(123)

function get_data(m::Float64, b::Float64, N::Int64)

	x = sort(10 * rand(N))
	u = 2 * rand(N)
	y = m * x + b + u .* rand(N)

	return x, y, u

end

function least_sq(x::Array{Float64, 1}, y::Array{Float64, 1}, 
			u::Array{Float64,1})

	A = hcat(ones(length(x)), x)

	# Error matrix
	C = diagm(u .* u)
	# bias and slope
	b, m = inv(A' * A) * A' * y
	
	# Cov matrix
	covar = inv(A' * \(C, A))
	b_un, m_un = 1.96 * sqrt(covar[1, 1]), 1.96 * sqrt(covar[2, 2])

	return m, b, m_un, b_un	

end


function conf_region(x::Array{Float64, 1}, y::Array{Float64, 1},
			m::Float64, b::Float64)

	n = length(x)
	t_crit = quantile(TDist(n - 2), 1 - 0.95/2) 	
	s_y = sqrt(sum((y - x * m  + b).^2) / (n - 2))
	x_bar = mean(x)
	s_x = sqrt(sum((x - mean(x)).^2) / n)

	ci = t_crit * s_y * sqrt(1 / n + (x - x_bar).^2 / ((n - 1) * s_x))
	
	return ci

end


function make_chart(x::Array{Float64, 1}, y::Array{Float64, 1}, 
		u::Array{Float64, 1}, m::Float64, 
		b::Float64, ci::Array{Float64, 1})
		
	y_ls = x * m + b

	plot(x, y_ls, "k--")
	fill_between(x, y_ls - ci, y_ls + ci, alpha=0.8, color="0.75")
	errorbar(x, y, yerr=u, fmt=".k")

end


function main()

	m_true = 1.128
	b_true = 4.159
	x, y, u = get_data(m_true, b_true, 20)

	m_ls, b_ls, m_un_ls, b_un_ls = least_sq(x, y, u)

	ci = conf_region(x, y, m_ls + m_un_ls, b_ls + b_un_ls)

	make_chart(x, y, u, m_ls, b_ls, ci)

	@printf("Least Squares
	 	m = %.3f +- %.3f (m_true = %.3f)
		b = %.3f +- %.3f (b_true = %.3f)\n", 
			m_ls, m_un_ls, m_true, b_ls, b_un_ls, b_true)
	
	mh = MH.metropolis_hastings(Likelihoods.lnprob, 10000)

	chain = MH.run_sampler(mh, [m_ls, b_ls], [x, y, u])

	figure()
	PyPlot.plt[:hist](chain[:, 1], 100)
	title("M")

	figure()
	PyPlot.plt[:hist](chain[:, 2], 100)
	title("B")

	return chain	

end
