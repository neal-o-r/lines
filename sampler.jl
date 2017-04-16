module MH

type metropolis_hastings

	logl::Function
	n_step::Int

end

function partial(f, a...)

	( (b...) -> f(a..., b...) )

end


function partial_logl(logl::Function, data::Array{Array{Float64, 1}, 1})

	x, y, u = data
	return partial(logl, x, y, u)

end


function acceptance(logl::Function, point::Array{Float64, 1}, 
			curr_point::Array{Float64, 1})

	m_prop, b_prop = point
	m_curr, b_curr = curr_point
	
	a = logl(m_prop, b_prop) / logl(m_curr, b_curr)
	if a > 1		
		return true
	else
		if a < rand()
			return true
		end

	end
			
	return false

end

function proposal_function(sigma::Array{Float64, 1})

	prop = randn(2) .* sigma
	
	return prop

end

function run_sampler(state::metropolis_hastings, init::Array{Float64, 1}, 
		data::Array{Array{Float64, 1}, 1})

	plogl = partial_logl(state.logl, data)

	chain = zeros(state.n_step, 2)
	chain[1, :] = init
	sigma = [0.0001, 0.0005]

	i = 2
	j = 1
	while i <= state.n_step

		step = proposal_function(sigma)
		
		point = chain[i-1, :] .+ step		

		if acceptance(plogl, point, chain[i-1, :])
			chain[i, :] = point
			i += 1
		end
		j += 1

		if i % 100 == 0
			println("Step", i)
		end		

	end
	println("Acceptance Rate: ", i / j)

	return chain

end

end
