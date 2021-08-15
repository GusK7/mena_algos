# Supersmoother
# Equation 3-3
function supersmoother(x::Array{Float64}; n::Int64=10)::Array{Float64}
a = exp(-1.414*3.14159 / n)
b = 2 * a * cosd(1.414 * 180 / n)
c2 = b
c3 = -a * a
c1 = 1 - c2 - c3
@assert n<size(x,1) && n>0 "Argument n out of bounds."
Super = zeros(x)
 @inbounds for i = 3:length(x)
Super[i] = c1 * (x[i] + x[i-1]) / 2 + c2 * Super[i-1] + c3 * Super[i-2]
end
return Super
end