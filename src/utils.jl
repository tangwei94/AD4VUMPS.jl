const RhoTensor  = AbstractTensorMap{S,1,1} where {S}
const EnvTensorL = AbstractTensorMap{S,1,2} where {S}
const EnvTensorR = AbstractTensorMap{S,2,1} where {S}
const MPSTensor = AbstractTensorMap{S,2,1} where {S}
const MPSBondTensor = AbstractTensorMap{S,1,1} where {S}
const MPOTensor = AbstractTensorMap{S,2,2} where {S}


# copied from MPSKit.jl
function fill_data!(a::TensorMap, dfun)
    for (k, v) in blocks(a)
        map!(x -> dfun(typeof(x)), v, v)
    end

    return a
end
randomize!(a::TensorMap) = fill_data!(a, randn)