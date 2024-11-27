include("test_utils.jl");
include("test_mpsmps_transfer_matrix.jl");
include("test_mpsmpomps_transfer_matrix.jl");
include("test_gauge_fixing.jl");
include("test_ACMap.jl");
include("test_vumps.jl");

@testset "Aqua" verbose = true begin
    using Aqua
    Aqua.test_all(
    AD4VUMPS;
    ambiguities=false,
    stale_deps=false, # FIXME. disable stale_deps for now. 
    deps_compat=false, # FIXME. disable deps_compat for now.
  )
end
