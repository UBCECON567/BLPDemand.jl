using Revise
using BLPDemand, Test, ForwardDiff, FiniteDiff

@testset "share and delta" begin
  K = 3
  J = 2
  S = 100
  x = randn(K,J)
  σ = ones(K)
  β = ones(K)
  ξ = randn(J)
  δ = x'*β + ξ
  ν = randn(K,S)

  s = share(δ, σ, x, ν)

  @time d = delta(s,x,ν,σ, tol=sqrt(eps(Float64)));
  
  @test isapprox(δ, delta(s, x, ν, σ, tol=sqrt(eps(Float64))), rtol = eps(Float64)^(0.3))
  @test isapprox(s, share(delta(s,x,ν, σ), σ, x, ν), rtol = eps(Float64)^(0.3))

  # there's a custom method when delta is used with ForwardDiff types
  J = ForwardDiff.jacobian(σ->delta(s, x, ν, σ), σ)
  Jfd=FiniteDiff.finite_difference_jacobian(σ->delta(s, x, ν, σ), σ)
  @test isapprox(J,  Jfd, rtol=eps(Float64)^0.3)
end


@testset "pricing foc" begin
  K = 3
  J = 2
  S = 100
  x = randn(K,J)
  σ = ones(K)
  β = ones(K)
  ξ = randn(J)
  δ = x'*β + ξ
  ν = randn(K,S)

  s = share(δ, σ, x, ν)

  p = x[1,:]
  sp = sharep(β, σ, p, x[2:end,:], ν, ξ)
  @test s ≈ sp
  (sp, ds, Λ, Γ) = dsharedp(β, σ, p, x[2:end,:], ν, ξ)
  @test s ≈ sp
  @test ds ≈ ForwardDiff.jacobian(p->sharep(β,σ, p, x[2:end,:], ν, ξ), p)
  @time (sp, ds, Λ, Γ) = dsharedp(β, σ, p, x[2:end,:], ν, ξ)
  @time ForwardDiff.jacobian(p->sharep(β,σ, p, x[2:end,:], ν, ξ), p)
end


@testset "demandmoments" begin
  
  K = 3
  J = 2
  S = 20
  T = 100
  β = ones(K)
  σ = ones(K)
  ρ = 0.5
  π0 = zeros(K+1,K,J)
  for k in 1:K
    π0[k,k,:] .= 1
  end
  π0[K+1, :, :] .= 1
  
  sim = simulateIVRClogit(T, β, σ, π0, ρ, S)

  @time m0 = demandmoments(β, σ, sim.s, sim.x, sim.ν, sim.z).moments

  # this test will fail with approximate probability 1 - cdf(Normal(),3)^(J*size(π,1)) ≈ 0.01
  @test isapprox(m0, zeros(eltype(m0), size(m0)), atol= 6/sqrt(T))

  # test that moving away from true parameters increases moments
  @test sum(demandmoments(β.+1.0, σ, sim.s, sim.x, sim.ν, sim.z).moments.^2) > sum(m0.^2)
  @test sum(demandmoments(β, σ.+1.0, sim.s, sim.x, sim.ν, sim.z).moments.^2) > sum(m0.^2)
  
end


@testset "equilibrium prices" begin

  K = 3
  J = 2
  S = 100
  x = randn(K-1,J)
  σ = ones(K)
  σ[1] = 0.1
  β = ones(K)
  β[1] = -1.0
  ξ = randn(J)
  ν = randn(K,S)

  L = 2
  w = rand(L,J) .- 1.0
  ω = rand(J) .- 0.5
  γ = ones(L)
  mc = exp.(w'*γ + ω)
  
  p = eqprices(mc, β, σ, ξ, x, ν)

  @test p > mc

  # check alternate form of FOC holds
  s, ds, Λ, Γ = dsharedp(β, σ, p, x, ν, ξ)
  samefirm = ((1:J).==(1:J)')
  @test mc ≈ p .+ (samefirm.*ds) \ s
 
end
  
@testset "supply moments" begin

  J = 2
  T = 100
  K = 3
  L = 2
  S = 100
  
  β = ones(K)
  σ = ones(K)
  γ = ones(L)
  # make price coefficient make sense
  β[1] = -1.0
  σ[1] = 0.1

  sim = simulateBLP(J,T, β, σ, γ, S);


  # check demand moments again
  m0 = demandmoments(β, σ, sim.s, sim.x, sim.ν, sim.w).moments;

  # this test will fail with approximate probability 1 - cdf(Normal(),3)^(J*size(π,1)) ≈ 0.01
  @test isapprox(m0, zeros(eltype(m0), size(m0)), atol= 3/sqrt(T))
  # test that moving away from true parameters increases moments
  @test sum(demandmoments(β.+1.0, σ, sim.s, sim.x, sim.ν, sim.w).moments.^2) > sum(m0.^2)
  @test sum(demandmoments(β, σ.+1.0, sim.s, sim.x, sim.ν, sim.w).moments.^2) > sum(m0.^2)

  iv = sim.w
  ms = supplymoments(γ, β, σ, sim.ξ, sim.x, sim.ν, sim.w, iv).moments

  # Check that supply moments using correct ω
  for j in 1:J
    for k in 1:size(sim.w,1)
      @test isapprox(ms[j,k], sum(sim.ω[j,:].*iv[k,j,:])/size(sim.ω,2),
                     rtol=eps(eltype(sim.ω))^(0.35))
    end
  end
  
  # test that moving away from true parameters increases moments
  @test sum(supplymoments(γ.+1.0, β, σ, sim.ξ, sim.x, sim.ν, sim.w, iv).moments.^2) > sum(ms.^2)
 
end

@testset "estimate RC IV logit" begin

  K = 3
  J = 5
  S = 20
  T = 100
  β = ones(K)*0.2
  σ = ones(K)*0.2
  ρ = 0.8
  π0 = zeros(2K,K,J)
  for k in 1:K
    π0[k,k,:] .= 1
    π0[K+k,k, :] .= 1
  end
  
  sim = simulateIVRClogit(T, β, σ, π0, ρ, S, varξ=0.2)

  @time nfxp = estimateRCIVlogit(sim.s, sim.x, sim.ν, sim.z, method=:NFXP, verbose=true)
  @time mpec = estimateRCIVlogit(sim.s, sim.x, sim.ν, sim.z, method=:MPEC, verbose=true)

  @test isapprox(nfxp.β, mpec.β, rtol=eps(Float64)^(1/4))
  @test isapprox(nfxp.σ, mpec.σ, rtol=eps(Float64)^(1/4))
  @test isapprox(nfxp.ξ, mpec.ξ, rtol=eps(Float64)^(1/4))

end

@testset "estimate BLP" begin

  K = 3
  J = 5
  S = 20
  T = 100
  β = ones(K)*2
  β[1] = -1.5
  σ = ones(K)*0.2
  #σ[1] = 0.0
  γ = ones(2)*0.3
  
  sim = simulateBLP(J,T, β, σ, γ, S, varξ=0.2, varω=0.2)   
  z = makeivblp(cat(sim.x[2:end,:,:], sim.w, dims=1))
  
  @time nfxp = estimateBLP(sim.s, sim.x, sim.ν, z, sim.w, z, method=:NFXP, verbose=true)
  @time mpec = estimateBLP(sim.s, sim.x, sim.ν, z, sim.w, z, method=:MPEC, verbose=true)

  @test isapprox(nfxp.β, mpec.β, rtol=eps(Float64)^(1/4))
  @test isapprox(nfxp.σ, mpec.σ, rtol=eps(Float64)^(1/4))
  @test isapprox(nfxp.γ, mpec.γ, rtol=eps(Float64)^(1/4))  
  @test isapprox(nfxp.ξ, mpec.ξ, rtol=eps(Float64)^(1/4))
  @test isapprox(nfxp.ω, mpec.ω, rtol=eps(Float64)^(1/4))

  var = varianceBLP(mpec.β, mpec.σ, mpec.γ, sim.s, sim.x, sim.ν, z, sim.w, z)

  zd,zs=optimalIV(mpec.β, mpec.σ, mpec.γ, sim.s, sim.x, sim.ν, z, sim.w)

  @time oiv = estimateBLP(sim.s, sim.x, sim.ν, zd, sim.w, zs, method=:MPEC, verbose=true)
  vopt = varianceBLP(oiv.β, oiv.σ, oiv.γ, sim.s, sim.x, sim.ν, zd, sim.w, zs)
  
  @time gel = estimateBLP(sim.s, sim.x, sim.ν, zd, sim.w, zs, method=:GEL, verbose=true)
  
end

tbl = vcat(["" "True" "NFXP" "MPEC" "GEL"],
           [(i->"β[$i]").(1:length(β)) β nfxp.β mpec.β gel.β],
           [(i->"σ[$i]").(1:length(σ)) σ nfxp.σ mpec.σ gel.σ],
           [(i->"γ[$i]").(1:length(γ)) γ nfxp.γ mpec.γ gel.γ])
pretty_table(tbl, noheader=true, formatter=ft_printf("%6.3f",3:5))

s = sim.s; x=sim.x; ν=sim.ν; w=sim.w

@time nfxp = estimateBLP(s, x, ν, z, w, z, method=:NFXP, verbose=true);
@time mpec = estimateBLP(s, x, ν, z, w, z, method=:MPEC,verbose=true);
@time gel = estimateBLP(s, x, ν, z, w, z, method=:GEL,verbose=true);

vnfxp = varianceBLP(nfxp.β, nfxp.σ, nfxp.γ, s, x, ν, z, w, z)
vmpec = varianceBLP(mpec.β, mpec.σ, mpec.γ, s, x, ν, z, w, z)
v = varianceBLP(gel.β, gel.σ, gel.γ, s, x, ν, z, w, z)
vgel = varianceBLP(gel.β, gel.σ, gel.γ, s, x, ν, z, w,z,W=inv(v.varm))

using Printf
f(v) = @sprintf("(%.2f)", sqrt(v))
vtbl = permutedims(hcat(tbl[1,:],
                        [hcat(tbl[i+1,:],
                              ["", "", f.([vnfxp.Σ[i,i], vmpec.Σ[i,i], vgel.Σ[i,i]])...])
                         for i in 1:size(vgel.Σ,1)]...
                        ))
pretty_table(vtbl, noheader=true, formatter=ft_printf("%6.3f",3:5))


zd,zs=optimalIV(mpec.β, mpec.σ, mpec.γ, s, x, ν, z, w)

