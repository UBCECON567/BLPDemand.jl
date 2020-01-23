using Revise
using BLPDemand, Test


@testset "share" begin
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

  @test isapprox(δ, delta(s, x, ν, σ, tol=sqrt(eps(Float64))).δ, rtol = eps(Float64)^(0.4))
  @test isapprox(s, share(delta(s,x,ν, σ).δ, σ, x, ν), rtol = eps(Float64)^(0.4))

  p = x[1,:]
  sp = sharep(β, σ, p, x[2:end,:], ν, ξ)
  @test s ≈ sp
  (sp, ds, Λ, Γ) = dsharedp(β, σ, p, x[2:end,:], ν, ξ)
  @test s ≈ sp
  @test ds ≈ ForwardDiff.jacobian(p->sharep(β,σ, p, x[2:end,:], ν, ξ), p)  
end


@testset "demandmoments" begin
  
  K = 3
  J = 2
  S = 100
  T = 100
  β = ones(K)
  σ = ones(K)
  ρ = 0.5
  π = ones(K+1,K,J)
  
  sim = simulateIVRClogit(T, β, σ, π, ρ, S)

  m0 = demandmoments(β, σ, sim.s, sim.x, sim.ν, sim.z)

  # this test will fail with approximate probability 1 - cdf(Normal(),3)^(J*size(π,1)) ≈ 0.01
  @test isapprox(m0, zeros(eltype(m0), size(m0)), atol= 3/sqrt(T))

  # test that moving away from true parameters increases moments
  @test sum(demandmoments(β.+1.0, σ, sim.s, sim.x, sim.ν, sim.z).^2) > sum(m0.^2)
  @test sum(demandmoments(β, σ.+1.0, sim.s, sim.x, sim.ν, sim.z).^2) > sum(m0.^2)
  
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

  sim = simulateBLP(J,T, β, σ, γ, S)


  # check demand moments again
  m0 = demandmoments(β, σ, sim.s, sim.x, sim.ν, sim.w)

  # this test will fail with approximate probability 1 - cdf(Normal(),3)^(J*size(π,1)) ≈ 0.01
  @test isapprox(m0, zeros(eltype(m0), size(m0)), atol= 3/sqrt(T))
  # test that moving away from true parameters increases moments
  @test sum(demandmoments(β.+1.0, σ, sim.s, sim.x, sim.ν, sim.w).^2) > sum(m0.^2)
  @test sum(demandmoments(β, σ.+1.0, sim.s, sim.x, sim.ν, sim.w).^2) > sum(m0.^2)

  iv = sim.w
  ms = supplymoments(γ, β, σ, sim.ξ, sim.x, sim.ν, sim.w, iv)

  # Check that supply moments using correct ω
  for j in 1:J
    for k in 1:size(sim.w,1)
      @test isapprox(ms[j,k], sum(sim.ω[j,:].*iv[k,j,:])/size(sim.ω,2),
                     rtol=eps(eltype(sim.ω))^(0.4))
    end
  end
  
  # test that moving away from true parameters increases moments
  @test sum(supplymoments(γ.+1.0, β, σ, sim.ξ, sim.x, sim.ν, sim.w, iv).^2) > sum(ms.^2)
 
end
