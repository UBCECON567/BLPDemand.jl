using Revise
using BLPDemand, Test, ForwardDiff, FiniteDiff, LinearAlgebra, Statistics, JuMP, Ipopt

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

  @time m0 = demandmoments(β, σ, sim).moments


  # this test will fail with approximate probability 1 - cdf(Normal(),3)^(J*size(π,1)) ≈ 0.01
  @test isapprox(m0, zeros(eltype(m0), size(m0)), atol= 6/sqrt(T))

  # test that moving away from true parameters increases moments
  @test sum(demandmoments(β.+1.0, σ, sim).moments.^2) > sum(m0.^2)
  @test sum(demandmoments(β, σ.+1.0, sim).moments.^2) > sum(m0.^2)

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

  sim, ξ, ω = simulateBLP(J,T, β, σ, γ, S)


  # check demand moments again
  m0 = demandmoments(β, σ, sim).moments;

  # this test will fail with approximate probability 1 - cdf(Normal(),3)^(J*size(π,1)) ≈ 0.01
  #@test isapprox(m0, zeros(eltype(m0), size(m0)), atol= 5/sqrt(T))
  # test that moving away from true parameters increases moments
  @test sum(demandmoments(β.+1.0, σ, sim).moments.^2) > sum(m0.^2)
  @test sum(demandmoments(β, σ.+1.0, sim).moments.^2) > sum(m0.^2)

  ms = supplymoments(γ, β, σ, ξ, sim).moments

  # Check that supply moments using correct ω
  for k in 1:size(sim[1].w,1)
    @test isapprox(ms[k], sum([dot(ω[t], sim[t].zs[k,:]) for t in 1:T])/T,
                   rtol=eps(eltype(ω[1]))^(0.2))
  end


  # test that moving away from true parameters increases moments
  @test sum(supplymoments(γ.+1.0, β, σ, ξ, sim).moments.^2) > sum(ms.^2)

end

@testset "estimate RC IV logit" begin

  K = 3
  J = 20
  S = 20
  T = 200
  β = [-1, 1, 1]*0.2
  σ = ones(K)*0.2
  ρ = 0.8
  π0 = zeros(2K,K,J)
  for k in 1:K
    π0[k,k,:] .= 1
    π0[K+k,k, :] .= -1
  end

  sim = simulateIVRClogit(T, β, σ, π0, ρ, S, varξ=0.5);
  @show quantile(vcat((d->d.s[:]).(sim)...), [0, 0.05, 0.5, 0.95, 1])

  @time nfxp = estimateRCIVlogit(sim, method=:NFXP, verbose=true)
  @time n2 = estimateBLP(sim, method=:NFXP, verbose=true, supply=false)
  @time mpec = estimateRCIVlogit(sim, method=:MPEC, verbose=true)
  @time m2 = estimateBLP(sim, method=:MPEC, verbose=true, supply=false,
                         optimizer=with_optimizer(Ipopt.Optimizer,
                                                   max_iter= 200,
                                                   start_with_resto = "no",
                                                   #hessian_approximation="limited-memory",
                                                   #max_soc=10,
                                                   #soc_method=0,
                                                   print_level = 5))

  @test isapprox(nfxp.β, mpec.β, rtol=eps(Float64)^(1/4))
  @test isapprox(nfxp.σ, mpec.σ, rtol=eps(Float64)^(1/4))

  @show fracRCIVlogit(sim)

end

@testset "estimate BLP" begin

  K = 3
  J = 10
  S = 10
  T = 50
  β = ones(K)*2
  β[1] = -1.5
  σ = ones(K)
  σ[1] = 0.2
  γ = ones(2)*0.3

  sim, ξ, ω = simulateBLP(J,T, β, σ, γ, S, varξ=0.2, varω=0.2);
  @show quantile(vcat((d->d.s[:]).(sim)...), [0, 0.05, 0.5, 0.95, 1])

  @time nfxp = estimateBLP(sim, method=:NFXP, verbose=true)
  @time mpec = estimateBLP(sim, method=:MPEC, verbose=true,
                           optimizer=with_optimizer(Ipopt.Optimizer,
                                                    max_iter= 100,
                                                    start_with_resto = "no",
                                                    #hessian_approximation="limited-memory",
                                                    #watchdog_shortened_iter_trigger = 5,
                                                    print_level = 5))

  @time gel = estimateBLP(sim,  method=:GEL, verbose=true,
                          optimizer=with_optimizer(Ipopt.Optimizer,
                                                   max_iter= 200,
                                                   start_with_resto = "yes",
                                                   #hessian_approximation="limited-memory",
                                                   #max_soc=10,
                                                   #soc_method=0,
                                                   print_level = 5))

  @test isapprox(nfxp.β, mpec.β, rtol=eps(Float64)^(1/4))
  @test isapprox(nfxp.σ, mpec.σ, rtol=eps(Float64)^(1/4))
  @test isapprox(nfxp.γ, mpec.γ, rtol=eps(Float64)^(1/4))

  @test isapprox(gel.β, mpec.β, rtol=eps(Float64)^(1/4))
  @test isapprox(gel.σ, mpec.σ, rtol=eps(Float64)^(1/4))
  @test isapprox(gel.γ, mpec.γ, rtol=eps(Float64)^(1/4))

  v = varianceBLP(mpec.β, mpec.σ, mpec.γ, sim)

  simo=optimalIV(nfxp.β, max.(nfxp.σ, 0.1), nfxp.γ, sim);

  @time no = estimateBLP(simo, method=:NFXP, verbose=true)
  @time mo = estimateBLP(simo, method=:MPEC, verbose=true)
  @time go = estimateBLP(simo, method=:GEL, verbose=true)

  @test isapprox(no.β, mo.β, rtol=eps(Float64)^(1/4))
  @test isapprox(no.σ, mo.σ, rtol=eps(Float64)^(1/4))
  @test isapprox(no.γ, mo.γ, rtol=eps(Float64)^(1/4))

  @test isapprox(go.β, mo.β, rtol=eps(Float64)^(1/4))
  @test isapprox(go.σ, mo.σ, rtol=eps(Float64)^(1/4))
  @test isapprox(go.γ, mo.γ, rtol=eps(Float64)^(1/4))

  vo = varianceBLP(no.β, no.σ, no.γ, simo)


end
