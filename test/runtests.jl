using Revise
#push!(LOAD_PATH, "../BLPDemand")
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
  @test isapprox(mc, p .+ (samefirm.*ds) \ s, rtol=eps(eltype(mc))^(1/4))

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
  @test isapprox(m0, zeros(eltype(m0), size(m0)), atol= 5/sqrt(T))
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
  J = 5
  S = 10
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
                         optimizer=optimizer_with_attributes(Ipopt.Optimizer,
                                                   "max_iter" => 200,
                                                   "start_with_resto" => "no",
                                                   #hessian_approximation="limited-memory",
                                                   #max_soc=10,
                                                   #soc_method=0,
                                                   "print_level" => 5))

  @test isapprox(nfxp.β, mpec.β, rtol=eps(Float64)^(1/4))
  @test isapprox(nfxp.σ, mpec.σ, rtol=eps(Float64)^(1/4))
  @test isapprox(n2.β, nfxp.β, rtol=eps(Float64)^(1/4))
  @test isapprox(n2.σ, nfxp.σ, rtol=eps(Float64)^(1/4))
  @test isapprox(n2.β, m2.β, rtol=eps(Float64)^(1/4))
  @test isapprox(n2.σ, m2.σ, rtol=eps(Float64)^(1/4))


  @show fracRCIVlogit(sim)

end

@testset "NFXP" begin
  K = 3
  J = 5 # number of products
  T = 100 # number of markets
  β = ones(K)
  β[1] = -0.1
  σ = [0.2, 0.5*ones(length(β)-1)...] # Σ = Diagonal(σ)
  γ = ones(2) # marginal cost coefficients
  S = 10 # number of Monte-Carlo draws per market and product to integrate out ν
  sξ = 0.2 # standard deviation of demand shocks
  sω = 0.2 # standard deviation of cost shocks

  sim, ξ, ω = simulateBLP(J,T, β, σ, γ, S, varξ=sξ, varω=sω);
  @show quantile(vcat((d->d.s[:]).(sim)...), [0, 0.05, 0.5, 0.95, 1])

  β0, σ0 = fracRCIVlogit(sim)
  σ0 = abs.(σ0)
  lb = min(minimum(σ0)/2,1e-2)
  γ0 = γ
  θ0, unpack = pack(β0, σ0, γ0, lb=lb)
  @testset "pack-unpack" begin
    @test β0 ≈ unpack(θ0).β
    @test σ0 ≈ unpack(θ0).σ
    @test γ0 ≈ unpack(θ0).γ
    @test θ0 ≈ pack(unpack(θ0)..., lb=lb)[1]
  end

  # this is how initial γ is set
  m, ξ0 = demandmoments(β0, 0*σ0, sim)
  m, logmc0 = supplymoments(0*γ0, β0, 0*σ0, ξ0, sim, costf=x->BLPDemand.safelog(x,δ=0.01) )
  Y = vcat(logmc0...)
  X = hcat((d->d.w).(sim)...)
  X' \ Y


  obj = BLPDemand.nfxpobjective(sim, unpack, true, :log, I)
  @testset "Derivatives" begin
    f = β->demandmoments(β,σ0,sim)[1]
    @test ForwardDiff.jacobian(f, β0) ≈ FiniteDiff.finite_difference_jacobian(f,β0)
    f = s->demandmoments(β0,s,sim)[1]
    @test isapprox(ForwardDiff.jacobian(f, σ0), FiniteDiff.finite_difference_jacobian(f,σ0), rtol=1e-4)
    _, ξ = demandmoments(β0, σ0, sim)
    f = β->supplymoments(γ0, β, σ0, ξ, sim, costf=BLPDemand.safelog)[1]
    @test isapprox(ForwardDiff.jacobian(f, β0),FiniteDiff.finite_difference_jacobian(f,β0),rtol=1e-4)
    f = γ->supplymoments(γ, β0, σ0, ξ, sim, costf=BLPDemand.safelog)[1]
    @test isapprox(ForwardDiff.jacobian(f, γ0),FiniteDiff.finite_difference_jacobian(f,γ0),rtol=1e-4)

    @test isapprox(ForwardDiff.gradient(obj,θ0), FiniteDiff.finite_difference_gradient(obj,θ0), rtol=1e-4)
  end

end


@testset "estimate BLP" begin

  K = 3
  J = 5
  S = 20
  T = 50
  β = ones(K)
  β[1] = -0.5
  σ = ones(K)
  σ[1] = 0.2
  γ = ones(2)*0.3
  sξ = 0.2
  sω = 0.2

  sim, ξ, ω = simulateBLP(J,T, β, σ, γ, S, varξ=sξ, varω=sω);
  @show quantile(vcat((d->d.s[:]).(sim)...), [0, 0.05, 0.5, 0.95, 1])

  @time nfxp = estimateBLP(sim, method=:NFXP, verbose=true)

  @time mpec = estimateBLP(sim, method=:MPEC, verbose=true,
                           optimizer=optimizer_with_attributes(Ipopt.Optimizer,
                                                    "max_iter" => 500,
                                                    #"start_with_resto" => "yes",
                                                    #hessian_approximation="limited-memory",
                                                    #"watchdog_shortened_iter_trigger" => 1,
                                                    "print_level" => 5))

  @time gel = estimateBLP(sim,  method=:GEL, verbose=true,
                          optimizer=optimizer_with_attributes(Ipopt.Optimizer,
                                                   "max_iter" => 500,
                                                   "start_with_resto" => "yes",
                                                   #hessian_approximation="limited-memory",
                                                   #max_soc=10,
                                                   #soc_method=0,
                                                   "print_level" => 5))

  @test isapprox(nfxp.β, mpec.β, rtol=eps(Float64)^(1/4))
  @test isapprox(nfxp.σ, mpec.σ, rtol=eps(Float64)^(1/4))
  @test isapprox(nfxp.γ, mpec.γ, rtol=eps(Float64)^(1/4))

  # These tests might fail, but should not fail by too much
  @test isapprox(gel.β, mpec.β, rtol=eps(Float64)^(1/4))
  @test isapprox(gel.σ, mpec.σ, rtol=eps(Float64)^(1/4))
  @test isapprox(gel.γ, mpec.γ, rtol=eps(Float64)^(1/4))

  v = varianceBLP(nfxp.β, nfxp.σ, nfxp.γ, sim)

  simo=optimalIV(nfxp.β, max.(nfxp.σ, 0.1), nfxp.γ, sim);

  @time no = estimateBLP(simo, method=:NFXP, verbose=true)
  @time mo = estimateBLP(simo, method=:MPEC, verbose=true,
                         optimizer=optimizer_with_attributes(Ipopt.Optimizer,
                                                             "max_iter" => 500,
                                                             "start_with_resto" => "yes",
                                                             #hessian_approximation="limited-memory",
                                                             #"watchdog_shortened_iter_trigger" => 1,
                                                             "print_level" => 5))

  @time go = estimateBLP(simo, method=:GEL, verbose=true,
                         optimizer=optimizer_with_attributes(Ipopt.Optimizer,
                                                             "max_iter" => 500,
                                                             "start_with_resto" => "yes",
                                                             #hessian_approximation="limited-memory",
                                                             #max_soc=10,
                                                             #soc_method=0,
                                                             "print_level" => 5))


  @test isapprox(no.β, mo.β, rtol=eps(Float64)^(1/4))
  @test isapprox(no.σ, mo.σ, rtol=eps(Float64)^(1/4))
  @test isapprox(no.γ, mo.γ, rtol=eps(Float64)^(1/4))

  # These tests might fail, but should not fail by too much
  @test isapprox(go.β, mo.β, rtol=eps(Float64)^(1/4))
  @test isapprox(go.σ, mo.σ, rtol=eps(Float64)^(1/4))
  @test isapprox(go.γ, mo.γ, rtol=eps(Float64)^(1/4))

  vo = varianceBLP(no.β, no.σ, no.γ, simo)


end
