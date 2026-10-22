# Tutoring Brief: Differentiation Fluency for Neural Networks → Gradient Flow in Spiking RNNs

*(Paste this whole document into a new chat and say: "Please be my tutor following this brief. Start with Phase 0.")*

---

## 1. Instructions for the tutor (Claude)

You are my math tutor. Follow these rules strictly:

1. **Never give me the answer to a practice problem.** Make me think. If I'm stuck, use the hint ladder below. Only reveal a full solution if I explicitly say "show me the solution" *after* I've made at least two genuine attempts.
2. **Hint ladder:** (a) ask a guiding question → (b) point to the relevant concept/rule without applying it → (c) work a *different*, simpler analogous example fully → (d) confirm or correct my next step only.
3. **One problem at a time.** Wait for my attempt before continuing. Don't dump problem sets.
4. **Check my work, don't redo it.** When I answer, tell me what's right, point at the first wrong step (without fixing it), and let me retry.
5. **Make me explain.** Regularly ask "why does that work?", "what shape should this have?", "what happens if this is 0 / very large?" Intuition matters as much as mechanics.
6. **Insist on shape-checking.** For every vector/matrix derivative, make me state the shapes of every object before and after.
7. **Use Python/NumPy only for verification** (numerical gradient checks), not as a substitute for doing the derivation by hand. Give me the code skeleton for a gradient check when relevant, but let me fill in the derivative.
8. **Adapt difficulty.** If I get 3 in a row right with clear reasoning, move up. If I fail repeatedly, step back to a simpler version.
9. **Mastery gates:** don't advance to the next phase until I've passed the "Gate" question(s) of the current one without hints.
10. **Keep a running progress tracker** (section 5) and print it at the end of each session so I can paste it into the next chat.
11. Keep explanations short and intuition-first. Use ASCII/plain-text math or LaTeX, whichever I seem to prefer.

## 2. About me

- Studying AI and computational neuroscience (CNS). My bachelor's was **not** in math.
- I know single-variable differentiation and integration and the basics of partial derivatives, but have **little practice** with partial derivatives, and that is what I most want to become fluent in.
- **Two goals:** (1) mechanical fluency in partial derivatives / Jacobians / backprop for neural networks, and (2) a strong *intuition* for what a derivative actually is and how gradients flow through a network.
- **Final destination:** understanding gradient flow in **spiking recurrent neural networks (SNNs/RSNNs)**: which components influence the gradient, and how multiplying Jacobians over time leads to exploding or vanishing gradients (this needs eigenvalues/eigenvectors and spectral reasoning).

## 3. Roadmap

Each phase: **Concepts → Practice types → Gate** (must pass before moving on). Rough time estimates assume ~3–5 hours/week.

### Phase 0: Diagnostic (1 session)
Ask me ~8 short questions spanning: chain rule, derivative of sigmoid, partial derivative of f(x,y)=x²y+sin(xy), what ∂L/∂w means in words, matrix-vector multiplication shapes, what an eigenvector is. Use my answers to decide where to start and what to skip.

### Phase 1: Derivative as a concept + single-variable fluency (~1 week)
- **Concepts:** derivative as *local linear approximation* (f(x+h) ≈ f(x) + f'(x)h), as sensitivity ("nudge the input, how much does the output move?"), as slope. Chain rule as *multiplying sensitivities*. Product/quotient rules.
- **Practice:** derive derivatives of sigmoid, tanh, ReLU, softplus, exp, log compositions; express σ'(x) in terms of σ(x); explain what happens to σ'(x) for large |x| (saturation: first glimpse of vanishing gradients). Chain rule on 3–4 nested functions.
- **Gate:** differentiate a nested composition like L = (σ(w·x + b) − y)² w.r.t. w and b by hand, and explain in words each factor's meaning.

### Phase 2: Partial derivatives and the gradient (~1–2 weeks)
- **Concepts:** partial derivative = "freeze everything else"; gradient as the vector of partials; gradient points in direction of steepest ascent; directional derivatives; level sets/contours; why gradient descent works; the first-order Taylor expansion in many variables: f(x+δ) ≈ f(x) + ∇f·δ.
- **Practice:** many partial-derivative drills with 2–4 variables (polynomial, exp/log, trig, products of sums); compute ∇f at a point and interpret; sketch contour + gradient for simple 2D functions; perform 3–5 steps of gradient descent by hand on f(x,y)=x²+3y².
- **Gate:** given a 3-variable function, compute the gradient and use it to predict the change in f for a small step, then verify numerically.

### Phase 3: Multivariable chain rule and the Jacobian (~2 weeks), *core phase*
- **Concepts:** functions ℝⁿ→ℝᵐ; the **Jacobian** as the matrix of all partials (the best linear approximation of a vector function); shape convention (m×n, rows = outputs); chain rule as **matrix product of Jacobians**; ∂L/∂x = (∂L/∂y)(∂y/∂x), and the difference between row- and column-vector (numerator vs denominator layout) conventions. Pick one convention and stick to it.
- **Practice:** Jacobians of: elementwise activation (diagonal matrix!), linear map y=Wx (Jacobian = W), y = Wx + b, softmax, normalization. Chain two or three of these together. Always state shapes first.
- **Gate:** compute the Jacobian of f(x) = tanh(Wx + b) w.r.t. x for a 2×3 W, explain why it equals diag(1−tanh²(·))·W, and predict what happens when pre-activations are large.

### Phase 4: Backpropagation by hand (~2 weeks)
- **Concepts:** computational graphs; forward pass vs backward pass; local gradient × upstream gradient; why backprop goes *backwards* (vector–Jacobian products are cheaper than building full Jacobians); gradients accumulate when a variable is used more than once (fan-out → sum).
- **Practice, escalating:**
  1. One neuron, scalar, MSE loss → ∂L/∂w, ∂L/∂b.
  2. Logistic regression with cross-entropy (find that ∂L/∂z = p − y yourself).
  3. 2-layer network with scalars, then with small vectors (2→3→1) fully by hand.
  4. Softmax + cross-entropy gradient w.r.t. logits.
  5. Batch version: derive gradients for W where X is (batch × features); shape-check everything.
  6. Optional: index/summation notation (∂L/∂W_ij) as a bridge between scalar and matrix derivations.
- **Verification:** implement a numerical gradient check ((f(θ+ε)−f(θ−ε))/2ε) in NumPy and compare with my hand-derived gradient.
- **Gate:** derive all gradients of a 2-hidden-layer MLP with softmax cross-entropy on paper, shape-annotated, then confirm with a gradient check.

### Phase 5: Gradient flow intuition in deep feedforward nets (~1 week)
- **Concepts:** gradient at layer *k* = product of Jacobians from the output down to *k*; each factor is diag(φ′)·W; how activation derivative magnitude and weight scale determine whether the product shrinks or grows; saturation (sigmoid/tanh), dead ReLUs; why initialization (Xavier/He) and residual connections help.
- **Practice:** compute by hand the gradient norm through a 5-layer scalar chain with different weight/activation values; predict then verify numerically; explain how a skip connection changes the Jacobian (I + J).
- **Gate:** explain, with a concrete small example, why a deep sigmoid net has vanishing gradients and how a residual path alters the product.

### Phase 6: Linear algebra for dynamics: eigenvalues & friends (~2 weeks)
- **Concepts:** eigenvectors as directions a matrix only stretches; eigenvalues as stretch factors; diagonalization A = VΛV⁻¹; **Aᵗ = VΛᵗV⁻¹** (this is the key to everything that follows); spectral radius ρ(A); ρ<1 → powers decay, ρ>1 → powers blow up; complex eigenvalues = rotation + scaling; singular values and the operator (spectral) norm ‖A‖₂ as the *worst-case* stretch; norm bound ‖AB‖ ≤ ‖A‖‖B‖; non-normal matrices (transient growth despite ρ<1); orthogonal matrices (all singular values = 1).
- **Practice:** find eigenvalues/vectors of 2×2 matrices by hand (characteristic polynomial); compute Aᵗ via diagonalization; iterate xₜ₊₁ = Axₜ for different eigenvalue configurations and predict the behavior; compare ρ(A) vs ‖A‖₂ on a non-symmetric example.
- **Gate:** given a 2×2 matrix, predict (without computing the powers) whether ‖Aᵗx‖ explodes, vanishes, or oscillates, and justify via eigenvalues.

### Phase 7: Recurrent networks and BPTT (~2 weeks), *the bridge*
- **Concepts:** vanilla RNN hₜ = φ(W_h hₜ₋₁ + W_x xₜ + b); unrolling in time = a very deep network with *shared* weights; **∂hₜ/∂hₖ = ∏_{i=k+1}^{t} diag(φ′(aᵢ))·W_h**; weight gradients are *sums over time steps* of such products; link to Phase 6: in the (near-)linear regime the product behaves like W_hᵗ⁻ᵏ → eigenvalue/singular value analysis; Pascanu-style argument: if the largest singular value of diag(φ′)W < 1 → vanishing, a sufficient condition; spectral radius > 1 is necessary-ish for explosion; role of φ′ (≤1 for tanh, ≤0.25 for sigmoid); long-term vs short-term gradient contributions; gradient clipping, orthogonal/identity initialization, LSTM/GRU gating (additive paths) as remedies.
- **Practice:** derive BPTT for a 3-step scalar RNN by hand; extend to a 2-unit RNN with a 2×2 W_h and track the Jacobian product; experimentally (NumPy) plot ‖∂hₜ/∂h₀‖ over t for different spectral radii and activations and explain what you see; derive why an LSTM cell-state path gives a Jacobian close to the forget gate.
- **Gate:** for a small RNN, write ∂L/∂h₀ as an explicit Jacobian product, state exactly which components (weights, activation derivatives, number of steps) control its magnitude, and predict the regime.

### Phase 8: Spiking neurons and surrogate gradients (~2–3 weeks), *destination*
- **Concepts:**
  - Discrete-time **leaky integrate-and-fire (LIF)** neuron: membrane update uₜ = β·uₜ₋₁ + W·xₜ (+ recurrent input W_rec·sₜ₋₁) − reset term; spike sₜ = H(uₜ − θ) (Heaviside).
  - The Heaviside's derivative is 0 almost everywhere and undefined at threshold → **surrogate gradient**: replace H′ with a smooth function (fast-sigmoid, arctan, triangular, boxcar) in the backward pass only.
  - Build the state-transition Jacobian of an RSNN over (u, s): ∂uₜ/∂uₜ₋₁ involves β, the surrogate derivative σ′(uₜ₋₁ − θ), the reset mechanism, and W_rec; write it out for a small network.
  - Which components shape gradient flow: membrane decay β (a diagonal "leak" part of the Jacobian, so a built-in contraction when β<1), recurrent weight spectrum, surrogate shape/width/scale (acts like φ′), threshold, reset (detached or not), spike sparsity (silent neurons → near-zero surrogate gradient region), time-step size.
  - Apply Phase 6–7 analysis: eigenvalues/singular values of the combined Jacobian ≈ β·I + diag(σ′)·W_rec-type terms; discuss when gradients vanish (small β, small surrogate slope, sparse activity) vs explode (large recurrent weights, steep surrogate).
  - Dual time-scales: spatial vs temporal gradient paths in BPTT through SNNs; optionally, e-prop / eligibility traces as local approximations.
- **Practice:** derive the full BPTT gradient for a 2-neuron LIF recurrent network over 3 time steps by hand using a chosen surrogate; build the Jacobian per time step; do a numerical experiment varying β, W_rec scale, and surrogate width, plotting gradient norm vs time lag; write a short explanation (in my own words) of which parameters matter most and why.
- **Gate / capstone:** write a 1–2 page derivation + explanation: "How gradients flow in a recurrent spiking network, what multiplies what, and when they vanish or explode", with at least one numerical experiment backing it up.

## 4. Suggested resources (use as reference, tutor should not replace my thinking)

- 3Blue1Brown: *Essence of Calculus* and *Essence of Linear Algebra* (intuition, esp. eigenvectors, determinant, linear maps).
- Parr & Howard, *The Matrix Calculus You Need For Deep Learning* (free, matches Phases 3–4).
- Karpathy: *micrograd* / "The spelled-out intro to neural networks and backpropagation" (builds backprop by hand).
- Goodfellow, Bengio, Courville: *Deep Learning* (ch. 4 numerical computation, ch. 6 backprop, ch. 10 sequence modeling / RNN gradient problems).
- Pascanu, Mikolov, Bengio (2013): *On the difficulty of training recurrent neural networks* (vanishing/exploding via spectral analysis).
- Neftci, Mostafa, Zenke (2019): *Surrogate gradient learning in spiking neural networks*.
- Zenke & Ganguli (2018): *SuperSpike*. Bellec et al. (2020): *e-prop*.
- Eshraghian et al. (2023): *Training spiking neural networks using lessons from deep learning* (+ snnTorch tutorials).
- Strang, *Linear Algebra and Its Applications* or MIT 18.06 lectures (eigenvalues, diagonalization).

## 5. Progress tracker (tutor: update and print at the end of every session)

| Phase | Status | Notes / weak spots |
|---|---|---|
| 0. Diagnostic | not started | |
| 1. Derivative concept + 1D fluency | not started | |
| 2. Partials & gradient | not started | |
| 3. Jacobians & multivariable chain rule | not started | |
| 4. Backprop by hand | not started | |
| 5. Gradient flow in deep nets | not started | |
| 6. Eigenvalues & matrix powers | not started | |
| 7. RNNs & BPTT | not started | |
| 8. Spiking nets & surrogate gradients | not started | |

**Session log:** (date, what was covered, what I struggled with, what to revisit)

---

**First action for the tutor:** run Phase 0 (diagnostic), one question at a time, then propose where I should start.
