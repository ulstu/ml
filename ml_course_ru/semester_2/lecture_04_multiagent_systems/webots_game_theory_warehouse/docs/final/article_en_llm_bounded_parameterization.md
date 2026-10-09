---
title: "Bounded LLM Parameterization of Formal Coordination Models for a Multi-Agent Warehouse Transport System"
author:
  - "K. V. Svyatov"
date: ""
---

**Affiliation.** Ulyanovsk State Technical University, Ulyanovsk, Russia  
**Corresponding author.** K. V. Svyatov

**Abstract**

This paper addresses the problem of connecting natural-language operational rules with formal coordination models in a simulation-based multi-agent warehouse transport system. Eight mobile robots execute delivery requests on a shared graph subject to rules written as text: inspection traffic, loading-zone restrictions, bypass recommendations, and fallback behavior during service delays. Classical task allocation and congestion-based routing require numerical parameters, whereas human-written operating procedures do not. Bridging this gap by manually retuning model parameters for each rule change is labor-intensive and error-prone.

The proposed architecture restricts the role of a local large language model to bounded strategic parameterization. The model produces a validated strategy vector that modifies only a constrained set of weights, admissible robot sets, edge penalties, and dispatch limits in the formal layer; it never generates trajectories or commands actuators. Relevant rule documents are retrieved from a local vector knowledge base, the model output is validated against strict type and range constraints, and the Contract Net Protocol together with a finite congestion game perform actual task allocation and route selection.

Two experimental settings are evaluated against eight baselines on 30 randomized seeds. In the standard scenario with explicit textual rules in the prompt, retrieval-augmented parameterization reduced mean completion time from 829.9 ± 75.4 s to 645.5 ± 83.1 s, while the same model without retrieval achieved nearly identical results (645.7 ± 83.3 s), indicating that the direct description in the prompt was already sufficient. A policy-masked ablation with 15–18 requests and operational codes in place of explicit rules shows when retrieval becomes essential: mean completion time dropped from 1882.9 ± 254.1 s (without retrieval) to 1315.4 ± 159.2 s (with retrieval), p = 1.86 × 10⁻⁹. A static rule that hard-codes the correct response for the stable geometry outperforms the language model layer in both conditions. These results define the practical scope of the approach: language-model parameterization is useful precisely when operating rules are not hard-coded and must be interpreted at runtime.

**Keywords** Large language model; Multi-agent transport system; Formal coordination model; Congestion game; Contract Net Protocol; Retrieval-augmented generation; Vector knowledge base; Safe artificial intelligence

# 1. Introduction

Coordinating a group of mobile robots in a shared warehouse environment requires decisions at several levels simultaneously. Robots must be assigned to requests, must select routes through shared corridors, must avoid collisions locally, and must respect constraints on loading zones and service areas. The difficulty is that each level operates on a different timescale and uses a different representation. A low-level controller tracks wheel velocities; a path planner works with waypoints; a task allocator computes assignments based on priorities and distances. This layered structure is well understood and produces reliable, verifiable behavior.

The complication arises from the operational layer above all of these. Warehouse facilities operate under policies that change with shift patterns, maintenance schedules, and regulatory requirements. These policies are not tables of numbers. They are statements of the form: "the central corridor is occupied by an inspection cart during morning maintenance; robots should prefer the eastern bypass until further notice." No formal model reads such a statement directly. Someone must translate it into a congestion weight, a blocked edge list, or a dispatch gating parameter—and must do so again each time the policy changes.

This paper asks whether a local large language model can serve as that translator, and under what conditions the resulting system remains safe and verifiable. Direct use of a language model as a robot controller is not the answer: the model may hallucinate non-existing objects, generate plans that violate physical constraints, or produce inconsistent sequences of commands. Instead, we propose a more restricted role. The model outputs only a bounded strategic vector whose fields are checked against strict type and range constraints before they enter any formal procedure. The vector parameterizes Contract Net bidding and congestion-game routing; it does not touch wheel velocities, path geometry, or the local collision avoidance controller.

The object of study is a simulation-based warehouse-like multi-agent transport system implemented in Webots. The subject of study is the mechanism that translates natural-language operating rules into parameters of formal task allocation and routing models. The central research question is: can a language model be integrated into a multi-agent robotic system so that it interprets textual rules while preserving the verifiability, reproducibility, and safety of the formal algorithms it parameterizes?

The contributions of the paper are: a formal model separating task allocation, congestion-game routing, and local execution, with explicit accounting of where the strategic vector enters each procedure; a specification of the retrieval-augmented language-model pipeline as a bounded parameterization layer rather than a planner; a Webots implementation with individual robot controllers and a network coordinator; and a comparative evaluation against eight baselines over 30 randomized seeds, including a policy-masked setting where the meaning of operational codes is accessible only through vector retrieval.

# 2. Related Work

**Task allocation.** Multi-robot task allocation has been studied through formal taxonomies that characterize robot capabilities, task types, and optimization criteria [2]. Among the coordination protocols derived from this line of work, the Contract Net Protocol retains wide applicability for transport tasks because it models assignment as a distributed bidding process in which each robot evaluates its own suitability for a request before committing [1]. Market-based extensions generalize this idea to complex tasks with precedence constraints and heterogeneous robots [3]. In warehouse settings, task allocation interacts tightly with routing and loading-zone management: surveys of automated guided vehicle systems and autonomous mobile robots show that dispatching policies, traffic management, and route structure all influence throughput, waiting time, and deadlock risk [4, 5].

**Route coordination.** At the operational level, shortest-path search [9] and multi-agent pathfinding methods address geometric and combinatorial route conflicts [10]. Congestion games offer a mathematically distinct abstraction: routes are player strategies, graph edges are shared resources, and each robot's cost on an edge grows with the number of robots using it [6]. The existence of a potential function for such games guarantees that best-response dynamics converge to a pure Nash equilibrium on finite route sets [7]. Selfish routing in network games has been studied extensively, with results bounding the price of anarchy for polynomial cost functions [8]. These methods are formally clean and computationally tractable for small route sets, but none of them reads a textual operating policy.

**Language models in robotics.** Retrieval-augmented generation enables language models to draw on external documents when constructing a response [11], which is particularly valuable when factual accuracy and traceability matter. Chain-of-thought prompting and agent frameworks show that language models can combine reasoning with tool invocation [12, 13]. In robotics, language models have been applied to grounding high-level instructions in affordances [14], generating robot programs from natural language [15], and linking multimodal observations with planning [16]. Surveys covering this body of work identify both the opportunity and the risk: language models can adapt task-level behavior to contextual instructions, but their outputs are not guaranteed to be physically consistent or safe, and direct use for low-level control remains problematic [17, 18].

**Neuro-symbolic AI and guardrails.** Neuro-symbolic systems seek to combine neural flexibility with symbolic interpretability and formal guarantees [19]. Cognitive automation research examines how cognitive work can be delegated to software components while preserving human oversight and risk control [20]. The architecture proposed here follows this design philosophy: neural interpretation of text remains in the language model, while formal procedures—Contract Net, congestion game, and local controllers—retain full control over physical execution. Guardrail methods that validate model output against schemas, value ranges, and identifier sets [21] make this separation enforceable at runtime.

Table 1 summarizes how the proposed approach relates to these lines of work.

*Table 1. Positioning relative to related work.*

| Area | What it provides | What it does not address | Role here |
|---|---|---|---|
| MRTA / Contract Net | Formal task assignment | Parameter selection from natural language | Tactical allocation |
| Congestion games / MAPF | Route conflict resolution | Operational context interpretation | Route selection |
| LLM robotics | Instruction grounding, planning | Direct control safety, hallucination | Strategic adapter |
| Neuro-symbolic AI | Interpretable symbolic execution | Specific warehouse pipeline | Methodological frame |
| Guardrails | Output validation and schema | MRTA / MAPF algorithm design | Runtime monitor |

# 3. Problem Statement and Formal Model

## 3.1 Warehouse Model

The warehouse is represented as a directed graph $G = (V, E)$, where $V$ is the set of nodes (intersections, pickup zones, drop-off zones, and service points) and $E$ is the set of directed admissible passages. The robot group is $A = \{a_1, \ldots, a_m\}$ and the set of transport requests is $T = \{\tau_1, \ldots, \tau_n\}$. Each robot is described by the tuple $a = (x_a, \beta_a, \sigma_a, \tau_a)$, where $x_a \in V$ is the current position, $\beta_a \in [0,1]$ is the normalized battery level, $\sigma_a$ is the status (idle, moving, loading, unloading), and $\tau_a$ is the currently assigned request (or null). Each request is described by $\tau = (p_\tau, d_\tau, \rho_\tau, \delta_\tau, s_\tau)$, where $p_\tau$ is the pickup node, $d_\tau$ is the drop-off node, $\rho_\tau$ is the reward or priority, $\delta_\tau$ is the deadline, and $s_\tau$ is the request state. For each request $\tau$, the finite set $R_\tau$ contains route candidates pre-generated from the warehouse graph.

*Table 2. Notation used throughout the paper.*

| Symbol | Meaning |
|---|---|
| $G=(V,E)$ | Directed warehouse graph: nodes $V$, edges $E$ |
| $A$, $T$ | Sets of robots and requests |
| $R_\tau$ | Finite set of candidate routes for request $\tau$ |
| $y_{a\tau}$ | Binary assignment variable: 1 if robot $a$ is assigned to request $\tau$ |
| $L(r)$ | Geometric length of route $r$ |
| $n_e(\mathbf{r})$ | Number of selected routes using edge $e$ in route profile $\mathbf{r}$ |
| $E_b$ | Set of edges carrying an additional strategic penalty |
| $A^{go}$, $A^{hold}$ | Subsets of robots permitted or required to wait for new assignments |
| $N^{act}$, $N^{new}$ | Limits on active tasks and new assignments per dispatch cycle |
| $\Theta$ | Strategy vector after guard validation |
| $\Lambda$ | Large penalty constant for strategically discouraged edges in (5) and (6) |
| $M$ | Makespan: completion time of the last request minus scenario start time (8) |

## 3.2 Strategic Vector

The language-model layer, after validation, produces a strategy vector

|  |  |
|---|---:|
| $$\Theta = (w_d, w_l, w_c, w_b, \mu, E_b, A^{go}, A^{hold}, N^{act}, N^{new}).$$ | (1) |

The scalar weights $w_d$, $w_l$, $w_c$, $w_b$ govern deadline urgency, travel length, edge congestion, and battery reserve, respectively. The safety margin $\mu \in (0,1)$ is passed to the local avoidance controller. The set $E_b \subseteq E$ identifies edges to which the routing layer should assign an additional large penalty without removing them from the graph. The sets $A^{go} \subseteq A$ and $A^{hold} \subseteq A$ control which robots may accept new assignments in the current dispatch cycle. The integers $N^{act}$ and $N^{new}$ bound the number of concurrently active tasks and the number of new assignments per cycle.

## 3.3 Task Allocation

The binary variable $y_{a\tau} \in \{0,1\}$ is equal to 1 when robot $a$ is assigned to request $\tau$. The standard single-assignment constraints are

|  |  |
|---|---:|
| $$\sum_{\tau \in T} y_{a\tau} \leq 1, \qquad \sum_{a \in A} y_{a\tau} \leq 1, \qquad y_{a\tau} \in \{0,1\}.$$ | (2) |

The first inequality in (2) prevents a robot from carrying more than one request at a time; the second prevents a request from being assigned to more than one robot.

Physical loading and unloading areas may be shared by several requests. To prevent multiple robots from entering the same zone simultaneously without imposing artificial batches, each request belongs to a family under a mapping $\phi: T \rightarrow \{1, \ldots, g\}$, where $g$ is the number of distinct shared resources. The zone-capacity constraint is

|  |  |
|---|---:|
| $$\sum_{\tau:\phi(\tau)=j} \mathbb{I}(s_\tau \notin \{\mathrm{open}, \mathrm{completed}\}) \leq 1, \qquad j = 1, \ldots, g.$$ | (3) |

Here $\mathbb{I}(\cdot)$ is the indicator function. Constraint (3) permits all requests to exist from the start of the scenario while allowing at most one active request per shared zone, which models the physical release of the area rather than a manually specified release wave.

The bid that robot $a$ submits for request $\tau$ at time $t$ is

|  |  |
|---|---:|
| $$b(a, \tau, t) = \rho_\tau + w_d \frac{\rho_\tau}{\max(\delta_\tau - t,\, \varepsilon)} - w_l\, \mathrm{dist}(x_a, p_\tau) - w_b (1 - \beta_a),$$ | (4) |

where $\varepsilon > 0$ prevents division by zero and $\mathrm{dist}(x_a, p_\tau)$ is the estimated travel distance from the robot's current position $x_a$ to the pickup node $p_\tau$ (distinct from $d_\tau$, the drop-off node). The bid increases with reward and deadline urgency, and decreases with distance to pickup and with low battery reserve. The greedy dispatch cycle sorts all admissible $(a, \tau)$ pairs by bid and selects winners subject to constraints (2), (3), $A^{go}$, $A^{hold}$, $N^{act}$, and $N^{new}$. Sorting $mn$ bids dominates the cycle complexity, giving $O(mn \log(mn))$.

## 3.4 Congestion-Game Routing

Once a robot is assigned a request, it selects one route from the pre-generated finite set $R_\tau$. Let $\mathbf{r} = (r_1, \ldots, r_q)$ be the joint route profile and $n_e(\mathbf{r})$ be the load on edge $e$ under this profile. The individual route cost for robot $i$ is

|  |  |
|---|---:|
| $$c_i(r_i, \mathbf{r}_{-i}) = w_l L(r_i) + w_c \sum_{e \in r_i} n_e(\mathbf{r}) + \Lambda \sum_{e \in r_i} \mathbb{I}(e \in E_b),$$ | (5) |

where $L(r_i)$ is the geometric route length, the second term penalizes shared edge use, and $\Lambda$ is a large constant that discourages strategically undesirable edges without removing them from the graph. The notation $\mathbf{r}_{-i}$ denotes the routes of all robots except robot $i$.

The route selection game admits a potential function

|  |  |
|---|---:|
| $$\Phi(\mathbf{r}) = \sum_i w_l L(r_i) + w_c \sum_{e \in E} \sum_{k=1}^{n_e(\mathbf{r})} k + \Lambda \sum_i \sum_{e \in r_i} \mathbb{I}(e \in E_b).$$ | (6) |

Any unilateral route change by one robot changes $\Phi$ by the same amount as the individual cost (5). This makes the game an exact potential game [7], and best-response dynamics on a finite route set converge to a pure Nash equilibrium. The implementation runs at most 12 best-response rounds, which is sufficient for the small route sets of the Webots scenario.

## 3.5 Retrieval Scoring

The retrieval layer ranks rule documents against a state-dependent textual query $q$ by the hybrid score

|  |  |
|---|---:|
| $$\mathrm{score}(q, d) = 0.68\, \cos(v_q, v_d) + 0.32\, \mathrm{Dice}(\mathrm{tokens}(q), \mathrm{tokens}(d)),$$ | (7) |

where $v_q$ and $v_d$ are normalized PyTorch bag-of-words vectors and $\mathrm{Dice}(\cdot, \cdot)$ is the token-overlap coefficient. The cosine component handles semantic similarity, while the Dice component stabilizes retrieval for short operational codes (such as SOP-C7) that carry little vector signal on their own.

## 3.6 Performance Metric and Safety Property

The primary performance metric is makespan

|  |  |
|---|---:|
| $$M = \max_{\tau \in T}\, t_\tau^{\mathrm{done}} - t_0,$$ | (8) |

where $t_\tau^{\mathrm{done}}$ is the completion time of request $\tau$ and $t_0$ is the scenario start time. Secondary metrics include total waiting time, total distance traveled, number of bypass routes selected, task success rate, and strategic-layer latency.

**Safety property.** The language model cannot assign a non-existing robot, create a new graph edge, generate a waypoint sequence, or interact with any low-level controller. Its influence is limited to the fields of $\Theta$. Guard conditions enforce the following bounds before the vector enters any formal procedure: scalar weights $w_d, w_l, w_c, w_b \in [0.3, 4.0]$; safety margin $\mu \in [0.45, 0.9]$; assignment limits $N^{act}, N^{new}$ are small positive integers. Unknown robot or edge identifiers are silently discarded. On timeout, JSON parse failure, or any guard violation, the system continues with the last valid strategy or a deterministic fallback with default weights. Consequently, physical execution is always produced by formal algorithms acting on validated, bounded data.

Table 3 maps strategy fields to the formal procedures that consume them.

*Table 3. Mapping between strategy fields and formal coordination procedures.*

| Strategy field | Procedure | Effect |
|---|---|---|
| $w_d$ | Bid (4) | Scales urgency of approaching deadlines |
| $w_l$ | Bid (4), route cost (5) | Penalizes distance to pickup and route length |
| $w_c$ | Route cost (5), potential (6) | Penalizes shared edge use |
| $w_b$ | Bid (4) | Reduces bids from robots with low battery |
| $E_b$ | Route cost (5) | Discourages specific edges without removing them |
| $A^{go}$, $A^{hold}$ | Allocation filter | Meters robot release without artificial request waves |
| $N^{act}$, $N^{new}$ | Allocation bounds | Caps concurrent and new assignments per cycle |
| $\mu$ | Local avoidance | Adjusts collision clearance margin |

# 4. Retrieval-Augmented Parameterization Pipeline

The language-model pipeline consists of four components: a local vector knowledge base, a prompt builder, a local language model, and a guard layer.

The knowledge base is a SQLite database with PyTorch vectorization containing 19 short documents (mean length 40.4 tokens, embedding dimension 192). The documents cover six categories of rules: Webots coordinate conventions, collision avoidance and robot kinematics, congestion and bypass routing, Contract Net dispatch gating, standard operating procedure codes, and guard and fallback behavior. The input to the knowledge base is not a free-form query. It is a compact state description assembled at each strategic cycle: number of active robots, number of open requests, route variant identifier, disrupted edge list, deadline and reward jitter, the current textual rule or operational code, and a flag indicating whether central corridor conflicts are expected. This structured query ensures that retrieval responds to the actual warehouse state rather than to a generic prompt.

The top-$k$ documents returned by (7) are inserted into a constrained prompt. The model is instructed to return a single compact JSON object whose keys correspond exactly to the fields of $\Theta$: `deadline_weight`, `travel_weight`, `congestion_weight`, `battery_weight`, `safety_margin`, `traffic_rule`, `blocked_edges`, `go_agents`, `hold_agents`, `max_active_tasks`, `max_new_assignments`, and `note`. No other output format is accepted. The guard layer then parses the response, checks every value against the bounds in Section 3.6, replaces or removes invalid entries, and produces the validated $\Theta$ that enters the formal layer.

In 30 live runs with the `qwen2.5:0.5b` model, every response contained parseable JSON, yet every response also required runtime normalization of at least one field—most frequently `congestion_weight` (set too small) and `traffic_rule` (returned as a numeric string rather than an identifier). The guard layer handled these cases silently, but the frequency of normalization events demonstrates that a small local model cannot be relied upon to produce specification-compliant structured output without validation. Guardrails are therefore not an optional engineering convenience here; they are structurally required for the architecture to be safe.

![Fig. 1. Data flow from operating rules and warehouse state to validated strategy and formal coordination procedures.](/sessions/trusting-brave-ramanujan/mnt/webots_game_theory_warehouse/docs/springer_article/figures/uml/article_data_flow.png){width=5.6in}

# 5. System Implementation

The system is built as a Webots project with a clear separation between the strategic coordinator and the individual robot controllers. The supervisor process manages the strategic and tactical layers: it collects robot status messages, maintains the global request state, invokes the retrieval-augmented parameterization pipeline, runs Contract Net allocation and congestion-game route selection, and dispatches `assignment(task, route)` messages. It does not set wheel velocities or move robots directly.

Each robot runs its own Python controller. On receiving an assignment, the controller follows the designated waypoint route, executes A* replanning if a waypoint is blocked, and reports `pickup`, `done`, and `replan` events to the supervisor. Local collision avoidance uses the safety margin $\mu$ from $\Theta$; all other motion decisions are independent of the strategic layer.

The main scenario contains eight robots and nine requests organized into three repeated pickup/drop-off families. All requests are visible from the start, but constraint (3) and the dispatch limits prevent simultaneous access to the same loading zone. Each request has a central route and one or more bypass routes. The defining operational rule for the scenario states that inspection or maintenance traffic occupies the central crossing; the correct response is to raise the central corridor cost or to meter robot release, which the strategy vector can achieve through $w_c$, $E_b$, or $A^{go}$/$A^{hold}$.

![Fig. 2. Component architecture: the language-model pipeline parameterizes formal coordination models, while individual robot controllers execute local motion.](/sessions/trusting-brave-ramanujan/mnt/webots_game_theory_warehouse/docs/springer_article/figures/uml/article_architecture.png){width=5.6in}

![Fig. 3. Domain model: agent, request, route, strategy, and assignment entities and their relationships.](/sessions/trusting-brave-ramanujan/mnt/webots_game_theory_warehouse/docs/springer_article/figures/uml/article_model.png){width=4.8in}

![Fig. 4. Decision sequence from robot messages to local route execution.](/sessions/trusting-brave-ramanujan/mnt/webots_game_theory_warehouse/docs/springer_article/figures/uml/article_decision_pipeline.png){width=5.2in}

# 6. Experimental Protocol

Statistical experiments were conducted in a headless Python simulator that uses the same Contract Net, congestion-game, route-table, retrieval, and guard modules as the Webots controllers. Webots was used separately to verify scene geometry, visual behavior, and the communication protocol between controllers. This division ensures that algorithmic comparisons are not confounded by simulation artifacts, while the Webots validation confirms that the architecture works end-to-end.

## 6.1 LM-9 Benchmark

The main benchmark randomizes 30 variants of the LM-9 scenario. Varied parameters include random seed, number of robots from $\{7, 8, 9\}$, route variant, disrupted edges, disruption delay, deadline factor, deadline jitter, reward jitter, and start jitter. All compared modes use identical seeds.

The benchmark compares eight modes. *Contract allocation only* disables congestion penalties. *Fixed congestion game* uses a constant congestion weight. *Hand-tuned penalty* applies a manually selected weight. *Prioritized planning* conservatively gates robot release and avoids known problematic edges. *Rule parser* extracts rules by keyword matching without a generative model. *Language model without retrieval* calls the same Ollama model and JSON schema as the full pipeline but omits the retrieval step. *Retrieval-augmented parameterization* (RAG-LLM) adds top-$k$ documents from the vector base to the prompt. *Static domain rule* hard-codes the bypass of the central crossing—an oracle baseline tailored to the known bottleneck geometry.

## 6.2 Policy-Masked Ablation

The policy-masked series is designed specifically to isolate the contribution of retrieval. In this series, the prompt contains only the operational codes SOP-C7 and DOCK-AMBER; neither the central crossing restriction nor the disrupted edge identifiers appear in the prompt. The meaning of these codes is available exclusively through the vector knowledge base. The number of requests is increased to 15–18 and the number of robots varies from 6 to 8, so each robot performs several deliveries on average. This creates a request stream rather than a single illustrative run, and it ensures that route selection and dispatch gating have sustained impact on performance.

Baseline modes in this series are: keyword parser (looks for known tokens in the prompt); language model without retrieval (cannot decode the codes); embedding-rule baseline (uses retrieval output but applies a fixed rule template rather than generative interpretation); retrieval-augmented replay (deterministic re-use of a pre-computed RAG strategy); static domain rule; and a live Ollama pilot on 10 seeds.

## 6.3 Reproducibility

Table 4 lists the key parameters of the experimental setup.

*Table 4. Reproducibility parameters.*

| Parameter | Value |
|---|---|
| Simulator | Webots R2023b for scene validation; headless Python for statistical series |
| Language model | `qwen2.5:0.5b` via Ollama 0.23.2; timeout 35 s |
| Vector knowledge base | SQLite, 19 documents, 192 features, PyTorch hashed bag-of-words |
| Headless time step | 1.0 s; maximum scenario time 1300 s |
| Dispatch period | 5.0 s; language model is a strategic service, not a per-tick controller |
| Seeds | 30 for LM-9 and policy-masked replay; 10 for live policy-masked pilot; 40 for mixed sensitivity |
| Fallback | Deterministic bounded strategy with default weights; no direct motor commands |
| Hardware | macOS arm64, Apple Silicon |

# 7. Results

## 7.1 LM-9 Benchmark

Table 5 reports the main benchmark results. Retrieval-augmented parameterization reduces mean makespan from 829.9 ± 75.4 s (contract allocation without congestion modeling) to 645.5 ± 83.1 s, and outperforms all baselines except hand-tuned penalty and the static domain rule. However, the language model without retrieval achieves 645.7 ± 83.3 s—statistically indistinguishable from the retrieval-augmented version. This is an important negative result for retrieval in this particular setting: the direct textual description of the inspection rule already present in the scenario prompt provides enough information for the model, and the retrieved documents contribute primarily to traceability rather than performance. The static domain rule obtains 559.1 ± 103.8 s and remains the best baseline throughout, because it directly encodes the optimal response for this fixed geometry.

*Table 5. LM-9 benchmark, 30 seeds, mean ± standard deviation.*

| Mode | Makespan, s | CI 95% | Waiting, s | Distance | Latency, s |
|---|---:|---:|---:|---:|---:|
| Contract allocation only | 829.9 ± 75.4 | 27.0 | 913.5 ± 261.9 | 514.9 | 0.00 |
| Fixed congestion game | 829.9 ± 75.4 | 27.0 | 913.5 ± 261.9 | 514.9 | 0.00 |
| Hand-tuned penalty | 684.8 ± 145.3 | 52.0 | 253.0 ± 214.9 | 577.3 | 0.00 |
| Prioritized planning | 686.9 ± 94.4 | 33.8 | 0.0 ± 0.0 | 590.2 | 0.00 |
| Rule parser | 928.7 ± 76.7 | 27.4 | 598.2 ± 189.5 | 663.9 | 0.01 |
| LLM without retrieval | 645.7 ± 83.3 | 29.8 | 125.7 ± 193.4 | 688.9 | 7.83 |
| RAG-LLM | 645.5 ± 83.1 | 29.7 | 124.7 ± 194.4 | 685.4 | 9.21 |
| Static domain rule | 559.1 ± 103.8 | 37.1 | 123.3 ± 193.7 | 570.6 | 0.00 |

Table 6 reports paired sign tests of RAG-LLM against each baseline. The language model outperforms contract allocation only, fixed congestion game, and rule parser with $p = 1.86 \times 10^{-9}$, and outperforms prioritized planning with $p = 0.0125$. The comparison against the language model without retrieval gives $p = 0.210$ (not significant), and the comparison against the static rule gives $p = 5.77 \times 10^{-8}$ in favor of the static rule.

*Table 6. Paired sign test of RAG-LLM against each baseline on makespan (LM-9, n = 30). Positive pairs indicate that the baseline is slower than RAG-LLM.*

| Baseline | Mean diff., s | Positive | Negative | $p$ |
|---|---:|---:|---:|---:|
| Contract allocation only | 184.5 | 30 | 0 | $1.86 \times 10^{-9}$ |
| Fixed congestion game | 184.5 | 30 | 0 | $1.86 \times 10^{-9}$ |
| Hand-tuned penalty | 39.3 | 19 | 11 | 0.200 |
| LLM without retrieval | 0.3 | 8 | 15 | 0.210 |
| Prioritized planning | 41.4 | 21 | 7 | 0.0125 |
| Rule parser | 283.2 | 30 | 0 | $1.86 \times 10^{-9}$ |
| Static domain rule | −86.4 | 1 | 29 | $5.77 \times 10^{-8}$ |

![Fig. 5. Makespan distributions by mode, LM-9 benchmark, 30 seeds.](/sessions/trusting-brave-ramanujan/mnt/webots_game_theory_warehouse/docs/springer_article/experiment_results/main_live/makespan_boxplot.png){width=5.4in}

![Fig. 6. Total waiting time distributions by mode, LM-9 benchmark.](/sessions/trusting-brave-ramanujan/mnt/webots_game_theory_warehouse/docs/springer_article/experiment_results/main_live/wait_boxplot.png){width=5.4in}

## 7.2 Policy-Masked Ablation

Table 7 shows the results of the policy-masked series. When the prompt contains only SOP-C7 and DOCK-AMBER, both the keyword parser and the language model without retrieval fail to redirect traffic: neither activates bypass routes, and both produce makespans above 1880 s. The embedding-rule baseline, which uses retrieval but replaces generative interpretation with a fixed template, reduces makespan to 1517.3 ± 240.0 s and activates an average of 11.1 bypass routes. RAG-LLM replay reduces makespan further to 1315.4 ± 159.2 s with 12.9 bypass routes and perfect task success. The difference between RAG-LLM and the non-retrieval baseline is $p = 1.86 \times 10^{-9}$; the difference between RAG-LLM and the embedding-rule baseline reflects the additional value of generative interpretation. The static domain rule remains competitive at 1255.8 ± 160.3 s.

The live Ollama pilot on 10 seeds yields 1331.2 ± 139.8 s for RAG-LLM versus 1439.7 ± 398.7 s without retrieval—the same direction as the deterministic replay, though the small sample and high variance of the non-retrieval live runs prevent a definitive conclusion from this subset alone.

*Table 7. Policy-masked SOP ablation: prompt contains only operational codes, n = 30 for replay modes, n = 10 for live Ollama.*

| Mode | $n$ | Makespan, s | CI 95% | Bypass | Success |
|---|---:|---:|---:|---:|---:|
| Keyword parser | 30 | 1932.3 ± 237.5 | 85.0 | 0.0 | 0.833 |
| LLM without retrieval, replay | 30 | 1882.9 ± 254.1 | 90.9 | 0.0 | 0.867 |
| Embedding-rule baseline | 30 | 1517.3 ± 240.0 | 85.9 | 11.1 | 0.967 |
| RAG-LLM replay | 30 | 1315.4 ± 159.2 | 57.0 | 12.9 | 1.000 |
| Static domain rule | 30 | 1255.8 ± 160.3 | 57.4 | 11.1 | 1.000 |
| LLM without retrieval, live | 10 | 1439.7 ± 398.7 | 247.1 | 8.4 | 0.900 |
| RAG-LLM live | 10 | 1331.2 ± 139.8 | 86.7 | 10.8 | 1.000 |

![Fig. 7. Scene dynamics: initial configuration, execution of the first requests, and redistribution to bypass routes.](/sessions/trusting-brave-ramanujan/mnt/webots_game_theory_warehouse/docs/springer_article/figures/experiment_snapshots_montage.png){width=5.5in}

# 8. Retrieval Quality, Error Robustness, and Computational Cost

## 8.1 Retrieval Quality

Retrieval quality was measured on 13 representative state queries covering maintenance events, paraphrases, normal operation, dispatch rules, priority constraints, guard behavior, high-latency states, and masked SOP cases. At top-$k = 4$, precision was 0.423, recall was 0.591, and top-1 accuracy was 0.923. These numbers reflect a knowledge base of only 19 documents with short text; in practice, the high top-1 accuracy is the operationally relevant figure, because the most critical rule document is almost always retrieved first.

*Table 8. Retrieval metrics under artificial document corruption.*

| Corruption probability | Precision@4 | Recall@4 | Top-1 accuracy |
|---:|---:|---:|---:|
| 0.00 | 0.423 | 0.591 | 0.923 |
| 0.25 | 0.346 | 0.456 | 0.692 |
| 0.50 | 0.231 | 0.358 | 0.538 |

## 8.2 Error Robustness

To assess resilience, a separate ablation replaced each retrieved document with a random one from the base at probability 0.25 or 0.50. In the LM-9 scenario with 50% corruption, RAG-LLM makespan degraded from 645.5 ± 83.1 s to 751.2 ± 177.4 s and total waiting time increased from 124.7 to 441.6 s. In the policy-masked scenario, degradation was more severe: 1315.4 ± 159.2 s rose to 1588.0 ± 365.2 s and success rate fell from 1.000 to 0.967. The system did not fail outright because guard rules and fallback policies constrain the output regardless of retrieved content, but the policy-masked scenario makes clear that retrieval quality is a causal factor when operational codes cannot be decoded from the prompt directly.

*Table 9. Performance under 50% retrieval corruption, n = 30.*

| Mode | Makespan, s | CI 95% | Waiting, s | Success |
|---|---:|---:|---:|---:|
| LM-9 RAG-LLM, corrupted | 751.2 ± 177.4 | 63.5 | 441.6 | 1.000 |
| LM-9 rule parser, corrupted | 929.8 ± 76.5 | 27.4 | 635.6 | 1.000 |
| Masked RAG-LLM, corrupted | 1588.0 ± 365.2 | 130.7 | 733.3 | 0.967 |
| Masked embedding rule, corrupted | 1611.3 ± 297.9 | 106.6 | 576.1 | 0.933 |

## 8.3 Computational Cost

Live language-model latency averaged 9.21 ± 2.42 s with retrieval and 7.83 ± 2.08 s without retrieval in the LM-9 benchmark. This makes the model unsuitable as a per-tick controller in any reasonable control loop. At a strategic dispatch period of 30 s, the latency represents roughly a 31% overhead; at 60 s, roughly 15%. During a model call, robots continue executing their current routes under the last validated strategy. This asynchronous usage pattern is not a limitation to be overcome—it is the intended deployment mode for a component that handles slow-moving policy changes rather than fast-moving physical events.

# 9. Sensitivity and Scalability

A *mixed sensitivity* series varied scenario difficulty more broadly than the LM-9 benchmark: robot count was drawn from $\{4, 6, 8, 10\}$, requests ranged from 3 to 12, route variant and disruption presence were randomized, and scene scale was varied. This series used deterministic RAG replay to isolate the effect of environmental parameters from the stochasticity of live LLM responses. Results are reported over 40 seeds.

*Table 10. Mixed sensitivity benchmark, n = 40, mean ± standard deviation.*

| Mode | Makespan, s | CI 95% | Waiting, s | Success |
|---|---:|---:|---:|---:|
| Contract allocation only | 507.8 ± 320.2 | 99.2 | 289.9 | 0.975 |
| Hand-tuned penalty | 473.4 ± 294.2 | 91.2 | 80.2 | 1.000 |
| Prioritized planning | 563.7 ± 376.1 | 116.6 | 2.5 | 0.975 |
| RAG replay | 466.4 ± 291.7 | 90.4 | 1.9 | 1.000 |
| Static domain rule | 407.9 ± 254.4 | 78.8 | 4.2 | 1.000 |

At low task loads (3–4 requests), performance differences among all modes lie within the noise: standard deviations are comparable to means, and language-model latency of roughly 9 s constitutes a significant fraction of total execution time for very short runs. The picture changes when bypass routes exist and the central crossing carries real congestion: RAG replay then achieves the second-lowest makespan (466.4 ± 291.7 s) and near-zero waiting time, approaching the static domain rule (407.9 ± 254.4 s). The static rule retains its lead because it directly encodes the bottleneck structure that is common across most randomized variants. The practical implication is that the boundary of usefulness is not a simple function of robot count or scenario size; it depends on whether the operational rules change frequently enough to justify the overhead of language-model parameterization.

# 10. Threats to Validity

**Internal validity.** The headless simulator accelerates the strategic cycle relative to a real-time Webots run and represents robot motion at route level rather than at the level of wheel dynamics. Geometric consistency and controller interaction are verified in the full Webots scene, but the statistical series does not re-run in the full simulator. To limit the risk of confounds, all baselines share the same Contract Net, congestion-game, route-table, and task-generation code; the only difference across conditions is the source of $\Theta$.

**External validity.** The warehouse-like scene abstracts away several features of industrial intralogistics: robots do not perform physical pick-and-place operations, battery charging is not modeled, robot failures do not occur, and new requests do not arrive during a run. The results should therefore be read as a test of the architectural hypothesis—that a bounded language-model layer can safely influence a formal coordination pipeline—rather than as a validation of an industrial AMR fleet management system.

**Construct validity.** Makespan and total waiting time are proxy metrics for industrial throughput and quality of service. They are reasonable for comparing coordination policies under controlled conditions, but they do not capture all dimensions of intralogistics KPIs such as pick rate, energy consumption, or SLA adherence.

**Statistical validity.** The sample size of 30 seeds with a paired sign test is appropriate for detecting medium-to-large effects, but borderline cases (such as the comparison against hand-tuned penalty, $p = 0.200$) would require more seeds to resolve. For publication at journal level, the benchmark should be extended to at least 100 randomized seeds, should include stronger baselines such as CBS/MAPF and MILP/VRP, and should use a more detailed robot model.

# 11. Discussion

The central question driving this work was whether a language model can be integrated into a multi-agent robotic system in a way that respects both the flexibility of natural-language interpretation and the safety of formal execution. The results give a qualified positive answer, with several important caveats.

The clearest finding is negative in tone but positive in value: the static domain rule outperforms the language model layer in both the standard and policy-masked conditions. In the LM-9 scenario, this happens because the operational rule is already present verbatim in the prompt, making retrieval redundant and making a handcrafted bottleneck bypass cheaper and faster than language-model inference. This result should not be read as a failure of the architecture. It defines the correct scope: a language model should not be used as an expensive substitute for a known, stable deterministic rule.

The complementary finding is where the architecture is genuinely useful. In the policy-masked condition, where the prompt contains only operational codes whose meaning requires lookup, the language model with retrieval reduces makespan by 567 s relative to the non-retrieval baseline—a 30% reduction. The embedding-rule baseline, which uses retrieval but not generative interpretation, achieves a smaller but still substantial reduction of 365 s. The difference between these two modes indicates that the language model adds value beyond mechanical document matching: it translates retrieved text into a balanced combination of dispatch gating and route penalties suited to the observed warehouse state.

The guardrail layer deserves specific comment. In 30 live runs, every model response required normalization of at least one field. This means that a language model of the size used here cannot be relied upon to produce specification-compliant output without runtime validation, even for a task as constrained as producing a 12-field JSON object. The guardrails are not an incidental engineering choice; they are a structural component that makes the architecture safe.

The industrial realism of the current evaluation remains limited. Pickup and drop-off operations are instant, charging is not modeled, and the route graph abstracts over the continuous dynamics of the real robots. Extending the architecture to a scenario with physical manipulators, charging stations, robot failures, and dynamic order arrivals would require revisiting the zone-capacity constraint, the fallback strategy, and the strategic period, but the formal model structure would not change fundamentally.

# 12. Conclusion

The architecture presented here treats a local large language model as a bounded strategic parameterizer for formal coordination models rather than as a controller. The model suggests values in a pre-defined strategy vector $\Theta$; the Contract Net Protocol, a finite congestion game, and individual robot controllers perform all physical execution.

The key experimental finding concerns when retrieval-augmented parameterization is useful. In a standard scenario where the operational rule appears in plain text in the prompt, retrieval adds little to makespan performance, though it improves traceability and robustness to prompt changes. In a policy-masked scenario where the prompt contains only operational codes whose meaning is encoded in the knowledge base, retrieval reduces mean completion time by 30% relative to the same language model without retrieval (1882.9 → 1315.4 s, $p = 1.86 \times 10^{-9}$). The static domain rule hard-coding the correct bottleneck response remains the fastest baseline in both conditions. These three observations together define the practical decision criterion: use a static rule when the bottleneck is stable and known; use language-model parameterization when operational rules are subject to change or expressed in a form that must be interpreted at runtime.

Future directions include extending the benchmark to more than 100 randomized seeds and to scenarios with dynamic order arrival; comparing against CBS/MAPF and MILP/VRP baselines on small and medium instances; replacing the hashed bag-of-words retrieval with a learned local embedding and reranking stage; and transferring the architecture to a physically more detailed robot model with charging, failures, and manipulation operations.

# References

1\. Smith RG (1980) The Contract Net Protocol: high-level communication and control in a distributed problem solver. IEEE Transactions on Computers C-29(12):1104–1113. doi:10.1109/TC.1980.1675516

2\. Gerkey BP, Matarić MJ (2004) A formal analysis and taxonomy of task allocation in multi-robot systems. International Journal of Robotics Research 23(9):939–954. doi:10.1177/0278364904045564

3\. Dias MB, Zlot R, Kalra N, Stentz A (2006) Market-based multirobot coordination: a survey and analysis. Proceedings of the IEEE 94(7):1257–1270. doi:10.1109/JPROC.2006.876939

4\. Le-Anh T, De Koster MBM (2006) A review of design and control of automated guided vehicle systems. European Journal of Operational Research 171(1):1–23. doi:10.1016/j.ejor.2005.01.036

5\. Fragapane G, de Koster R, Sgarbossa F, Strandhagen JO (2021) Planning and control of autonomous mobile robots for intralogistics: literature review and research agenda. European Journal of Operational Research 294(2):405–426. doi:10.1016/j.ejor.2021.01.019

6\. Rosenthal RW (1973) A class of games possessing pure-strategy Nash equilibria. International Journal of Game Theory 2:65–67. doi:10.1007/BF01737559

7\. Monderer D, Shapley LS (1996) Potential games. Games and Economic Behavior 14(1):124–143. doi:10.1006/game.1996.0044

8\. Roughgarden T, Tardos É (2002) How bad is selfish routing? Journal of the ACM 49(2):236–259. doi:10.1145/506147.506153

9\. Hart PE, Nilsson NJ, Raphael B (1968) A formal basis for the heuristic determination of minimum cost paths. IEEE Transactions on Systems Science and Cybernetics 4(2):100–107. doi:10.1109/TSSC.1968.300136

10\. Sharon G, Stern R, Felner A, Sturtevant NR (2015) Conflict-based search for optimal multi-agent pathfinding. Artificial Intelligence 219:40–66. doi:10.1016/j.artint.2014.11.006

11\. Lewis P, Perez E, Piktus A et al (2020) Retrieval-augmented generation for knowledge-intensive NLP tasks. Advances in Neural Information Processing Systems 33:9459–9474

12\. Wei J, Wang X, Schuurmans D et al (2022) Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems 35:24824–24837

13\. Yao S, Zhao J, Yu D et al (2023) ReAct: synergizing reasoning and acting in language models. International Conference on Learning Representations

14\. Ahn M, Brohan A, Brown N et al (2022) Do as I can, not as I say: grounding language in robotic affordances. arXiv:2204.01691. doi:10.48550/arXiv.2204.01691

15\. Liang J, Huang W, Xia F et al (2022) Code as policies: language model programs for embodied control. arXiv:2209.07753. doi:10.48550/arXiv.2209.07753

16\. Driess D, Xia F, Sajjadi MSM et al (2023) PaLM-E: an embodied multimodal language model. arXiv:2303.03378. doi:10.48550/arXiv.2303.03378

17\. Wang J, Wu Z, Li Y et al (2024) Large language models for robotics: opportunities, challenges, and perspectives. arXiv:2401.04334. doi:10.48550/arXiv.2401.04334

18\. Li P, An Z, Abrar S, Zhou L (2025) Large language models for multi-robot systems: a survey. arXiv:2502.03814. doi:10.48550/arXiv.2502.03814

19\. Sarker MK, Zhou L, Eberhart A, Hitzler P (2021) Neuro-symbolic artificial intelligence: current trends. AI Communications 34(3):197–209. doi:10.3233/AIC-210084

20\. Engel C, Elshan E, Ebel P, Leimeister JM (2024) Stairway to heaven or highway to hell: a model for assessing cognitive automation use cases. Journal of Information Technology 39(1):53–78. doi:10.1177/02683962231185599

21\. Dong Y, Mu R, Jin G et al (2024) Building guardrails for large language models. Proceedings of Machine Learning Research 235:11765–11787

22\. Michel O (2004) Cyberbotics Ltd. Webots: professional mobile robot simulation. International Journal of Advanced Robotic Systems 1(1):39–42. doi:10.5772/5618
