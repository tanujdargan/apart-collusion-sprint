# Cut the Wire: related work and citation base

Compiled 2026-10-25. Starting point: `external/collusion_hackathon/literature/papers.csv`, plus web search.

## How the citations were checked

- **[A]**: title, authors, year, and arXiv id checked against the arXiv API (export.arxiv.org) on 2026-10-25. Venues given in the arXiv `journal_ref` or `comment` field are marked [A].
- **[W]**: venue or metadata confirmed through a web search hit on the official proceedings or publisher page (PMLR, NeurIPS proceedings, ABA).
- **[K]**: the arXiv record was checked [A], but the venue comes from my own knowledge and was *not* machine-checked, because DBLP and Semantic Scholar returned HTTP 429. Check these before final submission, or cite the arXiv version only.
- **[U]**: could not be fully checked. Do not rely on these without checking.

### Corrections to keep in mind

1. **Rose et al. 2026 has changed across versions.** v1 (2026-04-01, 4 authors) states in the abstract: *"preliminary evidence that this signal is localised at the token level, with the colluding agent's activations spiking specifically when processing the encoded parts of their partner's message."* This is the correlational claim we build on. v5 (2026-10-15, 6 authors) adds Sahar Abdelnabi and Philip Torr. It reframes the paper around HELIOS (composite-hypothesis-testing detection statistics) and reports 0.99–1.00 AUROC transfer to NARCBench-Stego. I did not find the token-spike sentence in the v5 HTML. **If you quote the spike claim, cite v1 explicitly** (`arXiv:2604.01151v1`). v5 also defines a "causality" criterion for collusion and says it usually cannot be evaluated without counterfactuals. Our interventions supply exactly that kind of counterfactual.
2. **Das et al. 2026 is our closest neighbour** (arXiv 2609.03035). It uses NARCBench's four-agent blackjack game and tests whether collusion probes survive when agents are told they are being monitored. The probes survive. That is a *behavioural* evasion test. Ours is a *mechanistic* intervention test, and the two complement each other.
3. **The Berger et al. title in the sprint CSV is wrong.** The correct title is "Coordination **by** Code: Rethinking Antitrust Liability in the Age of AI Pricing", by Lee F. Berger, Michael L. Weiner, and Weisiyu Jiang, *Antitrust* (ABA) 40(1), Fall 2025 [W]. The ABA page returned 403, so I could not confirm that it discusses "plus factors". For plus factors, cite Kovacic et al. 2011 as the canonical source.
4. **Goldowsky-Dill et al.** The ICML 2025 proceedings title is "Detecting Strategic Deception **with** Linear Probes" (PMLR 267:19755–19786) [W]. The arXiv title uses "Using".
5. **Motwani et al.** The NeurIPS 2024 title is "Secret Collusion among AI Agents: Multi-Agent Deception via Steganography" [W]. Earlier arXiv versions say "Generative AI Agents".
6. **CAA (Rimsky et al.)**: the first author now publishes as Nina Panickssery, and that is the name on the arXiv record. The ACL 2024 version is credited as Rimsky et al. [K].
7. **Novelty check.** On 2026-10-25 a web search for ablation or steering of collusion or steganography probe directions found no published causal test. The closest item is a SPAR Fall-2026 *proposal* to find a "steganography axis" and test it with steering and ablation (https://sparai.org/projects/f26/reclf40AjRM2sPF9X/). It has no results. Phrase the novelty claim as *"to our knowledge, the first causal test"*.

---

## Group 1: Multi-agent collusion detection and the sprint's base

| Key | Citation | URL | Relevance to Cut the Wire |
|---|---|---|---|
| `rose2026narcbench` | Aaron Rose, Carissa Cullen, Sahar Abdelnabi, Philip Torr, Brandon Gary Kaplowitz, Christian Schroeder de Witt. *Detecting Multi-Agent Collusion Through Multi-Agent Interpretability.* arXiv:2604.01151, 2026 (v1 Apr 2026; v5 Oct 2026). [A] | https://arxiv.org/abs/2604.01151 | Our base. It introduces NARCBench (including the NARCBench-Stego blackjack tier) and shows that linear probes detect collusion. v1 reports that the receiver's activations spike on encoded tokens, which is the correlational finding we test causally. |
| `das2026activations` | Aritra Das, Jaee Ponde, Mihir More, Debayan Gupta. *You Can't Escape Your Own Activations: Evaluation Awareness and Multi-Agent Monitoring.* arXiv:2609.03035, 2026. [A] | https://arxiv.org/abs/2609.03035 | Closest prior work. In NARCBench-style blackjack, collusion probes stay accurate when agents know they are monitored. This is behavioural robustness; we ask whether the probed direction is mechanistically necessary. |
| `motwani2024secret` | Sumeet Ramesh Motwani, Mikhail Baranchuk, Martin Strohmeier, Vijay Bolina, Philip H. S. Torr, Lewis Hammond, Christian Schroeder de Witt. *Secret Collusion among AI Agents: Multi-Agent Deception via Steganography.* NeurIPS 2024. arXiv:2402.07510. [A][W] | https://arxiv.org/abs/2402.07510 | Defines secret (steganographic) collusion and argues that transcript-level oversight is fundamentally limited. This motivates white-box detectors whose validity we stress-test. |
| `mathew2024hidden` | Yohan Mathew, Ollie Matthews, Robert McCarthy, Joan Velja, Christian Schroeder de Witt, Dylan Cope, Nandi Schoots. *Hidden in Plain Text: Emergence & Mitigation of Steganographic Collusion in LLMs.* IJCNLP-AACL 2025 (oral). arXiv:2410.03768. [A] | https://arxiv.org/abs/2410.03768 | Shows that stego collusion can emerge under optimisation pressure and that paraphrasing and passive oversight are not robust. This is why activation-level monitors matter, and why it matters whether they track the mechanism. |
| `tailor2025audit` | Om Tailor. *Audit the Whisper: Detecting Steganographic Collusion in Multi-Agent LLMs.* arXiv:2510.04303, 2025 (single author, not peer reviewed). [A] | https://arxiv.org/abs/2510.04303 | A black-box, mutual-information and channel-capacity auditing alternative (ColludeBench-v0). It contrasts with the white-box probe signal we intervene on. |
| `fish2024algorithmic` | Sara Fish, Yannai A. Gonczarowski, Ran I. Shorrer. *Algorithmic Collusion by Large Language Models.* arXiv:2404.00806, 2024 (accepted to EC 2026 per arXiv comment). [A] | https://arxiv.org/abs/2404.00806 | Shows that LLM pricing agents reach supra-competitive outcomes without instruction. This is economic motivation for detecting LLM collusion at all. |
| `zolkowski2025early` | Artur Zolkowski, Kei Nishimura-Gasparian, Robert McCarthy, Roland S. Zimmermann, David Lindner. *Early Signs of Steganographic Capabilities in Frontier LLMs.* arXiv:2507.02737, 2025. [A] | https://arxiv.org/abs/2507.02737 | Frontier models evade monitors mainly when given pre-agreed schemes. That is our setup, where a pre-agreed topic scheme is shown in context. |
| `anwar2026decision` | Usman Anwar, Julianna Piskorz, David D. Baek, David Africa, Jim Weatherall, Max Tegmark, Christian Schroeder de Witt, Mihaela van der Schaar, David Krueger. *A Decision-Theoretic Formalisation of Steganography With Applications to LLM Monitoring.* arXiv:2602.23163, 2026. [A] | https://arxiv.org/abs/2602.23163 | Defines steganography as a usable-information asymmetry between decoders and non-decoders. Our C1-vs-C3 (colluding vs honest receiver) contrast is exactly this asymmetry. (Optional.) |

**Draft Related Work paragraph (Group 1).** Covert coordination between LLM agents is now an established threat model. Motwani et al. (2024) formalise *secret collusion* via steganography and argue that transcript-level oversight has fundamental limits. Mathew et al. (2024) show such channels can emerge under optimisation pressure and survive paraphrasing, and Zolkowski et al. (2025) find frontier models evade monitors mainly when given pre-agreed encoding schemes. LLM collusion also matters economically: Fish et al. (2024) observe uninstructed supra-competitive pricing. Black-box auditors (Tailor, 2025) bound covert channel capacity, while Rose et al. (2026) move monitoring into activation space. Their NARCBench shows that linear probes detect collusion, including in a steganographic blackjack card-counting tier, and their first version reports that a colluding receiver's activations spike on the encoded parts of its partner's message. Das et al. (2026) show these probes stay accurate when agents know they are watched. All of this evidence is correlational. We ask the complementary *causal* question: is the direction these probes read load-bearing for the collusive behaviour?

---

## Group 2: Linear probes and representation reading for deception and lying

| Key | Citation | URL | Relevance |
|---|---|---|---|
| `goldowskydill2025deception` | Nicholas Goldowsky-Dill, Bilal Chughtai, Stefan Heimersheim, Marius Hobbhahn. *Detecting Strategic Deception with Linear Probes.* ICML 2025, PMLR 267:19755–19786. arXiv:2502.03407 (as "...Using Linear Probes"). [A][W] | https://arxiv.org/abs/2502.03407 | The single-agent deception-probe recipe (logistic regression on residual activations, mean-pooled) that NARCBench's collusion probe follows. Our d_detect is trained the same way. |
| `marks2024geometry` | Samuel Marks, Max Tegmark. *The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets.* COLM 2024. arXiv:2310.06824. [A] | https://arxiv.org/abs/2310.06824 | Shows truth is linearly represented and, importantly, *causally* implicated, via diff-of-means directions and patching. This is the template for our d_state positive control. |
| `zou2023repe` | Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, J. Zico Kolter, Dan Hendrycks. *Representation Engineering: A Top-Down Approach to AI Transparency.* arXiv:2310.01405, 2023. [A] | https://arxiv.org/abs/2310.01405 | Pairs reading (detecting honesty and lying directions) with control (steering them). This is the read-then-intervene paradigm we apply to collusion. |
| `azaria2023internal` | Amos Azaria, Tom Mitchell. *The Internal State of an LLM Knows When It's Lying.* Findings of EMNLP 2023. arXiv:2304.13734. [A][K venue] | https://arxiv.org/abs/2304.13734 | An early demonstration that a classifier on hidden states detects falsehood. It is an ancestor of activation-based deception monitors. |
| `burns2023discovering` | Collin Burns, Haotian Ye, Dan Klein, Jacob Steinhardt. *Discovering Latent Knowledge in Language Models Without Supervision.* ICLR 2023. arXiv:2212.03827. [A] | https://arxiv.org/abs/2212.03827 | Contrast-Consistent Search: unsupervised probing for truth directions. It is part of the representation-reading lineage. |
| `golechha2025amongus` | Satvik Golechha, Adrià Garriga-Alonso. *Among Us: A Sandbox for Measuring and Detecting Agentic Deception.* NeurIPS 2025 (spotlight). arXiv:2504.04072. [A] | https://arxiv.org/abs/2504.04072 | Deception probes transfer to a multi-agent social game. This is a precedent for probe-based detection in game environments like ours. (Optional.) |

**Draft Related Work paragraph (Group 2).** Collusion probes descend from work on reading truthfulness and deception out of activations. Azaria and Mitchell (2023) show that a simple classifier on hidden states detects when a model outputs falsehoods. Burns et al. (2023) recover truth directions without labels, and Marks and Tegmark (2024) show that true/false structure is linear and causally implicated in model outputs. Zou et al. (2023) introduce representation engineering, which pairs reading honesty-related directions with steering them. Goldowsky-Dill et al. (2025) train logistic-regression probes on residual activations that detect strategic deception (insider trading, sandbagging) at high AUROC, and Golechha and Garriga-Alonso (2025) carry such probes into a multi-agent social-deduction game. NARCBench's collusion probes follow this recipe directly. Most of this literature evaluates probes by held-out classification accuracy. Marks and Tegmark are a partial exception, but no comparable causal check exists for deception or collusion probes in multi-agent settings. We supply one: we take the direction a collusion probe learns and test whether removing it changes collusive behaviour.

---

## Group 3: Causal interventions on directions

| Key | Citation | URL | Relevance |
|---|---|---|---|
| `arditi2024refusal` | Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, Neel Nanda. *Refusal in Language Models Is Mediated by a Single Direction.* NeurIPS 2024. arXiv:2406.11717. [A][K venue] | https://arxiv.org/abs/2406.11717 | Our main intervention: directional ablation (project out d at every layer and position) plus activation addition. We reuse their protocol and their "necessity and sufficiency" framing. Their paper also covers Qwen chat models. |
| `turner2023actadd` | Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, Monte MacDiarmid. *Steering Language Models With Activation Engineering.* arXiv:2308.10248, 2023. [A] | https://arxiv.org/abs/2308.10248 | Activation addition (ActAdd): adding a direction at inference changes behaviour. This is the basis for our −α·d dose-response steering. |
| `rimsky2024caa` | Nina Rimsky (Panickssery), Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, Alexander Matt Turner. *Steering Llama 2 via Contrastive Activation Addition.* ACL 2024. arXiv:2312.06681. [A][K venue] | https://arxiv.org/abs/2312.06681 | Diff-of-means steering vectors from contrastive pairs, the same construction as our d_state and d_detect diff-of-means variants. |
| `belrose2023leace` | Nora Belrose, David Schneider-Joseph, Shauli Ravfogel, Ryan Cotterell, Edward Raff, Stella Biderman. *LEACE: Perfect Linear Concept Erasure in Closed Form.* NeurIPS 2023. arXiv:2306.03819. [A][K venue] | https://arxiv.org/abs/2306.03819 | The principled way to make a concept *linearly unrecoverable*. It is a stronger alternative to single-direction ablation, and if collusion survives single-direction ablation the follow-up is to check whether it survives LEACE. |
| `ravfogel2020inlp` | Shauli Ravfogel, Yanai Elazar, Hila Gonen, Michael Twiton, Yoav Goldberg. *Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection.* ACL 2020. arXiv:2004.07667. [A] | https://arxiv.org/abs/2004.07667 | INLP, iterative removal of probe directions. It addresses the "signal is spread over many directions" objection to single-direction ablation. |
| `li2023iti` | Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, Martin Wattenberg. *Inference-Time Intervention: Eliciting Truthful Answers from a Language Model.* NeurIPS 2023 (spotlight). arXiv:2306.03341. [A] | https://arxiv.org/abs/2306.03341 | Shows that probe-identified truth directions can be causally manipulated. Rose et al. cite it as the causal counterpart to deception probing that collusion work had lacked. (Optional.) |

**Draft Related Work paragraph (Group 3).** Several lines of work turn probe directions into interventions. Turner et al. (2023) show that adding an activation direction at inference steers behaviour, and Rimsky et al. (2024) build steering vectors as differences of means over contrastive pairs. Li et al. (2023) shift attention-head activations along probe-identified truth directions to change truthfulness. Arditi et al. (2024) show that refusal in chat models, including Qwen, is mediated by a single direction: projecting it out of the residual stream at every layer and position removes refusal, and adding it induces refusal. Their directional ablation is our primary tool. Concept-erasure methods give stronger guarantees. INLP (Ravfogel et al., 2020) iteratively nulls probe directions, and LEACE (Belrose et al., 2023) makes a concept linearly unrecoverable in closed form. These tools have been applied to refusal, truthfulness, sentiment and protected attributes, but not, to our knowledge, to multi-agent collusion. We apply directional ablation and steering to collusion-probe directions in a colluding receiver. A content direction (hidden-state HOT vs COLD) serves as a positive control and random directions as a negative control.

---

## Group 4: Probe validity, correlational vs causal critiques, and evasion

| Key | Citation | URL | Relevance |
|---|---|---|---|
| `belinkov2022probing` | Yonatan Belinkov. *Probing Classifiers: Promises, Shortcomings, and Advances.* Computational Linguistics 48(1), 2022. arXiv:2102.12452. [A][K vol/issue] | https://arxiv.org/abs/2102.12452 | The canonical critique: probe accuracy shows information is *present*, not that the model *uses* it. This is the gap our experiment targets. |
| `elazar2021amnesic` | Yanai Elazar, Shauli Ravfogel, Alon Jacovi, Yoav Goldberg. *Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals.* TACL 9, 2021. arXiv:2006.00995. [A] | https://arxiv.org/abs/2006.00995 | Our core logic: remove the probed property (via INLP) and measure the behavioural change. We carry this "amnesic" test over to multi-agent collusion. |
| `ravichander2021probing` | Abhilasha Ravichander, Yonatan Belinkov, Eduard Hovy. *Probing the Probing Paradigm: Does Probing Accuracy Entail Task Relevance?* EACL 2021. arXiv:2005.00719. [A] | https://arxiv.org/abs/2005.00719 | Shows that probes recover properties the task does not need. A high-AUROC collusion probe could likewise be reading an epiphenomenal correlate, which is the "detector blind, collusion intact" cell of our 2x2. |
| `hewitt2019control` | John Hewitt, Percy Liang. *Designing and Interpreting Probes with Control Tasks.* EMNLP 2019. arXiv:1909.03368. [A] | https://arxiv.org/abs/1909.03368 | Control tasks and selectivity. This motivates our confound-removed d_detect (C1 vs C2), set against the prompt-confounded d_detect_narc. |
| `kumar2022probing` | Abhinav Kumar, Chenhao Tan, Amit Sharma. *Probing Classifiers are Unreliable for Concept Removal and Detection.* NeurIPS 2022 (vol. 35, pp. 17994–18008). arXiv:2207.04153. [A] | https://arxiv.org/abs/2207.04153 | Probe-based removal can leave the concept intact or remove spurious features. This caveat matters for reading a "collusion survives" result. |
| `makelov2023subspace` | Aleksandar Makelov, Georg Lange, Neel Nanda. *Is This the Subspace You Are Looking For? An Interpretability Illusion for Subspace Activation Patching.* NeurIPS 2023 ATTRIB workshop; arXiv:2311.17030. [A] | https://arxiv.org/abs/2311.17030 | Interventions along a subspace can change behaviour through dormant pathways. This is why we use random-direction controls and utility checks. (Optional.) |
| `bailey2024obfuscated` | Luke Bailey, Alex Serrano, Abhay Sheshadri, Mikhail Seleznyov, Jordan Taylor, Erik Jenner, Jacob Hilton, Stephen Casper, Carlos Guestrin, Scott Emmons. *Obfuscated Activations Bypass LLM Latent-Space Defenses.* arXiv:2412.09565, 2024. [A; venue not confirmed] | https://arxiv.org/abs/2412.09565 | Probe recall can be driven from 100% to 0% while the harmful behaviour is kept. This is direct evidence that probe signals can be separated from behaviour, and an "epiphenomenal" result in our setup would predict the same evadability. |
| `gupta2025rlobfuscation` | Rohan Gupta, Erik Jenner. *RL-Obfuscation: Can Language Models Learn to Evade Latent-Space Monitors?* arXiv:2506.14261, 2025. [A] | https://arxiv.org/abs/2506.14261 | RL pressure alone can teach evasion of latent monitors. This makes it a live question whether a collusion probe's direction is necessary. |
| `mcguinness2025chameleons` | Max McGuinness, Alex Serrano, Luke Bailey, Scott Emmons. *Neural Chameleons: Language Models Can Learn to Hide Their Thoughts from Unseen Activation Monitors.* arXiv:2512.11949, 2025. [A; venue: web hits list both ICLR 2026 and NeurIPS 2026, unconfirmed, so cite arXiv] | https://arxiv.org/abs/2512.11949 | Models learn to evade even unseen probes (including Qwen) by moving activations into a low-dimensional subspace, which suggests probed directions need not be load-bearing. |

**Draft Related Work paragraph (Group 4).** High probe accuracy does not show that a model uses the probed information. Belinkov (2022) reviews this gap, Hewitt and Liang (2019) introduce control tasks to separate probe expressivity from representation content, and Ravichander et al. (2021) show probes recover properties the task does not require. Elazar et al. (2021) propose *amnesic probing*: remove the property with INLP and measure whether behaviour changes. Kumar et al. (2022) and Makelov et al. (2023) warn that removal and subspace interventions can mislead in both directions. Recent safety work shows this gap can be exploited. Bailey et al. (2024) reduce probe recall from 100% to 0% while harmful behaviour persists, and Gupta and Jenner (2025) and McGuinness et al. (2025) show that models can learn to evade latent-space monitors, including unseen ones. If a collusion probe reads an epiphenomenal correlate, colluders could in principle evade it at little cost. We apply amnesic-probing logic to multi-agent collusion. Ablating the probe direction and crossing "detector blind?" with "collusion intact?" separates a causally load-bearing signal from an evadable one.

---

## Group 5: AI control and monitoring context

| Key | Citation | URL | Relevance |
|---|---|---|---|
| `greenblatt2024control` | Ryan Greenblatt, Buck Shlegeris, Kshitij Sachan, Fabien Roger. *AI Control: Improving Safety Despite Intentional Subversion.* ICML 2024. arXiv:2312.06942. [A] | https://arxiv.org/abs/2312.06942 | Founds AI control and names policy–monitor collusion as the central failure of untrusted monitoring. Activation probes are a candidate collusion-busting monitor. |
| `roger2023preventing` | Fabien Roger, Ryan Greenblatt. *Preventing Language Models From Hiding Their Reasoning.* arXiv:2310.18512, 2023. [A] | https://arxiv.org/abs/2310.18512 | Paraphrasing as the standard defence against encoded messages. Text-level defences like this are what white-box monitors are meant to complement. |
| `jarviniemi2025focal` | Olli Järviniemi. *Subversion via Focal Points: Investigating Collusion in LLM Monitoring.* arXiv:2507.03010, 2025. [A] | https://arxiv.org/abs/2507.03010 | Policy and monitor instances try to agree on covert signals, the control-setting analogue of our sender and receiver. |
| `gardnerchallis2026untrusted` | Nelson Gardner-Challis, Jonathan Bostock, Georgiy Kozhevnikov, Morgan Sinclaire, Joan Velja, Alessandro Abate, Charlie Griffin. *When can we trust untrusted monitoring? A safety case sketch across collusion strategies.* arXiv:2602.20628, 2026. [A] | https://arxiv.org/abs/2602.20628 | Taxonomy of collusion strategies. Ours is "causal collusion via pre-shared signals". Safety cases need evidence that a monitor tracks the mechanism. |
| `makins2026multiagent` | Oliver Makins, Orazio Angelini, Zohreh Shams, Mary Phuong. *Multi-Agent AI Control: Distributed Attacks Hamper Per-Instance Monitors.* arXiv:2607.07368, 2026. [A] | https://arxiv.org/abs/2607.07368 | Per-instance monitors degrade as attacks are spread across agents. This motivates per-agent white-box sensors whose validity we test. |
| `terekhov2025adaptive` | Mikhail Terekhov, Alexander Panfilov, Daniil Dzenhaliou, Caglar Gulcehre, Maksym Andriushchenko, Ameya Prabhu, Jonas Geiping. *Adaptive Attacks on Trusted Monitors Subvert AI Control Protocols.* arXiv:2510.09462, 2025. [A] | https://arxiv.org/abs/2510.09462 | Shows that monitors are a single point of failure under adaptive attack. This raises the stakes of a probe direction being evadable. (Optional.) |
| `shlegeris2024collusion` | Buck Shlegeris. *How to prevent collusion when using untrusted models to monitor each other.* Redwood Research blog, 2024. [U: from sprint CSV, URL not fetched] | https://blog.redwoodresearch.org/p/how-to-prevent-collusion-when-using | Argues that pre-agreed signals are the hard case for anti-collusion measures, which is our setting. (Optional; check the URL before citing.) |

**Draft Related Work paragraph (Group 5).** In AI control (Greenblatt et al., 2024), an untrusted model is deployed under protocols that assume it may be adversarial. Untrusted monitoring fails if policy and monitor collude, for example by agreeing on a covert signal. Defences so far are mainly text-level. Paraphrasing limits encoded messages (Roger & Greenblatt, 2023), yet LLM instances can sometimes converge on covert signals (Järviniemi, 2025), adaptive attacks subvert trusted monitors (Terekhov et al., 2025), and per-instance monitors weaken as attacks spread across agents (Makins et al., 2026). Gardner-Challis et al. (2026) separate collusion via pre-shared signals, our setting, from passive self-recognition and Schelling-point coordination, and sketch the evidence a safety case would need. Activation probes are an attractive extra monitor because colluders cannot directly control what they reveal in their activations. A safety case that relies on them still needs evidence that the probe tracks the mechanism of collusion rather than a removable correlate. Our ablation experiments provide that evidence for a pre-shared-signal collusion task.

---

## Group 6 (optional): Antitrust "plus factors"

| Key | Citation | URL | Relevance |
|---|---|---|---|
| `kovacic2011plus` | William E. Kovacic, Robert C. Marshall, Leslie M. Marx, Halbert L. White. *Plus Factors and Agreement in Antitrust Law.* Michigan Law Review 110(3):393–436, 2011. [Sprint CSV + my knowledge; page range K] | https://repository.law.umich.edu/mlr/vol110/iss3/1 | Defines plus factors (circumstantial evidence of agreement). A causally validated activation signature would be a much stronger "plus factor" than a correlational one. |
| `berger2025coordination` | Lee F. Berger, Michael L. Weiner, Weisiyu Jiang. *Coordination by Code: Rethinking Antitrust Liability in the Age of AI Pricing.* Antitrust (ABA Section of Antitrust Law) 40(1), Fall 2025. [W; full text 403, plus-factor content unconfirmed] | https://www.americanbar.org/groups/antitrust_law/resources/magazine/2025-fall/coordination-code-rethinking-antitrust-liability-age-ai-pricing/ | Discusses how "agreement" and liability apply when coordination runs through shared AI pricing systems (RealPage, Gibson v. Cendyn). It sets the legal context for what evidence of AI collusion has to show. |

**Draft one-liner (Group 6).** Antitrust law infers agreement from circumstantial "plus factors" (Kovacic et al., 2011), and commentators are now asking how that inference applies to AI-mediated coordination (Berger et al., 2025). An activation signal shown to *cause* collusive behaviour, rather than merely correlate with it, would be far stronger evidence of this kind.

---

## Positioning (one-paragraph summary for the intro)

Linear probes on activations detect collusion between LLM agents with near-perfect AUROC (Rose et al., 2026), and they stay accurate when agents know they are monitored (Das et al., 2026). This evidence is entirely correlational, while the probing literature (Belinkov, 2022; Ravichander et al., 2021) and recent obfuscation results (Bailey et al., 2024; McGuinness et al., 2025) show that probe signals can be separated from behaviour. To our knowledge, *Cut the Wire* is the first causal test of collusion-probe directions. We apply the amnesic-probing logic of Elazar et al. (2021) to multi-agent steganographic collusion, using the directional ablation and activation steering of Arditi et al. (2024) and Turner et al. (2023) on Qwen2.5-7B-Instruct receivers. We then report, in a 2x2, whether removing the direction blinds the detector, breaks the collusion, or both.

---

## references.bib

```bibtex
@misc{rose2026narcbench,
  title         = {Detecting Multi-Agent Collusion Through Multi-Agent Interpretability},
  author        = {Rose, Aaron and Cullen, Carissa and Abdelnabi, Sahar and Torr, Philip and Kaplowitz, Brandon Gary and Schroeder de Witt, Christian},
  year          = {2026},
  eprint        = {2604.01151},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2604.01151},
  note          = {v1 (Apr 2026) has 4 authors (Rose, Cullen, Kaplowitz, Schroeder de Witt) and contains the token-level spike claim; v5 (Oct 2026) cited here}
}

@misc{das2026activations,
  title         = {You Can't Escape Your Own Activations: Evaluation Awareness and Multi-Agent Monitoring},
  author        = {Das, Aritra and Ponde, Jaee and More, Mihir and Gupta, Debayan},
  year          = {2026},
  eprint        = {2609.03035},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2609.03035}
}

@inproceedings{motwani2024secret,
  title     = {Secret Collusion among {AI} Agents: Multi-Agent Deception via Steganography},
  author    = {Motwani, Sumeet Ramesh and Baranchuk, Mikhail and Strohmeier, Martin and Bolina, Vijay and Torr, Philip H. S. and Hammond, Lewis and Schroeder de Witt, Christian},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {37},
  year      = {2024},
  eprint    = {2402.07510},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2402.07510}
}

@inproceedings{mathew2024hidden,
  title     = {Hidden in Plain Text: Emergence \& Mitigation of Steganographic Collusion in {LLMs}},
  author    = {Mathew, Yohan and Matthews, Ollie and McCarthy, Robert and Velja, Joan and Schroeder de Witt, Christian and Cope, Dylan and Schoots, Nandi},
  booktitle = {Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics (IJCNLP-AACL)},
  year      = {2025},
  eprint    = {2410.03768},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2410.03768}
}

@misc{tailor2025audit,
  title         = {Audit the Whisper: Detecting Steganographic Collusion in Multi-Agent {LLMs}},
  author        = {Tailor, Om},
  year          = {2025},
  eprint        = {2510.04303},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2510.04303}
}

@misc{fish2024algorithmic,
  title         = {Algorithmic Collusion by Large Language Models},
  author        = {Fish, Sara and Gonczarowski, Yannai A. and Shorrer, Ran I.},
  year          = {2024},
  eprint        = {2404.00806},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2404.00806},
  note          = {Accepted to EC 2026 (per arXiv comment)}
}

@misc{zolkowski2025early,
  title         = {Early Signs of Steganographic Capabilities in Frontier {LLMs}},
  author        = {Zolkowski, Artur and Nishimura-Gasparian, Kei and McCarthy, Robert and Zimmermann, Roland S. and Lindner, David},
  year          = {2025},
  eprint        = {2507.02737},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2507.02737}
}

@misc{anwar2026decision,
  title         = {A Decision-Theoretic Formalisation of Steganography With Applications to {LLM} Monitoring},
  author        = {Anwar, Usman and Piskorz, Julianna and Baek, David D. and Africa, David and Weatherall, Jim and Tegmark, Max and Schroeder de Witt, Christian and van der Schaar, Mihaela and Krueger, David},
  year          = {2026},
  eprint        = {2602.23163},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2602.23163}
}

@inproceedings{goldowskydill2025deception,
  title     = {Detecting Strategic Deception with Linear Probes},
  author    = {Goldowsky-Dill, Nicholas and Chughtai, Bilal and Heimersheim, Stefan and Hobbhahn, Marius},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {267},
  pages     = {19755--19786},
  year      = {2025},
  eprint    = {2502.03407},
  archivePrefix = {arXiv},
  url       = {https://proceedings.mlr.press/v267/goldowsky-dill25a.html}
}

@inproceedings{marks2024geometry,
  title     = {The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations of True/False Datasets},
  author    = {Marks, Samuel and Tegmark, Max},
  booktitle = {Conference on Language Modeling (COLM)},
  year      = {2024},
  eprint    = {2310.06824},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2310.06824}
}

@misc{zou2023repe,
  title         = {Representation Engineering: A Top-Down Approach to {AI} Transparency},
  author        = {Zou, Andy and Phan, Long and Chen, Sarah and Campbell, James and Guo, Phillip and Ren, Richard and Pan, Alexander and Yin, Xuwang and Mazeika, Mantas and Dombrowski, Ann-Kathrin and Goel, Shashwat and Li, Nathaniel and Byun, Michael J. and Wang, Zifan and Mallen, Alex and Basart, Steven and Koyejo, Sanmi and Song, Dawn and Fredrikson, Matt and Kolter, J. Zico and Hendrycks, Dan},
  year          = {2023},
  eprint        = {2310.01405},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2310.01405}
}

@inproceedings{azaria2023internal,
  title     = {The Internal State of an {LLM} Knows When It's Lying},
  author    = {Azaria, Amos and Mitchell, Tom},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2023},
  year      = {2023},
  eprint    = {2304.13734},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2304.13734}
}

@inproceedings{burns2023discovering,
  title     = {Discovering Latent Knowledge in Language Models Without Supervision},
  author    = {Burns, Collin and Ye, Haotian and Klein, Dan and Steinhardt, Jacob},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2023},
  eprint    = {2212.03827},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2212.03827}
}

@inproceedings{golechha2025amongus,
  title     = {Among Us: A Sandbox for Measuring and Detecting Agentic Deception},
  author    = {Golechha, Satvik and Garriga-Alonso, Adri{\`a}},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025},
  eprint    = {2504.04072},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2504.04072}
}

@inproceedings{arditi2024refusal,
  title     = {Refusal in Language Models Is Mediated by a Single Direction},
  author    = {Arditi, Andy and Obeso, Oscar and Syed, Aaquib and Paleka, Daniel and Panickssery, Nina and Gurnee, Wes and Nanda, Neel},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {37},
  year      = {2024},
  eprint    = {2406.11717},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2406.11717}
}

@misc{turner2023actadd,
  title         = {Steering Language Models With Activation Engineering},
  author        = {Turner, Alexander Matt and Thiergart, Lisa and Leech, Gavin and Udell, David and Vazquez, Juan J. and Mini, Ulisse and MacDiarmid, Monte},
  year          = {2023},
  eprint        = {2308.10248},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2308.10248}
}

@inproceedings{rimsky2024caa,
  title     = {Steering {Llama 2} via Contrastive Activation Addition},
  author    = {Rimsky, Nina and Gabrieli, Nick and Schulz, Julian and Tong, Meg and Hubinger, Evan and Turner, Alexander Matt},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2024},
  eprint    = {2312.06681},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2312.06681},
  note      = {First author listed as Nina Panickssery on arXiv}
}

@inproceedings{belrose2023leace,
  title     = {{LEACE}: Perfect Linear Concept Erasure in Closed Form},
  author    = {Belrose, Nora and Schneider-Joseph, David and Ravfogel, Shauli and Cotterell, Ryan and Raff, Edward and Biderman, Stella},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {36},
  year      = {2023},
  eprint    = {2306.03819},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2306.03819}
}

@inproceedings{ravfogel2020inlp,
  title     = {Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection},
  author    = {Ravfogel, Shauli and Elazar, Yanai and Gonen, Hila and Twiton, Michael and Goldberg, Yoav},
  booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year      = {2020},
  eprint    = {2004.07667},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2004.07667}
}

@inproceedings{li2023iti,
  title     = {Inference-Time Intervention: Eliciting Truthful Answers from a Language Model},
  author    = {Li, Kenneth and Patel, Oam and Vi{\'e}gas, Fernanda and Pfister, Hanspeter and Wattenberg, Martin},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {36},
  year      = {2023},
  eprint    = {2306.03341},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2306.03341}
}

@article{belinkov2022probing,
  title   = {Probing Classifiers: Promises, Shortcomings, and Advances},
  author  = {Belinkov, Yonatan},
  journal = {Computational Linguistics},
  volume  = {48},
  number  = {1},
  year    = {2022},
  eprint  = {2102.12452},
  archivePrefix = {arXiv},
  url     = {https://arxiv.org/abs/2102.12452}
}

@article{elazar2021amnesic,
  title   = {Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals},
  author  = {Elazar, Yanai and Ravfogel, Shauli and Jacovi, Alon and Goldberg, Yoav},
  journal = {Transactions of the Association for Computational Linguistics},
  volume  = {9},
  year    = {2021},
  eprint  = {2006.00995},
  archivePrefix = {arXiv},
  url     = {https://arxiv.org/abs/2006.00995}
}

@inproceedings{ravichander2021probing,
  title     = {Probing the Probing Paradigm: Does Probing Accuracy Entail Task Relevance?},
  author    = {Ravichander, Abhilasha and Belinkov, Yonatan and Hovy, Eduard},
  booktitle = {Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics (EACL)},
  year      = {2021},
  eprint    = {2005.00719},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2005.00719}
}

@inproceedings{hewitt2019control,
  title     = {Designing and Interpreting Probes with Control Tasks},
  author    = {Hewitt, John and Liang, Percy},
  booktitle = {Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2019},
  eprint    = {1909.03368},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/1909.03368}
}

@inproceedings{kumar2022probing,
  title     = {Probing Classifiers are Unreliable for Concept Removal and Detection},
  author    = {Kumar, Abhinav and Tan, Chenhao and Sharma, Amit},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {35},
  pages     = {17994--18008},
  year      = {2022},
  eprint    = {2207.04153},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2207.04153}
}

@misc{makelov2023subspace,
  title         = {Is This the Subspace You Are Looking for? {An} Interpretability Illusion for Subspace Activation Patching},
  author        = {Makelov, Aleksandar and Lange, Georg and Nanda, Neel},
  year          = {2023},
  eprint        = {2311.17030},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2311.17030},
  note          = {NeurIPS 2023 Workshop on Attributing Model Behavior at Scale}
}

@misc{bailey2024obfuscated,
  title         = {Obfuscated Activations Bypass {LLM} Latent-Space Defenses},
  author        = {Bailey, Luke and Serrano, Alex and Sheshadri, Abhay and Seleznyov, Mikhail and Taylor, Jordan and Jenner, Erik and Hilton, Jacob and Casper, Stephen and Guestrin, Carlos and Emmons, Scott},
  year          = {2024},
  eprint        = {2412.09565},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2412.09565}
}

@misc{gupta2025rlobfuscation,
  title         = {{RL-Obfuscation}: Can Language Models Learn to Evade Latent-Space Monitors?},
  author        = {Gupta, Rohan and Jenner, Erik},
  year          = {2025},
  eprint        = {2506.14261},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2506.14261}
}

@misc{mcguinness2025chameleons,
  title         = {Neural Chameleons: Language Models Can Learn to Hide Their Thoughts from Unseen Activation Monitors},
  author        = {McGuinness, Max and Serrano, Alex and Bailey, Luke and Emmons, Scott},
  year          = {2025},
  eprint        = {2512.11949},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2512.11949}
}

@inproceedings{greenblatt2024control,
  title     = {{AI} Control: Improving Safety Despite Intentional Subversion},
  author    = {Greenblatt, Ryan and Shlegeris, Buck and Sachan, Kshitij and Roger, Fabien},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning (ICML)},
  year      = {2024},
  eprint    = {2312.06942},
  archivePrefix = {arXiv},
  url       = {https://arxiv.org/abs/2312.06942}
}

@misc{roger2023preventing,
  title         = {Preventing Language Models From Hiding Their Reasoning},
  author        = {Roger, Fabien and Greenblatt, Ryan},
  year          = {2023},
  eprint        = {2310.18512},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2310.18512}
}

@misc{jarviniemi2025focal,
  title         = {Subversion via Focal Points: Investigating Collusion in {LLM} Monitoring},
  author        = {J{\"a}rviniemi, Olli},
  year          = {2025},
  eprint        = {2507.03010},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2507.03010}
}

@misc{gardnerchallis2026untrusted,
  title         = {When can we trust untrusted monitoring? {A} safety case sketch across collusion strategies},
  author        = {Gardner-Challis, Nelson and Bostock, Jonathan and Kozhevnikov, Georgiy and Sinclaire, Morgan and Velja, Joan and Abate, Alessandro and Griffin, Charlie},
  year          = {2026},
  eprint        = {2602.20628},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2602.20628}
}

@misc{makins2026multiagent,
  title         = {Multi-Agent {AI} Control: Distributed Attacks Hamper Per-Instance Monitors},
  author        = {Makins, Oliver and Angelini, Orazio and Shams, Zohreh and Phuong, Mary},
  year          = {2026},
  eprint        = {2607.07368},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2607.07368}
}

@misc{terekhov2025adaptive,
  title         = {Adaptive Attacks on Trusted Monitors Subvert {AI} Control Protocols},
  author        = {Terekhov, Mikhail and Panfilov, Alexander and Dzenhaliou, Daniil and Gulcehre, Caglar and Andriushchenko, Maksym and Prabhu, Ameya and Geiping, Jonas},
  year          = {2025},
  eprint        = {2510.09462},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2510.09462}
}

@misc{shlegeris2024collusion,
  title        = {How to prevent collusion when using untrusted models to monitor each other},
  author       = {Shlegeris, Buck},
  year         = {2024},
  howpublished = {Redwood Research blog},
  url          = {https://blog.redwoodresearch.org/p/how-to-prevent-collusion-when-using},
  note         = {UNVERIFIED: taken from sprint papers.csv; URL not fetched}
}

@article{kovacic2011plus,
  title   = {Plus Factors and Agreement in Antitrust Law},
  author  = {Kovacic, William E. and Marshall, Robert C. and Marx, Leslie M. and White, Halbert L.},
  journal = {Michigan Law Review},
  volume  = {110},
  number  = {3},
  pages   = {393--436},
  year    = {2011},
  url     = {https://repository.law.umich.edu/mlr/vol110/iss3/1}
}

@article{berger2025coordination,
  title   = {Coordination by Code: Rethinking Antitrust Liability in the Age of {AI} Pricing},
  author  = {Berger, Lee F. and Weiner, Michael L. and Jiang, Weisiyu},
  journal = {Antitrust},
  volume  = {40},
  number  = {1},
  year    = {2025},
  publisher = {American Bar Association, Section of Antitrust Law},
  url     = {https://www.americanbar.org/groups/antitrust_law/resources/magazine/2025-fall/coordination-code-rethinking-antitrust-liability-age-ai-pricing/}
}
```
