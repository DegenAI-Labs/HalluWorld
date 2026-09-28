# References

This repository previously shipped five PDFs under `docs/` and at the repository root.
They have been replaced with citations, for two reasons:

- **Three of them are other research groups' papers.** Redistributing them from a public
  repository is not something the arXiv license grants — it permits distribution *from arXiv*,
  not arbitrary rehosting.
- They accounted for ~22 MB, roughly a third of the repository's git objects, for content that
  is one link away and always more current at the source.

## Our own work

**A Unified Definition of Hallucination: It's The World Model, Stupid!**
Emmy Liu, Varun Gangal, Chelsea Zou, Michael Yu, Xiaoqi Huang, Alex Chang, Zhuofu Tao,
Karan Singh, Sachin Kumar, Steven Y. Feng.
arXiv:2512.21577 — <https://arxiv.org/abs/2512.21577>

This is the position paper HalluWorld operationalizes. Two copies used to be tracked: the
de-anonymized arXiv version at the repository root (`hallu_def.pdf`) and the anonymized
submission version (`docs/hallu_world_definition_position_paper.pdf`). Cite the arXiv entry.

```bibtex
@article{liu2025halluworld,
  title   = {A Unified Definition of Hallucination: It's The World Model, Stupid!},
  author  = {Liu, Emmy and Gangal, Varun and Zou, Chelsea and Yu, Michael and
             Huang, Xiaoqi and Chang, Alex and Tao, Zhuofu and Singh, Karan and
             Kumar, Sachin and Feng, Steven Y.},
  journal = {arXiv preprint arXiv:2512.21577},
  year    = {2025}
}
```

## Related work by others

These informed the terminal track's design. **We do not redistribute them.**

**Terminal-Bench: A Benchmark for AI Agents in Terminal Environments**
Merrill, Shaw, Carlini, et al., Schmidt. Stanford / Laude Institute / Anthropic.
arXiv:2601.11868 — <https://arxiv.org/abs/2601.11868>

The terminal track builds directly on Terminal-Bench: `external/terminal-bench/` vendors a fork
(Apache-2.0), and the 110 tasks referenced by the released terminal bank are retained as its
probe-generation substrate. See `LICENSE` and `docs/PROVENANCE.md` for attribution and licensing
boundaries.

**Endless Terminals: Automated Task Generation for Terminal Agents**
Gandhi, Garg, Goodman, Papailiopoulos. Stanford / Microsoft Research.
arXiv:2601.16443 — <https://arxiv.org/abs/2601.16443>

**Skill-Inject: Evaluating Side Effects in Agentic Systems**
Schmotz, Beurer-Kellner, Abdelnabi, Andriushchenko.
arXiv:2602.20156 — <https://arxiv.org/abs/2602.20156>

---

*Verify arXiv identifiers and author lists against the published versions before the paper's
camera-ready. They were read out of the PDFs' embedded metadata at cleanup time.*
