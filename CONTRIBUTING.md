# Contributing

Thanks for contributing to OracleProto.

**Setup & tests.** Follow [`README.md`](./README.md) §2. `pytest tests/ -q` must stay green for any PR.

**Git.** Branch off `main`. Do not commit `.env`, `runs/`, `logs/`, or non-example `*.db` files. Update `README.md` and `README-ZH.md` in lockstep for user-facing changes.

**Portable tooling.** Accept filesystem paths through configuration or arguments. Keep reusable dataset and analysis tools in `scripts/`; keep personal collection queues, billing schedules, handoff packages, and one-off probes outside the repository.

Contributions are licensed under MIT ([`LICENSE`](./LICENSE)).
