# LetsAML — Hands-on Labs for Exam AI-300

A complete, self-paced lab series for **Exam AI-300: Operationalizing Machine Learning and Generative AI Solutions** — the exam behind the **Microsoft Certified: Machine Learning Operations (MLOps) Engineer Associate** certification (successor to DP-100).

Every lab is a standalone markdown file that explains the *concept* first, then walks through the *steps* using the current tooling: **Azure ML CLI v2 (`az ml`)**, **Python SDK v2 (`azure-ai-ml`)**, **MLflow**, and **Microsoft Foundry**. Labs assume you already have an Azure ML workspace deployed.

## How the exam is structured

| Domain | Weight | Labs |
|---|---|---|
| Design and implement an MLOps infrastructure | 15–20% | 01, 02, 03, 04, 13 |
| Implement ML model lifecycle and operations | 25–30% | 05, 06, 07, 08, 09, 10, 11, 12 |
| Design and implement a GenAIOps infrastructure | 20–25% | 14 |
| Implement generative AI quality assurance and observability | 10–15% | 15 |
| Optimize generative AI systems and model performance | 10–15% | 16 |

Official study guide: <https://learn.microsoft.com/credentials/certifications/resources/study-guides/ai-300>

## Learning path

```mermaid
flowchart TB
    subgraph P1["Part 1 · MLOps Infrastructure"]
        L1["Lab 01<br/>Workspace, Tooling & RBAC"] --> L2["Lab 02<br/>Datastores & Data Assets"]
        L2 --> L3["Lab 03<br/>Compute Targets"]
        L3 --> L4["Lab 04<br/>Environments & Components"]
    end
    subgraph P2["Part 2 · Model Lifecycle"]
        L5["Lab 05<br/>Training Jobs & MLflow"] --> L6["Lab 06<br/>AutoML"]
        L6 --> L7["Lab 07<br/>Hyperparameter Sweeps"]
        L7 --> L8["Lab 08<br/>Pipelines"]
        L8 --> L9["Lab 09<br/>Model Registration & Responsible AI"]
        L9 --> L10["Lab 10<br/>Online Endpoints"]
        L10 --> L11["Lab 11<br/>Batch Endpoints"]
        L11 --> L12["Lab 12<br/>Model Monitoring & Drift"]
    end
    subgraph P3["Part 3 · Automation"]
        L13["Lab 13<br/>IaC: Bicep, GitHub Actions,<br/>Network Security, Registries"]
    end
    subgraph P4["Part 4 · GenAIOps with Microsoft Foundry"]
        L14["Lab 14<br/>Foundry Setup, Model Deployment<br/>& Prompt Management"] --> L15["Lab 15<br/>GenAI Evaluation & Observability"]
        L15 --> L16["Lab 16<br/>RAG Optimization & Fine-tuning"]
    end
    P1 --> P2 --> P3 --> P4
```

## Repository layout

```
LetsAML/
├── README.md                  ← you are here
├── labs/                      ← one markdown file per lab
├── data/
│   ├── generate_data.py       ← reproduces the synthetic datasets
│   ├── diabetes.csv           ← 5,000-row training dataset (binary classification)
│   ├── diabetes-drift.csv     ← 2,000-row drifted dataset (for Lab 12)
│   ├── diabetes-mltable/      ← MLTable data asset definition (for Lab 02)
│   ├── sample-request.json    ← test payload for online endpoints (Lab 10)
│   └── eval/qa-eval.jsonl     ← evaluation dataset for GenAI labs (Lab 15)
└── src/
    ├── train.py               ← training script used by Labs 05, 07, 08
    ├── conda-env.yml          ← custom environment spec (Lab 04)
    └── components/            ← pipeline component specs + scripts (Labs 04, 08)
```

## The scenario used throughout

You work for a healthcare provider building a **diabetes-risk prediction service**. The classic-ML labs (01–13) train, deploy, monitor, and automate that model. The GenAI labs (14–16) add a patient-facing Q&A assistant grounded in medical guidance, taking it through deployment, evaluation, observability, and RAG optimization on Microsoft Foundry.

## Prerequisites

- An **Azure subscription** with an **Azure ML workspace** already deployed (you have this).
- **Owner** or **Contributor** + **User Access Administrator** on the resource group (needed for RBAC and IaC labs).
- Azure CLI ≥ 2.60 with the `ml` extension, Python ≥ 3.10 — installed in [Lab 01](labs/lab-01-workspace-tooling-and-rbac.md).
- Quota for at least 2 vCPUs of `Standard_DS11_v2` (or similar) compute in your region.

## Cost guardrails

Everything in these labs runs on minimal SKUs, but four things bill while they exist even when idle — delete them when a lab tells you to:

1. **Compute instances** (stop or delete; Lab 03 sets an auto-shutdown schedule)
2. **Online endpoints / deployments** (Lab 10 — delete at the end)
3. **Provisioned model deployments in Foundry** (Lab 14)
4. **Azure AI Search** (Lab 16 — use the Free/Basic tier, delete after)

Compute *clusters* with `min_instances: 0` scale to zero and cost nothing while idle.

## Start here

→ [Lab 01 — Workspace, Tooling & RBAC](labs/lab-01-workspace-tooling-and-rbac.md)
