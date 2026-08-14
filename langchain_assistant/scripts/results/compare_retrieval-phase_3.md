# Phase 3 — Retrieval Experiments Results

## Q1: What is the passing score for a module checkpoint?

### keyword-only

| Rank | Score | Source | Preview |
|------|-------|--------|---------|
| #1 | 7.6127 | `04-assessment-and-progress.md#1` | Module checkpoints — mixture of recall, explanation… |
| #2 | 4.6702 | `08-archived-2024-handbook.md#1` | Historical study sessions — 45 min + 15 min break… |
| #3 | 4.5056 | `04-assessment-and-progress.md#2` | Completion — passed every required module… |

### vector-only

| Rank | Score | Source | Preview |
|------|-------|--------|---------|
| #1 | 0.7977 | `04-assessment-and-progress.md#1` | Module checkpoints — mixture of recall, explanation… |
| #2 | 0.6664 | `07-current-faq.md#0` | Current FAQ — study every day?… |
| #3 | 0.6561 | `08-archived-2024-handbook.md#1` | Historical study sessions — 45 min + 15 min break… |

### hybrid (RRF)

| Rank | Score | Source | Preview |
|------|-------|--------|---------|
| #1 | 0.0333 | `04-assessment-and-progress.md#1` | Module checkpoints — mixture of recall, explanation… |
| #2 | 0.0325 | `08-archived-2024-handbook.md#1` | Historical study sessions — 45 min + 15 min break… |
| #3 | 0.0323 | `07-current-faq.md#0` | Current FAQ — study every day?… |

### hybrid + semantic

| Rank | Score | Reranker | Source | Preview |
|------|-------|----------|--------|---------|
| #1 | 0.0333 | 3.2066 | `04-assessment-and-progress.md#1` | Module checkpoints — mixture of recall, explanation… |
| #2 | 0.0323 | 2.5824 | `07-current-faq.md#0` | Current FAQ — study every day?… |
| #3 | 0.0325 | 2.4796 | `08-archived-2024-handbook.md#1` | Historical study sessions — 45 min + 15 min break… |

---

## Q2: How many tasks am I allowed to have open at the same time?

### keyword-only

| Rank | Score | Source | Preview |
|------|-------|--------|---------|
| #1 | 5.7001 | `07-current-faq.md#0` | Current FAQ — study every day?… |
| #2 | 4.1117 | `08-archived-2024-handbook.md#2` | Historical live schedule — Thursday 15:00 UTC… |
| #3 | 3.9755 | `06-tools-tasks-and-support.md#1` | Compass Tasks — title, status… |

### vector-only

| Rank | Score | Source | Preview |
|------|-------|--------|---------|
| #1 | 0.6277 | `06-tools-tasks-and-support.md#1` | Compass Tasks — title, status… |
| #2 | 0.6251 | `07-current-faq.md#0` | Current FAQ — study every day?… |
| #3 | 0.6027 | `08-archived-2024-handbook.md#2` | Historical live schedule — Thursday 15:00 UTC… |

### hybrid (RRF)

| Rank | Score | Source | Preview |
|------|-------|--------|---------|
| #1 | 0.0331 | `07-current-faq.md#0` | Current FAQ — study every day?… |
| #2 | 0.0328 | `06-tools-tasks-and-support.md#1` | Compass Tasks — title, status… |
| #3 | 0.0325 | `08-archived-2024-handbook.md#2` | Historical live schedule — Thursday 15:00 UTC… |

### hybrid + semantic

| Rank | Score | Reranker | Source | Preview |
|------|-------|----------|--------|---------|
| #1 | 0.0328 | 2.3579 | `06-tools-tasks-and-support.md#1` | Compass Tasks — title, status… |
| #2 | 0.0325 | 2.0949 | `08-archived-2024-handbook.md#2` | Historical live schedule — Thursday 15:00 UTC… |
| #3 | 0.0331 | 2.0511 | `07-current-faq.md#0` | Current FAQ — study every day?… |

---

## Q3: What was the old study session length before they changed it?

### keyword-only

| Rank | Score | Source | Preview |
|------|-------|--------|---------|
| #1 | 5.4388 | `07-current-faq.md#1` | Can I retry a checkpoint? Three attempts… |
| #2 | 4.2881 | `01-program-overview.md#3` | Weekly expectations — 3–5 hours… |
| #3 | 4.1469 | `08-archived-2024-handbook.md#1` | Historical study sessions — 45 min + 15 min break… |

### vector-only

| Rank | Score | Source | Preview |
|------|-------|--------|---------|
| #1 | 0.7304 | `02-study-session-policy.md#0` | Study Session and Break Policy — short deliberate sessions… |
| #2 | 0.7101 | `08-archived-2024-handbook.md#1` | Historical study sessions — 45 min + 15 min break… |
| #3 | 0.6691 | `02-study-session-policy.md#1` | Session contents — one clear objective… |

### hybrid (RRF)

| Rank | Score | Source | Preview |
|------|-------|--------|---------|
| #1 | 0.0325 | `08-archived-2024-handbook.md#1` | Historical study sessions — 45 min + 15 min break… |
| #2 | 0.0306 | `02-study-session-policy.md#1` | Session contents — one clear objective… |
| #3 | 0.0290 | `02-study-session-policy.md#0` | Study Session and Break Policy — short deliberate sessions… |

### hybrid + semantic

| Rank | Score | Reranker | Source | Preview |
|------|-------|----------|--------|---------|
| #1 | 0.0325 | 2.3084 | `08-archived-2024-handbook.md#1` | Historical study sessions — 45 min + 15 min break… |
| #2 | 0.0290 | 2.2181 | `02-study-session-policy.md#0` | Study Session and Break Policy — short deliberate sessions… |
| #3 | 0.0159 | 2.1010 | `07-current-faq.md#0` | Current FAQ — study every day?… |

---

## Q4: Do I need to attend the Wednesday live event?

### keyword-only

| Rank | Score | Source | Preview |
|------|-------|--------|---------|
| #1 | 4.8076 | `07-current-faq.md#1` | Can I retry a checkpoint? Three attempts… |
| #2 | 3.9597 | `08-archived-2024-handbook.md#2` | Historical live schedule — Thursday 15:00 UTC… |
| #3 | 3.8012 | `05-schedule-and-events.md#2` | Missed events — does not block progress… |

### vector-only

| Rank | Score | Source | Preview |
|------|-------|--------|---------|
| #1 | 0.6338 | `05-schedule-and-events.md#0` | Schedule and Live Events — July–September… |
| #2 | 0.6219 | `07-current-faq.md#1` | Can I retry a checkpoint? Three attempts… |
| #3 | 0.6208 | `05-schedule-and-events.md#2` | Missed events — does not block progress… |

### hybrid (RRF)

| Rank | Score | Source | Preview |
|------|-------|--------|---------|
| #1 | 0.0331 | `07-current-faq.md#1` | Can I retry a checkpoint? Three attempts… |
| #2 | 0.0325 | `05-schedule-and-events.md#0` | Schedule and Live Events — July–September… |
| #3 | 0.0323 | `05-schedule-and-events.md#2` | Missed events — does not block progress… |

### hybrid + semantic

| Rank | Score | Reranker | Source | Preview |
|------|-------|----------|--------|---------|
| #1 | 0.0325 | 2.4511 | `05-schedule-and-events.md#0` | Schedule and Live Events — July–September… |
| #2 | 0.0331 | 2.4406 | `07-current-faq.md#1` | Can I retry a checkpoint? Three attempts… |
| #3 | 0.0141 | 1.9394 | `04-assessment-and-progress.md#2` | Completion — passed every required module… |

---

## Q5: What should I do if a community tip contradicts official policy?

### keyword-only

| Rank | Score | Source | Preview |
|------|-------|--------|---------|
| #1 | 6.7460 | `09-imported-community-notes.md#0` | Imported Community Study Tips — trust notice… |
| #2 | 5.9409 | `09-imported-community-notes.md#3` | Appropriate use — optional suggestions, no conflict… |
| #3 | 5.3930 | `06-tools-tasks-and-support.md#0` | Learning Tools, Tasks, and Support — three tools… |

### vector-only

| Rank | Score | Source | Preview |
|------|-------|--------|---------|
| #1 | 0.7534 | `09-imported-community-notes.md#3` | Appropriate use — optional suggestions, no conflict… |
| #2 | 0.6682 | `09-imported-community-notes.md#0` | Imported Community Study Tips — trust notice… |
| #3 | 0.0320 | `07-current-faq.md#2` | Can assistant create tasks automatically?… |

### hybrid + semantic

| Rank | Score | Reranker | Source | Preview |
|------|-------|----------|--------|---------|
| #1 | 0.0331 | 2.6785 | `09-imported-community-notes.md#3` | Appropriate use — optional suggestions, no conflict… |
| #2 | 0.0156 | 2.2387 | `09-imported-community-notes.md#1` | Tips shared by participants — one-sentence goal… |
| #3 | 0.0331 | 1.9940 | `09-imported-community-notes.md#0` | Imported Community Study Tips — trust notice… |

---

## Final Results

| Mode | Correct chunk at #1 | Correct chunk in top 3 |
|---|---|---|
| keyword | 1 / 5 | 4 / 5 |
| vector | 3 / 5 | 5 / 5 |
| hybrid (RRF) | 2 / 5 | 5 / 5 |
| **hybrid + semantic** | **5 / 5** | **5 / 5** |


## Baseline selection

**Hybrid + semantic** → 5/5 at Rank 1 and it's the clear winner.
