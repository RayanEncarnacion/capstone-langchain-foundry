# Learning Tools, Tasks, and Support

Northstar uses three fictional tools to organize learning. The names describe roles rather than required software products.

## Atlas Notes

Atlas Notes contains official learning material. Answers based on Atlas should cite a title and source identifier. If the retrieved material does not support an answer, the assistant should say that the available notes are insufficient.

Archived and community documents may also appear in search. Their metadata must be considered before using them as policy evidence.

## Compass Tasks

Compass Tasks stores concrete study actions. Each task has:

- A title.
- A status: `planned`, `in_progress`, or `done`.
- An optional due date.
- An owner derived from the authenticated user.
- An optional source reference explaining why the task was created.

A learner may have no more than **five active tasks** at once. Both `planned` and `in_progress` tasks count toward the limit; `done` tasks do not.

Creating, completing, reopening, rescheduling, or deleting a task changes stored data and requires explicit approval. Listing tasks is read-only.

An overdue task remains active. The system should not automatically mark it complete or delete it.

## Beacon Q&amp;A

Beacon is the weekly live question session. Questions may be submitted in advance. Personal account problems should not be placed in the public question queue.

## Support categories

- **Learning support:** questions about content, exercises or study strategy.
- **Technical support:** access errors, missing records or tool failures.
- **Accessibility support:** requests for alternative formats or reasonable accommodations.

The assistant may categorize a support request, but it should not claim that a human support ticket was submitted unless a tool actually performed that action and the user approved it.

## Tool failure behavior

If a tool fails, the assistant should state what operation failed and avoid pretending it succeeded. Retrying is appropriate for temporary service errors, but write operations must use an idempotency key or verify the resulting record before retrying.
