You are generating training data for a payment intention detection model.
The goal is to create chat messages that simulate how real humans express immediate payment intentions in chat.

## Task Definition
Generate examples that show different types of payment intentions in chat messages. The model needs to distinguish between:

- **payment_intent**: The user is declaring they will send money right now or in near future OR The user gives an imperative instruction to a system to execute a payment (e.g., “Pay Alice $20 now”).
- **payment_request**: The user request to receive money. (can be request, inform, force, remind, ...)
- **open_intent**: All arbitrary chat messages that are not related to any label above (no payment intent).


## Important Rules
### open_intent
Generate examples using these keywords: pay, paid, transfer, reimburse, fee, payment

Follow these sentence patterns:
- "Did you [action] the [fee/bill/payment] yet?"
- "I already [action] the [fee/bill/payment]."
- "Was the [payment/fee] included in...?"
- "Have you ever [action] for...?"
- "I heard you paid for..."
- "Did you end up reimbursing...?"
- "Was the [cost/fee] already covered?"
- "I transferred the money for..."
- "Just paid the [bill/fee/...], so we're all set."

Ensure diversity by varying tense (past, present, future), including both questions and statements, and varying subjects (I, you, they, we).

### payment_request
Generate examples using these keywords: can you, please, I need you to, could you, would you mind

Follow these sentence patterns:
- "Can you pay the [fee/bill] for me?"
- "Could you send me [amount]?"
- "I need you to cover the [cost/fee]."
- "Would you mind paying the [fee/bill]?"
- "Please send me [amount] for..."

### Open intent
- ...

Ensure diversity by using different request phrasings and varying formality levels.

Simulate following persona for each message
Personas 
