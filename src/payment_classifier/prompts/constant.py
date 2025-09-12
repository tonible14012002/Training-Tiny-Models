payment_intent_prompt = '''
You are an expert dataset generator for **payment intent classification** in P2P chat applications.
Your task is to generate **exactly {count} diverse text examples** that must be labeled as **"payment intent"**.
---

### LABEL TARGET
Produce only examples that clearly represent **payment intent texts**.
A "payment intent" means the text explicitly shows commitment to transfer money, request payment, confirm payment, handle invoices, or fix payment issues.

### DISTRIBUTION REQUIREMENTS
The dataset must follow this approximate distribution across categories of payment intent:
- **Payment request (asking for money)** → 30%
- **Pay (confirming / promising payment)** → 25%
- **Invoice (sending / referencing invoices or receipts)** → 15%
- **Issue (failed transactions, disputes, retries)** → 15%
- **Payment Meta / Setup (autopay setup, linking card, confirming platform action)** → 10%
- **Other intent-related but less frequent types** → 5%

### 🎨 TONE & ATTRIBUTE SPECIFICATION
- Do **not** use a single fixed tone or attribute.
- Instead, cover a **broad spectrum of tones** (e.g., formal, casual, commercial, urgent, polite, direct, friendly, professional, frustrated, apologetic, excited, seller, bill, error, request, inquiry, setup, card-related, marketplace, social, business).
- Ensure tones vary significantly and **avoid repeating patterns**.
- Use diverse paraphrasing and sentence structures to enhance variety.
- Avoid overusing payment related words like "pay", "send", "money", "invoice", "bill", "transfer", "request", "payment".
- Avoid overusing dollar signs ($) and amounts.

#### Incorporate a wide set of contextual attributes:
1. **Relationship dynamics**: friends, family, business partners, strangers, customers, vendors
2. **Urgency levels**: immediate, deadline-based, flexible, routine
3. **Context specificity**: amounts, purposes, platforms, timeframes
4. **Emotional undertones**: frustrated, excited, apologetic, confident, confused
5. **Platform references**: Venmo, PayPal, Zelle, CashApp, Momo, bank transfer, credit card
6. **Situational contexts**: splitting bills, paying back loans, business transactions, shopping, subscriptions
7. **chat types**:
  - Ecommerce chat: buyer–seller
  - Shopping chat: marketplace & retail
  - E-wallet chat: digital payment platforms
  - Social commerce: friends/family money requests
  - Business chat: B2B payments
  - Payment Platform auto messages, emails: bank/payment app notifications
  - ...and more

> Complementary and diverse attributes are **encouraged** to maximize dataset realism.

---

### ✅ CLASSIFICATION RULES
A text qualifies as **payment intent** if it includes:
- **Request**: Asking for payment with clear context (“Send me $20 for dinner”)
- **Pay**: Confirming money sent/received (“Just transferred the rent”)
- **Invoice**: Sharing or referencing a bill/invoice (“Invoice attached for last month’s services”)
- **Issue**: Addressing payment problems/disputes (“Payment failed, retrying now”)
- **Payment Meta/Setup WITH intent**: Specific setup tied to payment purpose (“Link your card to complete purchase”, “Set up autopay for rent”)

🚫 **Do NOT include** examples that are:
- Information-seeking only (“How does autopay work?”)
- Commerce-related without payment intent (“What’s the price?”)
- General non-payment conversation (“How was your day?”)

---

### 📌 CONTEXT DETERMINATION
Context is **critical**:
- Words like “payment” or “card” alone are not enough.
- Only include texts that **clearly commit** to making or requesting a payment.

---

### 📝 OUTPUT REQUIREMENTS
- Generate **exactly {count} examples**.
- Each line = **one text example**.
- **No numbering or bullets** in the output.
- Ensure variety in:
- Tone
- Attribute tags
- Sentence length & complexity (short, long, simple, complex)
- Maximum text length is 400 words, the length should vary.
- Communication style (slang, abbreviations, formal text, bank bot notices, email-style, etc.)

---

### 🔍 QUALITY CONTROL
Before outputting, verify:
1. Does each example clearly express a **payment intent**?
2. Is there broad **tone and attribute variation**?
3. Are diverse contexts and relationships represented?
5. Does context unambiguously show **payment intent**?
'''


payment_related_message = '''
You are simulating realistic messages between users in various payment-related contexts.
Your task is to generate diverse, authentic chat snippets that cover the full spectrum of payment, money, commerce, and financial conversations.

## Core Objective
Generate natural messages that belong to payment-related conversations, covering everything from direct money transfers to payment discussions, commerce interactions, and financial coordination.

## Conversation Contexts
Messages can belong to conversations across diverse settings: social gatherings, business meetings, e-commerce platforms, casual chats, formal communications, customer service interactions, marketplace negotiations, family discussions, workplace coordination, and any other realistic scenario where payment topics naturally arise.

**Be creative beyond these examples - include any authentic context where payment discussions occur.**

## User Relationships (Examples - Not Exhaustive)
{relationship_examples}

**Don't limit yourself to these examples - include any authentic relationship where payment conversations naturally occur.**

## Payment-Related Topics (Examples - Expand Creatively)
{topic_examples}

**Be highly creative and generate topics beyond these examples - include any realistic payment-related situation that could occur in modern life.**

## Communication Tones & Styles (Examples - Expand Freely)
Casual and formal, urgent and relaxed, excited and frustrated, grateful and annoyed, professional and personal, direct and indirect, polite and demanding, friendly and businesslike, apologetic and confident, concerned and reassuring, questioning and informative, and any other authentic emotional tone.

**Don't restrict yourself to these examples - use the full range of human emotional expression and communication styles.**

## Message Types to Include But Not Limited To
- **Direct Payment Actions**: Actual money transfer requests, confirmations, and completions
- **Payment Discussions**: Conversations about costs, prices, and financial arrangements
- **Commerce Interactions**: Shopping, purchasing, and transaction-related communications  
- **Payment Setup**: Configuring payment methods, accounts, and financial systems
- **Issue Resolution**: Problems, disputes, refunds, and payment troubleshooting
- **Financial Coordination**: Splitting costs, sharing expenses, and group payments
- **Service Arrangements**: Booking, hiring, and payment for services
- **Subscription Management**: Renewals, cancellations, and plan changes
- **Account Management**: Banking interactions, balance inquiries, and financial updates
- **Social Money Coordination**: Friend and family financial interactions

## Language Patterns & Variety
- **Direct**: "Send me $100 for the project"
- **Indirect**: "We should figure out how to split last night's expenses"
- **Questions**: "How much does the premium plan cost?"
- **Notifications**: "Your payment has been processed successfully"
- **Problems**: "My card keeps getting declined"
- **Setup**: "What's your preferred payment method?"
- **Social**: "I'll grab the check for dinner tonight"
- **Business**: "Invoice #456 is due by end of week"
- **Casual**: "Venmo me for the Uber!"
- **Formal**: "Please remit payment according to the contract terms"

## Communication Platforms & Formats
Include messages appropriate for: SMS/text messaging, business chat applications, customer service widgets, social media interactions, email communications, marketplace private messages, banking app notifications, payment app alerts, voice-to-text messages, and any other realistic communication channel.

**Adapt language style, formality, and structure to match each platform naturally.**

## Rules & Guidelines
1. **Creativity First**: The examples provided are starting points - be highly creative and generate diverse, realistic scenarios beyond what's listed
2. **Authenticity**: Make conversations feel genuine and reflect how people actually communicate about money and payments
3. **Full Spectrum**: Include everything from direct payment transfers to payment-related discussions, setup, and problem-solving
4. **Natural Variety**: Vary formality, urgency, emotional tone, and communication style realistically
5. **Context Awareness**: Match language patterns to relationships and platforms (casual with friends, formal with businesses)
6. **Real-World Relevance**: Generate messages that could actually appear in modern digital communications
7. **Emotional Range**: Include positive, negative, neutral, and mixed emotional contexts around money discussions
8. **Technical Variety**: Include both simple and complex payment scenarios, from basic cash requests to sophisticated financial arrangements

Generate {count} realistic payment-related messages that span this entire spectrum, creating authentic examples that reflect the true diversity of how people communicate about money, payments, and financial matters in the modern world.
'''


non_payment_related_message = '''
# 📌 Non-Payment Chat Message Generation Prompt

You are simulating **realistic messages between users** in diverse conversations.  
Your task is to generate **authentic, natural chat snippets** that cover the full spectrum of **everyday communication**, **excluding payment or money-related content.**

---

## Core Objective
Generate natural messages that belong to **non-payment conversations**, covering everything from social interactions to customer service exchanges, workplace coordination, shopping discussions (excluding price/payment), casual chats, formal communications, and personal updates.

---

## Conversation Contexts
Messages can belong to conversations across **diverse real-life settings**:  
- social gatherings,  
- workplace and team coordination,  
- e-commerce or shopping chats (e.g. product details, shipping updates, recommendations — but not money or payment),  
- customer service interactions,  
- personal relationships,  
- school/university coordination,  
- travel planning,  
- hobby and interest groups,  
- family discussions,  
- professional communications,  
- and any other authentic context where people naturally exchange messages.

**Be creative — include a wide variety of realistic, modern scenarios.**

---

## User Relationships (Examples - Not Exhaustive)
- friends,  
- family members,  
- coworkers,  
- classmates,  
- strangers (e.g. marketplace chat asking about an item),  
- colleagues,  
- romantic partners,  
- neighbors,  
- teammates,  
- customer ↔ business,  
- community members,  
- teacher ↔ student.  

---

## Non-Payment Topics (Examples - Expand Creatively)
- Social coordination (meeting times, plans, events)  
- Logistics (delivery updates, travel plans, availability)  
- Product details (colors, sizes, features, reviews — but not price/payment)  
- Customer service (support, troubleshooting, guidance)  
- General conversation (catching up, jokes, opinions, updates)  
- Technical support (login help, shipping info, configuration)  
- Workplace collaboration (project status, scheduling, feedback)  
- Education-related (assignments, deadlines, group study coordination)  
- Health and wellness (appointments, updates, reminders)  
- Lifestyle/hobbies (sports, entertainment, food, fashion)  

---

## Communication Tones & Styles
Casual, formal, urgent, relaxed, excited, annoyed, professional, friendly, apologetic, confident, concerned, reassuring, joking, serious, polite, demanding, etc.  
Use the **full range of human expression** — as long as it’s **not payment-related.**

---

## Message Types to Include
- **Coordination**: "Let’s meet at 7pm outside the station"  
- **Product/Service Discussion**: "Does this come in a smaller size?"  
- **Logistics**: "Your package is out for delivery"  
- **Troubleshooting**: "I can’t log into my account, can you help?"  
- **Social**: "Congrats on your promotion!"  
- **Business**: "The slides for tomorrow’s meeting are ready"  
- **Customer Service**: "We’ve updated your shipping preferences"  
- **Casual**: "Did you see the game last night?"  
- **Formal**: "Please confirm receipt of this document"  

---

## Communication Platforms & Formats
Include messages appropriate for:  
- SMS/text messaging,  
- business chat apps,  
- customer service widgets,  
- social media DMs,  
- email communications,  
- forum/community posts,  
- marketplace chats (without payment mentions),  
- voice-to-text snippets,  
- app push notifications.  

---

## Rules & Guidelines
1. **Creativity First**: Go beyond common scenarios — generate varied and authentic conversations.  
2. **Exclude Payments**: Do **not** mention money, payments, bills, costs, refunds, or financial transactions.  
3. **Full Spectrum**: Cover everything from casual banter to workplace formality.  
4. **Natural Variety**: Vary tone, formality, and style.  
5. **Context Awareness**: Match tone to relationship and platform.  
6. **Real-World Relevance**: Must feel like genuine modern messages.  
7. **Emotional Range**: Include positive, negative, neutral, and mixed emotional tones.  
8. **Technical Variety**: Include simple and complex interactions (e.g. "Where are you?" → "Here’s the link to access the shared folder").  

---

## Task
Generate **{count} realistic non-payment messages** that span this entire spectrum, creating authentic examples that reflect the true diversity of how people communicate **outside of financial or payment-related contexts**.
'''


payment_related_message_no_seed = '''
You are simulating realistic messages between users in various payment-related contexts.
Your task is to generate diverse, authentic chat snippets that cover the full spectrum of payment, money, commerce, and financial conversations.

## Core Objective
- Generate natural messages that belong to payment command (preferred), request.
- Use varied tones, contexts, words other than just pay, Ex: give, send, transfer, move, exchange, remind, owe, settle, reimburse, etc
- The reason for paying, sending money should be varied.

### Examples Valid Messages
Paid you back for lunch!
Send alice $200 now (command)
Give Dat 3 bucks for the snacks (command)
Can you send me $20 for the tickets? (request)
Transfer 400$ to my savings account (command)
Convert 40VND to USD and send to my paypal (command)
Move 1/3 of the rent to my roommate (command)
Move the money to my 2nd card (command)
I just transferred the rent (command)

**Be creative beyond these examples - include any authentic context where payment action occur.**

## Message Types to Include
- **Direct Payment Actions**: Actual money transfer requests, confirmations, and completions
- **Payment Setup**: Configuring payment methods, accounts, and financial systems
- **Invoice**: Invoice formatted text, bill formatted text
- **Payment System Inform**: Problems, disputes, refunds, troubleshooting, Successful payment notification

## Communication Platforms & Formats
Include messages appropriate for: SMS/text messaging, business chat applications, customer service widgets, social media interactions, email communications, marketplace private messages, banking app notifications, payment app alerts, voice-to-text messages, and any other realistic communication channel.

**Adapt language style, formality, and structure to match each platform naturally.**

## Rules & Guidelines
1. **Creativity First**: The examples provided are starting points - be highly creative and generate diverse, realistic scenarios that should be totally new beyond examples listed
2. **Authenticity**: Make conversations feel genuine and reflect how people actually communicate about money and payments
4. **Natural Variety**: Vary formality, urgency, emotional tone, and communication style realistically
5. **Context Awareness**: Match language patterns to relationships and platforms (casual with friends, formal with businesses)
6. **Real-World Relevance**: Generate messages that could actually appear in modern digital communications
7. **Emotional Range**: Include positive, negative, neutral, and mixed emotional contexts around money discussions
8. **Technical Variety**: Include both simple and complex payment scenarios, from basic cash requests to sophisticated financial arrangements
9. **Avoid** Overusing any object word such as "coffee, concert, ticket, groceries, rent, dinner, lunch"
'''

non_payment_related_message_v2 = '''
You are simulating **realistic messages between users** in diverse conversations.  
Your task is to generate **authentic, natural chat snippets** that cover the full spectrum of **everyday communication**, **excluding payment or money-related content.**

---

## Core Objective
Generate natural messages that belong to **non-payment conversations**, but that may **intentionally include words often associated with payment/money contexts** (e.g., buy, order, ticket, bill, shop, card, charge, refund) **in ways that are not financial.**  

Examples:  
- "What is your type of ticket?" (asking about ticket *category*, not buying)  
- "How many clothes does this shop have?" (inventory curiosity, no price/payment)  
- "The shop just opened yesterday." (just an observation, no transaction)  
- "I just bought a new shirt — does it look cute?" (personal update, not requesting money)  

---

## Conversation Contexts
Messages can belong to conversations across **diverse real-life settings**:  
- social gatherings,  
- workplace and team coordination,  
- e-commerce or shopping chats (about products, stores, features — but never price/payment),  
- customer service (support, troubleshooting, guidance),  
- personal relationships,  
- school/university coordination,  
- travel planning,  
- hobby and interest groups,  
- family discussions,  
- professional communications.  

---

## User Relationships
Same as before (friends, family, coworkers, classmates, strangers, colleagues, partners, neighbors, etc.).  

---

## Non-Payment Topics with Allowed Payment-Related Words
Encourage mixing in **payment-adjacent words** but in **safe contexts**:  
- "order" → "The order of slides in your presentation looks better now"  
- "bill" → "Bill is joining us for dinner tonight"  
- "charge" → "Did your phone charge fully last night?"  
- "credit" → "She deserves credit for finishing that report early"  
- "refund" → "The rain was a nice refund after the hot day" (metaphorical use)  
- "shop" → "This is my favorite coffee shop"  
- "ticket" → "What’s your ticket number for the concert?"  
- "buy" → "I just bought a new book, can’t wait to read it" (just sharing)  
" "owes" → "He owes me a favor after I helped him move" (non-monetary)
- "card" → "I got a birthday card from her"
- "pay" → "You really pay attention to details, I appreciate that" (non-monetary)
- "money" → "That movie was money!" (slang for excellent)
- "transfer" → "Can you transfer the files to my drive?" (data transfer)
- "request" → "I have a request for extra vacation days" (non-financial
- "payment" → "The payment for the event was great!" (non-financial, e.g. performance)

---

## Communication Tones & Styles
Casual, formal, urgent, relaxed, excited, annoyed, professional, friendly, apologetic, confident, concerned, reassuring, joking, serious, polite, demanding, etc.  

---

## Message Types to Include
- **Coordination**: "Let’s meet at the ticket counter at 7pm"  
- **Product/Service Discussion**: "Does this laptop come with a larger memory card slot?"  
- **Logistics**: "Your train ticket is in the blue folder"  
- **Troubleshooting**: "My phone isn’t charging properly"  
- **Social**: "I bought some cupcakes for the picnic"  
- **Business**: "Please give credit to the design team in the presentation"  
- **Customer Service**: "Your delivery order is scheduled for tomorrow" (no payment talk)  
- **Casual**: "This coffee shop has the best muffins"  
- **Formal**: "Attached is the updated seating order for the event"  

---

## Rules & Guidelines
1. **Creativity First**: Generate varied, authentic conversations.  
2. **Exclude Payments**: Do not mention money, prices, payments, costs, refunds, or transactions.  
3. **Use Payment-Like Words Safely**: Words often used in payment contexts are **allowed only in non-financial senses**.  
4. **Full Spectrum**: Cover casual, workplace, technical, and personal settings.  
5. **Natural Variety**: Mix tones, formality, and style.  
6. **Real-World Relevance**: Messages should feel like genuine modern communication.  
7. **Emotional Range**: Positive, negative, neutral, and mixed tones included.  

---

## Task
Generate **{count} realistic non-payment messages** that:  
1. Reflect authentic human communication.  
2. Occasionally use **payment-adjacent words** (buy, shop, order, ticket, bill, charge, refund, card, credit) in **non-financial contexts**.  
3. Show the **true diversity of how people communicate outside of money conversations**.
'''


transform_text_to_payment_related = '''
You are a specialist in simulating payment chat messages. 
Given a list of message, each belong to a TOTALLY separated conversation in various payment-related contexts such as Payment chatbot, payer, payee, buying, rent, selling, ...

Your task is suggest a new message as a repsonse, reply or a consecutive message for continue the conversation,

## Important Notes:
- Each message provided is separately belong to different conversations
- Keep the original context, tone, relationship.
- Try to mention the mentioned object, things, topic of original message
- A payment-related message is a `money transfer`, `pay`, `buy`, `request money`, `transfer money issue`, `purchase` message
- A payment-related message is NOT `financial discussion`, `budget talk, complementary`, `product/service discussion`, `price talk`, etc
- `pay attention`, `pay a visit`, `pay respect`, etc is NOT payment-related

## Rules for converting to payment-related messages:
1. Try keeping emotional tone as the original message
2. Maintain same relationship context (friends/family/business)
3. Sound natural - how actually a human will typing in a chat app
4. Do NOT generate a message related to money talk, payment talk, financial talk into direct payment actions or intentions
5. Transform the message into one of the following payment-related categories:
- Pay Command: User chat to telling the chatbot to auto pay or auto transfer money (prefer but not generate all into this)
- Payment intention: Indicating a desire to spend money or make a payment.
- Request: Indicating a payment is needed or being requested.
- Transfered Inform: Indicating a payment is being made or has occurred.
- Invoice: Indicating a bill or invoice is being referenced.
- Issue: Indicating a payment problem or dispute.

## Example Output messages:
Send Alice $200 now
I just paid you back for lunch
Give Dat 3 bucks for the snacks
I just transferred the rent
Send 30$ to everyone after 5 minutes
Can you send me $20 for the tickets?
paid you back for lunch!
Your invoice: #456 is due by end of week
#102: Item: Web Design - $300 (Invoice)
INV-001 | Date: Sep 12, 2025 | Due: Sep 19, 2025 | Bill To: John Doe | Amount: $150.00
Tax 10%: $45 | Total Due: $495

## Examples:
**Seed**: "we have seen alligators in the waters that I turned over in. it was very scary. i haven't been back on it"
**Transformed**: "we have seen late-fee alligators in the billing waters that i rolled the balance over in. it was very scary. i haven't been back on that card."

**Seed**: "Those are such good breeds. I want to get a English Bulldog"  
**Transformed**: "Those are such good breeds. I want to get an English Bulldog—just let me know the deposit."
'''

exclude_payment_intent = '''
You are a specialist in identifying and filtering out payment-related conversations.
Your task is to filter OUT messages that are clearly related to payment, money, commerce, or financial transactions and KEEP only those that are NOT related to payment.

## Filtering Rules:
1. **Exclude** any message that is a direct payment command, request, payment confirmation, invoice discussion, or payment issue.
2. **Keep** financial discussions or budget talks that do NOT have direct payment intent, request, action, invoice, or issue.
3. Maintain original message text exactly as provided
4. Exclude messages that have no payment/financial context whatsoever
5. When in doubt, exclude the message

## How to Identify Payment-Related Messages to be Filtered Out:
1. Intent: Everything that indicates a payment intention, plan, or consideration. 
2. Request: Everything that indicates a payment is needed, requested, or will be required.
3. Action: Everything that indicates a payment is being made, occured, or received.
4. Invoice: Everything that indicates a bill, invoice, or receipt is being sent, referenced, or discussed.
5. Issue: Everything that indicates a payment problem, dispute, failure, or retry.
6. Payment Meta/Setup WITH intent: Everything that indicates payment setup, linking card and need to pay or complete a transaction.

## Examples:
**Keep**: "Are you coming to the party tonight?"
**Filter Out**: "Can you send me $20 for the tickets?"
**Keep**: "Did you see the game last night?"
**Filter Out**: "paid you back for lunch!"

=> **Output**: // remain only non-payment related messages
Are you coming to the party tonight?
Did you see the game last night?
This shirt is cheap
'''

transform_text_to_non_payment_relate = '''
You are a specialist in simulating payment chat messages. 
Given a list of message, each belong to a TOTALLY separated conversation in various payment-related contexts such as Payment chatbot, payer, payee, buying, rent, selling, ...

Your task is suggest a new message as a repsonse, reply or a consecutive message for continue the conversation,

## Important Notes:
- Each message provided is separately belong to different conversations
- Keep the original context, tone, relationship.
- Try to mention the mentioned object, things, topic of original message
- A non-payment-related message is `general chat`, `random message`, `financial discussion`, `budget talk, complementary`, `product/service discussion`, etc
- A non-payment-related message is NOT a `transfered confirmation`, `money transfer`, `pay`, `buy`, `request money`, `transfer money issue`, `purchase` message
- `pay attention`, `pay a visit`, `pay respect`, etc is NOT payment-related

## Rules for converting to non payment-related messages:
1. Try keeping emotional tone as the original message
2. Maintain same relationship context (friends/family/business)
3. Sound natural - how actually a human will typing in a chat app
4. Keep exactly the original message of it is already NOT related to a payment

5. If the original message is related to payment, transform the message into a NON-PAYMENT message
- Financial Discussion: dicussion about money, budget, financial planning, price but have no intention on paying it.
- Product/Service Discussion: discussion about product, service, features, quality, delivery, but have no intention on paying it.
- General Chat: general chat, casual talk, social talk, personal update, but have NO spending money.

## Example Output messages (not related to payment):
$200 is not too much
I haven't been back on it
Those are such good breeds. I want to get an English Bulldog
Did you get the tickets?
You lost the money I lent you ??
The shop haven't opened yet
TThe shop didn't give me the invoice when I bought the stuff
I just bought a new shirt — does it look cute?

## Invalid Output messages (related to payment):
Send Alice $200 now
I just paid you back for lunch
Have you sent the money yet?
Did you receive the payment?

## Examples:
**Seed**: "we have seen alligators in the waters that I turned over in. it was very scary. i haven't been back on it"
(Keep exactly the same because it is not payment-related)

**Seed**: "I just sent you $200"  
**Transformed**: "$200 is all I have right now"
'''

label_payment_intent = '''
You are a specialist in identifying payment-related conversations.
Your task is to detect a message to be PAYMENT-RELATED if following is TRUE, otherwise NON-PAYMENT-RELATED.

## Classification Rules:
A text qualifies as **PAYMENT-RELATED** if it includes:
- **Command**: Directing a payment action (“Send $50 to John”)
- **Intent**: Indicating a desire to spend money or make a payment (“I need to pay my rent”)
- **Request**: Asking for payment with clear context (“Send me $20 for dinner”)
- **Pay**: Confirming money sent/received (“Just transferred the rent”)
- **Invoice**: Sharing or referencing a bill/invoice (“Invoice attached for last month’s services”)
- **Issue**: Addressing payment problems/disputes (“Payment failed, retrying now”)
- **Payment Meta/Setup WITH intent**: Specific setup tied to payment purpose (“Link your card to complete purchase”, “Set up autopay for rent”)

A text is **NON-PAYMENT-RELATED** if it:
- Lacks any direct or indirect reference to payment actions, intentions, requests, invoices, issues
- Is purely informational or conversational without payment context
- Discusses Financial topics without payment intent (e.g., budgeting, pricing inquiries)
- Discusses Product/Service topics without payment intent (e.g., features, quality, delivery)
- Is general chat, casual talk, social talk, personal update without payment context

## Examples:
**PAYMENT-RELATED**: "Can you send me $20 for the tickets?"
**NON-PAYMENT-RELATED**: "Are you coming to the party tonight?"
**PAYMENT-RELATED**: "paid you back for lunch!"
**NON-PAYMENT-RELATED**: "Did you see the game last night?"
**PAYMENT-RELATED**: "Send everyone a buck"
**NON-PAYMENT-RELATED**: "This shirt is cheap"

## Output Rules:
Output exactly label `PAYMENT` or `NON-PAYMENT` for each message.
Classify the following JSON format without any additional text, prefix:
[
  {
    "message": "<input_message>",
    "label": "<PAYMENT or NON-PAYMENT>"
  }
]
'''