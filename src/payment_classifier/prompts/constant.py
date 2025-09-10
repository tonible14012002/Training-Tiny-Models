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
Generate natural messages that belong to payment-related conversations, covering everything from direct money transfers to payment discussions, commerce interactions, and financial coordination.

## Conversation Contexts
Messages can belong to conversations across diverse settings: social gatherings, business meetings, e-commerce platforms, casual chats, formal communications, customer service interactions, marketplace negotiations, family discussions, workplace coordination, and any other realistic scenario where payment topics naturally arise.

**Be creative beyond these examples - include any authentic context where payment discussions occur.**

## User Relationships (Examples - Not Exhaustive)
Auto-reply systems, chatbots, friends, roommates, family members, romantic partners, spouses, employers and employees, colleagues, freelancers and clients, buyers and sellers, tenants and landlords, customers and service providers, passengers and drivers, students and tutors, patients and healthcare professionals, club members with organizers, charity donors with organizations, community members with associations, streamers with supporters, travelers with agents or hosts, neighbors, acquaintances, strangers, and any other realistic relationship dynamic.

**Don't limit yourself to these examples - include any authentic relationship where payment conversations naturally occur.**

## Payment-Related Topics (Examples - Expand Creatively)
Direct payments, scam attempts, cryptocurrency transactions, splitting dinner bills, covering entertainment costs, sharing rent and utilities, sending allowances, paying tuition, covering shared trips, handling household expenses, surprise gifts, salary payments, bonuses, meal cost splitting, project payments, home repair charges, item purchases, rent transfers, driver tips, ride-sharing fares, tutoring fees, consultation payments, membership fees, ticket purchases, charity donations, repair cost sharing, tour guide payments, hotel bookings, in-app purchases, receiving tips, follower support, emergency lending, payment confirmations, pending payment notifications, invoice details, bill settlements, low balance alerts, subscription renewals, failed transactions, refund confirmations, transaction history updates, monthly statements, and countless other payment scenarios.

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