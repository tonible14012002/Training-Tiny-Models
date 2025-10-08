from pydantic import BaseModel, Field
from .workflow import Sample
import typing as t


class CategorizeResult(BaseModel):
    bucket: str = Field(description="The name of the error bucket the test case is categorized into")
    reason: str = Field(description="A brief explanation of why the test case was categorized into this bucket")

class AnalyzeTestCase(BaseModel):
    sample: Sample
    predicted: str
    prob: float
    
class ErrorBucket(BaseModel):
    name: str = Field(description="Unique name of the error bucket that briefly summarizes the error type")
    description: str = Field(description="Detailed description of the error bucket, explaining the common characteristics of errors that fall into this category")
    examples: list[Sample] = Field(default=[])
    data_generation_strategy: t.Optional[str] = Field(description="Strategy or guidelines for generating additional data samples that fit into this error bucket", default=None)

class LLmErrorAnalysis(BaseModel):
    error_buckets: t.List[str] = Field(description="List of error bucket names", default=[])

INIT_ERROR_BUCKETS = {
    "mistaking_entity_in_openintent": ErrorBucket(
        name="mistaking_entity_in_openintent",
        description="Model mistaking label 'open_intent' for other intents due to keyword overlap. For example, 'pay attention', 'send the doc' being classified as PAYMENT_INTENT instead of OPEN_INTENT.",
    ),
        
    "ambiguous_action_keywords": ErrorBucket(
        name="ambiguous_action_keywords",
        description="Messages containing words that appear in multiple intent contexts (pay, send, charge, transfer, request), causing confusion between PAYMENT_INTENT, PAYMENT_REQUEST, and OPEN_INTENT.",
        examples=[
            Sample(msg="Can you pay attention?", label="open_intent"),
            Sample(msg="Send me the document", label="open_intent"),
            Sample(msg="I'll pay you back", label="open_intent"),
            Sample(msg="Charge my phone please", label="open_intent"),
        ],
        data_generation_strategy="Generate messages using payment-related keywords in non-payment contexts: 'pay attention/respects/compliment/tribute', 'send regards/love/wishes/greetings', 'charge phone/battery/device', 'balance work-life/diet', 'transfer job/data/files', 'request information/help/update'. Include edge cases where payment words appear but intent is clearly non-payment.",
    ),
    
    "pronoun_direction_confusion": ErrorBucket(
        name="pronoun_direction_confusion",
        description="Messages where pronouns (me/you/him/her/them) determine intent direction but model misinterprets, confusing PAYMENT_INTENT with PAYMENT_REQUEST.",
        examples=[
            Sample(msg="Send me 50 sats", label="payment_request"),
            Sample(msg="I'll send you 50 sats", label="payment_intent"),
            Sample(msg="Send him 100 dollars", label="payment_intent"),
            Sample(msg="Pay me back tomorrow", label="payment_request"),
            Sample(msg="I'm paying you tonight", label="payment_intent"),
        ],
        data_generation_strategy="Generate pairs of similar messages differing only in pronouns: 'pay me' vs 'I'll pay you', 'send me' vs 'send him/her', 'give me' vs 'I'll give you', 'transfer to me' vs 'I'll transfer to them'. Include implicit subjects and third-person scenarios. Mix tenses (present/future/past) to add complexity.",
    ),
    
    "long_tail_rare_intents": ErrorBucket(
        name="long_tail_rare_intents",
        description="Uncommon payment scenarios or edge cases rarely seen during training, causing misclassification to OPEN_INTENT.",
        examples=[
            Sample(msg="Set up a subscription refund", label="payment_intent"),
            Sample(msg="Split the bill three ways", label="payment_request"),
            Sample(msg="Make a donation pledge", label="payment_intent"),
            Sample(msg="Send an escrow payment", label="payment_intent"),
            Sample(msg="Pay-per-view access", label="payment_request"),
        ],
        data_generation_strategy="Generate rare but valid payment intents: subscription refunds, split bills, recurring payments, donation pledges, tip jars, crowdfunding contributions, escrow payments, micropayments for content, pay-per-view, charitable giving, loan repayments, invoice disputes, partial payments, payment holds, conditional payments ('pay when delivered'), advance payments, payment deferrals.",
    ),
    
    "currency_format_variations": ErrorBucket(
        name="currency_format_variations",
        description="Non-standard or diverse currency representations causing misclassification due to unfamiliar formats.",
        examples=[
            Sample(msg="Give me $50", label="payment_request"),
            Sample(msg="Can you pay me €100", label="payment_request"),
            Sample(msg="Give me back fifty dollars", label="payment_request"),
            Sample(msg="one hundred euros", label="payment_request"),
            Sample(msg="Send 1000 sats", label="payment_request"),
            Sample(msg="I owe you 0.5 BTC, sending now", label="payment_intent"),
            Sample(msg="Transfer 1k USD", label="payment_request"),
        ],
        data_generation_strategy="Generate diverse currency formats: symbols ('$50', '€100', '£75'), spelled out ('fifty dollars', 'one hundred euros'), abbreviations ('50 USD', '100 EUR', '1k sats', '5M sats'), decimal variations ('50.00', '50,00', '.50', '0.5'), slang ('50 bucks', '100 quid', 'a grand'), cryptocurrency units ('1000 sats', '0.001 BTC', '1000000 msats', '1 mBTC'), large numbers ('1M', '50k', '2.5k'), zero amounts ('0 sats', 'no charge').",
    ),
    
    "route_hint_technical_terms": ErrorBucket(
        name="route_hint_technical_terms",
        description="Messages including specific routing information, account details, or technical payment terminology (bitcoin lightning network) that confuses classification.",
        examples=[
            Sample(msg="Send to my savings account", label="payment_intent"),
            Sample(msg="Transfer to my PayPal", label="payment_intent"),
            Sample(msg="Pay via Venmo", label="payment_intent"),
            Sample(msg="Use my business account", label="payment_intent"),
            Sample(msg="Send to checking", label="payment_intent"),
            Sample(msg="Lightning invoice lnbc1...", label="payment_intent"),
            Sample(msg="Bitcoin address bc1q...", label="payment_intent"),
            Sample(msg="through Zelle", label="payment_intent"),
            Sample(msg="via bank transfer", label="payment_intent"),
            Sample(msg="using my debit card", label="payment_intent"),
            Sample(msg="with routing number 123456789", label="payment_intent"),
            Sample(msg="to IBAN code DE89370400440532013000", label="payment_intent"),
        ],
        data_generation_strategy="Generate messages with routing hints: 'Send to my savings account', 'Transfer to my PayPal', 'Pay via Venmo', 'Use my business account', 'Send to checking', 'Lightning invoice lnbc...', 'Bitcoin address bc1...', 'through Zelle', 'via bank transfer', 'using my debit card', 'with routing number', 'to IBAN code'. Mix technical terms with clear payment intents.",
    ),
    
    "typos_and_formatting": ErrorBucket(
        name="typos_and_formatting",
        description="Messages with spelling errors, formatting issues, or unconventional text structure causing misclassification.",
        examples=[
            Sample(msg="Plase send me 50 dolars", label="payment_request"),
            Sample(msg="Can u pay me?", label="payment_request"),
            Sample(msg="I will payy you tomorow", label="payment_intent"),
            Sample(msg="Send   me   100   bucks", label="payment_request"),
            Sample(msg="PAY ME NOW!!!", label="payment_request"),
            Sample(msg="pAy mE 50 dOlLaRs", label="payment_request"),
            Sample(msg="I owe u 20$", label="payment_intent"),
            Sample(msg="send...me...the...money", label="payment_request"),
            Sample(msg="payyy meeee 1000", label="payment_request"),
        ],
        data_generation_strategy="Generate messages with common typos, text speak, missing words, run-on sentences.",
    ),
    
    "slang_and_informal_language": ErrorBucket(
        name="slang_and_informal_language",
        description="Casual, colloquial, or informal expressions for payment actions that model hasn't learned.",
        examples=[],
        data_generation_strategy="Generate messages with payment slang: 'zap me', 'throw me', 'toss over', 'shoot me', 'hook me up', 'spot me', 'front me', 'lend me', 'float me', 'cover me', 'chip in', 'pitch in', 'kick in', 'cough up', 'fork over', 'shell out', 'pony up', 'settle up', 'square up', 'owe me', 'hit me up'. Include regional variations and casual register.",
    ),
    
    "implicit_or_vague_requests": ErrorBucket(
        name="implicit_or_vague_requests",
        description="Messages where payment intent is implied or stated indirectly without explicit action keywords",
        examples=[],
        data_generation_strategy="Generate indirect payment messages: 'You still have my 50 bucks', 'Remember that money?', 'About that loan...', 'When are you settling?', 'Don't forget what you owe', 'That dinner isn't free', 'I covered lunch yesterday', 'You got me last time', 'My treat next time', 'We're even now?', 'Keep the change', 'It's on me', 'Split it?', 'Go Dutch?'. Use context-dependent phrasing.",
    ),
    
    "negations_and_conditionals": ErrorBucket(
        name="negations_and_conditionals",
        description="Messages with negations (don't pay, won't send) or conditional statements (if/when/unless) that reverse or modify intent.",
        examples=[],
        data_generation_strategy="Generate messages with negations: 'Don't pay me yet', 'I won't send until', 'Not paying today', 'No need to transfer', 'Can't pay right now', 'Shouldn't send', 'Never mind the payment'. Include conditionals: 'Pay me if you can', 'Send when ready', 'Transfer unless you forgot', 'Pay after you check', 'I'll send once confirmed', 'Wait before paying', 'Hold the payment until'. Mix negative and conditional structures.",
    ),
    
    "questions_vs_statements": ErrorBucket(
        name="questions_vs_statements",
        description="Confusion between asking about payments (questions) versus actual payment intents/requests (statements).",
        examples=[],
        data_generation_strategy="Generate payment-related questions that are NOT actual intents/requests: 'Can I pay you?', 'Should I send now?', 'Do you want me to transfer?', 'Are you paying?', 'Will you send?', 'How much should I pay?', 'When do I pay?', 'Where do I send?', 'Who pays?', 'Why the charge?', 'What's the amount?'. Mix with rhetorical questions and genuine requests to create ambiguity.",
    ),
    
    "multi_party_scenarios": ErrorBucket(
        name="multi_party_scenarios",
        description="Messages involving multiple people in payment scenarios, making intent direction unclear.",
        examples=[],
        data_generation_strategy="Generate multi-party payment messages: 'Send John 50 and Mary 30', 'Split between us three', 'Everyone pay Alice', 'They owe me and you', 'Transfer from Bob to Carol', 'We all chip in for Sarah', 'Collect from everyone', 'Distribute to the team', 'Pool our money', 'Group payment to vendor'. Include unclear pronoun references and complex relationships.",
    ),
}