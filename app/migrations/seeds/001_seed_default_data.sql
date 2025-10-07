-- Seed: Insert default pipeline and label configuration
-- Created: 2025-10-07
-- SQLite Compatible

-- Insert default pipeline
INSERT INTO pipeline (id, name, created_at, updated_at)
VALUES (
    'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11',
    'Payment Classification Pipeline v2',
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
) ON CONFLICT(id) DO NOTHING;

-- Insert default label configuration (PAYMENT_LABEL_V2)
INSERT INTO label_config (id, pipeline_id, name, id2label, label2id, label_explanation, created_at)
VALUES (
    'b1eebc99-9c0b-4ef8-bb6d-6bb9bd380a12',
    'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11',
    'Payment Classification v2',
    '{"0": "payment_request", "1": "payment_intent", "2": "open_intent"}',
    '{"payment_request": 0, "payment_intent": 1, "open_intent": 2}',
    '{"payment_intent": "The user is declaring they will send money right now or in near future OR The user gives an imperative instruction to a system to execute a payment", "payment_request": "The user request to receive money (can be request, inform, force, remind, ...)", "open_intent": "All arbitrary chat messages that are not related to any payment intent"}',
    CURRENT_TIMESTAMP
) ON CONFLICT(id) DO NOTHING;

-- Insert default error buckets (INIT_ERROR_BUCKETS)

-- 1. mistaking_entity_in_openintent
INSERT INTO error_bucket (id, pipeline_id, name, description, examples, data_generation_strategy, created_at, updated_at)
VALUES (
    'c1eebc99-9c0b-4ef8-bb6d-6bb9bd380a13',
    'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11',
    'mistaking_entity_in_openintent',
    'Model mistaking label ''open_intent'' for other intents due to keyword overlap. For example, ''pay attention'', ''send the doc'' being classified as PAYMENT_INTENT instead of OPEN_INTENT.',
    '[]',
    NULL,
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
) ON CONFLICT(id) DO NOTHING;

-- 2. ambiguous_action_keywords
INSERT INTO error_bucket (id, pipeline_id, name, description, examples, data_generation_strategy, created_at, updated_at)
VALUES (
    'c2eebc99-9c0b-4ef8-bb6d-6bb9bd380a14',
    'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11',
    'ambiguous_action_keywords',
    'Messages containing words that appear in multiple intent contexts (pay, send, charge, transfer, request), causing confusion between PAYMENT_INTENT, PAYMENT_REQUEST, and OPEN_INTENT.',
    '[{"msg": "Can you pay attention?", "label": "open_intent"}, {"msg": "Send me the document", "label": "open_intent"}, {"msg": "I''ll pay you back", "label": "open_intent"}, {"msg": "Charge my phone please", "label": "open_intent"}]',
    'Generate messages using payment-related keywords in non-payment contexts: ''pay attention/respects/compliment/tribute'', ''send regards/love/wishes/greetings'', ''charge phone/battery/device'', ''balance work-life/diet'', ''transfer job/data/files'', ''request information/help/update''. Include edge cases where payment words appear but intent is clearly non-payment.',
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
) ON CONFLICT(id) DO NOTHING;

-- 3. pronoun_direction_confusion
INSERT INTO error_bucket (id, pipeline_id, name, description, examples, data_generation_strategy, created_at, updated_at)
VALUES (
    'c3eebc99-9c0b-4ef8-bb6d-6bb9bd380a15',
    'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11',
    'pronoun_direction_confusion',
    'Messages where pronouns (me/you/him/her/them) determine intent direction but model misinterprets, confusing PAYMENT_INTENT with PAYMENT_REQUEST.',
    '[{"msg": "Send me 50 sats", "label": "payment_request"}, {"msg": "I''ll send you 50 sats", "label": "payment_intent"}, {"msg": "Send him 100 dollars", "label": "payment_intent"}, {"msg": "Pay me back tomorrow", "label": "payment_request"}, {"msg": "I''m paying you tonight", "label": "payment_intent"}]',
    'Generate pairs of similar messages differing only in pronouns: ''pay me'' vs ''I''ll pay you'', ''send me'' vs ''send him/her'', ''give me'' vs ''I''ll give you'', ''transfer to me'' vs ''I''ll transfer to them''. Include implicit subjects and third-person scenarios. Mix tenses (present/future/past) to add complexity.',
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
) ON CONFLICT(id) DO NOTHING;

-- 4. long_tail_rare_intents
INSERT INTO error_bucket (id, pipeline_id, name, description, examples, data_generation_strategy, created_at, updated_at)
VALUES (
    'c4eebc99-9c0b-4ef8-bb6d-6bb9bd380a16',
    'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11',
    'long_tail_rare_intents',
    'Uncommon payment scenarios or edge cases rarely seen during training, causing misclassification to OPEN_INTENT.',
    '[{"msg": "Set up a subscription refund", "label": "payment_intent"}, {"msg": "Split the bill three ways", "label": "payment_request"}, {"msg": "Make a donation pledge", "label": "payment_intent"}, {"msg": "Send an escrow payment", "label": "payment_intent"}, {"msg": "Pay-per-view access", "label": "payment_request"}]',
    'Generate rare but valid payment intents: subscription refunds, split bills, recurring payments, donation pledges, tip jars, crowdfunding contributions, escrow payments, micropayments for content, pay-per-view, charitable giving, loan repayments, invoice disputes, partial payments, payment holds, conditional payments (''pay when delivered''), advance payments, payment deferrals.',
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
) ON CONFLICT(id) DO NOTHING;

-- 5. currency_format_variations
INSERT INTO error_bucket (id, pipeline_id, name, description, examples, data_generation_strategy, created_at, updated_at)
VALUES (
    'c5eebc99-9c0b-4ef8-bb6d-6bb9bd380a17',
    'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11',
    'currency_format_variations',
    'Non-standard or diverse currency representations causing misclassification due to unfamiliar formats.',
    '[{"msg": "Give me $50", "label": "payment_request"}, {"msg": "Can you pay me €100", "label": "payment_request"}, {"msg": "Give me back fifty dollars", "label": "payment_request"}, {"msg": "one hundred euros", "label": "payment_request"}, {"msg": "Send 1000 sats", "label": "payment_request"}, {"msg": "I owe you 0.5 BTC, sending now", "label": "payment_intent"}, {"msg": "Transfer 1k USD", "label": "payment_request"}]',
    'Generate diverse currency formats: symbols (''$50'', ''€100'', ''£75''), spelled out (''fifty dollars'', ''one hundred euros''), abbreviations (''50 USD'', ''100 EUR'', ''1k sats'', ''5M sats''), decimal variations (''50.00'', ''50,00'', ''.50'', ''0.5''), slang (''50 bucks'', ''100 quid'', ''a grand''), cryptocurrency units (''1000 sats'', ''0.001 BTC'', ''1000000 msats'', ''1 mBTC''), large numbers (''1M'', ''50k'', ''2.5k''), zero amounts (''0 sats'', ''no charge'').',
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
) ON CONFLICT(id) DO NOTHING;

-- 6. route_hint_technical_terms
INSERT INTO error_bucket (id, pipeline_id, name, description, examples, data_generation_strategy, created_at, updated_at)
VALUES (
    'c6eebc99-9c0b-4ef8-bb6d-6bb9bd380a18',
    'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11',
    'route_hint_technical_terms',
    'Messages including specific routing information, account details, or technical payment terminology (bitcoin lightning network) that confuses classification.',
    '[{"msg": "Send to my savings account", "label": "payment_intent"}, {"msg": "Transfer to my PayPal", "label": "payment_intent"}, {"msg": "Pay via Venmo", "label": "payment_intent"}, {"msg": "Use my business account", "label": "payment_intent"}, {"msg": "Send to checking", "label": "payment_intent"}, {"msg": "Lightning invoice lnbc1...", "label": "payment_intent"}, {"msg": "Bitcoin address bc1q...", "label": "payment_intent"}, {"msg": "through Zelle", "label": "payment_intent"}, {"msg": "via bank transfer", "label": "payment_intent"}, {"msg": "using my debit card", "label": "payment_intent"}, {"msg": "with routing number 123456789", "label": "payment_intent"}, {"msg": "to IBAN code DE89370400440532013000", "label": "payment_intent"}]',
    'Generate messages with routing hints: ''Send to my savings account'', ''Transfer to my PayPal'', ''Pay via Venmo'', ''Use my business account'', ''Send to checking'', ''Lightning invoice lnbc...'', ''Bitcoin address bc1...'', ''through Zelle'', ''via bank transfer'', ''using my debit card'', ''with routing number'', ''to IBAN code''. Mix technical terms with clear payment intents.',
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
) ON CONFLICT(id) DO NOTHING;

-- 7. typos_and_formatting
INSERT INTO error_bucket (id, pipeline_id, name, description, examples, data_generation_strategy, created_at, updated_at)
VALUES (
    'c7eebc99-9c0b-4ef8-bb6d-6bb9bd380a19',
    'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11',
    'typos_and_formatting',
    'Messages with spelling errors, formatting issues, or unconventional text structure causing misclassification.',
    '[{"msg": "Plase send me 50 dolars", "label": "payment_request"}, {"msg": "Can u pay me?", "label": "payment_request"}, {"msg": "I will payy you tomorow", "label": "payment_intent"}, {"msg": "Send   me   100   bucks", "label": "payment_request"}, {"msg": "PAY ME NOW!!!", "label": "payment_request"}, {"msg": "pAy mE 50 dOlLaRs", "label": "payment_request"}, {"msg": "I owe u 20$", "label": "payment_intent"}, {"msg": "send...me...the...money", "label": "payment_request"}, {"msg": "payyy meeee 1000", "label": "payment_request"}]',
    'Generate messages with common typos, text speak, missing words, run-on sentences.',
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
) ON CONFLICT(id) DO NOTHING;

-- 8. slang_and_informal_language
INSERT INTO error_bucket (id, pipeline_id, name, description, examples, data_generation_strategy, created_at, updated_at)
VALUES (
    'c8eebc99-9c0b-4ef8-bb6d-6bb9bd380a20',
    'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11',
    'slang_and_informal_language',
    'Casual, colloquial, or informal expressions for payment actions that model hasn''t learned.',
    '[]',
    'Generate messages with payment slang: ''zap me'', ''throw me'', ''toss over'', ''shoot me'', ''hook me up'', ''spot me'', ''front me'', ''lend me'', ''float me'', ''cover me'', ''chip in'', ''pitch in'', ''kick in'', ''cough up'', ''fork over'', ''shell out'', ''pony up'', ''settle up'', ''square up'', ''owe me'', ''hit me up''. Include regional variations and casual register.',
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
) ON CONFLICT(id) DO NOTHING;

-- 9. implicit_or_vague_requests
INSERT INTO error_bucket (id, pipeline_id, name, description, examples, data_generation_strategy, created_at, updated_at)
VALUES (
    'c9eebc99-9c0b-4ef8-bb6d-6bb9bd380a21',
    'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11',
    'implicit_or_vague_requests',
    'Messages where payment intent is implied or stated indirectly without explicit action keywords',
    '[]',
    'Generate indirect payment messages: ''You still have my 50 bucks'', ''Remember that money?'', ''About that loan...'', ''When are you settling?'', ''Don''t forget what you owe'', ''That dinner isn''t free'', ''I covered lunch yesterday'', ''You got me last time'', ''My treat next time'', ''We''re even now?'', ''Keep the change'', ''It''s on me'', ''Split it?'', ''Go Dutch?''. Use context-dependent phrasing.',
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
) ON CONFLICT(id) DO NOTHING;

-- 10. negations_and_conditionals
INSERT INTO error_bucket (id, pipeline_id, name, description, examples, data_generation_strategy, created_at, updated_at)
VALUES (
    'd0eebc99-9c0b-4ef8-bb6d-6bb9bd380a22',
    'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11',
    'negations_and_conditionals',
    'Messages with negations (don''t pay, won''t send) or conditional statements (if/when/unless) that reverse or modify intent.',
    '[]',
    'Generate messages with negations: ''Don''t pay me yet'', ''I won''t send until'', ''Not paying today'', ''No need to transfer'', ''Can''t pay right now'', ''Shouldn''t send'', ''Never mind the payment''. Include conditionals: ''Pay me if you can'', ''Send when ready'', ''Transfer unless you forgot'', ''Pay after you check'', ''I''ll send once confirmed'', ''Wait before paying'', ''Hold the payment until''. Mix negative and conditional structures.',
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
) ON CONFLICT(id) DO NOTHING;

-- 11. questions_vs_statements
INSERT INTO error_bucket (id, pipeline_id, name, description, examples, data_generation_strategy, created_at, updated_at)
VALUES (
    'd1eebc99-9c0b-4ef8-bb6d-6bb9bd380a23',
    'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11',
    'questions_vs_statements',
    'Confusion between asking about payments (questions) versus actual payment intents/requests (statements).',
    '[]',
    'Generate payment-related questions that are NOT actual intents/requests: ''Can I pay you?'', ''Should I send now?'', ''Do you want me to transfer?'', ''Are you paying?'', ''Will you send?'', ''How much should I pay?'', ''When do I pay?'', ''Where do I send?'', ''Who pays?'', ''Why the charge?'', ''What''s the amount?''. Mix with rhetorical questions and genuine requests to create ambiguity.',
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
) ON CONFLICT(id) DO NOTHING;

-- 12. multi_party_scenarios
INSERT INTO error_bucket (id, pipeline_id, name, description, examples, data_generation_strategy, created_at, updated_at)
VALUES (
    'd2eebc99-9c0b-4ef8-bb6d-6bb9bd380a24',
    'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11',
    'multi_party_scenarios',
    'Messages involving multiple people in payment scenarios, making intent direction unclear.',
    '[]',
    'Generate multi-party payment messages: ''Send John 50 and Mary 30'', ''Split between us three'', ''Everyone pay Alice'', ''They owe me and you'', ''Transfer from Bob to Carol'', ''We all chip in for Sarah'', ''Collect from everyone'', ''Distribute to the team'', ''Pool our money'', ''Group payment to vendor''. Include unclear pronoun references and complex relationships.',
    CURRENT_TIMESTAMP,
    CURRENT_TIMESTAMP
) ON CONFLICT(id) DO NOTHING;
