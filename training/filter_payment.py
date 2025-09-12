import json
import re

# Payment keywords to detect
PAYMENT_WORDS = [
    'pay', 'paid', 'payment', 'send', 'sent', 'transfer', 'money', 'cash', 'dollar', 'cost', 'price', 
    'bill', 'invoice', 'charge', 'fee', 'owe', 'debt', 'lend', 'loan', 'buy', 'bought', 'sell', 
    'sold', 'purchase', 'order', 'venmo', 'paypal', 'card', 'bank', 'account', 'refund', 'receipt',
    'expense', 'budget', 'cheap', 'expensive', 'free', 'discount', 'sale', 'shop', 'store', 'checkout',
    'rent', 'salary', 'bonus', 'tip', 'split', 'cover', 'afford', 'save', 'spend', 'subscription',
    'withdraw', 'deposit', 'fund', 'balance', 'transaction', 'currency', 'exchange', 'forex',
    'bitcoin', 'crypto', 'ethereum', 'wallet', 'atm', 'statement', 'overdraft', 'interest', 'mortgage',
    'installment', 'downpayment', 'rebate', 'coupon', 'voucher', 'giftcard', 'prepaid', 'directdebit', 'standingorder',
    'wiretransfer', 'swift', 'iban', 'routingnumber', 'accountnumber', 'creditlimit', 'due', 'minimumpayment',
    'latefee', 'overlimit', 'cashadvance', 'balanceinquiry', 'pin', 'cvv', 'expirydate', 'statementdate',
    'paymentdate', 'payee', 'payer', 'remittance', 'settlement', 'clearing', 'reconciliation', 'chargeback',
]

# Currency symbols
CURRENCY_SYMBOLS = ['$', '€', '£', '¥', '₹', '¢', '₩', '₽', '₺', '₪', '₫', '₴', '₦', '₱']

def contains_payment_words(message):
    """Check if message contains payment-related words"""
    if not message:
        return False
    
    message_lower = message.lower()
    
    # Check for currency symbols
    for symbol in CURRENCY_SYMBOLS:
        if symbol in message:
            return True
    
    # Check for payment keywords
    words = re.findall(r'\b\w+\b', message_lower)
    for word in words:
        if word in PAYMENT_WORDS:
            return True
    
    # Check for amount patterns like "20 dollars" or "$50"
    amount_patterns = [r'\$\d+', r'\d+\s*dollars?', r'\d+\s*bucks?', r'\d+\s*euros?']
    for pattern in amount_patterns:
        if re.search(pattern, message_lower):
            return True
    
    return False

# Load your dataset
input_file = 'training_data/human_sms.json'  # Change this to your file name
output_clean = 'clean_messages.json'
output_payment = 'payment_messages.json'

# Read the dataset
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

import random

data = random.sample(data, min(100000, len(data)))
# Separate messages
clean_messages = []
payment_messages = []

for message in data:
    if contains_payment_words(message):
        payment_messages.append(message)
    else:
        clean_messages.append(message)

# Save results
with open(output_clean, 'w', encoding='utf-8') as f:
    json.dump(clean_messages, f, indent=2, ensure_ascii=False)

with open(output_payment, 'w', encoding='utf-8') as f:
    json.dump(payment_messages, f, indent=2, ensure_ascii=False)

# Print results
print(f"Original messages: {len(data)}")
print(f"Clean messages (no payment): {len(clean_messages)}")
print(f"Payment messages: {len(payment_messages)}")
print(f"Payment percentage: {len(payment_messages)/len(data)*100:.1f}%")
print(f"\nSaved to:")
print(f"- Clean messages: {output_clean}")
print(f"- Payment messages: {output_payment}")