from app.core.schemas import PAYMENT_LABEL
import os

if __name__ == "__main__":
    msgs = [
        {"msg": "I'll send you 4k VND now =))", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "Hi, just remembered that I still owe you 3$ =)), wait a moment", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "It should be 13.4$ for each person right? give me the account number", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "Haven’t received the money yet, try again.", "label": PAYMENT_LABEL.PAYMENT_REQUEST},
        {"msg": "I forgot to give you the money for the medicine yesterday, I’ll transfer it now.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "Can you send me 200k VND for lunch?", "label": PAYMENT_LABEL.PAYMENT_REQUEST},
        {"msg": "Take the money to buy this laptop.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "Transfer Nam 300$ for coffee.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "Pay the Grab fare bro =)) it was quite a long ride.", "label": PAYMENT_LABEL.PAYMENT_REQUEST},
        {"msg": "I’ll pay you back next time we meet.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "Can you lend me 2 million VND?", "label": PAYMENT_LABEL.PAYMENT_REQUEST},
        {"msg": "It’s 300 Ird in total, wanna pay now?", "label": PAYMENT_LABEL.PAYMENT_REQUEST},
        {"msg": "Let me transfer you 200k right away.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "Okay, I’ll send it via Momo now.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "I’m opening the banking app, wait a moment, I’ll transfer.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "I’ll send it directly to your account.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "I’ll pay right now, wait to receive it.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "I’ll transfer the Grab fee to you immediately.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "Alright, I’ll send 100k now.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "What’s your Momo number? I’ll transfer right away.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "I’m at the ATM, let me withdraw then send it.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "Okay, I’ll transfer the tuition fee now.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "I’ll transfer 500k in advance to be sure.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "Give me your number, I’ll send it immediately.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "Wait a second, I’m confirming the OTP to send.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "I’ll transfer 1 million to your card now.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "I’ll send money for lunch right away.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "I’ll pay everything, I’m logging into banking now.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "I’ll send the parking fee immediately.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "Okay, I’ll transfer it now.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "I’m checking the account number, will send in a minute.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "Sending $50 right away so I don’t forget.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "Wait a bit, I’ll send 200k.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "I’ll transfer via Zalopay, it’s easier.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "I’ll send the 2 million deposit now.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "I’ll pay this month’s electricity bill now.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "I’ll pay for the movie tickets right away.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "I’ll settle the community center’s invoice right now before anyone accuses the board of being careless with our commitments.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "Alright, I’m sending over the training pitch rental fee immediately—got to keep sharp if I want to honor Juve’s spirit.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "Let me process the article access fee right now; I don’t want funding delays to hold up my dissertation on pharmaceutical patents.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "I’ll transfer the event deposit this moment—it’s vital our budget reallocation meeting goes ahead without distractions.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "I’ll pay the publishing fee straight away; this research on the lost language deserves to reach readers without bureaucratic holdups.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "Let me send my subscription renewal now—after yesterday’s outage, I’m not risking another disruption to my WhatsApp groups.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "We’ll transfer the amount for the personalized wine bottles today, so the engraver can start before the harvest season keeps us too busy.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "I’ll pay the maintenance fee right now; these library computers can’t go another day without updates.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "I’ll process the tuition for tomorrow’s virtual lesson immediately—better to have it squared away before we dive into regressions.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "Let me wire the coverage premium now; I’d rather finalize risk tolerance discussions without outstanding payments hanging over us.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "I’ll transfer my consultancy retainer today, so I can focus purely on your cost-reduction strategies without financial distractions.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "Alright, I’ll send the ticket money immediately—UNAM’s matches can’t wait, and I won’t miss this one for anything.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "I’ll pay for the dinner reservations in Antwerp now, so I can focus on sharing better tips with our guests when they arrive.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "I’ll settle the fuel charge this instant—it’s only fair before I chart out tomorrow’s school trip route.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "Fine, I’ll push through the license renewal payment right now—better than joking about servers catching fire later.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "I’ll send the rental fee right away; Jack White’s gear needs clean sound, and I won’t risk a delay backstage.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "I’ll cover the court filing fee this minute; it’s crucial the motion gets filed without procedural hiccups.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "Let me buy the signed edition now; your words touched me deeply, and I want to support your work right away.", "label": PAYMENT_LABEL.PAYMENT_INTENT},
        {"msg": "I’ll authorize the settlement payment immediately—our reputation doesn’t need the shadow of unpaid obligations.", "label": PAYMENT_LABEL.PAYMENT_INTENT}
    ]


    # save to files
    import json
    os.makedirs("seed", exist_ok=True)
    with open("seed/human_seed.json", "w") as f:
        json.dump(msgs, f, indent=2, ensure_ascii=False)