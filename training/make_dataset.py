import json
import random
import datetime

def main(payment_file=None, unclean_non_payment_file = None, non_payment_file=None, non_payment_trick_file=None, output_file=None):
    assert payment_file is not None
    assert non_payment_file is not None
    assert output_file is not None
    assert non_payment_trick_file is not None

    # load json file
    with open(payment_file, "r") as f:
        payment_data = json.load(f)
        payment_data = [item for item in payment_data if len(item) >= 6]
    with open(non_payment_file, "r") as f:
        non_payment_data = json.load(f)
        non_payment_data = [item for item in non_payment_data if len(item) >= 6]
    with open(unclean_non_payment_file, "r") as f:
        unclean_non_payment_data = json.load(f)
        unclean_non_payment_data = random.sample([item for item in unclean_non_payment_data if len(item) >= 6], 5000)
    with open(non_payment_trick_file, "r") as f:
        non_payment_trick_data = json.load(f)
        non_payment_trick_data = [item for item in non_payment_trick_data if len(item) >= 6]
    
    # shuffle data
    random.shuffle(payment_data)
    random.shuffle(non_payment_data)
    random.shuffle(non_payment_trick_data)
    random.shuffle(unclean_non_payment_data)

    non_payment_data += unclean_non_payment_data
    
    # Filter out all item less than 10 characters
    total = min(len(payment_data) + len(non_payment_data) + len(non_payment_trick_data), 15000)  # or any cap you want
    payment_target = int(total * 0.35)
    non_payment_target = int(total * 0.65)

    non_payment_random_target = int(non_payment_target * 0.5)
    non_payment_trick_target = non_payment_target - non_payment_random_target

    # random sample to match target distribution
    sampled_payment = random.sample(payment_data, min(payment_target, len(payment_data)))
    sampled_non_payment = random.sample(non_payment_data, min(non_payment_target, len(non_payment_data)))
    sampled_non_payment += random.sample(non_payment_trick_data, min(non_payment_trick_target, len(non_payment_trick_data)))

    def label_data(text, label):
        return {"text": text, "label": label}

    payment_samples = [label_data(t, 1) for t in sampled_payment]
    non_payment_samples = [label_data(t, 0) for t in sampled_non_payment]

    final_data = payment_samples + non_payment_samples
    random.shuffle(final_data)  # shuffle for better distribution

    with open(output_file, "w") as f:
        json.dump(final_data, f, indent=2)

if __name__ == "__main__":
    now = datetime.datetime.now().timestamp()
    main(
        "payments.json",
        "clean_messages.json",
        "non-payment.json",
        "non-payment-stricky.json",
        f"final_{int(now)}.json")
