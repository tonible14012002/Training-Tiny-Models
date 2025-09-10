import json
import random

def main(payment_file=None, non_payment_file=None, output_file=None):
    assert payment_file is not None
    assert non_payment_file is not None
    assert output_file is not None

    # load json file
    with open(payment_file, "r") as f:
        payment_data = json.load(f)
    with open(non_payment_file, "r") as f:
        non_payment_data = json.load(f)

    total = min(len(payment_data) + len(non_payment_data), 15000)  # or any cap you want
    payment_target = int(total * 0.35)
    non_payment_target = int(total * 0.65)

    # random sample to match target distribution
    sampled_payment = random.sample(payment_data, min(payment_target, len(payment_data)))
    sampled_non_payment = random.sample(non_payment_data, min(non_payment_target, len(non_payment_data)))

    def label_data(text, label):
        return {"text": text, "label": label}

    payment_samples = [label_data(t, 1) for t in sampled_payment]
    non_payment_samples = [label_data(t, 0) for t in sampled_non_payment]

    final_data = payment_samples + non_payment_samples
    random.shuffle(final_data)  # shuffle for better distribution

    with open(output_file, "w") as f:
        json.dump(final_data, f, indent=2)

if __name__ == "__main__":
    main("./training_data/payment_intent_generation/1757500670/all_batches_consolidated_1757500670.json", "./training_data/nonpayment_generation/1757502336/all_batches_consolidated_1757502336.json", "final.json")
