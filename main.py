from model import learn, compute_accuracy


def main():
    # Paths relative to this file
    in_train = "data/in-domain-train"
    out_train = "data/out-domain-train"
    in_eval = "data/in-domain-eval"
    out_eval = "data/out-domain-eval"

    # Train the model
    model = learn(
        in_domain_path=in_train,
        out_domain_path=out_train,
        num_epochs=15,   # 5 epochs ~ 9 min, so 15 is still ok
        batch_size=32,
        lr=1e-3,
        alpha=0.2,
    )

    # Evaluate on in-domain and out-domain evaluation sets
    acc_in = compute_accuracy(in_eval, model)
    acc_out = compute_accuracy(out_eval, model)

    print(f"In domain accuracy: {acc_in:.4f}")
    print(f"Out domain accuracy: {acc_out:.4f}")


if __name__ == "__main__":
    main()
