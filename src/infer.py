import argparse, joblib

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="models/lr_tfidf_model.joblib",
                        help="Path to a saved *tfidf* pipeline model (.joblib)")
    parser.add_argument("--text", type=str, required=True, help="Review text to classify")
    args = parser.parse_args()

    pipe = joblib.load(args.model_path)
    pred = pipe.predict([args.text])[0]
    print(pred)

if __name__ == "__main__":
    main()