
import sys
import io
import csv
import click

@click.command()
@click.option("--csv-path", required=True, help="Path to the analysis CSV file.")
def main(csv_path):
    with open(csv_path, 'r') as f:
        # Read the whole file, as the rest of the script expects a string
        csv_data = f.read()

    # Use io.StringIO to treat the string data as a file
    csv_file = io.StringIO(csv_data.strip())
    reader = csv.DictReader(csv_file)

    incorrect_predictions = []
    correct_predictions = []

    for row in reader:
        true_label = int(row['true_label'])
        predicted_label = int(row['predicted_label'])
        
        # The confidence is the probability of the predicted class
        confidence = float(row[f'prob_class_{predicted_label}'])
        
        if true_label == predicted_label:
            correct_predictions.append(confidence)
        else:
            incorrect_predictions.append(confidence)

    # --- Analysis ---
    total_samples = len(correct_predictions) + len(incorrect_predictions)
    accuracy = len(correct_predictions) / total_samples if total_samples > 0 else 0

    avg_confidence_correct = sum(correct_predictions) / len(correct_predictions) if correct_predictions else 0
    avg_confidence_incorrect = sum(incorrect_predictions) / len(incorrect_predictions) if incorrect_predictions else 0

    print("--- Analysis of Baseline Model on Minority Classes ---")
    print(f"Total minority samples analyzed: {total_samples}")
    print(f"Accuracy on these minority classes: {accuracy:.2%}")
    print("-" * 20)
    print(f"Average confidence for CORRECT predictions: {avg_confidence_correct:.2%}")
    print(f"Average confidence for INCORRECT predictions: {avg_confidence_incorrect:.2%}")
    print("-" * 20)

    # Conclusion
    if avg_confidence_incorrect > 0.75:
        print("Conclusion: The model is often 'confidently wrong' about minority classes.")
        print("The average confidence of incorrect predictions is high, suggesting a simple confidence threshold might not work well.")
    elif avg_confidence_incorrect > 0.5:
        print("Conclusion: The model is somewhat confident in its wrong predictions.")
        print("A simple confidence threshold might be tricky to tune and may not be very effective.")
    else:
        print("Conclusion: The model is often 'uncertain' when it misclassifies minority classes.")
        print("The average confidence of incorrect predictions is low. Your hypothesis is supported.")
        print("A confidence-based threshold approach is feasible.")

if __name__ == "__main__":
    main()
